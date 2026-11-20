use ai_data_reader::{IndexedReader, ReaderState, Result};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

const HELP: &str = "ai-data-read read RELEASE [options]\n\
ai-data-read info RELEASE [options]\n\
\n\
Stream original JSONL rows from indexed mixtures or packed sequences.\n\
\n\
  --rank N              Reader rank (default 0)\n\
  --world-size N        Number of ranks (default 1)\n\
  --drop-remainder      Give all ranks equal counts by dropping the global tail\n\
  --offset N            Skip N rows assigned to this rank\n\
  --limit N             Emit at most N rows\n\
  --state-in FILE       Restore a Python/Rust reader state (same topology)\n\
  --state-out FILE      Save the next cursor after stdout has flushed\n\
\n\
read writes rows only to stdout; info writes a compact JSON description.\n\
The offset is rank-local. --offset and --state-in are mutually exclusive.\n";

struct Options {
    operation: String,
    path: PathBuf,
    rank: u64,
    world_size: u64,
    drop_remainder: bool,
    offset: Option<u64>,
    limit: Option<u64>,
    state_in: Option<PathBuf>,
    state_out: Option<PathBuf>,
}

fn parse() -> Result<Option<Options>> {
    let mut args = env::args().skip(1);
    let operation = match args.next().as_deref() {
        None | Some("--help" | "-h") => {
            print!("{HELP}");
            return Ok(None);
        }
        Some("--version") => {
            println!("ai-data-read {}", env!("CARGO_PKG_VERSION"));
            return Ok(None);
        }
        Some(value @ ("read" | "info")) => value.to_string(),
        Some(value) => return Err(format!("unknown operation {value:?}; use read or info").into()),
    };
    let path = args.next().ok_or("missing release directory")?;
    if path == "--help" || path == "-h" {
        print!("{HELP}");
        return Ok(None);
    }
    let mut result = Options {
        operation,
        path: path.into(),
        rank: 0,
        world_size: 1,
        drop_remainder: false,
        offset: None,
        limit: None,
        state_in: None,
        state_out: None,
    };
    while let Some(option) = args.next() {
        if option == "--drop-remainder" {
            result.drop_remainder = true;
            continue;
        }
        if option == "--help" || option == "-h" {
            print!("{HELP}");
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {option}"))?;
        match option.as_str() {
            "--rank" => result.rank = value.parse()?,
            "--world-size" => result.world_size = value.parse()?,
            "--offset" => result.offset = Some(value.parse()?),
            "--limit" => result.limit = Some(value.parse()?),
            "--state-in" => result.state_in = Some(value.into()),
            "--state-out" => result.state_out = Some(value.into()),
            _ => return Err(format!("unknown option {option:?}").into()),
        }
    }
    if result.offset.is_some() && result.state_in.is_some() {
        return Err("--offset and --state-in cannot be used together".into());
    }
    if result.operation == "info" && (result.limit.is_some() || result.state_out.is_some()) {
        return Err("--limit and --state-out apply to read".into());
    }
    Ok(Some(result))
}

fn save_state(path: &Path, state: &ReaderState) -> Result<()> {
    let parent = path
        .parent()
        .filter(|part| !part.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or("state output must be a file path")?
        .to_string_lossy();
    let temporary = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    let result = (|| {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        serde_json::to_writer(&mut output, state)?;
        output.write_all(b"\n")?;
        output.flush()?;
        drop(output);
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn run() -> Result<()> {
    let Some(options) = parse()? else {
        return Ok(());
    };
    let mut reader = IndexedReader::open(
        &options.path,
        options.rank,
        options.world_size,
        options.drop_remainder,
    )?;
    if let Some(state) = &options.state_in {
        reader.load_state(serde_json::from_reader(File::open(state)?)?)?;
    }
    if let Some(offset) = options.offset {
        reader.seek_rank_offset(offset)?;
    }
    let stdout = io::stdout();
    let mut output = BufWriter::with_capacity(64 * 1024, stdout.lock());
    if options.operation == "info" {
        serde_json::to_writer(&mut output, &reader.info())?;
        output.write_all(b"\n")?;
    } else {
        let mut row = Vec::new();
        let mut emitted = 0_u64;
        while options.limit.map_or(true, |limit| emitted < limit) && reader.read_next(&mut row)? {
            output.write_all(&row)?;
            emitted += 1;
        }
    }
    output.flush()?;
    if let Some(path) = options.state_out {
        save_state(&path, &reader.state())?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ai-data-read: {error}");
        std::process::exit(1);
    }
}
