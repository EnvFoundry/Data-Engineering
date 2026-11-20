# Native indexed reading

`ai-data-read` streams published mixture and packed-sequence shards into an
external trainer or process. It reads the manifest and byte-offset indexes in
Rust, then copies each selected JSONL row unchanged. Python owns preparation,
composition and packing; the native reader shares their file format and cursor.

## Build and consume

Install [Rust](https://rust-lang.org/tools/install/) if needed, then build from
the repository root. Python installation does not require Rust.

```sh
cargo build --release --locked --manifest-path rust/Cargo.toml
rust/target/release/ai-data-read info runs/training-example/mixture
rust/target/release/ai-data-read read runs/training-example/mixture \
  --rank 1 --world-size 4 --drop-remainder \
  --limit 64 --state-out reader-state.json > worker-batch.jsonl
```

On Windows the executable is `rust/target/release/ai-data-read.exe`. The same
command accepts the output of `ai-data pack`. Mixture rows contain the original
record and its source; packed rows contain aligned input/target IDs, masks and
segment metadata. Output preserves every byte of those rows, including their
trailing newline. The reader does not tokenize, decode payloads or alter masks.

## Partition and continuation

Global row `rank + consumed * world_size` is next. `--drop-remainder` drops only
the global tail, giving ranks equal counts. Without it, all rows are retained
and rank lengths can differ by one. `--offset` skips a number of rows assigned
to this rank, rather than a global number of rows.

```sh
rust/target/release/ai-data-read read runs/training-example/mixture \
  --rank 1 --world-size 4 --drop-remainder \
  --state-in reader-state.json --limit 64 --state-out next-state.json \
  > next-batch.jsonl
```

State files use the exact Python `MixtureReader.state_dict()` /
`PackedReader.state_dict()` fields. Python can continue from a native cursor and
the native reader can continue from Python. Release, rank, worker count and tail
policy must agree. `--offset` and `--state-in` cannot be combined.

The CLI publishes the new state after all emitted rows have flushed to stdout.
A failed output pipe leaves an existing state file unchanged and returns a
nonzero exit code. This is a **delivery cursor**: the trainer saves it with the
corresponding optimizer state after consuming the batch. If a trainer prefetches,
its outstanding queue belongs in that checkpoint as well.

Only one shard's data/index pair and a reusable row buffer stay open. Memory
scales with shard metadata and the largest selected row, not the full dataset.
The reader consumes complete releases produced by the Python operations; it
checks index ranges and file sizes while reading, without decoding records or
recomputing release contents. Use the Python preparation path for record-level
transformations and interpretation.

## Embed in a Rust consumer

The crate also exposes `IndexedReader` and `ReaderState`:

```rust,no_run
use ai_data_reader::IndexedReader;

fn consume() -> ai_data_reader::Result<()> {
    let mut reader = IndexedReader::open("runs/training-example/mixture", 1, 4, true)?;
    let mut row = Vec::new();
    while reader.read_next(&mut row)? {
        // Send this row to the consumer, or decode only the fields it needs.
        let checkpoint = reader.state();
        // Persist checkpoint alongside the corresponding consumer progress.
    }
    Ok(())
}
```

`read_next` advances after one complete row has been read. It does no background
prefetch. `load_state` and `seek_rank_offset` reposition the same reader.

Run the cross-language example after building:

```sh
cargo test --locked --manifest-path rust/Cargo.toml
python rust/compare_python.py
```

The example builds real Python releases from authored Unicode records, compares
byte-for-byte output across ranks and tail policies, and resumes Python → Rust →
Python. It also packs supplied token IDs and reads those blocks through both
implementations. These operations establish data-flow compatibility; model
throughput depends on decoding, collation and the trainer.
