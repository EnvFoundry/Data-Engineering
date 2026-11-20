//! Byte-transparent reading of published indexed mixtures and packed sequences.
//!
//! A reader owns one rank's strided stream. Its checkpoint is interchangeable
//! with Python's `MixtureReader` / `PackedReader`; no prefetch advances its cursor.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Component, Path, PathBuf};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReaderState {
    pub schema_version: String,
    pub release_id: String,
    pub rank: u64,
    pub world_size: u64,
    pub drop_remainder: bool,
    pub consumed: u64,
}

#[derive(Debug, Deserialize)]
struct Shard {
    path: String,
    index_path: String,
    first_index: u64,
    records: u64,
    bytes: u64,
    index_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct Counts {
    records: u64,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: String,
    format: String,
    name: String,
    release_id: String,
    counts: Counts,
    shards: Vec<Shard>,
}

#[derive(Debug, Serialize)]
pub struct ReaderInfo<'a> {
    pub format: &'a str,
    pub name: &'a str,
    pub release_id: &'a str,
    pub records: u64,
    pub rank_records: u64,
    pub shards: usize,
    pub rank: u64,
    pub world_size: u64,
    pub drop_remainder: bool,
    pub consumed: u64,
}

struct OpenShard {
    number: usize,
    data: BufReader<File>,
    index: BufReader<File>,
    data_position: u64,
    index_position: u64,
}

/// Reuses one caller-owned row buffer and one pair of buffered shard files.
/// Memory depends on shard metadata and the largest emitted row, not row count.
pub struct IndexedReader {
    root: PathBuf,
    manifest: Manifest,
    state: ReaderState,
    effective_total: u64,
    current: Option<OpenShard>,
}

fn relative_path(root: &Path, name: &str) -> Result<PathBuf> {
    let relative = Path::new(name);
    if name.is_empty()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
    {
        return Err(invalid("shard paths must be relative to the release directory").into());
    }
    Ok(root.join(relative))
}

fn open_file(root: &Path, name: &str, expected_bytes: u64) -> Result<File> {
    let path = relative_path(root, name)?.canonicalize()?;
    if !path.starts_with(root) {
        return Err(invalid("shard resolves outside its release directory").into());
    }
    let file = File::open(path)?;
    if file.metadata()?.len() != expected_bytes {
        return Err(invalid(format!("shard size differs from the manifest: {name}")).into());
    }
    Ok(file)
}

fn seek_to(reader: &mut BufReader<File>, current: &mut u64, target: u64) -> Result<()> {
    let delta = i128::from(target) - i128::from(*current);
    let delta =
        i64::try_from(delta).map_err(|_| invalid("seek distance exceeds supported range"))?;
    reader.seek_relative(delta)?;
    *current = target;
    Ok(())
}

impl IndexedReader {
    pub fn open(
        path: impl AsRef<Path>,
        rank: u64,
        world_size: u64,
        drop_remainder: bool,
    ) -> Result<Self> {
        if world_size == 0 || rank >= world_size {
            return Err(
                invalid("world_size must be positive and rank must be in [0, world_size)").into(),
            );
        }
        let root = path.as_ref().canonicalize()?;
        let manifest: Manifest = serde_json::from_reader(File::open(root.join("manifest.json"))?)?;
        if manifest.schema_version != "1.0"
            || !matches!(
                manifest.format.as_str(),
                "ai-data-mixture" | "ai-data-packed"
            )
            || manifest.release_id.is_empty()
        {
            return Err(
                invalid("expected an indexed mixture or packed release with schema 1.0").into(),
            );
        }
        let mut total = 0_u64;
        let mut paths = HashSet::new();
        for shard in &manifest.shards {
            let index_bytes = shard
                .records
                .checked_add(1)
                .and_then(|count| count.checked_mul(8))
                .ok_or_else(|| invalid("shard index length overflow"))?;
            if shard.records == 0
                || shard.first_index != total
                || shard.index_bytes != index_bytes
                || shard.bytes == 0
            {
                return Err(
                    invalid("shard counts, ranges or index length are inconsistent").into(),
                );
            }
            for name in [&shard.path, &shard.index_path] {
                relative_path(&root, name)?;
                if !paths.insert(name) {
                    return Err(invalid("shard paths are repeated").into());
                }
            }
            total = total
                .checked_add(shard.records)
                .ok_or_else(|| invalid("release row count overflow"))?;
        }
        if total != manifest.counts.records {
            return Err(invalid("shard rows do not sum to the release count").into());
        }
        let effective_total = if drop_remainder {
            total - total % world_size
        } else {
            total
        };
        let state = ReaderState {
            schema_version: "1.0".into(),
            release_id: manifest.release_id.clone(),
            rank,
            world_size,
            drop_remainder,
            consumed: 0,
        };
        Ok(Self {
            root,
            manifest,
            state,
            effective_total,
            current: None,
        })
    }

    /// Total number assigned to this rank, including already-consumed rows.
    pub fn len(&self) -> u64 {
        if self.effective_total <= self.state.rank {
            0
        } else {
            1 + (self.effective_total - 1 - self.state.rank) / self.state.world_size
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn state(&self) -> ReaderState {
        self.state.clone()
    }

    pub fn load_state(&mut self, state: ReaderState) -> Result<()> {
        let mut expected = self.state();
        expected.consumed = state.consumed;
        if state != expected || state.consumed > self.len() {
            return Err(invalid(
                "reader state belongs to another release/topology or has an invalid cursor",
            )
            .into());
        }
        self.state = state;
        self.current = None;
        Ok(())
    }

    pub fn seek_rank_offset(&mut self, consumed: u64) -> Result<()> {
        let mut state = self.state();
        state.consumed = consumed;
        self.load_state(state)
    }

    pub fn info(&self) -> ReaderInfo<'_> {
        ReaderInfo {
            format: &self.manifest.format,
            name: &self.manifest.name,
            release_id: &self.manifest.release_id,
            records: self.manifest.counts.records,
            rank_records: self.len(),
            shards: self.manifest.shards.len(),
            rank: self.state.rank,
            world_size: self.state.world_size,
            drop_remainder: self.state.drop_remainder,
            consumed: self.state.consumed,
        }
    }

    /// Read one complete original JSONL row. The cursor advances only on success.
    /// `false` means this rank is exhausted. The row is not decoded or rewritten.
    pub fn read_next(&mut self, row: &mut Vec<u8>) -> Result<bool> {
        row.clear();
        if self.state.consumed == self.len() {
            self.current = None;
            return Ok(false);
        }
        let ordinal = self.state.rank + self.state.consumed * self.state.world_size;
        let number = self
            .manifest
            .shards
            .partition_point(|shard| shard.first_index <= ordinal)
            - 1;
        let shard = &self.manifest.shards[number];
        if self.current.as_ref().map(|opened| opened.number) != Some(number) {
            let data = open_file(&self.root, &shard.path, shard.bytes)?;
            let index = open_file(&self.root, &shard.index_path, shard.index_bytes)?;
            self.current = Some(OpenShard {
                number,
                data: BufReader::with_capacity(64 * 1024, data),
                index: BufReader::with_capacity(64 * 1024, index),
                data_position: 0,
                index_position: 0,
            });
        }
        let opened = self.current.as_mut().unwrap();
        let local = ordinal - shard.first_index;
        seek_to(&mut opened.index, &mut opened.index_position, local * 8)?;
        let mut offsets = [0_u8; 16];
        opened.index.read_exact(&mut offsets)?;
        opened.index_position += 16;
        let start = u64::from_le_bytes(offsets[..8].try_into().unwrap());
        let end = u64::from_le_bytes(offsets[8..].try_into().unwrap());
        if start >= end
            || end > shard.bytes
            || (local == 0 && start != 0)
            || (local + 1 == shard.records && end != shard.bytes)
        {
            return Err(invalid(
                "row offsets are outside the shard or do not cover its boundaries",
            )
            .into());
        }
        let length =
            usize::try_from(end - start).map_err(|_| invalid("row exceeds addressable memory"))?;
        seek_to(&mut opened.data, &mut opened.data_position, start)?;
        row.try_reserve(length)
            .map_err(|_| invalid("unable to allocate the indexed row"))?;
        row.resize(length, 0);
        opened.data.read_exact(row)?;
        opened.data_position = end;
        if row.last() != Some(&b'\n') {
            return Err(invalid("indexed rows must end with a newline").into());
        }
        self.state.consumed += 1;
        Ok(true)
    }
}
