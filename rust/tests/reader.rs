use ai_data_reader::IndexedReader;
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new(format: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "ai-data-reader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let mut shards = vec![];
        for first in [0, 4, 8] {
            let count = 4.min(11 - first);
            let mut data = vec![];
            let mut index = vec![];
            for ordinal in first..first + count {
                index.extend_from_slice(&(data.len() as u64).to_le_bytes());
                let text = if ordinal == 5 {
                    "α中🙂".repeat(30_000)
                } else {
                    format!("row {ordinal}")
                };
                data.extend_from_slice(
                    serde_json::to_string(&json!({
                        "sample_index": ordinal, "record": {"text": text}
                    }))
                    .unwrap()
                    .as_bytes(),
                );
                data.push(b'\n');
            }
            index.extend_from_slice(&(data.len() as u64).to_le_bytes());
            let name = format!("shard-{first}.jsonl");
            let index_name = format!("shard-{first}.idx");
            fs::write(path.join(&name), &data).unwrap();
            fs::write(path.join(&index_name), &index).unwrap();
            shards.push(json!({"path": name, "index_path": index_name,
                "first_index": first, "records": count, "bytes": data.len(), "index_bytes": index.len()}));
        }
        fs::write(
            path.join("manifest.json"),
            serde_json::to_vec(&json!({
                "schema_version": "1.0", "format": format, "name": "reader-fixture",
                "release_id": "reader-fixture-1", "counts": {"records": 11}, "shards": shards
            }))
            .unwrap(),
        )
        .unwrap();
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn ordinals(reader: &mut IndexedReader) -> Vec<u64> {
    let mut values = vec![];
    let mut row = vec![];
    while reader.read_next(&mut row).unwrap() {
        let value: serde_json::Value = serde_json::from_slice(&row).unwrap();
        values.push(value["sample_index"].as_u64().unwrap());
    }
    values
}

#[test]
fn ranks_partition_rows_and_tail_on_both_formats() {
    for format in ["ai-data-mixture", "ai-data-packed"] {
        let fixture = Fixture::new(format);
        for workers in [1_u64, 2, 4, 13] {
            for drop in [false, true] {
                let total = if drop { 11 - 11 % workers } else { 11 };
                let mut merged = vec![];
                for rank in 0..workers {
                    let mut reader = IndexedReader::open(&fixture.0, rank, workers, drop).unwrap();
                    let expected: Vec<_> = (rank..total).step_by(workers as usize).collect();
                    assert_eq!(reader.len(), expected.len() as u64);
                    assert_eq!(ordinals(&mut reader), expected);
                    assert_eq!(reader.state().consumed, expected.len() as u64);
                    merged.extend(expected);
                }
                merged.sort();
                assert_eq!(merged, (0..total).collect::<Vec<_>>());
            }
        }
    }
}

#[test]
fn checkpoint_restarts_at_next_unconsumed_row() {
    let fixture = Fixture::new("ai-data-mixture");
    let mut reader = IndexedReader::open(&fixture.0, 1, 3, false).unwrap();
    let mut row = vec![];
    reader.read_next(&mut row).unwrap();
    let state = reader.state();
    assert_eq!(ordinals(&mut reader), vec![4, 7, 10]);
    reader.load_state(state.clone()).unwrap();
    assert_eq!(ordinals(&mut reader), vec![4, 7, 10]);
    let mut fresh = IndexedReader::open(&fixture.0, 1, 3, false).unwrap();
    fresh.load_state(state).unwrap();
    assert_eq!(ordinals(&mut fresh), vec![4, 7, 10]);
    fresh.seek_rank_offset(2).unwrap();
    assert_eq!(ordinals(&mut fresh), vec![7, 10]);
}

#[test]
fn cursor_requires_same_release_topology_and_valid_position() {
    let fixture = Fixture::new("ai-data-mixture");
    let mut reader = IndexedReader::open(&fixture.0, 1, 3, false).unwrap();
    for field in [
        "schema_version",
        "release_id",
        "rank",
        "world_size",
        "drop_remainder",
        "consumed",
    ] {
        let mut state = reader.state();
        match field {
            "schema_version" => state.schema_version = "2.0".into(),
            "release_id" => state.release_id = "another-release".into(),
            "rank" => state.rank = 2,
            "world_size" => state.world_size = 4,
            "drop_remainder" => state.drop_remainder = true,
            "consumed" => state.consumed = 5,
            _ => unreachable!(),
        }
        assert!(reader.load_state(state).is_err(), "{field}");
        assert_eq!(reader.state().consumed, 0);
    }
    assert!(IndexedReader::open(&fixture.0, 0, 0, false).is_err());
    assert!(IndexedReader::open(&fixture.0, 3, 3, false).is_err());
}

#[test]
fn short_files_and_outside_offsets_do_not_advance_cursor() {
    let fixture = Fixture::new("ai-data-mixture");
    let index_path = fixture.0.join("shard-0.idx");
    let original = fs::read(&index_path).unwrap();
    fs::write(&index_path, &original[..8]).unwrap();
    let mut reader = IndexedReader::open(&fixture.0, 0, 1, false).unwrap();
    let mut row = vec![];
    assert!(reader.read_next(&mut row).is_err());
    assert_eq!(reader.state().consumed, 0);
    let mut invalid_offsets = original;
    invalid_offsets[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    fs::write(&index_path, invalid_offsets).unwrap();
    assert!(reader.read_next(&mut row).is_err());
    assert_eq!(reader.state().consumed, 0);
}

#[test]
fn an_empty_release_has_an_empty_cursor() {
    let fixture = Fixture::new("ai-data-mixture");
    let path = fixture.0.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["counts"]["records"] = json!(0);
    manifest["shards"] = json!([]);
    fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let mut reader = IndexedReader::open(&fixture.0, 7, 8, false).unwrap();
    assert!(reader.is_empty());
    assert!(ordinals(&mut reader).is_empty());
    assert!(reader.seek_rank_offset(1).is_err());
}
