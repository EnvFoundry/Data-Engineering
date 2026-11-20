"""Exercise the compiled reader against actual Python-composed releases.

From the repository: python rust/compare_python.py [--binary PATH]
Uses temporary authored records; no model inference or external services.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from ai_data import MixtureReader, build  # noqa: E402
from ai_data.composition import compose  # noqa: E402
from ai_data.contracts import dumps  # noqa: E402


def exercise(binary: Path, root: Path):
    source = root / "records.jsonl"
    with source.open("w", encoding="utf-8", newline="\n") as stream:
        for index in range(17):
            record = {
                "schema_version": "1.0", "id": f"document:{index}", "kind": "document",
                "payload": {"text": (f"Document {index}: α, 中文, 🙂\n" * (4096 if index == 5 else index + 1)),
                            "input_ids": list(range(10 + index, 14 + index + index % 7)),
                            "loss_mask": [0, 0] + [1] * (2 + index % 7)},
                "source": {"uri": "example://native-reader", "revision": "1"},
                "split_group": f"document:{index}", "split": "train", "allowed_uses": ["train"],
            }
            stream.write(dumps(record) + "\n")
    recipe = {"schema_version": "1.0", "name": "native-reader-source",
              "sources": [{"connector": "jsonl", "path": source.name}], "use": "train", "seed": "17",
              "splits": {"train": 1, "validation": 0, "test": 0}}
    (root / "source.json").write_text(dumps(recipe), encoding="utf-8")
    release = build(root / "source.json", root / "source")
    mixture = {"schema_version": "1.0", "name": "native-reader-mixture",
               "sources": [{"name": "text", "path": "source", "release_id": release["release_id"],
                            "split": "train", "weight": 1}], "seed": "17",
               "budget": {"unit": "records", "value": 37}, "shard_records": 5, "exhaustion": "repeat"}
    (root / "mixture.json").write_text(dumps(mixture), encoding="utf-8")
    manifest = compose(root / "mixture.json", root / "mixture")
    calls = 0
    for world_size in (1, 4, 41):
        for drop in (False, True):
            merged = []
            for rank in range(world_size):
                args = [str(binary), "read", str(root / "mixture"), "--rank", str(rank), "--world-size", str(world_size)]
                if drop:
                    args.append("--drop-remainder")
                with MixtureReader(root / "mixture", rank=rank, world_size=world_size, drop_remainder=drop) as reader:
                    expected = list(reader)
                actual = subprocess.check_output(args)
                assert actual == b"".join((dumps(row) + "\n").encode("utf-8") for row in expected)
                merged.extend(row["sample_index"] for row in expected)
                calls += 1
            total = 37 - 37 % world_size if drop else 37
            assert sorted(merged) == list(range(total))

    options = {"rank": 1, "world_size": 4, "drop_remainder": True}
    with MixtureReader(root / "mixture", **options) as reader:
        prefix = [next(reader), next(reader)]
        (root / "python-state.json").write_text(dumps(reader.state_dict()), encoding="utf-8")
        expected = [next(reader), next(reader), next(reader)]
        expected_state = reader.state_dict()
        suffix = list(reader)
    command = [str(binary), "read", str(root / "mixture"), "--rank", "1", "--world-size", "4", "--drop-remainder",
               "--state-in", str(root / "python-state.json"), "--limit", "3", "--state-out", str(root / "rust-state.json")]
    output = subprocess.check_output(command)
    assert [json.loads(line) for line in output.splitlines()] == expected
    state = json.loads((root / "rust-state.json").read_text())
    assert state == expected_state
    with MixtureReader(root / "mixture", **options) as reader:
        reader.load_state_dict(state)
        assert list(reader) == suffix
    offset_output = subprocess.check_output([str(binary), "read", str(root / "mixture"), "--rank", "1",
                                            "--world-size", "4", "--drop-remainder", "--offset", "2", "--limit", "3"])
    assert offset_output == output
    assert len(prefix + expected + suffix) == 9
    info = json.loads(subprocess.check_output([str(binary), "info", str(root / "mixture"), "--rank", "1", "--world-size", "4"]))
    assert info["records"] == 37 and info["rank_records"] == 9
    assert info["shards"] == len(manifest["shards"])
    state_path = root / "rust-state.json"
    saved = state_path.read_bytes()
    interrupted = subprocess.Popen([str(binary), "read", str(root / "mixture"), "--state-out", str(state_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    interrupted.stdout.close()
    error = interrupted.stderr.read()
    assert interrupted.wait() != 0 and error
    assert state_path.read_bytes() == saved

    from ai_data.packing import PackedReader, pack

    packed = pack(root / "mixture", root / "packed", sequence_length=9,
                  tokenizer={"name": "authored-token-ids", "revision": "1"},
                  loss_mask="/payload/loss_mask", shard_records=3)
    packed_calls = 0
    for world_size in (1, 4):
        for drop in (False, True):
            for rank in range(world_size):
                options = {"rank": rank, "world_size": world_size, "drop_remainder": drop}
                with PackedReader(root / "packed", **options) as reader:
                    expected = list(reader)
                command = [str(binary), "read", str(root / "packed"), "--rank", str(rank), "--world-size", str(world_size)]
                if drop:
                    command.append("--drop-remainder")
                assert subprocess.check_output(command) == b"".join((dumps(row) + "\n").encode("utf-8") for row in expected)
                packed_calls += 1
    with PackedReader(root / "packed", rank=1, world_size=4) as reader:
        next(reader)
        (root / "packed-state.json").write_text(dumps(reader.state_dict()), encoding="utf-8")
        expected = list(reader)
    command = [str(binary), "read", str(root / "packed"), "--rank", "1", "--world-size", "4",
               "--state-in", str(root / "packed-state.json")]
    assert [json.loads(line) for line in subprocess.check_output(command).splitlines()] == expected
    print(dumps({"native_rank_reads": calls, "records": 37, "shards": len(manifest["shards"]),
                 "byte_identical": True, "python_to_rust_to_python_resume": True, "offset_matches_state": True,
                 "failed_pipe_preserves_state": True,
                 "packed_rank_reads": packed_calls, "packed_blocks": packed["counts"]["records"]}))


def main():
    parser = argparse.ArgumentParser()
    suffix = ".exe" if sys.platform == "win32" else ""
    parser.add_argument("--binary", type=Path, default=REPOSITORY / "rust" / "target" / "release" / f"ai-data-read{suffix}")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ai-data-native-") as temporary:
        exercise(args.binary.resolve(strict=True), Path(temporary))


if __name__ == "__main__":
    main()
