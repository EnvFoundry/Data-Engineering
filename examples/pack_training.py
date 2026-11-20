"""Pack authored token IDs and compare the packed objective with record-level loss.

The vocabulary and IDs are fixtures, not a tokenizer or a model experiment.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile

from ai_data import MixtureReader, PackedReader, build, compose, pack
from ai_data.contracts import dumps


def run(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    sequences = [
        ([1, 4, 7, 2], [0, 1, 1, 1]),
        ([1, 5, 6, 3, 8, 9, 10, 11, 2], [0, 0, 0, 0, 1, 1, 1, 1, 1]),
        ([1, 12, 13, 14, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2], [0] + [1] * 13),
    ]
    records = [{"schema_version": "1.0", "id": f"sequence:{index}", "kind": "tokenized",
                "payload": {"input_ids": ids, "loss_mask": mask},
                "source": {"uri": "example://authored-token-ids", "revision": "1"},
                "split_group": f"document:{index}", "allowed_uses": ["train"], "split": "train"}
               for index, (ids, mask) in enumerate(sequences)]
    (root / "records.jsonl").write_text("".join(dumps(row) + "\n" for row in records), encoding="utf-8")
    source = {"schema_version": "1.0", "name": "authored-token-sequences",
              "sources": [{"connector": "jsonl", "path": "records.jsonl"}], "use": "train", "seed": "packing-1",
              "splits": {"train": 1, "validation": 0, "test": 0}}
    (root / "source.json").write_text(dumps(source) + "\n", encoding="utf-8")
    release = build(root / "source.json", root / "release")
    mixture = {"schema_version": "1.0", "name": "packing-example", "seed": "1", "shuffle": False,
               "budget": {"unit": "records", "value": len(records)}, "shard_records": 2,
               "exhaustion": "renormalize", "sources": [{"name": "training", "path": "release",
                   "release_id": release["release_id"], "split": "train", "weight": 1}]}
    (root / "mixture.json").write_text(dumps(mixture) + "\n", encoding="utf-8")
    compose(root / "mixture.json", root / "mixture")
    manifest = pack(root / "mixture", root / "packed", sequence_length=5, shard_records=2,
                    tokenizer={"name": "authored-16-token-vocabulary", "revision": "1"},
                    loss_mask="/payload/loss_mask")

    # Fixed logits demonstrate the objective, independently of a neural framework.
    logits = [token / 16 for token in range(16)]
    log_partition = math.log(sum(math.exp(value) for value in logits))
    nll = lambda target: log_partition - logits[target]
    expected_sum = sum(nll(token) * mask[i] for ids, mask in sequences for i, token in enumerate(ids) if i > 0)
    expected_count = sum(sum(mask[1:]) for _, mask in sequences)
    expected_pairs = []
    with MixtureReader(root / "mixture") as reader:
        for sample in reader:
            ids = sample["record"]["payload"]["input_ids"]
            expected_pairs.extend(zip(ids[:-1], ids[1:]))
    pairs, loss_sum, trainable = [], 0.0, 0
    with PackedReader(root / "packed") as reader:
        blocks = list(reader)
    for block in blocks:
        for token, target, mask, occupied in zip(block["input_ids"], block["target_ids"],
                                                 block["loss_mask"], block["attention_mask"]):
            if occupied:
                pairs.append((token, target))
            loss_sum += nll(target) * mask
            trainable += mask
    assert pairs == expected_pairs
    assert trainable == expected_count == 21
    assert math.isclose(loss_sum, expected_sum, rel_tol=1e-12)
    rank_counts = []
    for rank in range(2):
        with PackedReader(root / "packed", rank=rank, world_size=2) as reader:
            rank_counts.append(len(reader))
    with PackedReader(root / "packed", rank=0, world_size=2) as reader:
        next(reader)
        state = reader.state_dict()
        expected_next = next(reader)
    with PackedReader(root / "packed", rank=0, world_size=2) as reader:
        reader.load_state_dict(state)
        assert next(reader) == expected_next
    (root / "reader-state.json").write_text(dumps(state) + "\n", encoding="utf-8")
    return {"counts": manifest["counts"], "label_alignment": manifest["label_alignment"],
            "objective": {"trainable_tokens": trainable, "mean_nll": loss_sum / trainable},
            "rank_blocks": rank_counts, "resumed_block": expected_next["sample_index"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        with tempfile.TemporaryDirectory() as directory:
            result = run(Path(directory) / "example")
    else:
        result = run(args.output.resolve())
    print(dumps(result))


if __name__ == "__main__":
    main()
