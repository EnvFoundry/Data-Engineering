import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ai_data import MixtureReader, PackedReader, build, compose, pack
from ai_data.contracts import dumps
from ai_data.packing import packed_page
from ai_data.views import samples_page


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer = {"name": "authored-vocabulary", "revision": "1"}

    def source(self, sequences, masks=None):
        records = [{"schema_version": "1.0", "id": f"record:{i:03}", "kind": "tokenized",
                    "payload": {"input_ids": sequence,
                                "loss_mask": masks[i] if masks is not None else [1] * len(sequence)},
                    "metadata": {"sequence": i}, "source": {"uri": "example://tokens", "revision": "1"},
                    "split_group": f"group:{i}", "allowed_uses": ["train"], "split": "train"}
                   for i, sequence in enumerate(sequences)]
        (self.root / "records.jsonl").write_text("".join(dumps(row) + "\n" for row in records), encoding="utf-8")
        source = {"schema_version": "1.0", "name": "tokens", "sources": [{"path": "records.jsonl", "connector": "jsonl"}],
                  "use": "train", "seed": "1", "splits": {"train": 1, "validation": 0, "test": 0}}
        (self.root / "source.json").write_text(dumps(source), encoding="utf-8")
        manifest = build(self.root / "source.json", self.root / "release")
        mixture = {"schema_version": "1.0", "name": "mixture", "seed": "1", "shuffle": False,
                   "budget": {"unit": "records", "value": max(1, len(records))}, "shard_records": 3,
                   "exhaustion": "renormalize", "sources": [{"name": "tokens", "path": "release",
                       "release_id": manifest["release_id"], "split": "train", "weight": 1}]}
        (self.root / "mixture.json").write_text(dumps(mixture), encoding="utf-8")
        compose(self.root / "mixture.json", self.root / "mixture")

    def pack(self, name="packed", **kwargs):
        return pack(self.root / "mixture", self.root / name, sequence_length=kwargs.pop("sequence_length", 4),
                     tokenizer=kwargs.pop("tokenizer", self.tokenizer), **kwargs)

    def rows(self, **kwargs):
        with PackedReader(self.root / "packed", **kwargs) as reader:
            return list(reader)

    def test_causal_pairs_survive_long_record_block_boundaries_exactly_once(self):
        self.source([list(range(18))])
        manifest = self.pack(shard_records=2)
        rows = self.rows()
        pairs = [(x, y) for row in rows for x, y, active in
                 zip(row["input_ids"], row["target_ids"], row["attention_mask"]) if active]
        self.assertEqual(pairs, list(zip(range(17), range(1, 18))))
        self.assertEqual([row["segments"][0]["token_start"] for row in rows], [0, 4, 8, 12, 16])
        self.assertEqual([s["records"] for s in manifest["shards"]], [2, 2, 1])
        self.assertEqual(manifest["counts"]["target_tokens"], 17)
        self.assertEqual(manifest["counts"]["padding_tokens"], 3)
        self.assertEqual(rows[-1]["loss_mask"], [1, 0, 0, 0])
        self.assertEqual(rows[-1]["segment_ids"], [0, -1, -1, -1])

    def test_no_cross_document_targets_and_segment_attention_layout(self):
        self.source([[10, 11, 12], [20, 21, 22, 23]])
        self.pack(sequence_length=5)
        row = self.rows()[0]
        self.assertEqual(row["input_ids"], [10, 11, 20, 21, 22])
        self.assertEqual(row["target_ids"], [11, 12, 21, 22, 23])
        self.assertEqual(row["segment_ids"], [0, 0, 1, 1, 1])
        self.assertEqual(row["position_ids"], [0, 1, 0, 1, 2])
        self.assertEqual(row["cu_seqlens"], [0, 2, 5])
        allowed = [[j <= i and row["segment_ids"][i] == row["segment_ids"][j]
                    for j in range(5)] for i in range(5)]
        self.assertEqual(allowed[2], [False, False, True, False, False])
        self.assertEqual(row["segments"][1]["source"]["record_id"], "record:001")
        self.assertEqual(row["segments"][1]["split_group"], "group:1")

    def test_target_position_masks_preserve_the_supervised_objective(self):
        sequences = [[1, 4, 2], [1, 5, 6, 7, 2]]
        masks = [[1, 1, 1], [0, 0, 0, 1, 1]]
        self.source(sequences, masks)
        manifest = self.pack(sequence_length=3, loss_mask="/payload/loss_mask")
        self.assertEqual(manifest["counts"]["trainable_tokens"], 4)
        probabilities = [.02, .05, .3, .03, .1, .15, .15, .2]
        expected = sum(-math.log(probabilities[ids[i]]) * mask[i]
                       for ids, mask in zip(sequences, masks) for i in range(1, len(ids)))
        actual = sum(-math.log(probabilities[target]) * mask for row in self.rows()
                     for target, mask in zip(row["target_ids"], row["loss_mask"]))
        self.assertAlmostEqual(actual, expected)
        self.assertEqual(manifest["counts"]["target_tokens"], 6)
        self.assertEqual(manifest["counts"]["source_tokens"], 8)

    def test_short_and_empty_documents_are_counted_without_fabricated_targets(self):
        self.source([[], [5], [8, 9]])
        manifest = self.pack()
        self.assertEqual(manifest["counts"]["documents"], 3)
        self.assertEqual(manifest["counts"]["documents_without_targets"], 2)
        self.assertEqual(manifest["counts"]["source_tokens"], 3)
        self.assertEqual(manifest["counts"]["target_tokens"], 1)
        self.assertEqual(self.rows()[0]["target_ids"], [9, 0, 0, 0])

    def test_empty_mixture_produces_empty_readable_release(self):
        self.source([])
        manifest = self.pack()
        self.assertEqual(manifest["shards"], [])
        self.assertEqual(manifest["counts"]["records"], 0)
        self.assertEqual(self.rows(), [])
        self.assertIsNone(packed_page(self.root / "packed")["next_offset"])

    def test_rank_partition_resume_and_equal_length_tail(self):
        self.source([list(range(22))])
        self.pack()
        all_rows = self.rows()
        for world_size in (1, 2, 4, 9):
            indices = []
            for rank in range(world_size):
                expected = all_rows[rank::world_size]
                with PackedReader(self.root / "packed", rank=rank, world_size=world_size) as reader:
                    prefix = [next(reader)] if expected else []
                    state = json.loads(json.dumps(reader.state_dict()))
                with PackedReader(self.root / "packed", rank=rank, world_size=world_size) as reader:
                    reader.load_state_dict(state)
                    self.assertEqual(prefix + list(reader), expected)
                indices.extend(row["sample_index"] for row in expected)
            self.assertEqual(sorted(indices), list(range(6)))
        self.assertEqual([len(self.rows(rank=i, world_size=4, drop_remainder=True)) for i in range(4)], [1] * 4)
        with PackedReader(self.root / "packed", world_size=2) as reader:
            with self.assertRaisesRegex(ValueError, "worker topology"):
                reader.load_state_dict(state)

    def test_reproducibility_and_tokenizer_revision_identity(self):
        self.source([[1, 4, 8, 2]])
        first = self.pack()
        second = self.pack("second")
        self.assertEqual(first, second)
        changed = self.pack("changed", tokenizer=self.tokenizer | {"revision": "2"})
        self.assertNotEqual(first["release_id"], changed["release_id"])
        with PackedReader(self.root / "packed") as reader:
            state = reader.state_dict()
        with PackedReader(self.root / "changed") as reader:
            with self.assertRaisesRegex(ValueError, "different release"):
                reader.load_state_dict(state)

    def test_bad_tokens_masks_and_options_leave_no_partial_release(self):
        self.source([[1, True, 2]])
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            self.pack()
        self.assertFalse((self.root / "packed").exists())
        self.assertFalse(list(self.root.glob(".ai-packed-*")))
        for options in ({"sequence_length": 0}, {"sequence_length": True}, {"tokens": "input_ids"},
                        {"tokenizer": {"name": "", "revision": "1"}}, {"pad_token_id": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.pack(**options)

    def test_mask_length_and_missing_pointer_are_explicit(self):
        self.source([[1, 2, 3]], [[0, 1]])
        with self.assertRaisesRegex(ValueError, "one binary value per token"):
            self.pack(loss_mask="/payload/loss_mask")
        with self.assertRaisesRegex(ValueError, "no value"):
            self.pack(tokens="/payload/missing")

    def test_bounded_pages_from_all_three_product_types(self):
        self.source([list(range(140)), [1, 2, 3], [4, 5, 6]])
        self.pack(sequence_length=100)
        for name in ("release", "mixture"):
            first = samples_page(self.root / name, limit=1)
            self.assertEqual(first["items"][0]["id"], "record:000")
            self.assertEqual(first["next_offset"], 1)
            self.assertNotIn("payload_preview", first["items"][0])
            rank = samples_page(self.root / name, rank=1, world_size=2, include_payload=True)
            self.assertEqual(rank["total"], 1)
            self.assertEqual(rank["items"][0]["id"], "record:001")
            self.assertIn("payload_preview", rank["items"][0])
        packed = samples_page(self.root / "packed", limit=1, include_payload=True)
        self.assertEqual(len(packed["items"][0]["input_ids"]), 64)
        self.assertTrue(packed["items"][0]["tokens_truncated"])
        self.assertEqual(packed["next_offset"], 1)
        self.assertEqual(samples_page(self.root / "packed", offset=99)["items"], [])

    def test_installed_cli_pack_and_paged_consumption(self):
        self.source([[1, 2, 3, 4, 5]])
        command = [sys.executable, "-m", "ai_data", "pack", str(self.root / "mixture"),
                   "--output", str(self.root / "packed"), "--sequence-length", "3",
                   "--tokenizer", "fixture", "--tokenizer-revision", "1", "--loss-mask", "/payload/loss_mask"]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["counts"]["target_tokens"], 4)
        page = subprocess.run([sys.executable, "-m", "ai_data", "samples", str(self.root / "packed"),
                               "--include-payload", "--rank", "1", "--world-size", "2"],
                              cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(page.returncode, 0, page.stderr)
        self.assertEqual(json.loads(page.stdout)["items"][0]["sample_index"], 1)


if __name__ == "__main__":
    unittest.main()
