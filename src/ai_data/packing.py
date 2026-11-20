"""Stream supplied token sequences into indexed causal-language-model blocks."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import shutil
import tempfile

from .composition import MixtureReader, _Shards, _U64, _manifest, _positive_int
from .contracts import confined, digest, dumps, nonempty, object_only


def _pointer(value: dict, pointer: str):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("token and loss-mask fields must be JSON pointers")
    try:
        for raw in pointer.split("/")[1:]:
            key = raw.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"record has no value at {pointer}") from exc


def _token_array(record: dict, tokens: str, mask: str | None) -> tuple[list[int], list[int]]:
    ids = _pointer(record, tokens)
    if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError(f"record {record.get('id')!r}: tokens must be a list of nonnegative integers")
    weights = [1] * len(ids) if mask is None else _pointer(record, mask)
    if (not isinstance(weights, list) or len(weights) != len(ids) or
            any(type(weight) not in (int, bool) or weight not in (0, 1) for weight in weights)):
        raise ValueError(f"record {record.get('id')!r}: loss mask must contain one binary value per token")
    return ids, weights


class _Block:
    def __init__(self, ordinal: int, length: int, pad: int):
        self.length, self.pad = length, pad
        self.row = {"sample_index": ordinal, "input_ids": [], "target_ids": [], "loss_mask": [],
                    "attention_mask": [], "segment_ids": [], "position_ids": [], "segments": [],
                    "cu_seqlens": [0]}

    @property
    def available(self) -> int:
        return self.length - len(self.row["input_ids"])

    def append(self, sample: dict, ids: list[int], mask: list[int], start: int, stop: int):
        row, count = self.row, stop - start
        segment = len(row["segments"])
        begin = len(row["input_ids"])
        row["input_ids"].extend(ids[start:stop])
        row["target_ids"].extend(ids[start + 1:stop + 1])
        row["loss_mask"].extend(int(weight) for weight in mask[start + 1:stop + 1])
        row["attention_mask"].extend([1] * count)
        row["segment_ids"].extend([segment] * count)
        row["position_ids"].extend(range(count))
        row["segments"].append({"start": begin, "end": begin + count,
                                "sample_index": sample["sample_index"], "source": sample["source"],
                                "split_group": sample["record"]["split_group"],
                                "token_start": start, "token_end": stop})
        row["cu_seqlens"].append(begin + count)

    def finish(self) -> dict:
        padding = self.available
        for field in ("input_ids", "target_ids"):
            self.row[field].extend([self.pad] * padding)
        for field in ("loss_mask", "attention_mask", "position_ids"):
            self.row[field].extend([0] * padding)
        self.row["segment_ids"].extend([-1] * padding)
        return self.row


def pack(mixture: str | Path, output: str | Path, *, sequence_length: int,
         tokenizer: dict, tokens: str = "/payload/input_ids", loss_mask: str | None = None,
         pad_token_id: int = 0, shard_records: int = 4096) -> dict:
    """Pack every within-record next-token pair exactly once.

    Input records already contain tokenizer-produced IDs. A supplied mask is
    aligned to original token positions. The output target at position i is
    predicted by the input at position i; consumers must not shift it again.
    """
    _positive_int(sequence_length, "sequence_length")
    _positive_int(shard_records, "shard_records")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    object_only(tokenizer, {"name", "revision"}, set(), "tokenizer")
    for key in ("name", "revision"):
        nonempty(tokenizer[key], f"tokenizer.{key}")
    for pointer in (tokens, loss_mask):
        if pointer is not None and (not isinstance(pointer, str) or not pointer.startswith("/")):
            raise ValueError("token and loss-mask fields must be JSON pointers")
    if tokens is None:
        raise ValueError("tokens must be a JSON pointer")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ai-packed-", dir=output.parent))
    lock = output.parent / ("." + output.name + ".lock")
    writer, locked = _Shards(staging, shard_records), False
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        locked = True
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        counts = Counter(documents=0, source_tokens=0, target_tokens=0, trainable_tokens=0,
                         padding_tokens=0, documents_without_targets=0)
        source_counts: dict[str, dict] = {}
        block = _Block(0, sequence_length, pad_token_id)
        with MixtureReader(mixture) as reader:
            source_manifest = reader.manifest
            for sample in reader:
                ids, mask = _token_array(sample["record"], tokens, loss_mask)
                pairs = max(0, len(ids) - 1)
                counts["documents"] += 1
                counts["source_tokens"] += len(ids)
                counts["target_tokens"] += pairs
                counts["trainable_tokens"] += sum(mask[1:])
                counts["documents_without_targets"] += int(pairs == 0)
                source = source_counts.setdefault(sample["source"]["name"],
                                                  {"documents": 0, "source_tokens": 0, "target_tokens": 0,
                                                   "trainable_tokens": 0})
                source["documents"] += 1
                source["source_tokens"] += len(ids)
                source["target_tokens"] += pairs
                source["trainable_tokens"] += sum(mask[1:])
                start = 0
                while start < pairs:
                    stop = min(pairs, start + block.available)
                    block.append(sample, ids, mask, start, stop)
                    start = stop
                    if not block.available:
                        writer.append(block.finish())
                        block = _Block(writer.total, sequence_length, pad_token_id)
        if block.available != sequence_length:
            counts["padding_tokens"] += block.available
            writer.append(block.finish())
        writer.finish()
        counts["records"] = writer.total
        manifest = {"schema_version": "1.0", "format": "ai-data-packed",
                    "name": source_manifest["name"] + "-packed",
                    "source_mixture": source_manifest["release_id"],
                    "tokenizer": dict(tokenizer), "sequence_length": sequence_length,
                    "recipe": {"tokens": tokens, "loss_mask": loss_mask, "pad_token_id": pad_token_id,
                               "shard_records": shard_records},
                    "label_alignment": "input_at_i_predicts_target_at_i",
                    "attention": "causal_within_segment", "positions": "reset_per_block_segment",
                    "algorithm": "causal-pairs-v1", "counts": dict(counts),
                    "source_counts": source_counts, "shards": writer.items}
        manifest["release_id"] = digest(manifest)
        (staging / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8", newline="\n")
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        staging.rename(output)
        return manifest
    finally:
        writer.close()
        if staging.exists():
            shutil.rmtree(staging)
        if locked:
            lock.unlink()


def _packed_manifest(root: Path) -> dict:
    manifest = _manifest(root)
    if manifest.get("format") != "ai-data-packed":
        raise ValueError("not a packed token release")
    _positive_int(manifest["sequence_length"], "sequence_length")
    if manifest.get("label_alignment") != "input_at_i_predicts_target_at_i":
        raise ValueError("unsupported packed label alignment")
    count, paths = 0, set()
    for shard in manifest["shards"]:
        if shard["first_index"] != count or type(shard["records"]) is not int or shard["records"] <= 0:
            raise ValueError("packed shard ranges are inconsistent")
        count += shard["records"]
        for name in (shard["path"], shard["index_path"]):
            confined(root, name)
            if name in paths:
                raise ValueError("packed shard paths are repeated")
            paths.add(name)
        if shard["index_bytes"] != (shard["records"] + 1) * _U64.size:
            raise ValueError("packed index size is inconsistent")
    if count != manifest["counts"]["records"]:
        raise ValueError("packed total count is inconsistent")
    return manifest


class PackedReader(MixtureReader):
    """Rank-strided blocks; checkpoints name the exact next unconsumed block."""

    def __init__(self, path: str | Path, *, rank: int = 0, world_size: int = 1,
                 drop_remainder: bool = False):
        _positive_int(world_size, "world_size")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("rank must be an integer in [0, world_size)")
        if type(drop_remainder) is not bool:
            raise ValueError("drop_remainder must be boolean")
        self.root = Path(path).resolve()
        self.manifest = _packed_manifest(self.root)
        self.rank, self.world_size, self.consumed = rank, world_size, 0
        self.drop_remainder = drop_remainder
        self.total = self.manifest["counts"]["records"]
        if drop_remainder:
            self.total -= self.total % world_size
        self.starts = [shard["first_index"] for shard in self.manifest["shards"]]
        self._shard = self._data = self._index = None


def packing_info(path: str | Path) -> dict:
    """Return the training semantics and realized counts without shard details."""
    value = _packed_manifest(Path(path).resolve())
    return {key: value[key] for key in ("format", "name", "release_id", "source_mixture", "tokenizer",
                                       "sequence_length", "counts", "label_alignment", "attention", "positions")} | {
        "shards": len(value["shards"]), "source_counts": value["source_counts"]}


def _block_preview(row: dict, include_tokens: bool) -> dict:
    segments = [{**segment, "source": {key: value for key, value in segment["source"].items()
                                        if key != "release_id"}} for segment in row["segments"][:5]]
    item = {"sample_index": row["sample_index"], "tokens": sum(row["attention_mask"]),
            "trainable_tokens": sum(row["loss_mask"]), "segments": segments,
            "segment_count": len(row["segments"])}
    if include_tokens:
        item.update({key: row[key][:64] for key in ("input_ids", "target_ids", "loss_mask",
                                                    "segment_ids", "position_ids")})
        item["tokens_truncated"] = len(row["input_ids"]) > 64
    return item


def packed_page(path: str | Path, *, offset: int = 0, limit: int = 5,
                include_tokens: bool = False) -> dict:
    """Read block metadata and optional bounded token previews for an agent."""
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("offset must be nonnegative and limit between 1 and 20")
    if type(include_tokens) is not bool:
        raise ValueError("include_tokens must be boolean")
    result = packing_info(path)
    items = []
    with PackedReader(path) as reader:
        reader.load_state_dict(reader.state_dict() | {"consumed": min(offset, len(reader))})
        for _ in range(limit):
            row = next(reader, None)
            if row is None:
                break
            items.append(_block_preview(row, include_tokens))
    next_offset = offset + len(items)
    return result | {"offset": offset, "limit": limit, "items": items,
                     "next_offset": next_offset if next_offset < result["counts"]["records"] else None}
