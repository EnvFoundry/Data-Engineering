"""Small pages of source records, mixture draws and training blocks."""

from __future__ import annotations

from pathlib import Path

from .composition import MixtureReader, _manifest
from .contracts import SPLITS, confined, dumps, strict_json
from .packing import PackedReader, _block_preview


def _record_view(row: dict, include_payload: bool) -> dict:
    record = row["record"]
    item = {"sample_index": row["sample_index"], "id": record["id"], "kind": record["kind"],
            "split": record["split"], "split_group": record["split_group"],
            "source": {key: value for key, value in row["source"].items() if key != "release_id"},
            "payload_fields": list(record["payload"])[:30]}
    if include_payload:
        value = dumps(record["payload"])
        item.update(payload_preview=value[:2000], payload_truncated=len(value) > 2000)
    return item


def samples_page(path: str | Path, *, split: str = "train", offset: int = 0, limit: int = 20,
                 rank: int = 0, world_size: int = 1, include_payload: bool = False) -> dict:
    """Page a rank's logical stream without returning the complete data product."""
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("offset must be nonnegative and limit between 1 and 100")
    if (type(world_size) is not int or world_size < 1 or type(rank) is not int or
            not 0 <= rank < world_size):
        raise ValueError("rank must be in [0, world_size), with a positive world_size")
    if split not in SPLITS:
        raise ValueError("unknown split")
    if type(include_payload) is not bool:
        raise ValueError("include_payload must be boolean")
    root = Path(path).resolve()
    manifest = _manifest(root)
    kind = manifest.get("format", "ai-data-release")
    items = []
    if kind in ("ai-data-mixture", "ai-data-packed"):
        reader_type = PackedReader if kind == "ai-data-packed" else MixtureReader
        with reader_type(root, rank=rank, world_size=world_size) as reader:
            total = len(reader)
            reader.load_state_dict(reader.state_dict() | {"consumed": min(offset, total)})
            for _ in range(limit):
                row = next(reader, None)
                if row is None:
                    break
                items.append(_block_preview(row, include_payload) if kind == "ai-data-packed"
                             else _record_view(row, include_payload))
    elif kind == "ai-data-release":
        global_count = manifest["counts"]["splits"][split]
        total = max(0, (global_count - rank + world_size - 1) // world_size)
        if offset < total:
            position = 0
            with confined(root, split + ".jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    index, position = position, position + 1
                    if index % world_size != rank or index // world_size < offset:
                        continue
                    record = strict_json(line)
                    items.append(_record_view({"sample_index": index, "source": record["source"],
                                               "record": record}, include_payload))
                    if len(items) == limit:
                        break
    else:
        raise ValueError("samples requires a canonical release, mixture or packed release")
    next_offset = offset + len(items)
    result = {"format": kind, "name": manifest["name"], "total": total, "offset": offset,
              "limit": limit, "rank": rank, "world_size": world_size, "items": items,
              "next_offset": next_offset if next_offset < total else None}
    if kind == "ai-data-release":
        result["split"] = split
    if kind == "ai-data-packed":
        result["label_alignment"] = manifest["label_alignment"]
        result["attention"] = manifest["attention"]
    return result
