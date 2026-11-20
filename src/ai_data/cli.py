"""Command-line and machine-readable entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .contracts import record_schema
from .pipeline import build, inspect_release, profile, verify_release
from .composition import compose, verify_mixture
from .packing import pack, packed_page, packing_info
from .views import samples_page


def capabilities() -> dict:
    from .guidance import GUIDES
    return {"record_schema": "1.0", "connectors": ["jsonl", "gym-jsonl"],
            "operations": ["profile", "build", "inspect", "samples", "compose", "pack", "packed", "guide"],
            "readers": ["MixtureReader", "PackedReader"], "sample_page_limit": 100,
            "packing": {"input": "supplied integer token arrays and optional target-position binary masks",
                        "label_alignment": "input_at_i_predicts_target_at_i", "attention": "causal_within_segment"},
            "guides": list(GUIDES), "entry_points": "ai_data.connectors",
            "scope": "local release preparation, indexed composition and training sequence packing"}


def main(argv=None) -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="ai-data", description="Build versioned AI data products.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("profile", "inspect", "verify", "verify-mixture"):
        command = commands.add_parser(name)
        command.add_argument("path")
    builder = commands.add_parser("build")
    builder.add_argument("recipe")
    builder.add_argument("--output", required=True)
    builder.add_argument("--root", help="source workspace; defaults to recipe directory")
    builder.add_argument("--allow-plugins", action="store_true", help="load trusted installed connector code")
    composer = commands.add_parser("compose", help="compose pinned source releases into indexed training shards")
    composer.add_argument("recipe")
    composer.add_argument("--output", required=True)
    composer.add_argument("--root", help="source workspace; defaults to recipe directory")
    packer = commands.add_parser("pack", help="pack supplied token arrays into causal training blocks")
    packer.add_argument("mixture")
    packer.add_argument("--output", required=True)
    packer.add_argument("--sequence-length", required=True, type=int)
    packer.add_argument("--tokenizer", required=True)
    packer.add_argument("--tokenizer-revision", required=True)
    packer.add_argument("--tokens", default="/payload/input_ids", help="JSON pointer inside each canonical record")
    packer.add_argument("--loss-mask", help="JSON pointer to one binary value per original token")
    packer.add_argument("--pad-token-id", type=int, default=0)
    packer.add_argument("--shard-records", type=int, default=4096)
    packed = commands.add_parser("packed", help="inspect packed blocks and optional token previews")
    packed.add_argument("path")
    packed.add_argument("--offset", type=int, default=0)
    packed.add_argument("--limit", type=int, default=5)
    packed.add_argument("--include-tokens", action="store_true")
    samples = commands.add_parser("samples", help="page source records, mixture draws or packed blocks")
    samples.add_argument("path")
    samples.add_argument("--split", choices=("train", "validation", "test"), default="train")
    samples.add_argument("--offset", type=int, default=0)
    samples.add_argument("--limit", type=int, default=20)
    samples.add_argument("--rank", type=int, default=0)
    samples.add_argument("--world-size", type=int, default=1)
    samples.add_argument("--include-payload", action="store_true")
    commands.add_parser("capabilities")
    guide = commands.add_parser("guide", help="read the method relevant to the current data decision")
    guide.add_argument("topic", nargs="?", default="data-design")
    commands.add_parser("schema")
    server = commands.add_parser("serve")
    server.add_argument("--root", required=True)
    config = commands.add_parser("connect")
    config.add_argument("--root", required=True)
    config.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(args.recipe, args.output, root=args.root, allow_plugins=args.allow_plugins)
        elif args.command == "compose":
            result = compose(args.recipe, args.output, root=args.root)
        elif args.command == "pack":
            pack(args.mixture, args.output, sequence_length=args.sequence_length,
                 tokenizer={"name": args.tokenizer, "revision": args.tokenizer_revision}, tokens=args.tokens,
                 loss_mask=args.loss_mask, pad_token_id=args.pad_token_id, shard_records=args.shard_records)
            result = packing_info(args.output)
        elif args.command == "packed":
            result = packed_page(args.path, offset=args.offset, limit=args.limit, include_tokens=args.include_tokens)
        elif args.command == "samples":
            result = samples_page(args.path, split=args.split, offset=args.offset, limit=args.limit,
                                  rank=args.rank, world_size=args.world_size, include_payload=args.include_payload)
        elif args.command == "capabilities":
            result = capabilities()
        elif args.command == "guide":
            from .guidance import get_guide
            result = get_guide(args.topic)
        elif args.command == "verify-mixture":
            result = verify_mixture(args.path)
        elif args.command == "profile":
            result = profile(args.path)
        elif args.command == "inspect":
            result = inspect_release(args.path)
        elif args.command == "verify":
            result = verify_release(args.path)
        elif args.command == "schema":
            result = record_schema()
        elif args.command == "connect":
            root = Path(args.root).resolve(strict=True)
            if not root.is_dir():
                raise ValueError("root must be a directory")
            result = {"mcpServers": {"data-engineering": {
                "command": sys.executable,
                "args": ["-m", "ai_data", "serve", "--root", str(root)],
            }}}
            with Path(args.output).open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        else:
            from .mcp_server import create_server
            create_server(args.root).run(transport="stdio")
            return 0
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
