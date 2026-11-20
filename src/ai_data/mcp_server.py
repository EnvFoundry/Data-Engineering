"""MCP is a transport over the same Python core, not a second implementation."""

from pathlib import Path

from .contracts import confined
from .pipeline import build, inspect_release, profile, verify_release
from .composition import compose, verify_mixture
from .packing import pack, packed_page, packing_info
from .views import samples_page


def create_server(root: str | Path):
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise ImportError('Install the MCP extra: python -m pip install -e ".[mcp]"') from exc
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("MCP workspace must be a directory")
    from .cli import capabilities
    from .guidance import (GUIDES, get_guide, design_data_product, prepare_training_data,
                           compare_data_recipes)
    server = FastMCP("Data Engineering", instructions=(
        "Start with the consumer, data unit and learning or research objective. "
        "Paths are relative to the configured workspace. Build source releases, compose explicit partitions, "
        "then pack supplied token IDs when the consumer needs causal training blocks. "
        "Use data_samples for small record pages; payloads and tokens are optional bounded previews. "
        "Packed input_ids[i] predicts target_ids[i]: do not shift targets again, and use within-segment "
        "causal attention. Read data://guides/ for the method relevant to the current decision."
    ))
    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=read)
    def data_capabilities() -> dict:
        """List implemented capabilities and data contract version."""
        return capabilities() | {"resources": [f"data://guides/{topic}" for topic in GUIDES],
                                  "prompts": ["design_data_product", "prepare_training_data", "compare_data_recipes"]}

    @server.tool(annotations=read)
    def data_profile(path: str) -> dict:
        """Profile canonical JSONL records at a workspace-relative path."""
        return profile(confined(root, path))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
    def data_build(recipe: str, output: str) -> dict:
        """Build a new local release. Source paths are workspace-relative. Existing outputs are refused."""
        return build(confined(root, recipe), confined(root, output), root=root, allow_plugins=False)

    @server.tool(annotations=read)
    def data_inspect(path: str) -> dict:
        """Read a release manifest, without claiming its files have been verified."""
        return inspect_release(confined(root, path))

    @server.tool(annotations=read)
    def data_verify(path: str) -> dict:
        """Check local release hashes, partitions, declared use and counts; not external artifact bytes."""
        return verify_release(confined(root, path))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
    def data_compose(recipe: str, output: str) -> dict:
        """Compose pinned releases into budgeted indexed shards; paths are workspace-relative."""
        result = compose(confined(root, recipe), confined(root, output), root=root)
        return {key: result[key] for key in ("release_id", "name", "counts", "stop_reason", "sampling")}

    @server.tool(annotations=read)
    def data_verify_mixture(path: str) -> dict:
        """Check composed shard bytes, indexes and realized source/budget counts."""
        return verify_mixture(confined(root, path))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
    def data_pack(mixture: str, output: str, sequence_length: int, tokenizer_name: str,
                  tokenizer_revision: str, tokens: str = "/payload/input_ids", loss_mask: str | None = None,
                  pad_token_id: int = 0, shard_records: int = 4096) -> dict:
        """Pack supplied token arrays into causal blocks; masks address original target-token positions."""
        destination = confined(root, output)
        pack(confined(root, mixture), destination, sequence_length=sequence_length,
             tokenizer={"name": tokenizer_name, "revision": tokenizer_revision}, tokens=tokens,
             loss_mask=loss_mask, pad_token_id=pad_token_id, shard_records=shard_records)
        return packing_info(destination)

    @server.tool(annotations=read)
    def data_packed(path: str, offset: int = 0, limit: int = 5, include_tokens: bool = False) -> dict:
        """Inspect packing semantics, source counts and up to 20 blocks with optional 64-token previews."""
        return packed_page(confined(root, path), offset=offset, limit=limit, include_tokens=include_tokens)

    @server.tool(annotations=read)
    def data_samples(path: str, split: str = "train", offset: int = 0, limit: int = 20,
                     rank: int = 0, world_size: int = 1, include_payload: bool = False) -> dict:
        """Read up to 100 source records, mixture draws or packed blocks; include bounded payloads on request."""
        return samples_page(confined(root, path), split=split, offset=offset, limit=limit,
                            rank=rank, world_size=world_size, include_payload=include_payload)

    for topic in GUIDES:
        def register_guide(name):
            @server.resource(f"data://guides/{name}")
            def guide() -> str:
                """Read one data research method."""
                return get_guide(name)["content"]
        register_guide(topic)

    server.prompt()(design_data_product)
    server.prompt()(prepare_training_data)
    server.prompt()(compare_data_recipes)

    return server
