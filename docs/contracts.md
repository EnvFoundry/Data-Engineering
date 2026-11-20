# Data contracts

The v0.3 implementation uses record and source-recipe schema `1.0`. The common envelope preserves domain payloads and their origin. [Composition](consumption.md) adds a separate mixture recipe, indexed shard manifest and consumer state.

## Record

```json
{
  "schema_version": "1.0",
  "id": "document:source-a:revision-7",
  "kind": "document",
  "payload": {"text": "An observed or authored document."},
  "source": {"uri": "example://source-a", "revision": "7"},
  "split_group": "source-a",
  "parents": [],
  "allowed_uses": ["train", "evaluate", "retrieve"],
  "metadata": {"language": "en"}
}
```

| Field | Contract |
| --- | --- |
| `id` | Stable, nonempty identity for this version of a record. IDs must be unique across a build. A changed record needs a new ID or an explicit new source version outside the old release. |
| `kind` | Lowercase domain profile, such as `document`, `conversation`, `preference`, `trajectory`, `measurement`, `table_row`, `media`. Custom names are allowed. |
| `payload` | JSON object preserving domain structure. Binary/large content belongs in referenced artifacts. |
| `source` | Nonempty `uri` and `revision`. The operator supplies a meaningful revision; a nonempty string alone does not prove remote immutability. Local input bytes are also hashed. |
| `split_group` | Domain independence unit: document, patient, scene, episode family, experimental specimen, etc. Use globally scoped values across sources. |
| `parents` | Optional parent record IDs, including external IDs. Shared ancestors connect sibling records for splitting. Full ancestry must be supplied by the producer; the core cannot infer unknown derivations. |
| `allowed_uses` | Declared subset of `train`, `evaluate`, `retrieve`, `analyze`, `redistribute`; empty means no declared use. This is an operational declaration, not a license classifier. |
| `split` | Optional fixed `train`, `validation`, `test`, or null. Related records inherit a fixed split; contradictory fixed splits fail. |
| `metadata` | Optional annotations and domain metadata. Different annotations are retained even when payloads match. |

`ai-data schema` emits JSON Schema for the envelope. Runtime checks also reject self-parent relationships, non-finite JSON values, duplicate JSON keys and duplicate record IDs. Domain profile validators are a planned extension; the core does not certify units, conversation formatting, preference validity or artifact content.

Domain payloads should retain what consumers need: conversation roles and tool-call IDs; preference candidates and comparison conditions; trajectories with ordered observations/actions and terminal status; feature entities with event/available times; measurements with units/calibration/uncertainty; media with encoding and spatial/temporal alignment. These are profile requirements to implement, not fields already validated by the generic envelope.

## Recipe and paths

```json
{
  "schema_version": "1.0",
  "name": "training-corpus",
  "sources": [{"connector": "jsonl", "path": "records.jsonl"}],
  "use": "train",
  "seed": "split-plan-1",
  "splits": {"train": 0.8, "validation": 0.1, "test": 0.1}
}
```

CLI/SDK source paths are relative to the recipe's directory unless `root` is specified. MCP source paths are relative to its configured workspace. Paths may not escape that root, including through resolved symlinks. Recipe paths and output paths passed to CLI/SDK may be absolute; MCP exposes only relative paths.

Split proportions apply only to components without a fixed split and are expected proportions, not exact row quotas. A small sample or large component may yield empty partitions. A zero weight does not override a fixed assignment.

The builder constructs connected components using equal `split_group`, equal `(kind, payload)` hashes and parent relationships **over every input record**, including records later excluded by use. It chooses each unspecified component's split using a seed and its sorted group identifiers. It then filters by declared use and removes exact duplicate payloads only where source, allowed-use set and metadata also match. The lexicographically first eligible ID is retained; lineage records all represented inputs and their parents.

This ordering prevents a filtered bridge or held-out copy from losing its influence on partitioning. It deliberately favors retaining differences in provenance or annotation over maximum compression. It does not perform semantic deduplication, template inference, time-based splitting or content decoding. Set `split_group`, `parents` and fixed `split` appropriately before building.

For identical input bytes, recipe, connector behavior and core code, releases are deterministic. Input order does not change record assignment. Appending data that merges components can change assignments; cross-release stable split ledgers and temporal split operators are future work. Carry explicit fixed splits when reusing an existing assignment.

## Release

| File | Purpose |
| --- | --- |
| `train.jsonl`, `validation.jsonl`, `test.jsonl` | Canonical retained records with assigned split. |
| `decisions.jsonl` | Every input ID, source, assigned component/split, and retained/duplicate/excluded decision. |
| `lineage.jsonl` | Retained output IDs, represented input IDs, parents and source. |
| `manifest.json` | Recipe, input byte hashes, connector identities, core source fingerprint, output byte hashes/sizes, counts and release ID. |

The release ID hashes the canonical manifest without its own `release_id` field. `verify` checks the manifest digest, declared output file set, file bytes, partition labels, IDs, allowed use and counts.

Builds stage outputs beside the destination and rename on success. A lock coordinates local writers. Each release uses a new destination; ordinary exceptions remove staging and the acquired lock. A killed process may leave a staging directory or lock that needs cleanup after the process has stopped.

Source bytes stay at their original location. The manifest hashes them but does not archive them; preserve input snapshots separately for reproduction. Unknown-use records are not copied to partition files; decision metadata still records their IDs and sources. Artifact URIs inside payloads are retained without fetching or verifying the referenced bytes. A withdrawn source will require affected releases and derived indexes to be rebuilt or retired; automated withdrawal propagation is planned.

`use="train"` does not turn validation/test files into training material. Consumers select the proper partition. Trainer formatting, tokenization, loss masks, on-policy log probabilities and sampling weights belong to explicit consumer adapters with their own versioned recipes.

## Compatibility

Keep incompatible envelope changes behind a new schema version and a migration. Preserve input IDs in migration lineage. Add domain structure inside a versioned payload profile rather than silently changing existing field meanings. Keep the repository/package version distinct from record schema, external connector version and data release ID.

## Packed sequences

[Sequence packing](packing.md) defines the separate packed-release format, aligned causal targets, loss masks, segment boundaries and consumption cursor. It consumes a mixture without changing the source-record contract.
