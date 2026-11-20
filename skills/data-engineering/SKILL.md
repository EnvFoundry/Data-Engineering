---
name: data-engineering
description: Prepare source releases, compose AI data mixtures, pack causal training sequences and resume distributed readers. Use for pretraining or SFT data, task and trajectory products, retrieval data, multimodal or scientific records, and experiments on data selection or composition.
---

# Data Engineering

Start from the consumer, the data change and the budget. Identify the sampling unit and source revision. Keep domain structure in the payload; a sensor stream, preference pair or scientific table need not become a chat message.

Use `ai-data` or `python -m ai_data`. Discover available operations through `--help` or MCP `data_capabilities`. Read only the guide needed for the current task with `ai-data guide TOPIC` or `data://guides/TOPIC`.

| Work | Guide | Operation |
| --- | --- | --- |
| Prepare a source product and its partitions | `data-design` | `profile`, `build`, `inspect` |
| Choose source weights, coverage and exhaustion | `mixture-design` | `compose`, `MixtureReader` |
| Deliver pretraining or SFT sequences | `causal-packing` | `pack`, `packed`, `PackedReader` |
| Resume rank-partitioned consumption | `consumption-restart` | Reader `state_dict` and `load_state_dict` |

Installed guides are the primary reference and remain available when a host copies this Skill separately. For source-level work, explicitly locate the [Data Engineering checkout](https://github.com/EnvFoundry/Data-Engineering), then read `docs/contracts.md` for the record envelope, `docs/consumption.md` for mixtures or `docs/packing.md` for training arrays. Document and SDK paths below are relative to that checkout, not the installed Skill location.

For a source release, preserve groups, parent relationships and fixed splits. Use `jsonl` for canonical records and `gym-jsonl` for Gym exports. Build a recipe into a new output directory. CLI source paths default to the recipe directory; MCP uses its configured workspace.

For a mixture, pin release identities and partitions, choose source weights and budget, then run `compose`. Read realized counts and exhaustion: source weights select records, not token fractions. For packing, use actual token arrays and the named tokenizer revision. `input_ids[i]` predicts `target_ids[i]`; targets are already aligned. Implement segmented causal attention in the learner and keep target-position loss masks. Save the reader cursor with the corresponding model and optimizer state.

Use `docs/integrations.md` for connector extensions, `sdk/typescript/README.md` for a Node harness and `docs/native-reader.md` for Rust shard streaming. Keep tensor collation, inference and optimization with the consumer. Use `docs/research.md` to connect a data change to a measured downstream effect at a comparable budget.

Example request: “Prepare two source mixtures at the same training-token budget, retain the same evaluation split, and produce packed inputs for our existing learner. Show realized source coverage, supervised targets and the command that starts each consumer run.” Return recipes and output paths, a short record or block example, and the result that answers the consumer question.
