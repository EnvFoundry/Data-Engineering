# Implementation status

Version 0.3 supplies a standalone source-to-training-input workflow. The [design](design.zh-CN.md) describes the wider data domain; [research methods](research.md) connect data decisions to consumer outcomes.

| Area | Working behavior | Next workload-driven extension |
| --- | --- | --- |
| Source preparation | Record envelope, JSONL/Gym imports, group/parent relationships, exact duplicates, fixed partitions and lineage. | Out-of-memory curation through an established columnar backend; persistent split assignments across evolving corpora. |
| Distribution construction | Source weights, selected partitions, record/supplied-token budgets, exhaustion policies and indexed shards. | Source allocation experiments with a real learner and matched compute. |
| Training inputs | Supplied tokenizer IDs, explicit target masks, complete within-document causal pairs, fixed blocks and segment boundaries. | Model/framework adapters for the chosen attention implementation and checkpoint convention. |
| Consumption | Python mixture/packed readers and a compiled Rust reader with interoperable rank cursors. | Measured trainer throughput with decoding, collation and accelerator transfer included. |
| Agent access | Python, CLI, MCP, TypeScript clients, installed guides/prompts, bounded samples and source plugin launchers. | Remote execution when a concrete source or consumer requires it. |
| Domain data | Structured payloads, artifact references and preserved task/trajectory metadata. | Complete multimodal alignment, incremental retrieval and time-aware feature workflows. |

Source preparation currently keeps records in memory. Composition and packing write indexed shards; readers access assigned rows directly. Tokenization, domain decoding and model optimization remain explicit consumer operations. The Rust reader copies published rows; it does not change their semantics.

Run `python examples/pack_training.py` for the complete local data path and an independently calculated masked prediction objective. `examples/compose_training.py` demonstrates source weighting and rank continuation; `examples/measure_composition.py` separates preparation, composition and reading costs. `rust/compare_python.py` exercises shared native/Python cursors. The TypeScript example drives actual Python operations.

The next research result should establish when a data decision helps a consumer: a changed learning curve, transfer result, retrieval outcome or scientific prediction at comparable resources. Authored examples demonstrate working operations, not downstream capability gains. Optional Gym experience reuse is documented in [interfaces](integrations.md); default development and execution remain independent.
