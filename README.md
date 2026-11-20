# Data Engineering

**Prepare data, construct learning distributions, and deliver model inputs with explicit semantics.**

[中文](README.zh-CN.md) · [Consumption](docs/consumption.md) · [Sequence packing](docs/packing.md) · [Research](docs/research.md) · [Interfaces](docs/integrations.md)

Data Engineering is an independent research environment for the AI data lifecycle: pretraining, post-training, retrieval, agents, multimodal systems and scientific workloads. Records preserve domain structure; source preparation, sampling and consumer formatting are separate operations. A dataset change becomes a research result when it changes the consumer's behavior under a stated budget.

## Run the data-to-training workflow

Python 3.11 or newer. The Python core uses the standard library.

```sh
git clone https://github.com/EnvFoundry/Data-Engineering.git
cd Data-Engineering
python -m pip install -e .
python examples/pack_training.py --output runs/packed-training
```

The standalone example builds authored token records, composes a distribution, packs causal prediction pairs and resumes a partitioned reader. It demonstrates data and objective semantics; it does not train a model or report a learning improvement.

| Operation | Result |
| --- | --- |
| Prepare | Source releases with explicit partitions, group relationships, duplicate decisions and lineage. |
| Compose | Weighted draws from selected source partitions, record or supplied-token budgets, indexed shards and realized mixture counts. |
| Pack | Fixed-length causal prediction pairs, target loss masks, document segments, token positions and source references. |
| Consume | Python or optional Rust indexed readers, rank assignment and restorable cursors. |
| Inspect | Bounded record pages and installed research guides through CLI or MCP. |

```sh
ai-data profile examples/records.jsonl
ai-data build examples/local.recipe.json --output runs/source
python examples/compose_training.py --output runs/mixture-example
ai-data samples runs/mixture-example/mixture --limit 3
```

Use `python -m ai_data` in place of `ai-data` where needed. The [record contract](docs/contracts.md) preserves preference pairs, trajectories, image/audio references, sensor streams and tables without forcing them into chat messages. Modality decoding and model-specific tokenization remain consumer operations.

## Preserve the learning objective

Packing consumes token IDs produced by the chosen tokenizer and records its name and revision. Each `input_ids[i]` predicts `target_ids[i]`; the stored mask selects which targets contribute to the loss. These targets are already aligned with logits. Models that shift labels internally need the explicit adapter described in [sequence packing](docs/packing.md).

Segments retain document boundaries and support separate causal attention. Long records preserve every within-record next-token pair across block boundaries. Padding, masked targets and rank tail handling remain explicit. Save the reader cursor alongside the model and optimizer checkpoint so consumption and learning resume together.

```python
from ai_data import MixtureReader

with MixtureReader("runs/mixture-example/mixture", rank=1, world_size=4) as reader:
    record = next(reader)["record"]
    cursor = reader.state_dict()
```

[Consumption](docs/consumption.md) explains source exhaustion, weighting and rank assignment. [Native reading](docs/native-reader.md) describes the optional Rust executable and its shared cursor format. Python remains sufficient for the complete workflow.

## Work with a harness or Agent

```sh
python -m pip install -e '.[mcp]'
ai-data capabilities
ai-data connect --root /absolute/path/to/data-workspace --output data-engineering.local.json
```

CLI, Python and MCP use the same data operations. Agent responses begin with counts and small pages; token arrays and payload details are requested when needed. Installed method resources and prompts cover distribution design, packing and interpretation of downstream results. The repository includes a common operating Skill and native plugin manifests; see [interfaces and installation](docs/integrations.md).

The [TypeScript SDK](sdk/typescript/README.md) starts the installed Python package with typed methods and ordinary argument arrays. It supports cancellation, process errors and bounded responses without duplicating data processing. The optional [Rust reader](docs/native-reader.md) streams rows from byte indexes for consumers that need a native process.

## Research and extension

Study source allocation, synthetic candidate selection, task difficulty, experience reuse, multimodal alignment and feedback freshness through a concrete consumer experiment. Separate the chosen distribution, systems throughput and learning outcomes; faster reading alone does not establish better learning. [Research methods](docs/research.md) and [public frontier workloads](docs/frontier-workloads.md) connect these decisions to published AI work.

Connectors are installed Python entry points under `ai_data.connectors`. Canonical JSONL and Gym trajectory imports are included. Optional task-to-experience examples are described in [interfaces](docs/integrations.md); no other EnvFoundry repository is required for the default workflow.

Source preparation currently holds its input records in memory. Composition, packing and consumption use disk-backed shards. [Implementation status](docs/roadmap.md) states the working paths and the next workload-driven extensions.

Implementation lives in `src/ai_data/`, `rust/` and `sdk/typescript/`; runnable workflows live in `examples/`. Read [AGENTS.md](AGENTS.md) before contributing. MIT license.
