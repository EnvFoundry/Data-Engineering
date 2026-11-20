"""Installed, task-specific guidance for data research and consumption."""

from __future__ import annotations

import json


GUIDES = {
    "data-design": (
        "Design a data product around its consumer",
        """Describe the consumer, the decision its data supports and the unit sampled
from the world. A document, conversation, trajectory, specimen and sensor window
are different units. Keep their domain structure in payload; retain units,
timestamps, missing observations and the source revision needed to interpret it.
Choose split_group from the relationship that must stay together, not from an
arbitrary row number. Link derivatives with parents and preserve fixed splits.

Inspect a few records and run profile on canonical JSONL. Build a release from a
recipe with explicit source paths, intended use, seed and split weights. The
jsonl connector consumes the native envelope; gym-jsonl consumes Gym exports.
CLI source paths default to the recipe directory; MCP paths use its workspace.
inspect returns the release's counts, decisions and lineage. Source preparation
currently holds its grouping population in memory; large upstream curation can
produce the same record contract through an existing processing backend.

A useful data change has a proposed mechanism: broader coverage, better labels,
more relevant contexts, useful difficult cases or fresher feedback. Hold the
consumer and evaluation population fixed when estimating its effect. Compare at
the budget the question requires, such as optimizer tokens or actual compute.
Report what changed in the data and what changed in the consumer separately.

Deliver a runnable recipe, the source and output locations, an explanation of
the sampling unit and one consumer operation. A compact sample should illustrate
the payload and relationships; the complete dataset belongs in the data files.""",
    ),
    "mixture-design": (
        "Choose and measure a training distribution",
        """Compose pinned source releases and explicit partitions. Each weight is
a probability of choosing a record source. A 3:1 source ratio is not necessarily
a 3:1 token ratio: record lengths and repetition change the realized distribution.
Token budgets use supplied counts from a named tokenizer or encoder field; the
composer stops before the next whole record exceeds that ceiling.

Choose exhaustion deliberately. stop ends when a source supplies its final
record; renormalize continues with remaining sources; repeat starts a new source
permutation until the budget is reached. Record counts, source cycles and unique
coverage answer different questions. More draws from a small source increase its
influence without adding independent examples.

Use compose to materialize indexed shards and MixtureReader to consume them.
Read the realized source counts and stop reason before interpreting a recipe as
the intended distribution. For a source-weight experiment, keep source snapshots,
consumer settings and evaluation cases fixed while changing the selected weights.
Measure the learner at comparable budgets; data throughput or packing utilization
alone does not establish better learning.

When introducing synthetic examples, preserve their generation condition and
relationship to original tasks. An outcome selected by the same score used for
evaluation can change what the comparison measures. Choose independent cases for
the downstream question and retain unsuccessful or unscored experience when it is
part of the intended consumer distribution.""",
    ),
    "causal-packing": (
        "Pack token sequences without changing the objective",
        """Start with token arrays produced by the consumer's chosen tokenizer and
formatting rules. Include any intended BOS, EOS, role or modality tokens upstream.
For supervised fine-tuning, define the loss mask on target-token positions so the
objective can distinguish prompt context from response targets. Keep tokenization
and formatting settings with the release used by the learner.

The packer emits already-aligned causal pairs: input_ids[i] predicts target_ids[i].
For tokens [a,b,c], the prediction pairs are (a,b) and (b,c); the first token has
no predecessor. Every within-record next-token pair is retained, including at
block boundaries.
Do not shift these targets a second time in a model wrapper. Padding and masked
targets contribute no loss. The consumer must implement the recorded segment
boundaries as block-diagonal causal attention or equivalent segmented execution;
a loss mask by itself does not prevent attention across documents.

Packing multiple documents changes tensor layout, not their source identities.
Use segment metadata and source offsets to connect predictions back to records.
A long document split across blocks has a shorter available context in later
blocks unless the consumer explicitly supplies context. State this choice when
comparing sequence lengths or packing policies.

Run the authored packing example, inspect one block's inputs, targets, masks and
segments, then adapt the consumer's tensor collation to that convention. Compare
valid prediction pairs, supervised targets, padding and source coverage. These
describe the delivered objective; a learning experiment measures its usefulness.""",
    ),
    "consumption-restart": (
        "Consume shards and resume the same training stream",
        """A reader assigns each rank a disjoint stride through the materialized
global stream. The default keeps the tail, so ranks may differ by one record or
block. Equal-rank tail dropping is a separate option; it changes consumed coverage.
Map distributed rank and local loader worker to one combined rank/world-size pair
when each worker opens its own reader.

Save reader.state_dict() with the model and optimizer state for the data actually
consumed. Restore through load_state_dict() on the same release and rank topology.
The next read follows the saved cursor. A model checkpoint taken before a batch
and a reader cursor saved after it would silently skip that batch on restart.
Prefetch queues and gradient accumulation belong to the same checkpoint boundary.

Use MixtureReader for record draws and PackedReader for materialized training
blocks. The optional Rust ai-data-read command streams either format and exchanges
cursor state with the Python readers. Keep the source recipe, encoding convention
and pack settings available
alongside the checkpoint. Changing world size or packing changes the consumption
plan; do not interpret an old per-rank cursor as a cursor in the new plan.

For throughput work, measure source preparation, composition, packing and reading
separately. Include record sizes, filesystem, worker count and cache conditions.
An optional native reader reads the same indexed data format; it does not
change sampling weights, loss masks or the learner's objective.""",
    ),
}


def get_guide(topic: str = "data-design") -> dict:
    """Read the guide for the current data decision."""
    if topic not in GUIDES:
        raise ValueError(f"unknown guide; choose one of: {', '.join(GUIDES)}")
    title, content = GUIDES[topic]
    return {"topic": topic, "title": title, "content": content}


def design_data_product(consumer: str, sources: str, budget: str) -> str:
    """Turn a consumer workload into a runnable data product."""
    context = json.dumps({"consumer": consumer, "sources": sources, "budget": budget}, ensure_ascii=False)
    return ("Prepare a usable data product for this workload: " + context + "\n"
            "Read data://guides/data-design and discover the installed operations. Identify the sampling "
            "unit, source revisions, payload structure and intended partition. Inspect representative "
            "records, build the recipe and run the available consumer path. Return the recipe and output "
            "locations, realized composition and the few choices that affect downstream behavior.")


def prepare_training_data(mixture: str, encoding: str, objective: str) -> str:
    """Deliver a packed stream with an explicit causal objective."""
    context = json.dumps({"mixture": mixture, "encoding": encoding, "objective": objective}, ensure_ascii=False)
    return ("Prepare this training stream: " + context + "\n"
            "Read data://guides/causal-packing and data://guides/consumption-restart. Use the actual token "
            "and loss-mask fields, state the label alignment and attention convention, and materialize "
            "the blocks. Show one short decoded or token-level example and the consumer's reader/checkpoint "
            "usage. Report supervised targets, padding and source coverage from the produced stream.")


def compare_data_recipes(question: str, baseline: str, candidate: str) -> str:
    """Study a data change through its effect on the intended consumer."""
    context = json.dumps({"question": question, "baseline": baseline, "candidate": candidate}, ensure_ascii=False)
    return ("Investigate this proposed data change: " + context + "\n"
            "Read data://guides/mixture-design. Identify the changed sampling or transformation choice "
            "and the mechanism it could affect. Produce both data streams under the stated budget, "
            "then use the available consumer and fixed evaluation population. Explain the observed "
            "data difference separately from the measured learning or retrieval effect. If no consumer "
            "run is available, return the prepared experiment without inventing a downstream result.")
