# Research agenda

The object of study is a data policy: how sources are acquired, transformed, selected and sampled for a particular consumer. Its value is the resulting capability at a specified cost. The same policy can help one domain and reduce performance in another, so experiments report the affected task distribution alongside aggregate results.

[Public frontier workloads](frontier-workloads.md) ground the engineering requirements. [Consumption](consumption.md) describes the current experimental machinery: pinned source releases, budgeted mixtures, indexed shards and restorable rank readers. Verifiable Agent Gym adds finite task suites, repeated attempts, policy identity and environment feedback.

## Decisions worth studying

| Decision | Intervention and mechanism | Evidence that changes the next data decision |
| --- | --- | --- |
| Source allocation | Vary domain/language/source weights at matched consumption budgets; separate additional coverage from repeated exposure. | Learning curves and held-out domain results show where the next unit of compute should be allocated. |
| Synthetic candidate selection | Vary teacher diversity, attempts per task and selection rules separately. Track shared source and generator ancestry. | Accepted candidates improve a consumer on held-out mechanisms or source families enough to justify generation and verification cost. |
| Task distribution and curriculum | Change task families, difficulty and composition while fixing the learner and interaction budget. | Transfer and sample efficiency improve; per-family results identify which behaviors were learned. |
| Experience reuse | Compare fresh and replayed trajectories at explicit policy lag, feedback type and generation cost. | Reuse yields useful learning per unit of compute, with the effect of stale behavior and selection made measurable. |
| Representation and alignment | Change document packing, temporal windows, sensor alignment or multimodal sampling while preserving the underlying observations. | Relevant downstream outcomes identify which boundaries, timing or spatial relations the consumer needs. |
| Feedback and refresh | Compare collection and update policies using event time, availability time and versioned consumer outcomes. | Adaptation, retrieval freshness or scientific information gain improves at a measured acquisition and update cost. |

Each question can use a small working experiment before requiring large infrastructure. Full-scale training, multimodal decoding and online acquisition enter through consumer and source adapters appropriate to that experiment.

## Studies enabled by the current implementation

**Mixture and delivery.** Build two or more source releases, pin them in a recipe and vary weights under a record or supplied-token budget. Use `MixtureReader` to keep sample assignment and restart behavior explicit. First measure source exposure, bytes, throughput and memory with [the composition script](../examples/measure_composition.py); then connect the same products to a fixed trainer and measure learning. Source draw weights apply to records, so differences in record length must be measured when interpreting token exposure.

**Task data to grouped experience.** Prepare a versioned SQL task pack or generated scientific task suite. Hold the environment and task distribution fixed, collect several policy attempts per task and compare how candidate count or selection changes useful experience per model call. Keep task groups intact across ranks. The runnable starting points are Gym's [SQL workload](https://github.com/Cherish0925/Verifiable-Agent-Gym/tree/main/examples/sql-workload) and [batched rollout example](https://github.com/Cherish0925/Verifiable-Agent-Gym/blob/main/examples/batched_rollouts.py).

**Feedback to the next data product.** Import rollout exports through `gym-jsonl`, retain policy/task identities and compare selection or replay recipes. Measure downstream effect with a fixed consumer or the necessary crossed data/consumer comparisons. Scored failures, truncations and environment errors lead to different data decisions; preserve their observed status when constructing the next product.

**Encoding and objective equivalence.** Use packed token positions, target masks and segment boundaries to compare sequence layouts while preserving the intended prediction pairs. The [packing example](packing.md) computes the same masked objective before and after packing. Then measure actual learner throughput and held-out behavior.

These paths supply the data and execution operations. Their next research result is a measured consumer effect, such as a changed learning curve or transfer result.

The [experience-cycle example](../examples/experience_cycle.py) connects the last two paths: SQL task release → eight grouped attempts → trajectory release → three mixture shards → two reader ranks and restart. It gives selection and replay studies an executable starting point.

## Experimental unit and attribution

Start with the target behavior, consumer, data-generating process and independent sample unit. Count independent documents, entities, scenes, task mechanisms or experiments alongside rows and tokens. Related fragments and generated variants retain their parent group. Choose time, ancestry, domain or mechanism holdouts according to the generalization question.

A useful experiment records the manipulated data decision, source and product versions, consumer/policy revision, execution conditions, consumed units and downstream outcome. Compare at matched budgets or report the cost/performance curve. Repeat runs when sampling or optimization variation could change the decision. When data and consumer both change, fix one or use crossed comparisons to identify the cause.

Measure three stages separately: **construction** determines which data exists and its acquisition cost; **delivery** determines whether the consumer can obtain the intended stream efficiently; **learning or use** determines whether the data improves the system. Fast composition enables a larger experiment but is a different result from improved model capability.

Scientific and embodied workloads additionally retain units, calibration, uncertainty, synchronization, coordinate systems and intervention history. An observed measurement, a simulation output and a model estimate have different roles in the inference even when their numerical values agree.

## Reusable outputs

Store methods, recipes, adapters, compact results and the interpretation of changed decisions in Git. Reference large data and consumer checkpoints by version. A completed study should leave another researcher able to repeat the intervention, understand its cost and apply the resulting method to a new workload.
