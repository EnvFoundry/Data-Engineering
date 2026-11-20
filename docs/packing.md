# Causal training blocks

`pack` converts tokenized mixture records into fixed-length prediction blocks.
It supports pretraining and supervised fine-tuning without choosing a model,
tokenizer, chat template or trainer for the consumer. Source records already
contain the intended token IDs, including any BOS, EOS, role or modality tokens.
The tokenizer name and revision identify that upstream encoding.

```sh
python examples/pack_training.py --output runs/packed-training
ai-data samples runs/packed-training/packed --limit 1 --include-payload
```

The authored example has 27 source tokens in three documents. Packing delivers
24 next-token pairs, of which 21 receive loss, in five blocks of length five.
One slot is padding. A fixed-logit calculation produces the same summed loss
before and after packing, including the long document's block boundaries.
These fixtures demonstrate the objective and reader behavior; they do not
measure a trained model.

## Encode the objective

One canonical record can contain:

```json
{
  "payload": {
    "input_ids": [1, 8, 4, 6, 2],
    "loss_mask": [0, 0, 0, 1, 1]
  }
}
```

This is a payload excerpt; the complete record retains its usual ID, source,
group, split and declared-use fields. The loss mask addresses **original target
token positions**. Here `[1,8,4]` is context and `[6,2]` supplies supervised targets.
Every mask value is zero or one. If no mask field is requested, every token
after the first is a target. The first token has no predecessor within its
record. Empty and one-token records contribute no prediction pairs and are
counted as `documents_without_targets`.

```sh
ai-data pack runs/mixture --output runs/packed \
  --sequence-length 2048 \
  --tokenizer my-tokenizer --tokenizer-revision revision-7 \
  --tokens /payload/input_ids --loss-mask /payload/loss_mask \
  --pad-token-id 0 --shard-records 4096
```

`tokens` and `loss_mask` are JSON pointers into each canonical record. Token
values must be nonnegative integers. Omitting `--loss-mask` gives the pretraining
objective. The caller's tokenizer controls the vocabulary and padding token.
The packer applies the same declared encoding to every record in the mixture.

## Prediction and attention semantics

Each output position is an already-aligned pair:
**`input_ids[i]` predicts `target_ids[i]`. Do not shift the targets again.**

| Original tokens | Block length | Output inputs | Output targets |
| --- | ---: | --- | --- |
| `[a,b,c,d,e]` | 2 | `[a,b]`, `[c,d]` | `[b,c]`, `[d,e]` |
| `[a,b,c]`, `[x,y]` | 3 | `[a,b,x]` | `[b,c,y]` |

Every within-record next-token pair appears once. In the first row, `c` is a
target in the first block and becomes context in the second; its prediction
loss is not duplicated. In the second row, no target connects `c` to `x`.
No implicit separator is inserted. Long documents continue in the next block
with fresh attention context. Position IDs restart at zero for each block
segment; source offsets preserve the original token positions.

| Block field | Meaning |
| --- | --- |
| `sample_index` | Global packed-block ordinal. |
| `input_ids`, `target_ids` | Equal-length, prealigned input and next-token arrays. |
| `loss_mask` | One for supervised target positions; zero for prompt-only targets and padding. |
| `attention_mask` | One for occupied positions; zero for padding. This is a padding mask, not the complete attention rule. |
| `segment_ids` | Equal IDs identify one document fragment inside the block; padding has ID `-1`. |
| `position_ids` | Positions within each block segment, starting at zero. |
| `cu_seqlens` | Segment boundaries over nonpadding positions: `[0, end₀, end₁, ...]`. |
| `segments` | Block range, original mixture sample/source/group, and original input-token range. `token_end` is exclusive. |

Attention from query position `i` to key position `j` is allowed when `j <= i`,
both positions are occupied, and their segment IDs agree. Use that
block-diagonal causal mask, or run each segment separately. A one-dimensional
padding mask alone would allow later documents to attend to earlier documents.
A variable-length attention implementation can consume the unpadded inputs and
segment boundaries; its adapter owns the tensor layout.

## Connect a learner

```python
from ai_data import PackedReader

with PackedReader("runs/packed", rank=0, world_size=4) as reader:
    block = next(reader)
    # Tensorize input_ids, target_ids, loss_mask and segment information.
    # Save this cursor with the model state after consuming the block.
    data_state = reader.state_dict()
```

For a model wrapper that already shifts its `labels` internally, request logits
without passing `target_ids` as labels. The following PyTorch adapter evaluates
segments independently, then computes the explicitly aligned cross-entropy.
`model` is a caller-supplied causal model returning logits at every input
position. This portable example prioritizes the objective; a fused segmented
attention adapter can process the complete packed tensor in one forward call.

```python
import torch
import torch.nn.functional as F

def block_objective(model, block, device):
    loss_sum = None
    target_count = 0
    for segment in block["segments"]:
        start, end = segment["start"], segment["end"]
        x = torch.tensor([block["input_ids"][start:end]], device=device)
        y = torch.tensor(block["target_ids"][start:end], device=device)
        mask = torch.tensor(block["loss_mask"][start:end], device=device, dtype=torch.bool)
        if not mask.any():
            continue
        positions = torch.arange(end - start, device=device).unsqueeze(0)
        logits = model(input_ids=x, position_ids=positions, use_cache=False).logits[0]
        value = F.cross_entropy(logits[mask], y[mask], reduction="sum")
        loss_sum = value if loss_sum is None else loss_sum + value
        target_count += int(mask.sum())
    return loss_sum, target_count
```

Sum losses and target counts over the intended optimizer batch, then divide
once by the total target count. Averaging per-block means would overweight
blocks with few supervised tokens. A batch with zero supervised targets has no
objective contribution. Distributed and gradient-accumulation adapters must
use the intended global denominator and their framework's gradient-reduction
convention. [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
defines this summed-loss operation; the
[CodeGen model interface](https://huggingface.co/docs/transformers/model_doc/codegen#transformers.CodeGenForCausalLM.forward)
documents an example of internal label shifting.

## Reading, paging and scale

The output contains indexed JSONL shards and a manifest. `counts.records` is
the number of blocks; `documents` counts mixture draws, `source_tokens` counts
their original tokens, `target_tokens` counts next-token pairs, and
`trainable_tokens` counts the selected targets. Source-level counts retain the
realized training distribution. The full final block is padded; packing itself
does not drop targets.

`PackedReader` assigns disjoint strided block ordinals to ranks and restores
the exact next block through `state_dict()` / `load_state_dict()`. It performs
no prefetch. By default it retains every block, so rank lengths can differ by
one. `drop_remainder=True` drops the global block tail to equalize rank lengths;
use it only when that consumption choice is intended. Save any downstream
prefetch and accumulation state at the same model-checkpoint boundary.

```sh
ai-data packed runs/packed --offset 0 --limit 2 --include-tokens
ai-data samples runs/packed --rank 1 --world-size 4 --limit 2
ai-data samples runs/mixture --limit 2 --include-payload
ai-data samples runs/source --split validation --limit 2
ai-data guide causal-packing
```

The generic sample view defaults to identities and field names. Requested
payload previews stop at 2,000 characters; packed token previews stop at 64
positions. The Python reader returns complete blocks. `offset` counts samples
within the selected rank, and `next_offset` continues that page.

Packing reads one mixture record at a time and writes each completed block.
Memory grows with the largest tokenized record, one block, and source/shard
metadata. It never materializes the complete training stream. Immutable input
mixtures, tokenizer identity, packing settings and actual output determine the
packed release. Tokenization, model execution and optimization remain with the
consumer.
