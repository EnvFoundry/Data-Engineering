# Data Engineering for TypeScript harnesses

This Node client prepares releases, composes mixtures and packs supplied token sequences through the installed Python implementation. It has no runtime npm dependencies. The package is supplied in this repository; it is not required by Python or MCP consumers.

From the repository root:

```sh
python -m pip install -e .
cd sdk/typescript
npm ci
npm run build
node examples/pack-training.mjs
```

The example authors three token records, builds a release and mixture, and packs four blocks. Thirteen within-document targets retain twelve trainable positions after masking. It writes its output directory to stdout. Token IDs are illustrative; an actual training pipeline supplies its tokenizer's IDs and identity.

```ts
import { DataClient } from "@envfoundry/data-engineering";

const data = new DataClient({ python: "/path/to/venv/bin/python", cwd: "/data/study" });
const packed = await data.pack("mixture", "packed", {
  sequenceLength: 2048,
  tokenizer: "my-tokenizer",
  tokenizerRevision: "2026-11-20",
  lossMask: "/payload/loss_mask",
});
const page = await data.packed("packed", { limit: 3, includeTokens: true });
```

`input_ids[i]` predicts the already aligned `target_ids[i]`. A model adapter must implement the reported within-segment causal attention and target mask. The page method returns short previews; Python `PackedReader` or the native reader supplies full training blocks. See [packing semantics](../../docs/packing.md).

| Method | Result |
| --- | --- |
| `capabilities`, `guide` | Operations and concise operating guidance |
| `profile`, `schema`, `inspect` | Source structure, record contract and release summary |
| `build`, `compose` | Materialized releases and mixtures |
| `pack`, `packed` | Training blocks and bounded previews |
| `samples` | Paged release, mixture or packed records, with optional rank selection |

Paths resolve from `cwd`, or the host process's working directory. `python` names one executable; arguments are passed directly. The Python environment supplies installed connectors. `env` can set `PYTHONPATH` for an uninstalled source checkout.

Calls accept an optional final `{ signal, timeoutMs, maxOutputBytes }`. The default command deadline is five minutes; increase it for a large build. Complete responses are bounded at 16 MiB by default. Request pages for large populations. A failed Python operation throws `ProcessError` with `exitCode`, `stderr` and any `stdout`; output directories remain available for inspection.

Run `npm test` for the real Python data path. `npm pack` builds an ESM package with declaration files; install that tarball in a Node harness. The SDK uses [Node subprocesses](https://nodejs.org/api/child_process.html) and [TypeScript declaration exports](https://www.typescriptlang.org/docs/handbook/declaration-files/publishing.html). Node 20 or newer is required.
