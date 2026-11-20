import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { DataClient } from "../dist/index.js";

export async function packTraining(output, options = {}) {
  await mkdir(output, { recursive: false });
  const client = new DataClient(options);
  // Authored token IDs demonstrate packing; no trained tokenizer or model is claimed.
  const documents = [[1, 10, 11, 12, 2], [1, 20, 21, 2], [1, 30, 31, 32, 33, 34, 2]];
  const rows = documents.map((ids, index) => ({
    schema_version: "1.0", id: `document:${index}`, kind: "document",
    payload: { input_ids: ids, loss_mask: ids.map((_, token) => Number(token > (index === 0 ? 1 : 0))) },
    source: { uri: "example://typescript-packing", revision: "1" },
    split_group: `document:${index}`, split: "train", allowed_uses: ["train"],
  }));
  await writeFile(join(output, "records.jsonl"), rows.map(row => JSON.stringify(row) + "\n").join(""));
  const recipe = { schema_version: "1.0", name: "token-records", use: "train", seed: "1",
    sources: [{ connector: "jsonl", path: "records.jsonl" }], splits: { train: 1, validation: 0, test: 0 } };
  await writeFile(join(output, "source.json"), JSON.stringify(recipe));
  const release = await client.build(join(output, "source.json"), join(output, "source"));
  const mixture = { schema_version: "1.0", name: "packing-stream", seed: "1", exhaustion: "stop", shuffle: false,
    sources: [{ name: "tokens", path: "source", release_id: release.release_id, split: "train", weight: 1 }],
    budget: { unit: "records", value: 3 }, shard_records: 2 };
  await writeFile(join(output, "mixture.json"), JSON.stringify(mixture));
  await client.compose(join(output, "mixture.json"), join(output, "mixture"));
  const packed = await client.pack(join(output, "mixture"), join(output, "packed"), {
    sequenceLength: 4, tokenizer: "authored-token-ids", tokenizerRevision: "1",
    lossMask: "/payload/loss_mask", shardRecords: 2,
  });
  const page = await client.packed(join(output, "packed"), { limit: 5, includeTokens: true });
  return { output, packed, page };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const parent = process.argv[2] ? null : await mkdtemp(join(tmpdir(), "ai-data-typescript-"));
  const output = process.argv[2] ? resolve(process.argv[2]) : join(parent, "study");
  const result = await packTraining(output, { python: process.env.PYTHON ?? "python" });
  console.log(JSON.stringify({ output, counts: result.packed.counts, page: result.page }, null, 2));
}
