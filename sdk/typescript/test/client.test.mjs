import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DataClient, ProcessError } from "../dist/index.js";
import { packTraining } from "../examples/pack-training.mjs";

const options = { python: process.env.PYTHON ?? "python" };
test("Python release, mixture, training blocks and rank page through the compiled client", async () => {
  const root = await mkdtemp(join(tmpdir(), "data sdk "));
  try {
    const output = join(root, "experiment");
    const result = await packTraining(output, options);
    assert.equal(result.packed.counts.records, 4);
    assert.equal(result.packed.counts.target_tokens, 13);
    assert.equal(result.packed.counts.trainable_tokens, 12);
    assert.deepEqual(result.page.items[0].input_ids, [1, 10, 11, 12]);
    assert.deepEqual(result.page.items[0].target_ids, [10, 11, 12, 2]);
    assert.deepEqual(result.page.items[0].loss_mask, [0, 1, 1, 1]);
    const client = new DataClient(options);
    const page = await client.samples(join(output, "mixture"), { rank: 1, worldSize: 2, limit: 1 });
    assert.equal(page.total, 1);
    assert.equal(page.items.length, 1);
    assert.equal(page.next_offset, null);
    const capabilities = await client.capabilities();
    assert.ok(capabilities);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("Python operation errors retain exit status and stderr", async () => {
  await assert.rejects(new DataClient(options).inspect("/path/that/does/not/exist"), error =>
    error instanceof ProcessError && error.exitCode === 2 && error.stderr.includes("error"));
});
