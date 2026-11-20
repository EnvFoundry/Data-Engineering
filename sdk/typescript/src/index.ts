import { PythonCLI, flag, type CallOptions, type JsonObject, type ProcessOptions } from "./process.js";
export { ProcessError, PythonCLI } from "./process.js";
export type { CallOptions, Json, JsonObject, ProcessOptions } from "./process.js";

export interface DataRecord {
  schema_version: "1.0";
  id: string;
  kind: string;
  payload: JsonObject;
  source: { uri: string; revision: string };
  split_group: string;
  allowed_uses: ("train" | "evaluate" | "retrieve" | "analyze" | "redistribute")[];
  parents?: string[];
  split?: "train" | "validation" | "test" | null;
  metadata?: JsonObject;
}

export interface PackOptions {
  sequenceLength: number;
  tokenizer: string;
  tokenizerRevision: string;
  /** JSON pointer into each source record. Default /payload/input_ids. */
  tokens?: string;
  /** JSON pointer to binary target-position masks. */
  lossMask?: string;
  padTokenId?: number;
  shardRecords?: number;
}

export interface PackedPageOptions {
  offset?: number;
  limit?: number;
  includeTokens?: boolean;
}

export interface PackingCounts {
  documents: number; documents_without_targets: number; records: number;
  source_tokens: number; target_tokens: number; trainable_tokens: number; padding_tokens: number;
}
export interface PackingSummary {
  format: "ai-data-packed"; name: string; release_id: string; source_mixture: string;
  tokenizer: { name: string; revision: string }; sequence_length: number; counts: PackingCounts;
  label_alignment: "input_at_i_predicts_target_at_i"; attention: "causal_within_segment";
  positions: "reset_per_block_segment"; shards: number;
}
export interface PackedItem {
  sample_index: number; tokens: number; trainable_tokens: number; segments: JsonObject[]; segment_count: number;
  input_ids?: number[]; target_ids?: number[]; loss_mask?: number[]; segment_ids?: number[]; position_ids?: number[];
  tokens_truncated?: boolean;
}
export interface PackedPage extends PackingSummary {
  offset: number; limit: number; items: PackedItem[]; next_offset: number | null;
}

/** Calls the installed Python core. Paths resolve from options.cwd. */
export class DataClient {
  readonly cli: PythonCLI;
  constructor(options: ProcessOptions = {}) { this.cli = new PythonCLI("ai_data", options); }
  capabilities(call?: CallOptions) { return this.cli.json(["capabilities"], call); }
  guide(topic: string, call?: CallOptions) { return this.cli.json(["guide", topic], call); }
  profile(path: string, call?: CallOptions) { return this.cli.json(["profile", path], call); }
  inspect(path: string, call?: CallOptions) { return this.cli.json(["inspect", path], call); }
  schema(call?: CallOptions) { return this.cli.json(["schema"], call); }

  build(recipe: string, output: string, options: { root?: string; allowPlugins?: boolean } = {}, call?: CallOptions) {
    const args = ["build", recipe, "--output", output];
    flag(args, "root", options.root); flag(args, "allow-plugins", options.allowPlugins);
    return this.cli.json(args, call);
  }

  compose(recipe: string, output: string, options: { root?: string } = {}, call?: CallOptions) {
    const args = ["compose", recipe, "--output", output];
    flag(args, "root", options.root);
    return this.cli.json(args, call);
  }

  pack(mixture: string, output: string, options: PackOptions, call?: CallOptions) {
    const args = ["pack", mixture, "--output", output, "--sequence-length", String(options.sequenceLength),
      "--tokenizer", options.tokenizer, "--tokenizer-revision", options.tokenizerRevision];
    flag(args, "tokens", options.tokens); flag(args, "loss-mask", options.lossMask);
    flag(args, "pad-token-id", options.padTokenId); flag(args, "shard-records", options.shardRecords);
    return this.cli.json<PackingSummary>(args, call);
  }

  packed(path: string, options: PackedPageOptions = {}, call?: CallOptions) {
    const args = ["packed", path];
    flag(args, "offset", options.offset); flag(args, "limit", options.limit);
    flag(args, "include-tokens", options.includeTokens);
    return this.cli.json<PackedPage>(args, call);
  }

  samples(path: string, options: { split?: "train" | "validation" | "test"; offset?: number;
      limit?: number; rank?: number; worldSize?: number; includePayload?: boolean } = {}, call?: CallOptions) {
    const args = ["samples", path];
    flag(args, "split", options.split); flag(args, "offset", options.offset); flag(args, "limit", options.limit);
    flag(args, "rank", options.rank); flag(args, "world-size", options.worldSize); flag(args, "include-payload", options.includePayload);
    return this.cli.json(args, call);
  }
}
