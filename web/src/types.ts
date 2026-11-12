import type { components } from './api.generated';
export type RewardSpec = components['schemas']['RewardSpec'];
export interface Run { run_id: string; name: string; material: string; model_id: string; framework: string;
  framework_version: string; recipe_id: string; snapshot_id: string; group_count: number; completion_count: number;
  capture: {tier: string; source: string; sampling_probability: number}; coverage: {fraction: number|null; captured_groups: number}; }
export interface Metric {generation_step: number; name: string; value: number; source: string}
export interface Group { group_id: string; prompt: string|null; prompt_id: string; generation_step: number;
  expected_group_size: number; recorded: number; variance: number|null; reward_mean: number|null; }
export interface Completion {completion_id: string; group_id: string; sequence_index: number; text: string|null;
  reward_total: number|null; reward_status: string; reward_components: Record<string, number|null>;
  inputs: Record<string, number|boolean|null>; grade: {source: string; value: number}|null;
  completion_token_count: number|null; advantage: number|null; stop_reason: string|null;}
export interface GroupDetail extends Group {run_id: string; completions: Completion[]; recipe_id: string;
  advantage_available: boolean; unavailable: {message: string; code: string}|null;
  batch: {generation_step: number; policy_version: string; generation_batch_id: string}; snapshot_id: string;}
export interface Anomaly {id: string; title: string; why_flagged: string; supporting_group_ids: string[];
  limits: string; detector_version: string; coverage: {fraction: number|null; captured_groups: number};}
export interface Change {completion_id: string; group_id: string; reward_before: number; reward_after: number;
  rank_before: number; rank_after: number; advantage_before: number; advantage_after: number; affected: boolean; sign_flip: boolean;}
export interface Analysis {analysis_id: string; status: string; result_hash?: string;
  run_id?: string; reward_spec?: RewardSpec; group_filter?: {group_ids?: string[]|null};
  summary?: {groups: number; completions: number; affected_groups: number; rank_reversals: number; sign_flips: number; zero_variance_rescued: number};
  changes?: Change[]; error?: {code: string; message: string}; coverage?: {selected_groups: number};}
export interface Contract {id: string; origin: {reviewed: boolean; run_id: string; analysis_id: string|null};
  fixtures: Completion[]; assertions: Record<string, unknown>[];}
export interface ContractResult {contract_id: string; passed: boolean; scores: Record<string, number>;
  assertions: {type: string; passed: boolean; actual_delta: number}[];}
