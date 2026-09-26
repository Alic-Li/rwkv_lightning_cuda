/**
 * Control-plane schema. Field names are the wire names from
 * `RWKV_Lightning_CUDA_Launcher/docs/control-plane-api.md` — do not rename
 * them, the Agent rejects unknown fields (`DisallowUnknownFields`).
 */

export const CAPABILITIES = [
  "runtime",
  "tuning_state",
  "tuning_miss",
  "quantization",
  "metrics",
  "fs",
  // The Agent can hand a node-local .pth to the runtime's multipart state
  // upload — a hop the native API has no path form for.
  "state_import",
  "host_dialog",
  "inference",
] as const;

/** `capabilities` is an open set: unknown members must be ignored, not rejected. */
export type Capability = (typeof CAPABILITIES)[number] | (string & {});

export type BackendKind = "agent" | "inference_only" | "";

/** `GET /api/v1/backends` entry. `token` is never returned by the Agent. */
export interface BackendView {
  id: string;
  name: string;
  base_url: string;
  has_token: boolean;
  kind: BackendKind;
  capabilities: Capability[];
  reachable: boolean;
  last_probe: number;
  probe_error: string;
}

export interface BackendListResponse {
  backends: BackendView[];
}

export interface AddBackendRequest {
  name: string;
  base_url: string;
  token: string;
}

/**
 * A partial update: omitted fields are left alone, and an empty `token` keeps
 * the stored one — the console never receives a token, so it cannot mean
 * "clear it".
 */
export interface UpdateBackendRequest {
  name?: string;
  base_url?: string;
  token?: string;
}

export type ProcessState =
  | "offline"
  | "starting"
  | "ready"
  | "stopping"
  | "error"
  | "running"
  | "completed";

export interface JobProgress {
  step?: number;
  total?: number;
  epoch?: number;
  epochs?: number;
  loss?: number;
  lr?: number;
  tokens_per_second?: number;
  eta?: number;
}

export interface ProcessStatus {
  status: ProcessState;
  running: boolean;
  error: string;
  logs: string[];
  elapsed: number;
  checkpoint: string;
  progress: JobProgress | null;
  losses: { step: number; loss: number }[];
  available?: boolean;
  miss_available?: boolean;
  output_path?: string;
}

export type JobID = "tuning" | "quantization";

/** Wire shape of `GET /api/v1/jobs`. */
export interface JobsResponse {
  jobs: Record<JobID, ProcessStatus>;
}

/** Unwrapped job map as stored in a node snapshot. */
export type Jobs = Record<JobID, ProcessStatus>;

/** `GET /api/v1/runtime` / `GET /api/v1/node` payload. */
export interface RuntimeState extends ProcessStatus {
  config: RuntimeConfig;
  base_url: string;
  translation_adapter: boolean;
  available: boolean;
  visible_devices: string;
  /**
   * The Agent's `--card` pin, empty when unpinned. A hard restriction, not a
   * default: the Agent refuses to start anything outside this spec, so
   * device pickers must lock to it.
   */
  card?: string;
  backend?: Record<string, unknown>;
  /**
   * False when the ready runtime was started outside this launcher (the
   * Agent only reports what it can see, it cannot stop a process it does
   * not own) — stop/restart must be disabled, start is port-conflicted.
   */
  managed?: boolean;
}

export interface NodeInfo extends RuntimeState {
  role: "agent" | string;
  version: string;
  capabilities: Capability[];
}

export interface GpuMetric {
  index: number;
  name: string;
  memory_total_bytes: number;
  memory_used_bytes: number;
  /** Optional: the Agent omits the field entirely when it cannot sample it. */
  utilization_percent?: number;
  /**
   * Of `memory_used_bytes`, what the inference runtime on that node holds.
   * Omitted when the driver cannot attribute memory per process, or when no
   * runtime of ours is running — the console then falls back to the device
   * spec, which can only split a card all-or-nothing.
   */
  own_memory_bytes?: number;
  temperature_c?: number;
  power_watts?: number;
}

export interface MetricsResponse {
  available: boolean;
  vendor: string;
  sampled_at?: number;
  reason?: string;
  gpus: GpuMetric[];
}

export interface FsEntry {
  name: string;
  is_dir: boolean;
  size?: number;
}

export interface FsRoots {
  roots: string[];
  /** Where the browser should open when the form field is empty: appDir. */
  default?: string;
}

export interface FsDirectory {
  path: string;
  /** Empty string means the whitelist root: cannot go further up. */
  parent: string;
  entries: FsEntry[];
}

export type FsResponse = FsRoots | FsDirectory;

/** `{}` answers with the whitelist roots; `{path}` with a directory listing. */
export function isFsRoots(response: FsResponse): response is FsRoots {
  return "roots" in response;
}

/**
 * `visible_devices` is a string with three distinct states:
 *  - `undefined`  -> omit the field, inherit the Agent `--card` / env
 *  - `""`         -> explicitly inject nothing
 *  - `"0,1"`      -> explicit device set
 */
export type DeviceSelection =
  | { mode: "inherit" }
  | { mode: "none" }
  | { mode: "explicit"; value: string };

export interface RuntimeConfig {
  model_path: string;
  vocab_path: string;
  port: string;
  password: string;
  use_wkv32: boolean;
  chunk_load: boolean;
  enable_dynamic_loading: boolean;
  chunk_size: number;
  state_db_path: string;
  tune_cache: string;
  visible_devices?: string;
}

export type TuningMethod = "state" | "miss";
export type TuningOptimizer = "adam" | "muon";

export const MISS_TARGETS = [
  "att.receptance.weight",
  "att.key.weight",
  "att.value.weight",
  "att.output.weight",
  "ffn.key.weight",
  "ffn.value.weight",
] as const;

export interface TuningConfig {
  method: TuningMethod;
  model: string;
  data: string;
  output: string;
  vocab: string;
  ctx: number;
  chunk: number;
  epochs: number;
  batch_size: number;
  max_steps: number;
  lr: number;
  lr_final: number;
  warmup_steps: number;
  save_every: number;
  seed: number;
  optimizer: TuningOptimizer;
  wkv_tape: boolean;
  rank: number;
  alpha: number;
  targets: string;
  state: string;
  resume: string;
  visible_devices?: string;
}

export type QuantizationFormat = "w8a16" | "w4a16";

export interface QuantizationConfig {
  input_path: string;
  output_path: string;
  format: QuantizationFormat;
  group_size: 32 | 128;
  visible_devices?: string;
}

/** Session/state files uploaded to the native server. */
export interface UploadedState {
  state_id: string;
  filename: string;
  size_bytes: number;
  tensor_count: number;
  created: number;
}

export interface AdapterEntry {
  id: string;
  version: string;
  manifest: { rank: number; scale: number; targets: unknown[] };
}

export interface AdapterListResponse {
  data: AdapterEntry[];
  ram_bytes: number;
  gpu_bytes: number;
  uploads: number;
}

export interface ModelListResponse {
  data: { id: string }[];
  available?: string[];
  loaded?: string | null;
}

/**
 * `POST /api/v1/runtime/load`: put the runtime (dynamic mode: also a model)
 * onto a card by restarting with `CUDA_VISIBLE_DEVICES` injected. Inherently
 * interrupting — CUDA binds devices at process init.
 */
export interface RuntimeLoadRequest {
  model?: string;
  visible_devices?: string;
}

export interface RuntimeLoadResponse {
  ok: boolean;
  visible_devices?: string;
  model?: string;
}
