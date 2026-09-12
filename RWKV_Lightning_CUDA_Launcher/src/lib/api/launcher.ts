import { request } from "./client";
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
}
export type RuntimeStatus =
  | "offline"
  | "starting"
  | "ready"
  | "stopping"
  | "error";
export interface ProcessStatus {
  status: RuntimeStatus | "running" | "completed";
  running: boolean;
  error: string;
  logs: string[];
  elapsed: number;
  checkpoint?: string;
  progress?: {
    step: number;
    total: number;
    epoch: number;
    epochs: number;
    loss: number;
    lr: number;
    tokens_per_second: number;
    eta: number;
  };
  losses?: { step: number; loss: number }[];
  available?: boolean;
}
export interface RuntimeState extends ProcessStatus {
  config?: RuntimeConfig;
  base_url?: string;
  backend?: {
    model?: { id?: string; name?: string; path?: string; loaded?: boolean };
    prefill_queue?: Record<string, number>;
  };
}
export const defaultRuntime: RuntimeConfig = {
  model_path: "",
  vocab_path: "./rwkv_vocab_v20230424.txt",
  port: "8000",
  password: "",
  use_wkv32: false,
  chunk_load: false,
  enable_dynamic_loading: false,
  chunk_size: 128,
  state_db_path: "rwkv_sessions.db",
  tune_cache: "",
};
export interface TuningConfig {
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
}
export const defaultTuning: TuningConfig = {
  model: "",
  data: "",
  output: "./state_output",
  vocab: "./rwkv_vocab_v20230424.txt",
  ctx: 128,
  chunk: 64,
  epochs: 1,
  batch_size: 1,
  max_steps: 0,
  lr: 1,
  lr_final: 0.01,
  warmup_steps: 10,
  save_every: 0,
  seed: 1234,
};
export class LauncherClient {
  getStatus(signal?: AbortSignal) {
    return request<RuntimeState>("/api/status", undefined, signal);
  }
  start(config: RuntimeConfig) {
    return request("/api/start", config);
  }
  stop() {
    return request("/api/stop", {});
  }
  restart() {
    return request("/api/restart", {});
  }
  pickFile() {
    return request<{ path: string }>("/api/pick-file", {});
  }
}
export class StateTuningClient {
  getStatus(signal?: AbortSignal) {
    return request<ProcessStatus>("/api/tuning/status", undefined, signal);
  }
  start(config: TuningConfig) {
    return request("/api/tuning/start", config);
  }
  stop() {
    return request("/api/tuning/stop", {});
  }
  validate(path: string) {
    return request<{ samples: number }>("/api/tuning/validate", { path });
  }
  openFolder() {
    return request("/api/tuning/open-folder", {});
  }
}
export const launcher = new LauncherClient();
export const tuning = new StateTuningClient();
