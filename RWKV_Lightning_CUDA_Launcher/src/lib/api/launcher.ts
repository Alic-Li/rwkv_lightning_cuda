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
export interface QuantizationStatus extends ProcessStatus {
  output_path?: string;
}
export interface QuantizationConfig {
  input_path: string;
  output_path: string;
  format: "w8a16" | "w4a16";
  group_size: 32 | 128;
}
export const defaultQuantization: QuantizationConfig = {
  input_path: "",
  output_path: "",
  format: "w4a16",
  group_size: 128,
};
export function suggestedQuantizedPath(
  input: string,
  format: QuantizationConfig["format"],
) {
  const base = input.replace(/\.pth$/i, "");
  return base ? `${base}.${format}.rwkvq` : "";
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
  optimizer: "adam" | "muon";
  wkv_tape: boolean;
}
export const defaultTuning: TuningConfig = {
  model: "",
  data: "",
  output: "./state_output",
  vocab: "./rwkv_vocab_v20230424.txt",
  ctx: 512,
  chunk: 128,
  epochs: 1,
  batch_size: 16,
  max_steps: 0,
  lr: 0.0005,
  lr_final: 0.0001,
  warmup_steps: 10,
  save_every: 100,
  seed: 1234,
  optimizer: "adam",
  wkv_tape: false,
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
  pickDirectory() {
    return request<{ path: string }>("/api/pick-directory", {});
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
export class QuantizationClient {
  getStatus(signal?: AbortSignal) {
    return request<QuantizationStatus>(
      "/api/quantization/status",
      undefined,
      signal,
    );
  }
  start(config: QuantizationConfig) {
    return request("/api/quantization/start", config);
  }
  stop() {
    return request("/api/quantization/stop", {});
  }
}
export const launcher = new LauncherClient();
export const tuning = new StateTuningClient();
export const quantization = new QuantizationClient();
