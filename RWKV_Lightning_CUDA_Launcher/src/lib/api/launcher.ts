import type {
  DeviceSelection,
  QuantizationConfig,
  RuntimeConfig,
  TuningConfig,
} from "./types";

export type * from "./types";

/**
 * First-run defaults. Note that a runtime `start` decodes onto the previously
 * saved config, so omitting a field preserves the stored value — always submit
 * the complete form.
 */
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

/** MiSS is adam-only; `state` keeps whatever the user picked. */
export function effectiveOptimizer(
  config: TuningConfig,
): TuningConfig["optimizer"] {
  return config.method === "miss" ? "adam" : config.optimizer;
}

export const defaultTuning: TuningConfig = {
  method: "state",
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
  rank: 16,
  alpha: 16,
  targets: "all",
  state: "",
  resume: "",
};

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

/**
 * `visible_devices` is a three-state string on the wire. `inherit` omits the
 * field so the Agent keeps its `--card` / inherited env; `none` sends `""` to
 * explicitly inject nothing.
 */
export function deviceValue(selection: DeviceSelection) {
  switch (selection.mode) {
    case "inherit":
      return undefined;
    case "none":
      return "";
    case "explicit":
      return selection.value.trim();
  }
}

export function applyDevice<T extends { visible_devices?: string }>(
  body: T,
  selection: DeviceSelection,
): T {
  const value = deviceValue(selection);
  if (value === undefined) {
    // `visible_devices` must be absent (not empty) to inherit the Agent config.
    const rest = { ...body };
    delete rest.visible_devices;
    return rest;
  }
  return { ...body, visible_devices: value };
}

/** Rebuild the tri-state selector from a value echoed by the Agent. */
export function deviceSelection(value: string | undefined): DeviceSelection {
  if (value === undefined) return { mode: "inherit" };
  if (value === "") return { mode: "none" };
  return { mode: "explicit", value };
}
