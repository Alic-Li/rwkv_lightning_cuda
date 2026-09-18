import { useEffect, useRef, useState } from "react";
import {
  Play,
  Square,
  FolderOpen,
  CheckCircle2,
  Circle,
  RotateCcw,
  Zap,
  Terminal,
} from "lucide-react";
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import {
  defaultTuning,
  tuning,
  launcher,
  type TuningConfig,
} from "../lib/api/launcher";
import { useRuntime } from "../stores/runtime";
import { storage } from "../stores/settings";
import {
  Panel,
  Field,
  PathField,
  Console,
  ErrorPanel,
  Dialog,
  CopyButton,
} from "../components/common";
const useTuningForm = create(
  persist<{
    config: TuningConfig;
    set: (v: Partial<TuningConfig>) => void;
    resetParameters: () => void;
  }>(
    (set) => ({
      config: defaultTuning,
      set: (v) => set((s) => ({ config: { ...s.config, ...v } })),
      resetParameters: () =>
        set((s) => ({
          config: {
            ...defaultTuning,
            model: s.config.model,
            data: s.config.data,
            output: s.config.output,
            vocab: s.config.vocab,
          },
        })),
    }),
    { name: "rwkv-tuning-form-v2", storage: createJSONStorage(() => storage) },
  ),
);
const numeric: {
  key: keyof TuningConfig;
  label: string;
  min: number;
  max?: number;
  step?: number | "any";
  hint: string;
}[] = [
  {
    key: "ctx",
    label: "Context length",
    min: 1,
    hint: "Maximum tokens per sample. Recommended default: 512.",
  },
  {
    key: "chunk",
    label: "Recompute chunk",
    min: 1,
    hint: "Activation recompute length. Recommended default: 128.",
  },
  {
    key: "batch_size",
    label: "Batch size",
    min: 1,
    max: 128,
    hint: "Samples accumulated per optimizer update. Range: 1–128; default: 16.",
  },
  {
    key: "epochs",
    label: "Epochs",
    min: 1,
    hint: "Number of complete dataset passes. Default: 1.",
  },
  {
    key: "lr",
    label: "Learning rate",
    min: 0.0000001,
    step: "any",
    hint: "Initial learning rate; MiSS default: 0.0001, state default: 0.0005.",
  },
  {
    key: "lr_final",
    label: "Final learning rate",
    min: 0.0000001,
    step: "any",
    hint: "Final learning rate; MiSS default: 0.00001, state default: 0.0001.",
  },
  {
    key: "save_every",
    label: "Save every N steps",
    min: 0,
    hint: "Periodic checkpoints. Default: every 100 updates; 0 disables it.",
  },
  {
    key: "warmup_steps",
    label: "Warmup steps",
    min: 0,
    hint: "Linear warmup updates. Default: 10.",
  },
  {
    key: "max_steps",
    label: "Max steps",
    min: 0,
    hint: "Optional optimizer-update cap. 0 runs all configured epochs.",
  },
  {
    key: "seed",
    label: "Seed",
    min: 0,
    hint: "Deterministic seed. Default: 1234.",
  },
];
const recommendedParameters: Partial<TuningConfig> = {
  ctx: 512,
  chunk: 128,
  epochs: 1,
  batch_size: 16,
  lr: 0.0005,
  lr_final: 0.0001,
  warmup_steps: 10,
  save_every: 100,
  max_steps: 0,
  seed: 1234,
  optimizer: "adam",
  wkv_tape: false,
};
const parameterPresets: {
  label: string;
  hint: string;
  values: Partial<TuningConfig>;
}[] = [
  {
    label: "Recommended",
    hint: "Balanced starting point",
    values: {},
  },
  {
    label: "Low memory",
    hint: "Smaller context and batch",
    values: { ctx: 256, chunk: 64, batch_size: 4 },
  },
  {
    label: "Quick check",
    hint: "Short pipeline validation",
    values: {
      ctx: 128,
      chunk: 64,
      batch_size: 2,
      max_steps: 10,
      save_every: 5,
    },
  },
];
export function StateTuningPage() {
  const { config: saved, set, resetParameters } = useTuningForm();
  const config = { ...defaultTuning, ...saved };
  const miss = config.method === "miss";
  // Persisted v2 forms created before these options existed have no optimizer.
  const optimizer = config.optimizer === "muon" ? "muon" : "adam";
  const form = useRef<HTMLFormElement>(null);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (
        (event.ctrlKey || event.metaKey) &&
        event.key === "Enter" &&
        !document.querySelector("dialog[open]")
      ) {
        event.preventDefault();
        form.current?.requestSubmit();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  const { runtime, tuning: state, connected, refresh } = useRuntime();
  const available = miss ? state.miss_available : state.available;
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [validated, setValidated] = useState<{
    path: string;
    samples: number;
  } | null>(null);
  const progress = state.progress;
  const losses = state.losses || [];
  const min = Math.min(...losses.map((p) => p.loss)),
    max = Math.max(...losses.map((p) => p.loss));
  const datasetReady = validated?.path === config.data;
  const pathsReady = Boolean(config.model && config.data && config.output);
  const parametersReady =
    config.ctx > 0 &&
    config.chunk > 0 &&
    config.chunk <= config.ctx &&
    config.epochs > 0 &&
    config.batch_size > 0 &&
    config.batch_size <= 128 &&
    config.lr > 0 &&
    config.lr_final > 0 &&
    config.max_steps >= 0 &&
    config.warmup_steps >= 0 &&
    config.save_every >= 0 &&
    config.seed >= 0 &&
    (optimizer === "adam" || optimizer === "muon") &&
    (!miss ||
      (Number.isInteger(config.rank) &&
        config.rank > 0 &&
        config.rank <= 1024 &&
        Number.isFinite(config.alpha) &&
        Boolean(config.targets.trim())));
  const canStart =
    connected &&
    pathsReady &&
    parametersReady &&
    !busy &&
    !state.running &&
    available !== false;
  const fullUpdateEstimate = datasetReady
    ? Math.ceil(validated.samples / config.batch_size) * config.epochs
    : 0;
  const estimatedUpdates = config.max_steps
    ? Math.min(fullUpdateEstimate, config.max_steps)
    : fullUpdateEstimate;
  const commandPreview = [
    miss ? "rwkv_miss_tune" : "rwkv_state_tune",
    miss
      ? `--rank ${config.rank} --alpha ${config.alpha} --targets ${JSON.stringify(config.targets)}`
      : "",
    miss && config.state ? `--state ${JSON.stringify(config.state)}` : "",
    miss && config.resume ? `--resume ${JSON.stringify(config.resume)}` : "",
    `--model ${JSON.stringify(config.model || "MODEL.pth")}`,
    `--data ${JSON.stringify(config.data || "DATA.jsonl")}`,
    `--output ${JSON.stringify(config.output)}`,
    config.vocab ? `--vocab ${JSON.stringify(config.vocab)}` : "",
    `--optimizer ${miss ? "adam" : optimizer}`,
    config.wkv_tape ? "--wkv_tape" : "",
    `--ctx ${config.ctx}`,
    `--chunk ${config.chunk}`,
    `--epochs ${config.epochs}`,
    `--lr ${config.lr}`,
    `--lr-final ${config.lr_final}`,
    `--warmup-steps ${config.warmup_steps}`,
    `--save-every ${config.save_every}`,
    `--batch-size ${config.batch_size}`,
    `--max-steps ${config.max_steps}`,
    `--seed ${config.seed}`,
  ]
    .filter(Boolean)
    .join(" ");
  const start = async (stopRuntime = false) => {
    setBusy(true);
    setError("");
    setConfirm(false);
    try {
      const result = await tuning.validate(config.data);
      setValidated({ path: config.data, samples: result.samples });
      if (stopRuntime) await launcher.stop();
      await tuning.start({ ...config, optimizer: miss ? "adam" : optimizer });
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  const inputs = numeric.map((f) => (
    <Field key={f.key} label={f.label} hint={f.hint}>
      <input
        type="number"
        required
        value={Number(config[f.key])}
        min={f.min}
        max={f.max}
        step={f.step ?? 1}
        onChange={(e) => set({ [f.key]: Number(e.target.value) })}
      />
    </Field>
  ));
  return (
    <div className="page">
      <header className="page-heading">
        <div>
          <div className="eyebrow">FROZEN BASE TRAINING</div>
          <h1>{miss ? "MiSS Training" : "State Tuning"}</h1>
          <p>
            {miss
              ? "Train compact MiSS adapters with chunked state passing."
              : "Train a reusable RWKV state from a JSONL dataset."}
          </p>
        </div>
        <span className="tag">CUDA · BF16 .pth</span>
      </header>
      <ErrorPanel error={error || state.error} />
      {available === false && (
        <p className="notice">
          Place the <code>{miss ? "rwkv_miss_tune" : "rwkv_state_tune"}</code>{" "}
          executable next to this Launcher to enable training.
        </p>
      )}
      <section className="tuning-overview" aria-label="State tuning readiness">
        {[
          [
            pathsReady,
            "Files",
            pathsReady ? "Paths ready" : "Choose model, data and output",
          ],
          [
            datasetReady,
            "Dataset",
            datasetReady
              ? `${validated.samples.toLocaleString()} valid samples`
              : "Run dataset validation",
          ],
          [
            parametersReady,
            "Parameters",
            parametersReady
              ? `${config.ctx} ctx · batch ${config.batch_size}`
              : "Review invalid values",
          ],
          [
            state.running,
            "Trainer",
            state.running ? "Training in progress" : state.status,
          ],
        ].map(([ready, label, detail]) => (
          <div className={ready ? "ready" : ""} key={String(label)}>
            {ready ? <CheckCircle2 size={17} /> : <Circle size={17} />}
            <span>{label}</span>
            <strong>{detail}</strong>
          </div>
        ))}
      </section>
      <form
        ref={form}
        onSubmit={(e) => {
          e.preventDefault();
          if (!canStart) return;
          if (runtime.running) setConfirm(true);
          else void start();
        }}
      >
        <fieldset disabled={busy || state.running}>
          <Panel title="Training method">
            <Field label="Method">
              <select
                value={config.method}
                onChange={(e) => {
                  const method = e.target.value as "state" | "miss";
                  set({
                    method,
                    optimizer: "adam",
                    resume: "",
                    output:
                      method === "miss" ? "./miss_output" : "./state_output",
                    lr: method === "miss" ? 0.0001 : defaultTuning.lr,
                    lr_final:
                      method === "miss" ? 0.00001 : defaultTuning.lr_final,
                  });
                }}
              >
                <option value="state">State tuning</option>
                <option value="miss">MiSS adapter</option>
              </select>
            </Field>
            {miss && (
              <>
                <div className="form-grid">
                  <Field label="Rank">
                    <input
                      type="number"
                      min={1}
                      max={1024}
                      required
                      value={config.rank}
                      onChange={(e) => set({ rank: Number(e.target.value) })}
                    />
                  </Field>
                  <Field
                    label="Alpha"
                    hint="Effective scale = alpha / rank. Set alpha equal to rank for scale 1."
                  >
                    <input
                      type="number"
                      step="any"
                      required
                      value={config.alpha}
                      onChange={(e) => set({ alpha: Number(e.target.value) })}
                    />
                  </Field>
                </div>
                <Field
                  label="Targets"
                  hint="all, or comma-separated att.receptance.weight, att.key.weight, att.value.weight, att.output.weight, ffn.key.weight, ffn.value.weight"
                >
                  <input
                    required
                    value={config.targets}
                    onChange={(e) => set({ targets: e.target.value })}
                  />
                </Field>
                <PathField
                  label="Initial state (.pth, optional)"
                  value={config.state}
                  onChange={(state) => set({ state })}
                />
                <PathField
                  label="Resume checkpoint directory (optional)"
                  directory
                  value={config.resume}
                  onChange={(resume) => set({ resume })}
                />
                <p className="small muted">
                  Resume with the original model, data, rank, targets and
                  schedule, and a new output directory. Batch size accumulates
                  independent microbatches before one update.
                </p>
              </>
            )}
          </Panel>
          <Panel title="Quick setup" hint="PRESETS">
            <div className="preset-grid">
              {miss && (
                <button
                  type="button"
                  onClick={() =>
                    set({
                      ctx: 4096,
                      chunk: 1024,
                      batch_size: 8,
                      epochs: 1,
                      rank: 16,
                      alpha: 16,
                      targets: "all",
                      lr: 0.0001,
                      lr_final: 0.00001,
                      warmup_steps: 10,
                      save_every: 100,
                      max_steps: 0,
                      optimizer: "adam",
                      wkv_tape: true,
                    })
                  }
                >
                  <Zap size={15} />
                  <span>
                    <strong>MiSS 4096</strong>
                    <small>
                      Rank 16 · chunk 1024 · batch 8 · shared WKV tape
                    </small>
                  </span>
                </button>
              )}
              {parameterPresets.map((preset) => (
                <button
                  type="button"
                  key={preset.label}
                  onClick={() =>
                    set({
                      ...recommendedParameters,
                      ...(miss ? { lr: 0.0001, lr_final: 0.00001 } : {}),
                      ...preset.values,
                    })
                  }
                >
                  <Zap size={15} />
                  <span>
                    <strong>{preset.label}</strong>
                    <small>{preset.hint}</small>
                  </span>
                </button>
              ))}
              <button
                type="button"
                onClick={() => {
                  resetParameters();
                  if (miss)
                    set({ method: "miss", lr: 0.0001, lr_final: 0.00001 });
                }}
              >
                <RotateCcw size={15} />
                <span>
                  <strong>Reset parameters</strong>
                  <small>Restore recommended defaults</small>
                </span>
              </button>
            </div>
          </Panel>
          <div className="form-grid runtime-grid">
            <Panel title="Base model & dataset" hint="01">
              <PathField
                label="Base model (.pth)"
                value={config.model}
                onChange={(model) => set({ model })}
              />
              <PathField
                label="JSONL dataset"
                value={config.data}
                onChange={(data) => {
                  set({ data });
                  setValidated(null);
                }}
              />
              <div className="toolbar">
                <button
                  type="button"
                  disabled={!config.data}
                  onClick={async () => {
                    setError("");
                    try {
                      const { samples } = await tuning.validate(config.data);
                      setValidated({ path: config.data, samples });
                    } catch (e) {
                      setError(String(e));
                    }
                  }}
                >
                  Validate dataset
                </button>
                {validated?.path === config.data && (
                  <span className="success">
                    ✓ JSONL · text field · {validated.samples.toLocaleString()}{" "}
                    samples
                  </span>
                )}
              </div>
              <p className="small muted">
                Each row must contain exactly one string field:{" "}
                <code>{'{"text":"..."}'}</code>. Validation reads the real file
                on the Launcher host.
              </p>
              <PathField
                label="Vocabulary"
                value={config.vocab}
                onChange={(vocab) => set({ vocab })}
              />
            </Panel>
            <Panel title="Training" hint="02">
              <div className="form-grid">
                <Field
                  label="Optimizer"
                  hint="Adam preserves the original behavior. Muon orthogonalizes each 64×64 state matrix."
                >
                  <select
                    value={miss ? "adam" : optimizer}
                    disabled={miss}
                    onChange={(e) =>
                      set({ optimizer: e.target.value as "adam" | "muon" })
                    }
                  >
                    <option value="adam">Adam</option>
                    <option value="muon">Muon</option>
                  </select>
                </Field>
                <Field
                  label="Shared WKV tape"
                  hint="Recompute WKV per layer during backward to reduce GPU memory usage."
                >
                  <label className="thinking-toggle">
                    <input
                      type="checkbox"
                      checked={Boolean(config.wkv_tape)}
                      onChange={(e) => set({ wkv_tape: e.target.checked })}
                    />
                    Enable <code>--wkv_tape</code>
                  </label>
                </Field>
              </div>
              <div className="form-grid">{inputs.slice(0, 6)}</div>
              {config.chunk > config.ctx && (
                <p className="inline-warning">
                  Recompute chunk should not be larger than context length.
                </p>
              )}
              {config.lr > 0.01 && (
                <p className="inline-warning">
                  This learning rate is unusually high for state tuning and may
                  make loss diverge.
                </p>
              )}
              <details>
                <summary>Advanced training</summary>
                <div className="form-grid">{inputs.slice(6)}</div>
              </details>
            </Panel>
          </div>
          <Panel
            title={miss ? "Adapter checkpoints" : "State checkpoints"}
            hint="03"
          >
            <PathField
              label="Output directory"
              value={config.output}
              directory
              placeholder="./state_output"
              onChange={(output) => set({ output })}
            />
            {miss ? (
              <p className="muted small">
                Resumable checkpoints: <code>checkpoint-N/training.pth</code>{" "}
                and <code>checkpoint.json</code>. Final export:{" "}
                <code>adapter-final.pth</code>. Register or upload either PTH in
                the MiSS adapters panel for inference.
              </p>
            ) : (
              <p className="muted small">
                Saved as <code>state-step-XXXXXXXX.pth</code> and{" "}
                <code>state-final.pth</code>. These contain trained state
                tensors, not model weights or optimizer state. The current CLI
                does not support checkpoint resume.
              </p>
            )}
            <div className="training-summary">
              <div>
                <span>Validated samples</span>
                <strong>
                  {datasetReady ? validated.samples.toLocaleString() : "—"}
                </strong>
              </div>
              <div>
                <span>Estimated updates</span>
                <strong>
                  {estimatedUpdates ? estimatedUpdates.toLocaleString() : "—"}
                </strong>
              </div>
              <div>
                <span>Checkpoint interval</span>
                <strong>
                  {config.save_every
                    ? `${config.save_every} steps`
                    : "Final only"}
                </strong>
              </div>
            </div>
          </Panel>
          <details className="command-preview">
            <summary>
              <Terminal size={15} /> Command preview
            </summary>
            <pre>{commandPreview}</pre>
            <CopyButton text={commandPreview} label="Copy command" />
          </details>
        </fieldset>
        <div className="action-strip">
          <button className="primary" disabled={!canStart}>
            <Play size={14} />{" "}
            {busy
              ? "Starting…"
              : miss
                ? "Start MiSS training"
                : "Start state tuning"}
          </button>
          <button
            type="button"
            disabled={busy || !state.running}
            onClick={async () => {
              setBusy(true);
              try {
                await tuning.stop();
                await refresh();
              } catch (e) {
                setError(String(e));
              } finally {
                setBusy(false);
              }
            }}
          >
            <Square size={13} /> Stop training
          </button>
          <span className="muted">
            {!connected
              ? "Launcher unavailable"
              : !pathsReady
                ? "Choose all required paths"
                : !parametersReady
                  ? "Review training parameters"
                  : `${state.status} · Ctrl/⌘ + Enter to start`}
          </span>
        </div>
      </form>
      {progress && (
        <Panel
          title={`Training progress · ${Math.round((progress.step / Math.max(progress.total, 1)) * 100)}%`}
        >
          <progress value={progress.step} max={progress.total} />
          <div className="metrics">
            {[
              ["Step", `${progress.step} / ${progress.total}`],
              ["Epoch", `${progress.epoch} / ${progress.epochs}`],
              ["Loss", progress.loss.toFixed(4)],
              ["LR", progress.lr],
              ["Tokens/s", progress.tokens_per_second],
              ["ETA", `${progress.eta}s`],
            ].map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
          {losses.length > 1 && (
            <div
              className="loss-chart"
              role="img"
              aria-label={`Training loss, ${losses.length} measured updates. Latest loss ${progress.loss}`}
            >
              <div className="chart-label">
                <span>LOSS</span>
                <span>
                  {max.toFixed(4)} → {min.toFixed(4)}
                </span>
              </div>
              <svg viewBox="0 0 1000 150" preserveAspectRatio="none">
                <polyline
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                  points={losses
                    .map(
                      (p, i) =>
                        `${(i / (losses.length - 1)) * 1000},${140 - ((p.loss - min) / (max - min || 1)) * 125}`,
                    )
                    .join(" ")}
                />
              </svg>
            </div>
          )}
        </Panel>
      )}
      {state.checkpoint && (
        <Panel
          title={
            state.status === "completed"
              ? "Training completed"
              : "Latest checkpoint"
          }
        >
          <code className="checkpoint-path">{state.checkpoint}</code>
          <div className="toolbar">
            <CopyButton text={state.checkpoint} label="Copy path" />
            <button
              onClick={async () => {
                try {
                  await tuning.openFolder();
                } catch (e) {
                  setError(String(e));
                }
              }}
            >
              <FolderOpen size={14} /> Open folder
            </button>
          </div>
        </Panel>
      )}
      <Console lines={state.logs} title="Training logs" />
      {confirm && (
        <Dialog
          title="Stop inference and start tuning?"
          onClose={() => setConfirm(false)}
        >
          <p>
            Inference is currently using the GPU. The Launcher will stop it,
            validate the dataset, then start training.
          </p>
          <p className="muted">
            This is a Launcher resource policy; the native executables do not
            coordinate GPU memory with each other.
          </p>
          <div className="toolbar">
            <button onClick={() => setConfirm(false)}>Cancel</button>
            <button className="primary" onClick={() => void start(true)}>
              Stop & start
            </button>
          </div>
        </Dialog>
      )}
    </div>
  );
}
