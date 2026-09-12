import { useEffect, useRef, useState } from "react";
import { Play, Square, FolderOpen } from "lucide-react";
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
  persist<{ config: TuningConfig; set: (v: Partial<TuningConfig>) => void }>(
    (set) => ({
      config: defaultTuning,
      set: (v) => set((s) => ({ config: { ...s.config, ...v } })),
    }),
    { name: "rwkv-tuning-form-v1", storage: createJSONStorage(() => storage) },
  ),
);
const numeric: {
  key: keyof TuningConfig;
  label: string;
  min: number;
  step?: number;
  hint: string;
}[] = [
  {
    key: "epochs",
    label: "Epochs",
    min: 1,
    hint: "Dataset passes. Default: 1.",
  },
  {
    key: "max_steps",
    label: "Max steps",
    min: 0,
    hint: "Optimizer updates; 0 means no step cap.",
  },
  {
    key: "lr",
    label: "Learning rate",
    min: 0.0000001,
    step: 0.0001,
    hint: "Initial learning rate. CLI default: 1.0.",
  },
  {
    key: "lr_final",
    label: "Final learning rate",
    min: 0.0000001,
    step: 0.0001,
    hint: "Final learning rate. Default: 0.01.",
  },
  {
    key: "ctx",
    label: "Context length",
    min: 1,
    hint: "Maximum tokens per sample. Default: 128.",
  },
  {
    key: "chunk",
    label: "Recompute chunk",
    min: 1,
    hint: "Activation recompute length. Default: 64.",
  },
  {
    key: "batch_size",
    label: "Samples per update",
    min: 1,
    hint: "Sequential gradient accumulation, not parallel GPU batches. Default: 1.",
  },
  {
    key: "save_every",
    label: "Save every N steps",
    min: 0,
    hint: "Periodic state checkpoints. 0 disables periodic saving.",
  },
  {
    key: "warmup_steps",
    label: "Warmup steps",
    min: 0,
    hint: "Linear warmup updates. Default: 10.",
  },
  {
    key: "seed",
    label: "Seed",
    min: 0,
    hint: "Deterministic seed. Default: 1234.",
  },
];
export function StateTuningPage() {
  const { config, set } = useTuningForm();
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
  const start = async (stopRuntime = false) => {
    setBusy(true);
    setError("");
    setConfirm(false);
    try {
      await tuning.validate(config.data);
      if (stopRuntime) await launcher.stop();
      await tuning.start(config);
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
        value={config[f.key]}
        min={f.min}
        step={f.step || 1}
        onChange={(e) => set({ [f.key]: Number(e.target.value) })}
      />
    </Field>
  ));
  return (
    <div className="page">
      <header className="page-heading">
        <div>
          <div className="eyebrow">STATE, NOT WEIGHTS</div>
          <h1>State Tuning</h1>
          <p>Train a reusable RWKV state from a JSONL dataset.</p>
        </div>
        <span className="tag">CUDA · BF16 .pth</span>
      </header>
      <ErrorPanel error={error || state.error} />
      {state.available === false && (
        <p className="notice">
          Place the CUDA <code>rwkv_state_tune</code> executable next to this
          Launcher to enable training.
        </p>
      )}
      <form
        ref={form}
        onSubmit={(e) => {
          e.preventDefault();
          if (!connected || busy || state.running || state.available === false)
            return;
          if (runtime.running) setConfirm(true);
          else void start();
        }}
      >
        <fieldset disabled={busy || state.running}>
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
              <div className="form-grid">{inputs.slice(0, 6)}</div>
              <details>
                <summary>Advanced training</summary>
                <div className="form-grid">{inputs.slice(6)}</div>
              </details>
            </Panel>
          </div>
          <Panel title="State checkpoints" hint="03">
            <Field label="Output directory">
              <input
                value={config.output}
                required
                onChange={(e) => set({ output: e.target.value })}
              />
            </Field>
            <p className="muted small">
              Saved as <code>state-step-XXXXXXXX.pth</code> and{" "}
              <code>state-final.pth</code>. These contain trained state tensors,
              not model weights or optimizer state. The current CLI does not
              support checkpoint resume.
            </p>
          </Panel>
        </fieldset>
        <div className="action-strip">
          <button
            className="primary"
            disabled={
              !connected || busy || state.running || state.available === false
            }
          >
            <Play size={14} /> {busy ? "Starting…" : "Start state tuning"}
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
            {state.status} · Inference and tuning run one at a time.
          </span>
        </div>
      </form>
      {progress && (
        <Panel title="Training progress">
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
              : "Latest saved state"
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
            validate the dataset, then start state tuning.
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
