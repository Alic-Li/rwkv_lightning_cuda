import { useEffect, useState } from "react";
import {
  Archive,
  Cpu,
  FolderOpen,
  LoaderCircle,
  SlidersHorizontal,
  Square,
} from "lucide-react";
import {
  useQuantizationForm,
  useRuntime,
  useRuntimeForm,
} from "../stores/runtime";
import { useSecret } from "../stores/settings";
import { RWKVClient } from "../lib/api/client";
import {
  Panel,
  PathField,
  Field,
  Console,
  ErrorPanel,
} from "../components/common";
import { RuntimeControls, RuntimeBadge } from "../components/RuntimeControls";
import { suggestedQuantizedPath } from "../lib/api/launcher";
export function RuntimePage() {
  const { config, set, recent } = useRuntimeForm();
  const quantize = useQuantizationForm((s) => s.config);
  const setQuantize = useQuantizationForm((s) => s.set);
  const {
    runtime,
    quantization,
    quantizationBusy,
    quantizationError,
    quantizationAction,
    error,
    connected,
  } = useRuntime();
  const key = useSecret((s) => s.key);
  const [models, setModels] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [loadError, setLoadError] = useState("");
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (runtime.status === "ready" && runtime.config?.enable_dynamic_loading) {
      new RWKVClient("", key)
        .models()
        .then((m) => {
          setModels(m.available || m.data.map((d) => d.id));
          setSelected(m.loaded || "");
        })
        .catch((e) => setLoadError(String(e)));
    }
  }, [runtime.status, runtime.config?.enable_dynamic_loading, key]);
  return (
    <div className="page">
      <header className="page-heading">
        <div>
          <div className="eyebrow">LOCAL ENGINE</div>
          <h1>Runtime</h1>
          <p>One model. Your hardware. Fully in your control.</p>
        </div>
        <RuntimeBadge />
      </header>
      <ErrorPanel
        error={
          !connected
            ? "Launcher unavailable. Open this UI through the Go Launcher on port 8088."
            : error || runtime.error
        }
      />
      <div className="form-grid runtime-grid">
        <Panel title="Model" hint="01">
          <PathField
            label="Model path"
            value={config.model_path}
            onChange={(model_path) => set({ model_path })}
            placeholder={
              config.enable_dynamic_loading
                ? "/path/to/models directory"
                : "/path/to/model.pth or model.rwkvq"
            }
          />
          {recent.length > 0 && (
            <Field label="Recent models">
              <select
                value=""
                onChange={(e) => set({ model_path: e.target.value })}
              >
                <option value="">Choose a recent model…</option>
                {recent.map((p) => (
                  <option key={p}>{p}</option>
                ))}
              </select>
            </Field>
          )}
          <PathField
            label="Vocabulary"
            value={config.vocab_path}
            onChange={(vocab_path) => set({ vocab_path })}
          />
          <label className="check">
            <input
              type="checkbox"
              checked={config.enable_dynamic_loading}
              onChange={(e) =>
                set({ enable_dynamic_loading: e.target.checked })
              }
            />{" "}
            Load models on demand from a directory
          </label>
        </Panel>
        <Panel title="Device & server" hint="02">
          <div className="device-info">
            <Cpu size={22} />
            <div>
              <strong>Native GPU backend</strong>
              <p>CUDA / HIP is selected when the runtime is compiled.</p>
            </div>
          </div>
          <p className="muted small">
            Uses the bundled rwkv_lighting_cuda executable. State tuning
            requires CUDA.
          </p>
          <div className="form-grid">
            <Field label="Bind address">
              <input readOnly value="127.0.0.1" />
            </Field>
            <Field label="Port" hint="Backend HTTP port. Default: 8000.">
              <input
                type="number"
                min="1"
                max="65535"
                value={config.port}
                onChange={(e) => set({ port: e.target.value })}
              />
            </Field>
          </div>
          <Field
            label="Runtime password"
            hint="Passed to the backend without logging; retained only for this browser session."
          >
            <input
              type="password"
              value={config.password}
              onChange={(e) => set({ password: e.target.value })}
              autoComplete="off"
              placeholder="Optional"
            />
          </Field>
        </Panel>
        <Panel title="Performance" hint="03">
          <Field
            label="Prefill chunk size"
            hint="--chunk-size: prompt tokens per prefill chunk; default 128."
          >
            <input
              type="number"
              min="1"
              value={config.chunk_size}
              onChange={(e) => set({ chunk_size: Number(e.target.value) })}
            />
          </Field>
          <label className="check">
            <input
              type="checkbox"
              checked={config.chunk_load}
              onChange={(e) => set({ chunk_load: e.target.checked })}
            />{" "}
            Chunk-load weights <span className="muted">Lower host RAM</span>
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={config.use_wkv32}
              onChange={(e) => set({ use_wkv32: e.target.checked })}
            />{" "}
            FP32 WKV state{" "}
            <span className="muted">Higher precision / VRAM</span>
          </label>
          <p className="small muted">
            Request concurrency is admitted dynamically by the backend according
            to free VRAM.
          </p>
        </Panel>
        <Panel title="State cache" hint="04">
          <Field
            label="State database"
            hint="--state-db-path defaults to rwkv_sessions.db in the executable directory."
          >
            <input
              value={config.state_db_path}
              onChange={(e) => set({ state_db_path: e.target.value })}
            />
          </Field>
          <details>
            <summary>
              <SlidersHorizontal size={14} /> Advanced
            </summary>
            <Field
              label="W8A16 tuning cache"
              hint="--tune-cache: defaults alongside the state database."
            >
              <input
                value={config.tune_cache}
                onChange={(e) => set({ tune_cache: e.target.value })}
                placeholder="Automatic · alongside state database"
              />
            </Field>
          </details>
          <p className="small muted">
            <FolderOpen size={13} /> Relative paths are resolved next to the Go
            Launcher.
          </p>
        </Panel>
      </div>
      <Panel title="Model quantization" hint="W8A16 / W4A16">
        <div className="device-info">
          <Archive size={22} />
          <div>
            <strong>Convert a BF16 checkpoint</strong>
            <p>
              Create a smaller .rwkvq model that the CUDA runtime detects
              automatically.
            </p>
          </div>
        </div>
        <div className="form-grid">
          <PathField
            label="BF16 input model (.pth)"
            value={quantize.input_path}
            onChange={(input_path) =>
              setQuantize({
                input_path,
                output_path: suggestedQuantizedPath(
                  input_path,
                  quantize.format,
                ),
              })
            }
          />
          <Field
            label="Quantized output (.rwkvq)"
            hint="A new file is required; existing files are never overwritten."
          >
            <input
              value={quantize.output_path}
              onChange={(e) => setQuantize({ output_path: e.target.value })}
              placeholder="/path/to/model.w4a16.rwkvq"
            />
          </Field>
          <Field label="Weight format">
            <select
              value={quantize.format}
              onChange={(e) => {
                const format = e.target.value as "w8a16" | "w4a16";
                setQuantize({
                  format,
                  output_path: suggestedQuantizedPath(
                    quantize.input_path,
                    format,
                  ),
                });
              }}
            >
              <option value="w4a16">W4A16 · smallest (recommended)</option>
              <option value="w8a16">W8A16 · higher fidelity</option>
            </select>
          </Field>
          <Field
            label="W4 group size"
            hint="128 uses less space; 32 keeps more per-group precision."
          >
            <select
              disabled={quantize.format !== "w4a16"}
              value={quantize.group_size}
              onChange={(e) =>
                setQuantize({ group_size: Number(e.target.value) as 32 | 128 })
              }
            >
              <option value="128">128 · recommended</option>
              <option value="32">32 · higher fidelity</option>
            </select>
          </Field>
        </div>
        <div className="toolbar quantization-actions">
          <button
            type="button"
            onClick={() => {
              const input_path = config.model_path;
              setQuantize({
                input_path,
                output_path: suggestedQuantizedPath(
                  input_path,
                  quantize.format,
                ),
              });
            }}
            disabled={
              quantization.running ||
              !config.model_path.toLowerCase().endsWith(".pth")
            }
          >
            Use current model
          </button>
          {quantization.running ? (
            <button
              type="button"
              disabled={quantizationBusy}
              onClick={() => void quantizationAction("stop")}
            >
              <Square size={13} /> Stop quantization
            </button>
          ) : (
            <button
              type="button"
              className="primary"
              disabled={
                !connected ||
                quantizationBusy ||
                quantization.available === false ||
                !quantize.input_path.trim() ||
                !quantize.output_path.trim()
              }
              onClick={() => void quantizationAction("start")}
            >
              {quantizationBusy ? (
                <LoaderCircle size={13} className="spin" />
              ) : (
                <Archive size={13} />
              )}{" "}
              Quantize model
            </button>
          )}
          {quantization.status === "completed" && quantization.output_path && (
            <button
              type="button"
              onClick={() => set({ model_path: quantization.output_path! })}
            >
              Use quantized model
            </button>
          )}
          <span className="muted small" role="status">
            {quantization.running
              ? `Quantizing · ${Math.round(quantization.elapsed)}s`
              : quantization.status === "completed"
                ? "Quantization completed"
                : quantization.available === false
                  ? "rwkv_quantize is not included beside the Launcher."
                  : "Conversion runs locally and may use substantial RAM."}
          </span>
        </div>
        <ErrorPanel error={quantizationError || quantization.error} />
        {quantization.logs.length > 0 && (
          <Console lines={quantization.logs} title="Quantization logs" />
        )}
      </Panel>
      <div className="action-strip">
        <RuntimeControls />
        <span className="muted">
          Ready is verified against the live backend.
        </span>
      </div>
      {runtime.config?.enable_dynamic_loading && runtime.status === "ready" && (
        <Panel title="Available models">
          <div className="toolbar">
            <select
              aria-label="Available model"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
            >
              <option value="">Select model…</option>
              {models.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
            <button
              disabled={!selected || loading}
              onClick={async () => {
                setLoading(true);
                setLoadError("");
                try {
                  await new RWKVClient("", key).loadModel(selected);
                  await useRuntime.getState().refresh();
                } catch (e) {
                  setLoadError(String(e));
                } finally {
                  setLoading(false);
                }
              }}
            >
              {loading ? "Loading…" : "Load model"}
            </button>
          </div>
          <ErrorPanel error={loadError} />
        </Panel>
      )}
      {runtime.backend?.prefill_queue && (
        <details className="panel">
          <summary>Live admission queue</summary>
          <pre>{JSON.stringify(runtime.backend.prefill_queue, null, 2)}</pre>
        </details>
      )}
      <Console lines={runtime.logs} />
    </div>
  );
}
