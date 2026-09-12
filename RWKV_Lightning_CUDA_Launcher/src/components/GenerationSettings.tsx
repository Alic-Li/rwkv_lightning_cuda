import {
  useSettings,
  type GenerationSettings as Values,
} from "../stores/settings";
import { Field } from "./common";
const fields: {
  key: keyof Values;
  label: string;
  hint: string;
  step: number;
  min: number;
  max?: number;
}[] = [
  {
    key: "temperature",
    label: "Temperature",
    hint: "Sampling temperature. Backend default: 1.0.",
    step: 0.05,
    min: 0,
  },
  {
    key: "top_p",
    label: "Top P",
    hint: "Nucleus sampling. Backend default: 0.3.",
    step: 0.05,
    min: 0,
    max: 1,
  },
  {
    key: "top_k",
    label: "Top K",
    hint: "Top-K sampling. Backend default: 20.",
    step: 1,
    min: 0,
  },
  {
    key: "max_tokens",
    label: "Max tokens",
    hint: "Maximum generated tokens. Backend default: 8192.",
    step: 1,
    min: 1,
  },
  {
    key: "alpha_presence",
    label: "Presence penalty",
    hint: "Repetition presence penalty. Default: 2.0.",
    step: 0.1,
    min: 0,
  },
  {
    key: "alpha_frequency",
    label: "Frequency penalty",
    hint: "Repetition frequency penalty. Default: 0.2.",
    step: 0.1,
    min: 0,
  },
  {
    key: "alpha_decay",
    label: "Penalty decay",
    hint: "Repetition penalty decay. Default: 0.996.",
    step: 0.001,
    min: 0,
    max: 1,
  },
  {
    key: "chunk_size",
    label: "Stream chunk",
    hint: "Output tokens per SSE update, not prefill size. Chat default: 1.",
    step: 1,
    min: 1,
  },
];
export function GenerationSettings({ compact = false }: { compact?: boolean }) {
  const generation = useSettings((s) => s.values.generation);
  const set = useSettings((s) => s.set);
  const update = (key: keyof Values, value: number | string) =>
    set({ generation: { ...generation, [key]: value } });
  const inputs = fields.map((f) => (
    <Field key={f.key} label={f.label} hint={f.hint}>
      <input
        type="number"
        required
        min={f.min}
        max={f.max}
        step={f.step}
        value={generation[f.key]}
        onChange={(e) => update(f.key, Number(e.target.value))}
      />
    </Field>
  ));
  return (
    <div className={compact ? "generation compact" : "generation"}>
      {compact ? (
        <>
          <div className="form-grid four">{inputs.slice(0, 4)}</div>
          <details>
            <summary>Advanced sampling</summary>
            <div className="form-grid">{inputs.slice(4)}</div>
            <Field label="Thinking">
              <select
                value={generation.think_type}
                onChange={(e) => update("think_type", e.target.value)}
              >
                {[
                  "none",
                  "fast",
                  "free",
                  "preferChinese",
                  "en",
                  "enShort",
                  "enLong",
                ].map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
            </Field>
            <Field
              label="Uploaded state ID"
              hint="Use the basename returned by /v1/state/upload."
            >
              <input
                value={generation.state_id}
                onChange={(e) => update("state_id", e.target.value)}
                placeholder="Optional state filename"
              />
            </Field>
          </details>
        </>
      ) : (
        <>
          <div className="form-grid four">{inputs}</div>
          <Field label="Thinking">
            <select
              value={generation.think_type}
              onChange={(e) => update("think_type", e.target.value)}
            >
              {[
                "none",
                "fast",
                "free",
                "preferChinese",
                "en",
                "enShort",
                "enLong",
              ].map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </Field>
        </>
      )}
    </div>
  );
}
