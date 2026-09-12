import { memo, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Play,
  Square,
  FileCode,
  Download,
  RotateCcw,
} from "lucide-react";
import { useTranslate, translationBody } from "../stores/translate";
import { useSettings } from "../stores/settings";
import { useRuntime } from "../stores/runtime";
import { chunkText, translationPrompt } from "../lib/translate/chunk";
import { orderedMerge, type TranslateChunk } from "../lib/translate/scheduler";
import {
  CopyButton,
  Dialog,
  ErrorPanel,
  Field,
  download,
} from "../components/common";
export const languages = [
  "Auto",
  "English",
  "Chinese",
  "Simplified Chinese",
  "Traditional Chinese",
  "Japanese",
  "Korean",
  "French",
  "German",
  "Spanish",
  "Portuguese",
  "Russian",
  "Italian",
];
export function LanguageInput({
  label,
  value,
  onChange,
  disabled = false,
}: {
  label: string;
  value: string;
  onChange: (s: string) => void;
  disabled?: boolean;
}) {
  return (
    <Field label={label}>
      <input
        list="languages"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Custom language name"
      />
    </Field>
  );
}
const PreviewChunk = memo(
  function PreviewChunk({ chunk }: { chunk: TranslateChunk }) {
    return (
      <section className={`translation-chunk ${chunk.status}`}>
        <span className="chunk-number">
          {String(chunk.id + 1).padStart(2, "0")}
        </span>
        <div>
          {chunk.translated || (
            <span className="muted">
              {chunk.status === "error"
                ? "Translation failed · retry this chunk"
                : "···"}
            </span>
          )}
        </div>
      </section>
    );
  },
  (a, b) =>
    a.chunk.translated === b.chunk.translated &&
    a.chunk.status === b.chunk.status,
);
export function TranslatePage() {
  const job = useTranslate();
  const values = useSettings((s) => s.values);
  const ready = useRuntime(
    (s) =>
      s.runtime.status === "ready" &&
      s.runtime.backend?.model?.loaded !== false,
  );
  const [inspect, setInspect] = useState<number | null>(null);
  const [check, setCheck] = useState(false);
  const [listPage, setListPage] = useState(0);
  const sourceLang = job.from || values.sourceLanguage,
    targetLang = job.to || values.targetLanguage;
  const concurrency = job.concurrency || values.concurrency,
    target = job.target || values.chunkTarget;
  const estimate = useMemo(() => {
    try {
      return chunkText(job.source, target);
    } catch {
      return [];
    }
  }, [job.source, target]);
  const done = job.chunks.filter((c) => c.status === "done").length;
  const failed = job.chunks.filter((c) => c.status === "error").length;
  const output = orderedMerge(job.chunks);
  const chars = job.chunks.reduce((n, c) => n + c.translated.length, 0);
  const prompt = translationPrompt(
    estimate[0] || job.source,
    sourceLang,
    targetLang,
  );
  const selected = job.chunks.find((c) => c.id === inspect);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        (e.ctrlKey || e.metaKey) &&
        e.key === "Enter" &&
        !document.querySelector("dialog[open]")
      ) {
        e.preventDefault();
        if (ready) void useTranslate.getState().run();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [ready]);
  return (
    <div className="page translate-page">
      <header className="page-heading">
        <div>
          <div className="eyebrow">MANY REQUESTS. ONE DOCUMENT.</div>
          <h1>Parallel Translate</h1>
          <p>
            Translate long documents with independent, concurrent continuations.
          </p>
        </div>
        <span className="tag">{concurrency} workers</span>
      </header>
      <datalist id="languages">
        {languages.map((l) => (
          <option key={l} value={l} />
        ))}
      </datalist>
      <div className="translation-bar panel">
        <LanguageInput
          label="Source language"
          value={sourceLang}
          onChange={(from) => job.set({ from })}
          disabled={job.busy}
        />
        <ArrowRight size={18} />
        <LanguageInput
          label="Target language"
          value={targetLang}
          onChange={(to) => job.set({ to })}
          disabled={job.busy}
        />
        <div className="spacer" />
        <button
          className="primary"
          disabled={job.busy || !job.source.trim() || !ready}
          onClick={() => void job.run()}
        >
          <Play size={14} /> Start translation
        </button>
        <button disabled={!job.busy} onClick={job.stop}>
          <Square size={13} /> Stop
        </button>
      </div>
      <ErrorPanel error={job.error} />
      {!ready && (
        <p className="notice">
          Start the local runtime and load a model before translating.{" "}
          <a href="#/runtime">Open Runtime →</a>
        </p>
      )}
      <div className="toolbar translate-options">
        <Field
          label="Concurrency"
          hint="Frontend hard limit; the runtime independently admits requests according to VRAM."
        >
          <input
            type="number"
            min="1"
            max="64"
            value={concurrency}
            disabled={job.busy}
            onChange={(e) => job.set({ concurrency: Number(e.target.value) })}
          />
        </Field>
        <Field label="Chunk target (characters)">
          <input
            type="number"
            min="32"
            step="1"
            value={target}
            disabled={job.busy}
            onChange={(e) => job.set({ target: Number(e.target.value) })}
          />
        </Field>
        <div className="spacer" />
        <button onClick={() => setCheck(true)}>
          <FileCode size={14} /> Check request
        </button>
        <button disabled={job.busy} onClick={job.clear}>
          Clear
        </button>
        <button
          disabled={job.busy}
          onClick={() => {
            job.clear();
            job.set({
              from: values.sourceLanguage,
              to: values.targetLanguage,
              concurrency: values.concurrency,
              target: values.chunkTarget,
            });
          }}
        >
          <RotateCcw size={13} /> Reset
        </button>
      </div>
      <div className="translate-columns">
        <section className="editor-panel">
          <div className="editor-heading">
            <h2>Source</h2>
            <span>
              {job.source.length.toLocaleString()} chars · {estimate.length}{" "}
              estimated chunks
            </span>
          </div>
          <textarea
            aria-label="Source document"
            value={job.source}
            disabled={job.busy}
            onChange={(e) => job.set({ source: e.target.value })}
            placeholder="Paste a long document here.\n\nRWKV will translate it with parallel short requests."
          />
        </section>
        <section className="editor-panel">
          <div className="editor-heading">
            <h2>Translation</h2>
            <span>
              {done} / {job.chunks.length} chunks · {chars.toLocaleString()}{" "}
              chars
            </span>
          </div>
          <progress value={done} max={job.chunks.length || 1} />
          <div className="translation-preview">
            {job.chunks.length ? (
              job.chunks.map((c) => <PreviewChunk key={c.id} chunk={c} />)
            ) : (
              <div className="preview-empty">
                <ArrowRight size={24} />
                <p>
                  A new language.
                  <br />
                  The same document.
                </p>
                <span>
                  Completed chunks appear here in their original order.
                </span>
              </div>
            )}
          </div>
          <div className="output-tools">
            <CopyButton text={output} />
            <button disabled={!chars} onClick={() => download(output, "txt")}>
              <Download size={13} /> TXT
            </button>
            <button disabled={!chars} onClick={() => download(output, "md")}>
              Markdown
            </button>
          </div>
        </section>
      </div>
      <div className="translation-stats">
        <span>
          {job.busy
            ? "Translating…"
            : job.chunks.length
              ? "Job results"
              : "Ready for a document"}
        </span>
        <span>
          {job.elapsed.toFixed(1)}s elapsed ·{" "}
          {job.elapsed > 0 ? (chars / job.elapsed).toFixed(1) : "0"} chars/s ·{" "}
          {failed} failed
        </span>
      </div>
      <details className="panel inspector">
        <summary>
          Chunk inspector <span>{job.chunks.length} independent requests</span>
        </summary>
        <div className="toolbar">
          <button
            disabled={job.busy || !failed || !ready}
            onClick={() =>
              void job.run(
                job.chunks.filter((c) => c.status === "error").map((c) => c.id),
              )
            }
          >
            Retry failed
          </button>
          <button
            disabled={
              job.busy ||
              !ready ||
              !job.chunks.some((c) => c.status === "pending")
            }
            onClick={() =>
              void job.run(
                job.chunks
                  .filter((c) => c.status === "pending")
                  .map((c) => c.id),
              )
            }
          >
            Resume pending
          </button>
          <div className="spacer" />
          <button
            disabled={!listPage}
            onClick={() => setListPage((p) => p - 1)}
          >
            Previous
          </button>
          <span>
            {listPage + 1} / {Math.max(1, Math.ceil(job.chunks.length / 50))}
          </span>
          <button
            disabled={(listPage + 1) * 50 >= job.chunks.length}
            onClick={() => setListPage((p) => p + 1)}
          >
            Next
          </button>
        </div>
        <div className="chunk-grid">
          {job.chunks.slice(listPage * 50, listPage * 50 + 50).map((c) => (
            <button
              key={c.id}
              className={`chunk-button ${c.status}`}
              onClick={() => setInspect(c.id)}
            >
              <strong>#{c.id + 1}</strong>
              <span>{c.status}</span>
              <small>
                {c.source.length} chars · {(c.elapsed || 0).toFixed(2)}s
              </small>
            </button>
          ))}
        </div>
      </details>
      {check && (
        <Dialog title="Check request" onClose={() => setCheck(false)}>
          <p className="muted">
            Preview for the first chunk of the next job. Existing chunk requests
            are available in the inspector.
          </p>
          <p>
            <code>POST /v1/chat/completions</code>
          </p>
          <p className="notice">
            Go Launcher forwards raw contents to the native{" "}
            <code>/v1/batch/completions</code> handler to preserve the exact
            continuation prompt. Chat requests use the native chat handler.
          </p>
          <h3>Prompt</h3>
          <pre>{prompt}</pre>
          <h3>Request body</h3>
          <pre>{JSON.stringify(translationBody(prompt), null, 2)}</pre>
          <CopyButton text={JSON.stringify(translationBody(prompt), null, 2)} />
        </Dialog>
      )}
      {selected && (
        <Dialog
          title={`Chunk #${selected.id + 1} · ${selected.status}`}
          onClose={() => setInspect(null)}
        >
          <h3>Source</h3>
          <pre>{selected.source}</pre>
          <h3>Prompt</h3>
          <pre>{selected.prompt}</pre>
          <h3>Raw output</h3>
          <pre>{selected.translated || "No output yet."}</pre>
          <ErrorPanel error={selected.error} />
          <div className="toolbar">
            <CopyButton text={selected.translated} />
            <button
              disabled={job.busy || !ready}
              onClick={() => {
                void job.run([selected.id]);
                setInspect(null);
              }}
            >
              Retry chunk
            </button>
          </div>
        </Dialog>
      )}
    </div>
  );
}
