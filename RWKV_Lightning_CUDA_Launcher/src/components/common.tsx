import { useEffect, useRef, useState, type ReactNode } from "react";
import { X, Copy, Search, FolderOpen, Check } from "lucide-react";
import { launcher } from "../lib/api/launcher";
export function ErrorPanel({
  error,
  children,
}: {
  error?: string;
  children?: ReactNode;
}) {
  return error ? (
    <div className="error-panel" role="alert">
      <strong>Something needs attention</strong>
      <div>{error}</div>
      {children}
    </div>
  ) : null;
}
export function Panel({
  title,
  children,
  hint,
}: {
  title: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <section className="panel">
      <div className="section-heading">
        <h2>{title}</h2>
        {hint && <span>{hint}</span>}
      </div>
      {children}
    </section>
  );
}
export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <span title={hint}>
        {label}
        {hint && (
          <span className="help" title={hint}>
            ⓘ
          </span>
        )}
      </span>
      {children}
    </label>
  );
}
export function PathField({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <div>
      <Field label={label}>
        <span className="path-input">
          <input
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder || "/path/to/file"}
          />
          <button
            type="button"
            title={`Browse ${label}`}
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              setError("");
              try {
                const { path } = await launcher.pickFile();
                if (path) onChange(path);
              } catch (e) {
                setError(String(e));
              } finally {
                setBusy(false);
              }
            }}
          >
            <FolderOpen size={16} />
            <span>Browse</span>
          </button>
        </span>
      </Field>
      {error && <p className="inline-error">{error}</p>}
    </div>
  );
}
export function CopyButton({
  text,
  label = "Copy",
}: {
  text: string;
  label?: string;
}) {
  const [state, setState] = useState("");
  return (
    <>
      <button
        type="button"
        title={label}
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(text);
            setState("Copied");
            setTimeout(() => setState(""), 1800);
          } catch {
            setState("Clipboard unavailable");
          }
        }}
      >
        {state === "Copied" ? <Check size={14} /> : <Copy size={14} />}{" "}
        {state || label}
      </button>
    </>
  );
}
export function Dialog({
  title,
  children,
  onClose,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current;
    const previous = document.activeElement as HTMLElement | null;
    el?.showModal();
    return () => {
      el?.close();
      previous?.focus();
    };
  }, []);
  return (
    <dialog
      ref={ref}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
    >
      <div className="dialog-heading">
        <h2>{title}</h2>
        <button title="Close dialog" onClick={onClose}>
          <X size={18} />
        </button>
      </div>
      {children}
    </dialog>
  );
}
export function Console({
  lines,
  title = "Console",
}: {
  lines: string[];
  title?: string;
}) {
  const [query, setQuery] = useState("");
  const [cleared, setCleared] = useState<string | undefined>();
  const box = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const start = cleared ? lines.lastIndexOf(cleared) + 1 : 0;
  const visible = lines
    .slice(start)
    .filter((l) => l.toLowerCase().includes(query.toLowerCase()));
  useEffect(() => {
    if (following.current && box.current)
      box.current.scrollTop = box.current.scrollHeight;
  }, [lines, query]);
  return (
    <details className="console-wrap" open>
      <summary>
        {title}
        <span>stdout / stderr · {lines.length} lines</span>
      </summary>
      <div className="console-tools">
        <span className="search-input">
          <Search size={14} />
          <input
            aria-label="Search logs"
            placeholder="Filter output…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </span>
        <CopyButton text={visible.join("\n")} />
        <button onClick={() => setCleared(lines.at(-1))}>Clear view</button>
      </div>
      <div
        className="console"
        ref={box}
        onScroll={() => {
          const el = box.current!;
          following.current =
            el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
      >
        {visible.length ? (
          visible.map((line, i) => (
            <div
              key={i}
              className={
                /error|failed|exception|out of memory/i.test(line)
                  ? "log-error"
                  : ""
              }
            >
              {line}
            </div>
          ))
        ) : (
          <span className="muted">Process output will appear here.</span>
        )}
      </div>
    </details>
  );
}
export function download(text: string, extension: string) {
  const url = URL.createObjectURL(
    new Blob([text], { type: "text/plain;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `rwkv-translation.${extension}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
