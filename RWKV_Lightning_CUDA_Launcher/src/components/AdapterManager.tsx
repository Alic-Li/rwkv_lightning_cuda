import { useEffect, useMemo, useState } from "react";
import { RWKVClient, type AdapterEntry } from "../lib/api/client";
import { useSecret, useSettings } from "../stores/settings";
import { ErrorPanel, Field } from "./common";

export function AdapterManager() {
  const { values, set } = useSettings();
  const key = useSecret((s) => s.key);
  const client = useMemo(
    () => new RWKVClient(values.baseURL, key),
    [values.baseURL, key],
  );
  const [entries, setEntries] = useState<AdapterEntry[]>([]);
  const [id, setId] = useState("");
  const [path, setPath] = useState("");
  const [file, setFile] = useState<File>();
  const [metadata, setMetadata] = useState<File>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [stats, setStats] = useState("");
  const generation = values.generation;
  const select = (adapter_id: string, adapter_version = "") => {
    const store = useSettings.getState();
    store.set({
      generation: {
        ...store.values.generation,
        adapter_id,
        adapter_version,
        adapter_scale: "",
      },
    });
  };
  useEffect(() => {
    const controller = new AbortController();
    client
      .listAdapters(controller.signal)
      .then((result) => {
        setEntries(result.data);
        setStats(
          `RAM ${(result.ram_bytes / 1048576).toFixed(1)} MiB · GPU ${(result.gpu_bytes / 1048576).toFixed(1)} MiB · H2D uploads ${result.uploads}`,
        );
      })
      .catch((e: unknown) => {
        if (!controller.signal.aborted) setError(String(e));
      });
    return () => controller.abort();
  }, [client, revision]);
  const register = async (upload: boolean) => {
    setBusy(true);
    setError("");
    try {
      const result = upload
        ? await client.uploadAdapter(id.trim(), file!, metadata)
        : await client.registerAdapter(id.trim(), path);
      select(id.trim(), result.version);
      setRevision((v) => v + 1);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="state-manager">
      <p>
        Register a MiSS adapter for chat and parallel translation. Registration
        caches D in server RAM; the first request loads it onto the GPU.
      </p>
      <Field label="Selected adapter">
        <select
          value={JSON.stringify([
            generation.adapter_id || "",
            generation.adapter_version || "",
          ])}
          disabled={busy}
          onChange={(e) => {
            const [id, version] = JSON.parse(e.target.value) as string[];
            select(id, version);
          }}
        >
          <option value={'["",""]'}>None (base model)</option>
          {generation.adapter_id &&
            !entries.some(
              (e) =>
                e.id === generation.adapter_id &&
                e.version === generation.adapter_version,
            ) && (
              <option
                value={JSON.stringify([
                  generation.adapter_id,
                  generation.adapter_version || "",
                ])}
              >
                {generation.adapter_id} (not listed; refresh server adapters)
              </option>
            )}
          {entries.map((entry) => (
            <option
              key={entry.id + entry.version}
              value={JSON.stringify([entry.id, entry.version])}
            >
              {entry.id} · {entry.version.slice(0, 12)} · rank{" "}
              {entry.manifest.rank}
            </option>
          ))}
        </select>
      </Field>
      <Field
        label="Scale override"
        hint="Leave empty to use the adapter scale. Zero disables its delta."
      >
        <input
          type="number"
          step="any"
          disabled={!generation.adapter_id}
          value={generation.adapter_scale || ""}
          onChange={(e) =>
            set({
              generation: { ...generation, adapter_scale: e.target.value },
            })
          }
        />
      </Field>
      <Field label="Adapter ID to register">
        <input
          value={id}
          onChange={(e) => setId(e.target.value)}
          placeholder="html"
        />
      </Field>
      <Field label="Upload PTH">
        <input
          type="file"
          accept=".pth"
          disabled={busy}
          onChange={(e) => setFile(e.target.files?.[0])}
        />
      </Field>
      <Field label="Legacy checkpoint.json (optional)">
        <input
          type="file"
          accept=".json"
          disabled={busy}
          onChange={(e) => setMetadata(e.target.files?.[0])}
        />
      </Field>
      <button
        type="button"
        disabled={busy || !file || !id.trim()}
        onClick={() => void register(true)}
      >
        Upload & select
      </button>
      <details>
        <summary>Register a file already on the server</summary>
        <Field
          label="Server PTH path"
          hint="adapter-final.pth or checkpoint-N/training.pth"
        >
          <input value={path} onChange={(e) => setPath(e.target.value)} />
        </Field>
        <button
          type="button"
          disabled={busy || !path.trim() || !id.trim()}
          onClick={() => void register(false)}
        >
          Register & select
        </button>
      </details>
      <p className="small muted">
        New training PTH files are self-contained. Old training checkpoints need
        checkpoint.json alongside the PTH. Registrations are lost when the
        server restarts. Selection applies to new requests; running requests
        keep their adapter.
      </p>
      <button
        type="button"
        disabled={busy}
        onClick={() => {
          setError("");
          setRevision((v) => v + 1);
        }}
      >
        Refresh adapters
      </button>
      <p className="small muted">{stats}</p>
      <ErrorPanel error={error} />
      {busy && <p role="status">Registering…</p>}
      {entries.map((entry) => (
        <div className="panel" key={entry.id + entry.version}>
          <strong>{entry.id}</strong>
          <p>
            <code>{entry.version}</code>
          </p>
          <button type="button" onClick={() => select(entry.id, entry.version)}>
            Use adapter
          </button>{" "}
          <button
            type="button"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              setError("");
              try {
                await client.deleteAdapter(entry.id, entry.version);
                const current = useSettings.getState().values.generation;
                if (
                  current.adapter_id === entry.id &&
                  current.adapter_version === entry.version
                )
                  select("");
                setRevision((v) => v + 1);
              } catch (e) {
                setError(String(e));
              } finally {
                setBusy(false);
              }
            }}
          >
            Delete version
          </button>
        </div>
      ))}
    </div>
  );
}
