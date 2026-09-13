import { useEffect, useMemo, useState } from "react";
import { RWKVClient, type UploadedState } from "../lib/api/client";
import { useSecret, useSettings } from "../stores/settings";
import { ErrorPanel, Field } from "./common";

export function StateManager() {
  const baseURL = useSettings((s) => s.values.baseURL);
  const key = useSecret((s) => s.key);
  const selected = useSettings((s) => s.values.generation.state_id);
  const client = useMemo(() => new RWKVClient(baseURL, key), [baseURL, key]);
  const [states, setStates] = useState<UploadedState[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const select = (state_id: string) => {
    const store = useSettings.getState();
    store.set({ generation: { ...store.values.generation, state_id } });
  };
  useEffect(() => {
    const controller = new AbortController();
    setBusy(true);
    setError("");
    client
      .listStates(controller.signal)
      .then((result) => setStates(result.data))
      .catch((e: unknown) => {
        if (!controller.signal.aborted) setError(String(e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false);
      });
    return () => controller.abort();
  }, [client, revision]);
  return (
    <div className="state-manager">
      <p>
        Upload a state checkpoint, then select it to initialize chat requests.
        Each request starts from this state and includes the conversation
        history.
      </p>
      <p className="muted">
        States belong to the current server and are lost when it restarts.
      </p>
      <Field label="Initial state">
        <select
          value={selected}
          disabled={busy}
          onChange={(e) => select(e.target.value)}
        >
          <option value="">None (default initial state)</option>
          {selected && !states.some((s) => s.state_id === selected) && (
            <option value={selected}>
              {selected} (not listed on this server)
            </option>
          )}
          {states.map((s) => (
            <option key={s.state_id} value={s.state_id}>
              {s.filename}
            </option>
          ))}
        </select>
      </Field>
      <Field label="Upload state (.pth)">
        <input
          type="file"
          accept=".pth"
          disabled={busy}
          onChange={async (e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (!file) return;
            setBusy(true);
            setError("");
            try {
              const state = await client.uploadState(file);
              select(state.state_id);
              setRevision((r) => r + 1);
            } catch (e) {
              setError(String(e));
            } finally {
              setBusy(false);
            }
          }}
        />
      </Field>
      <button
        type="button"
        disabled={busy}
        onClick={() => setRevision((r) => r + 1)}
      >
        Refresh states
      </button>
      <ErrorPanel error={error} />
      {busy && <p role="status">Loading…</p>}
      {!busy && !error && !states.length && (
        <p>No uploaded states on this server.</p>
      )}
      {states.map((state) => (
        <div className="panel" key={state.state_id}>
          <strong>{state.filename}</strong>
          <p>
            State ID: <code>{state.state_id}</code>
          </p>
          <p>
            {state.size_bytes.toLocaleString()} bytes · {state.tensor_count}{" "}
            tensors · {new Date(state.created * 1000).toLocaleString()}
          </p>
          <button
            type="button"
            disabled={busy}
            onClick={() => select(state.state_id)}
          >
            {selected === state.state_id ? "Selected" : "Use state"}
          </button>{" "}
          <button
            type="button"
            disabled={busy}
            onClick={async () => {
              if (!window.confirm(`Delete server state "${state.filename}"?`))
                return;
              setBusy(true);
              setError("");
              try {
                await client.deleteState(state.state_id);
                if (
                  useSettings.getState().values.generation.state_id ===
                  state.state_id
                )
                  select("");
                setStates((items) =>
                  items.filter((s) => s.state_id !== state.state_id),
                );
              } catch (e) {
                setError(String(e));
              } finally {
                setBusy(false);
              }
            }}
          >
            Delete
          </button>
        </div>
      ))}
    </div>
  );
}
