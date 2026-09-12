import { Play, Square, RotateCw, LoaderCircle } from "lucide-react";
import { useRuntime, useRuntimeForm } from "../stores/runtime";
export function RuntimeControls() {
  const { runtime, tuning, connected, busy, action } = useRuntime();
  const model = useRuntimeForm((s) => s.config.model_path);
  return (
    <div className="runtime-controls">
      <button
        className="primary"
        disabled={
          !connected ||
          busy ||
          runtime.running ||
          tuning.running ||
          !model.trim()
        }
        onClick={() => void action("start")}
      >
        {busy ? (
          <LoaderCircle size={13} className="spin" />
        ) : (
          <Play size={13} />
        )}{" "}
        Start
      </button>
      <button
        disabled={!connected || busy || !runtime.running}
        onClick={() => void action("stop")}
      >
        <Square size={12} /> Stop
      </button>
      <button
        title="Restart with the running configuration"
        disabled={!connected || busy || !runtime.running}
        onClick={() => void action("restart")}
      >
        <RotateCw size={13} />
        <span>Restart</span>
      </button>
    </div>
  );
}
export function RuntimeBadge() {
  const { runtime, connected } = useRuntime();
  const state = connected ? runtime.status : "offline";
  return (
    <span className={`status ${state}`}>
      {state === "starting" || state === "stopping" ? (
        <LoaderCircle size={12} className="spin" />
      ) : (
        <i />
      )}
      {!connected
        ? "Disconnected"
        : state === "ready"
          ? "Running"
          : state.charAt(0).toUpperCase() + state.slice(1)}
    </span>
  );
}
