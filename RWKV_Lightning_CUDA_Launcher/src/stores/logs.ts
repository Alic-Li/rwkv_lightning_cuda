import { create } from "zustand";
import { readLogStream } from "@/lib/api/runtime";
import { jobsApi } from "@/lib/api/jobs";

export type LogKind = "runtime" | "tuning" | "quantization";

export interface LogStream {
  lines: string[];
  status: "idle" | "connecting" | "open" | "error";
  error: string;
}

const MAX_LINES = 4000;

const streamKey = (backendId: string, kind: LogKind) => `${backendId}:${kind}`;

const pathFor = (kind: LogKind) =>
  kind === "runtime" ? "/api/v1/runtime/logs" : jobsApi.logsPath(kind);

/** Live SSE connections, keyed so switching nodes never mixes logs. */
const controllers = new Map<string, AbortController>();

interface LogsState {
  streams: Record<string, LogStream>;
  open: (backendId: string, kind: LogKind) => void;
  close: (backendId: string, kind: LogKind) => void;
  closeBackend: (backendId: string) => void;
  clear: (backendId: string, kind: LogKind) => void;
}

export const useLogs = create<LogsState>((set, get) => {
  const update = (
    backendId: string,
    kind: LogKind,
    next: Partial<LogStream>,
  ) => {
    const key = streamKey(backendId, kind);
    set((s) => {
      const base: LogStream = s.streams[key] ?? {
        lines: [],
        status: "idle",
        error: "",
      };
      return { streams: { ...s.streams, [key]: { ...base, ...next } } };
    });
  };

  return {
    streams: {},

    open: (backendId, kind) => {
      if (!backendId) return;
      const key = streamKey(backendId, kind);
      const existing = controllers.get(key);
      if (existing) return;
      const controller = new AbortController();
      controllers.set(key, controller);
      // A reconnect replays the buffered log, so start from a clean slate.
      update(backendId, kind, { lines: [], status: "connecting", error: "" });
      void (async () => {
        try {
          await readLogStream(
            backendId,
            pathFor(kind),
            controller.signal,
            (line) => {
              const current = get().streams[key]?.lines ?? [];
              const lines =
                current.length >= MAX_LINES
                  ? [...current.slice(current.length - MAX_LINES + 1), line]
                  : [...current, line];
              update(backendId, kind, { lines, status: "open" });
            },
          );
          if (!controller.signal.aborted)
            update(backendId, kind, { status: "idle" });
        } catch (error) {
          if (controller.signal.aborted) return;
          update(backendId, kind, {
            status: "error",
            error: error instanceof Error ? error.message : String(error),
          });
        } finally {
          // Delete only our own controller. close(key) followed by open(key)
          // in the same tick (StrictMode's mount→unmount→mount, or any effect
          // re-run) registers a newer controller under this key; deleting by
          // key alone would drop it from the map while its stream is live,
          // leaving that stream unabortable and letting the next open() start
          // a second one that appends to the same lines.
          if (controllers.get(key) === controller) controllers.delete(key);
        }
      })();
    },

    close: (backendId, kind) => {
      const key = streamKey(backendId, kind);
      controllers.get(key)?.abort();
      controllers.delete(key);
      update(backendId, kind, { status: "idle" });
    },

    closeBackend: (backendId) => {
      for (const kind of ["runtime", "tuning", "quantization"] as LogKind[])
        get().close(backendId, kind);
    },

    clear: (backendId, kind) => update(backendId, kind, { lines: [] }),
  };
});

/**
 * Shared fallback snapshot. `useSyncExternalStore` compares snapshots by
 * identity, so a selector that returns a fresh literal would re-render forever.
 */
export const IDLE_STREAM: LogStream = {
  lines: [],
  status: "idle",
  error: "",
};

export function selectLogStream(
  streams: Record<string, LogStream>,
  backendId: string,
  kind: LogKind,
): LogStream {
  return streams[streamKey(backendId, kind)] ?? IDLE_STREAM;
}

export function useLogStream(backendId: string, kind: LogKind): LogStream {
  return useLogs((s) => selectLogStream(s.streams, backendId, kind));
}
