import { create } from "zustand";
import { applyDevice } from "@/lib/api/launcher";
import { inferenceApi } from "@/lib/api/inference";
import { jobsApi } from "@/lib/api/jobs";
import { nodeApi } from "@/lib/api/node";
import { runtimeApi } from "@/lib/api/runtime";
import { tNow } from "@/lib/i18n";
import type {
  DeviceSelection,
  JobID,
  Jobs,
  MetricsResponse,
  NodeInfo,
  QuantizationConfig,
  RuntimeConfig,
  RuntimeState,
  TuningConfig,
} from "@/lib/api/types";
import { useBackends } from "./backends";

export interface NodeSnapshot {
  info?: NodeInfo;
  runtime?: RuntimeState;
  /** Unwrapped `jobs` map, so consumers read `snapshot.jobs.tuning`. */
  jobs?: Jobs;
  metrics?: MetricsResponse;
  metricsError?: string;
  error: string;
  fetchedAt: number;
}

const EMPTY: NodeSnapshot = { error: "", fetchedAt: 0 };

/** Combine a caller signal with a status-poll deadline. */
function withTimeout(signal: AbortSignal | undefined, ms: number) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, ms);
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort, { once: true });
  return {
    signal: controller.signal,
    /** True only when our own deadline fired — not when the caller aborted. */
    get timedOut() {
      return timedOut;
    },
    done: () => {
      window.clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    },
  };
}

/**
 * A poll that outran its deadline means "slow node", not "broken node". The
 * raw DOMException reads as "signal is aborted without reason" and renders as
 * a red failure notice on a node that is merely busy.
 */
function pollError(error: unknown, timedOut: boolean) {
  if (timedOut) return tNow("node.pollTimeout");
  return error instanceof Error ? error.message : String(error);
}

interface NodesState {
  snapshots: Record<string, NodeSnapshot>;
  busy: Record<string, boolean>;
  refresh: (id: string, signal?: AbortSignal) => Promise<void>;
  refreshAll: (signal?: AbortSignal) => Promise<void>;
  refreshMetrics: (id: string, signal?: AbortSignal) => Promise<void>;
  startRuntime: (id: string, config: RuntimeConfig) => Promise<void>;
  stopRuntime: (id: string) => Promise<void>;
  restartRuntime: (id: string) => Promise<void>;
  startTuning: (id: string, config: TuningConfig) => Promise<void>;
  startQuantization: (id: string, config: QuantizationConfig) => Promise<void>;
  stopJob: (id: string, job: JobID) => Promise<void>;
  validateDataset: (id: string, path: string) => Promise<number>;
  loadModel: (id: string, model: string) => Promise<void>;
  /** Card-switch load: restart on the card, then (dynamic) load the model. */
}

const patch = (
  set: (fn: (s: NodesState) => Partial<NodesState>) => void,
  id: string,
  next: Partial<NodeSnapshot>,
) =>
  set((s) => ({
    snapshots: {
      ...s.snapshots,
      [id]: { ...(s.snapshots[id] ?? EMPTY), ...next },
    },
  }));

export const useNodes = create<NodesState>((set, get) => ({
  snapshots: {},
  busy: {},

  refresh: async (id, signal) => {
    if (!id) return;
    const backend = useBackends.getState().list.find((b) => b.id === id);
    // An inference-only node has no Agent API: nothing to poll.
    if (backend?.kind === "inference_only") {
      patch(set, id, { error: "", fetchedAt: Date.now() });
      return;
    }
    const timeout = withTimeout(signal, 6000);
    try {
      const [info, jobsResponse] = await Promise.all([
        nodeApi.info(id, timeout.signal),
        jobsApi.list(id, timeout.signal),
      ]);
      const runtime: RuntimeState = info;
      patch(set, id, {
        info,
        runtime,
        jobs: jobsResponse.jobs,
        error: "",
        fetchedAt: Date.now(),
      });
    } catch (error) {
      if (signal?.aborted) return;
      patch(set, id, {
        error: pollError(error, timeout.timedOut),
        fetchedAt: Date.now(),
      });
    } finally {
      timeout.done();
    }
  },

  refreshAll: async (signal) => {
    const ids = useBackends.getState().list.map((b) => b.id);
    await Promise.all(ids.map((id) => get().refresh(id, signal)));
  },

  refreshMetrics: async (id, signal) => {
    if (!id) return;
    const backend = useBackends.getState().list.find((b) => b.id === id);
    if (backend?.kind === "inference_only") {
      patch(set, id, { metrics: undefined, metricsError: "" });
      return;
    }
    const timeout = withTimeout(signal, 6000);
    try {
      const metrics = await nodeApi.metrics(id, timeout.signal);
      patch(set, id, { metrics, metricsError: "" });
    } catch (error) {
      if (signal?.aborted) return;
      patch(set, id, {
        metrics: undefined,
        metricsError: pollError(error, timeout.timedOut),
      });
    } finally {
      timeout.done();
    }
  },

  startRuntime: async (id, config) => {
    set((s) => ({ busy: { ...s.busy, [id]: true } }));
    try {
      await runtimeApi.start(id, config);
      await get().refresh(id);
    } finally {
      set((s) => ({ busy: { ...s.busy, [id]: false } }));
    }
  },

  stopRuntime: async (id) => {
    set((s) => ({ busy: { ...s.busy, [id]: true } }));
    try {
      await runtimeApi.stop(id);
      await get().refresh(id);
    } finally {
      set((s) => ({ busy: { ...s.busy, [id]: false } }));
    }
  },

  restartRuntime: async (id) => {
    set((s) => ({ busy: { ...s.busy, [id]: true } }));
    try {
      await runtimeApi.restart(id);
      await get().refresh(id);
    } finally {
      set((s) => ({ busy: { ...s.busy, [id]: false } }));
    }
  },

  startTuning: async (id, config) => {
    set((s) => ({ busy: { ...s.busy, [id]: true } }));
    try {
      await jobsApi.startTuning(id, config);
      await get().refresh(id);
    } finally {
      set((s) => ({ busy: { ...s.busy, [id]: false } }));
    }
  },

  startQuantization: async (id, config) => {
    set((s) => ({ busy: { ...s.busy, [id]: true } }));
    try {
      await jobsApi.startQuantization(id, config);
      await get().refresh(id);
    } finally {
      set((s) => ({ busy: { ...s.busy, [id]: false } }));
    }
  },

  stopJob: async (id, job) => {
    set((s) => ({ busy: { ...s.busy, [id]: true } }));
    try {
      await jobsApi.stop(id, job);
      await get().refresh(id);
    } finally {
      set((s) => ({ busy: { ...s.busy, [id]: false } }));
    }
  },

  validateDataset: async (id, path) => {
    const { samples } = await jobsApi.validateDataset(id, path);
    return samples;
  },

  loadModel: async (id, model) => {
    set((s) => ({ busy: { ...s.busy, [id]: true } }));
    try {
      await inferenceApi.loadModel(id, model);
      await get().refresh(id);
    } finally {
      set((s) => ({ busy: { ...s.busy, [id]: false } }));
    }
  },
}));

/** Read a snapshot without subscribing to the whole map. */
export function useSnapshot(id: string) {
  return useNodes((s) => s.snapshots[id]);
}

export function useNodeBusy(id: string) {
  return useNodes((s) => Boolean(s.busy[id]));
}

/** Apply the tri-state device selector to a request body. */
export function withDevices<T extends { visible_devices?: string }>(
  body: T,
  selection: DeviceSelection,
) {
  return applyDevice(body, selection);
}
