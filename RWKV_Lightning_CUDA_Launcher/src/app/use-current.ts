import { useMemo } from "react";
import type {
  BackendView,
  Jobs,
  MetricsResponse,
  NodeInfo,
  RuntimeState,
} from "@/lib/api/types";
import { useBackends } from "@/stores/backends";
import { useNodes, type NodeSnapshot } from "@/stores/nodes";

export interface CurrentContext {
  backendId: string;
  backend: BackendView | undefined;
  snapshot: NodeSnapshot | undefined;
  info: NodeInfo | undefined;
  runtime: RuntimeState | undefined;
  jobs: Jobs | undefined;
  metrics: MetricsResponse | undefined;
  metricsError: string | undefined;
  error: string;
  busy: boolean;
  hasAgent: boolean;
  runtimeReady: boolean;
  /**
   * Whether inference can be attempted. A bare inference node has no Agent
   * API at all, so `runtimeReady` can never become true there — reachability
   * plus the `inference_only` kind is the strongest signal available.
   */
  canInfer: boolean;
}

/** Everything the shell and pages need about the selected backend. */
export function useCurrent(): CurrentContext {
  const list = useBackends((s) => s.list);
  const backendId = useBackends((s) => s.currentId);
  const snapshot = useNodes((s) => s.snapshots[backendId]);
  const busy = useNodes((s) => Boolean(s.busy[backendId]));

  return useMemo(() => {
    const backend = list.find((b) => b.id === backendId);
    const runtime = snapshot?.runtime;
    const hasAgent =
      Boolean(backend) &&
      backend?.kind !== "inference_only" &&
      (backend?.kind === "agent" || (backend?.capabilities.length ?? 0) > 0);
    const runtimeReady = runtime?.status === "ready";
    return {
      backendId,
      backend,
      snapshot,
      info: snapshot?.info,
      runtime,
      jobs: snapshot?.jobs,
      metrics: snapshot?.metrics,
      metricsError: snapshot?.metricsError,
      error: snapshot?.error ?? "",
      busy,
      hasAgent,
      runtimeReady,
      canInfer:
        Boolean(backend?.reachable) &&
        (backend?.kind === "inference_only" || runtimeReady),
    };
  }, [list, backendId, snapshot, busy]);
}
