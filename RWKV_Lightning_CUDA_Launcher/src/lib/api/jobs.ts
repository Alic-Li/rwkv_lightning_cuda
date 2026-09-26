import { nodeRequest } from "./http";
import type {
  JobID,
  JobsResponse,
  ProcessStatus,
  QuantizationConfig,
  TuningConfig,
} from "./types";

/**
 * Training and quantization share one process slot each. `{id}` is the fixed
 * literal `tuning` / `quantization` — there is no task queue.
 */
export const jobsApi = {
  list: (backendId: string, signal?: AbortSignal) =>
    nodeRequest<JobsResponse>(backendId, "/api/v1/jobs", { signal }),

  status: (backendId: string, id: JobID, signal?: AbortSignal) =>
    nodeRequest<ProcessStatus>(backendId, `/api/v1/jobs/${id}`, { signal }),

  startTuning: (backendId: string, config: TuningConfig) =>
    nodeRequest<{ ok: boolean }>(backendId, "/api/v1/jobs/tuning", {
      body: config,
    }),

  validateDataset: (backendId: string, path: string) =>
    nodeRequest<{ samples: number }>(
      backendId,
      "/api/v1/jobs/tuning/validate",
      { body: { path } },
    ),

  startQuantization: (backendId: string, config: QuantizationConfig) =>
    nodeRequest<{ ok: boolean }>(backendId, "/api/v1/jobs/quantization", {
      body: config,
    }),

  /** Success is HTTP 200 with an empty body — never parse it unconditionally. */
  stop: (backendId: string, id: JobID) =>
    nodeRequest<{ ok: boolean } | undefined>(
      backendId,
      `/api/v1/jobs/${id}/stop`,
      { body: {} },
    ),

  logsPath: (id: JobID) => `/api/v1/jobs/${id}/logs`,
};
