import { tNow } from "@/lib/i18n";
import { nodeRequest, nodeStream } from "./http";
import { readSSE, type StreamEvent } from "./sse";
import type { ModelListResponse, UploadedState } from "./types";

export interface BatchCompletion {
  choices: {
    index: number;
    message?: { role?: string; content?: string };
    finish_reason?: string;
  }[];
}

/** OpenAI-compatible surface (`/v1/*`), always through the local Client. */
export const inferenceApi = {
  /**
   * Chat completions with server-side chat templating. Streams `data: {json}`
   * frames and terminates on `data: [DONE]`.
   */
  async streamChat(
    backendId: string,
    body: unknown,
    signal: AbortSignal,
    onEvent: (event: StreamEvent) => void,
  ) {
    const response = await nodeStream(backendId, "/v1/chat/completions", {
      method: "POST",
      body,
      signal,
      headers: { Accept: "text/event-stream" },
    });
    if (!response.body) throw new Error(tNow("error.noStream"));
    return readSSE(response.body, signal, onEvent);
  },

  /** Raw continuation. `contents[]` maps to `choices[].index` in the reply. */
  batchCompletions: (backendId: string, body: unknown, signal?: AbortSignal) =>
    nodeRequest<BatchCompletion>(backendId, "/v1/batch/completions", {
      body,
      signal,
    }),

  models: (backendId: string, signal?: AbortSignal) =>
    nodeRequest<ModelListResponse>(backendId, "/v1/models", { signal }),

  /** Dynamic-loading mode only; inference requests never switch the model. */
  loadModel: (backendId: string, model: string) =>
    nodeRequest<unknown>(backendId, "/v1/model/load", { body: { model } }),

  listStates: (backendId: string, signal?: AbortSignal) =>
    nodeRequest<{ data: UploadedState[] }>(backendId, "/v1/state/list", {
      signal,
    }),

  uploadState: (backendId: string, file: File) => {
    const body = new FormData();
    body.append("file", file);
    return nodeRequest<UploadedState>(backendId, "/v1/state/upload", { body });
  },

  deleteState: (backendId: string, stateId: string) =>
    nodeRequest<{ state_id: string; deleted: boolean }>(
      backendId,
      "/v1/state/delete",
      { body: { state_id: stateId } },
    ),
};
