import { tNow } from "@/lib/i18n";
import { nodeRequest, nodeStream } from "./http";
import type {
  RuntimeConfig,
  RuntimeLoadRequest,
  RuntimeLoadResponse,
  RuntimeState,
} from "./types";
import { SSEParser } from "./sse";

export const runtimeApi = {
  status: (backendId: string, signal?: AbortSignal) =>
    nodeRequest<RuntimeState>(backendId, "/api/v1/runtime", { signal }),

  start: (backendId: string, config: RuntimeConfig) =>
    nodeRequest<{ ok: boolean }>(backendId, "/api/v1/runtime/start", {
      body: config,
    }),

  stop: (backendId: string) =>
    nodeRequest<{ ok: boolean } | undefined>(
      backendId,
      "/api/v1/runtime/stop",
      {
        body: {},
      },
    ),

  /** Restarts with the configuration the process actually runs with. */
  restart: (backendId: string) =>
    nodeRequest<{ ok: boolean } | undefined>(
      backendId,
      "/api/v1/runtime/restart",
      { body: {} },
    ),

  /**
   * Hand a .pth that already sits on the node to the runtime's multipart
   * upload. The native API registers an *adapter* from a server-side path but
   * has no path form for a state, so the Agent does that hop
   * (`state_import` capability).
   */
  importState: (backendId: string, path: string) =>
    nodeRequest<{ state_id?: string }>(
      backendId,
      "/api/v1/runtime/state/import",
      { body: { path } },
    ),

  /** Card-switch load: restart on `visible_devices`, then (dynamic) load. */
  load: (backendId: string, body: RuntimeLoadRequest) =>
    nodeRequest<RuntimeLoadResponse>(backendId, "/api/v1/runtime/load", {
      body,
    }),

  logsUrl: (backendId: string) =>
    `/api/v1/backends/${backendId}/api/v1/runtime/logs`,
};

/**
 * Control-plane log SSE. Each `data:` payload is one raw text line — no JSON
 * envelope, no `[DONE]`, and the stream stays open after the process exits.
 */
export async function readLogStream(
  backendId: string,
  path: string,
  signal: AbortSignal,
  onLine: (line: string) => void,
) {
  const response = await nodeStream(backendId, path, {
    method: "GET",
    signal,
    headers: { Accept: "text/event-stream" },
  });
  const body = response.body;
  if (!body) throw new Error(tNow("error.noLogStream"));
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const cancel = () => void reader.cancel().catch(() => {});
  signal.addEventListener("abort", cancel, { once: true });
  const parser = new SSEParser((data) => {
    for (const line of data.split("\n")) onLine(line);
  });
  try {
    while (!signal.aborted) {
      const part = await reader.read();
      if (part.done) break;
      parser.push(decoder.decode(part.value, { stream: true }));
    }
  } finally {
    signal.removeEventListener("abort", cancel);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
