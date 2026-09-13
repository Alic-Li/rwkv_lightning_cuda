import { readSSE, type StreamEvent } from "./sse";
export async function request<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
  key = "",
): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
      ...(key ? { Authorization: `Bearer ${key}` } : {}),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok)
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  return response.json() as Promise<T>;
}
export interface UploadedState {
  state_id: string;
  filename: string;
  size_bytes: number;
  tensor_count: number;
  created: number;
}
export class RWKVClient {
  constructor(
    public baseURL = "",
    private key = "",
  ) {}
  async streamChat(
    body: unknown,
    signal: AbortSignal,
    onEvent: (event: StreamEvent) => void,
  ) {
    const response = await fetch(
      `${this.baseURL.replace(/\/$/, "")}/v1/chat/completions`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.key ? { Authorization: `Bearer ${this.key}` } : {}),
        },
        body: JSON.stringify(body),
        signal,
      },
    );
    if (!response.ok)
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    if (
      !response.body ||
      !response.headers.get("content-type")?.includes("text/event-stream")
    )
      throw new Error("Backend did not return an SSE stream");
    return readSSE(response.body, signal, onEvent);
  }
  completeChat(body: unknown, signal: AbortSignal) {
    return request<{
      choices: {
        index: number;
        message?: { role?: string; content?: string };
        finish_reason?: string;
      }[];
    }>(
      `${this.baseURL.replace(/\/$/, "")}/v1/chat/completions`,
      body,
      signal,
      this.key,
    );
  }
  listStates(signal?: AbortSignal) {
    return request<{ data: UploadedState[] }>(
      `${this.baseURL.replace(/\/$/, "")}/v1/state/list`,
      undefined,
      signal,
      this.key,
    );
  }
  async uploadState(file: File) {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch(
      `${this.baseURL.replace(/\/$/, "")}/v1/state/upload`,
      {
        method: "POST",
        headers: this.key ? { Authorization: `Bearer ${this.key}` } : {},
        body,
      },
    );
    if (!response.ok)
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    return response.json() as Promise<UploadedState>;
  }
  deleteState(state_id: string) {
    return request<{ state_id: string; deleted: boolean }>(
      `${this.baseURL.replace(/\/$/, "")}/v1/state/delete`,
      { state_id },
      undefined,
      this.key,
    );
  }
  loadModel(model: string) {
    return request(
      `${this.baseURL}/v1/model/load`,
      { model },
      undefined,
      this.key,
    );
  }
  models() {
    return request<{
      data: { id: string }[];
      available?: string[];
      loaded?: string | null;
    }>(`${this.baseURL}/v1/models`, undefined, undefined, this.key);
  }
}
