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
