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
export interface AdapterEntry {
  id: string;
  version: string;
  manifest: { rank: number; scale: number; targets: unknown[] };
}
export function adapterFields(v: {
  adapter_id?: string;
  adapter_version?: string;
  adapter_scale?: string;
}) {
  if (!v.adapter_id?.trim()) return {};
  const scale = v.adapter_scale?.trim();
  if (scale && !Number.isFinite(Number(scale)))
    throw new Error("Adapter scale must be finite");
  return {
    adapter_id: v.adapter_id.trim(),
    ...(v.adapter_version?.trim()
      ? { adapter_version: v.adapter_version.trim() }
      : {}),
    ...(scale ? { adapter_scale: Number(scale) } : {}),
  };
}
export function generationBody<
  T extends {
    adapter_id?: string;
    adapter_version?: string;
    adapter_scale?: string;
  },
>(v: T) {
  const { adapter_id, adapter_version, adapter_scale, ...rest } = v;
  return {
    ...rest,
    ...adapterFields({ adapter_id, adapter_version, adapter_scale }),
  };
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
  listAdapters(signal?: AbortSignal) {
    return request<{
      data: AdapterEntry[];
      ram_bytes: number;
      gpu_bytes: number;
      uploads: number;
    }>(
      `${this.baseURL.replace(/\/$/, "")}/v1/adapters`,
      undefined,
      signal,
      this.key,
    );
  }
  registerAdapter(adapter_id: string, path: string) {
    return request<{ adapter_id: string; version: string }>(
      `${this.baseURL.replace(/\/$/, "")}/v1/adapters`,
      { adapter_id, path },
      undefined,
      this.key,
    );
  }
  async uploadAdapter(adapter_id: string, file: File, metadata?: File) {
    const body = new FormData();
    body.append("adapter_id", adapter_id);
    body.append("file", file);
    if (metadata) body.append("metadata", metadata, "checkpoint.json");
    const response = await fetch(
      `${this.baseURL.replace(/\/$/, "")}/v1/adapters`,
      {
        method: "POST",
        headers: this.key ? { Authorization: `Bearer ${this.key}` } : {},
        body,
      },
    );
    if (!response.ok)
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    return response.json() as Promise<{ adapter_id: string; version: string }>;
  }
  async deleteAdapter(adapter_id: string, adapter_version: string) {
    const response = await fetch(
      `${this.baseURL.replace(/\/$/, "")}/v1/adapters`,
      {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
          ...(this.key ? { Authorization: `Bearer ${this.key}` } : {}),
        },
        body: JSON.stringify({ adapter_id, adapter_version }),
      },
    );
    if (!response.ok)
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
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
