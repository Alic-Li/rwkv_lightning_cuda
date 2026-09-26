/**
 * Same-origin transport. The browser only ever talks to the local Client:
 * every node-scoped call is addressed as `/api/v1/backends/{id}/...`, and the
 * Client forwards it to the Agent. Never fetch a backend `base_url` directly.
 */

export class ApiError extends Error {
  readonly status: number;
  readonly reason: string;

  constructor(message: string, status: number, reason = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.reason = reason;
  }

  /** `unsupported` (501) / `client_only` (404) style markers from the Agent. */
  get code() {
    return this.reason;
  }
}

export const backendPrefix = (id: string) =>
  `/api/v1/backends/${encodeURIComponent(id)}`;

/**
 * Node-scoped URL. `path` is either `/api/v1/...` (control plane) or
 * `/v1/...` (OpenAI-compatible inference).
 */
export const nodeUrl = (backendId: string, path: string) =>
  `${backendPrefix(backendId)}${path.startsWith("/") ? path : `/${path}`}`;

/** Client-scoped URL: the backend registry lives on the local Client only. */
export const clientUrl = (path: string) =>
  path.startsWith("/") ? path : `/${path}`;

interface ErrorBody {
  error?: unknown;
  reason?: unknown;
}

const asMessage = (value: unknown) =>
  typeof value === "string" ? value : value ? JSON.stringify(value) : "";

/**
 * Control-plane failures are usually `{"error":"…","reason":"…"}`, but proxy
 * failures can be plain text and stop endpoints can return an empty body.
 */
export async function readError(response: Response) {
  let text = "";
  try {
    text = await response.text();
  } catch {
    text = "";
  }
  let reason = "";
  let message = text.trim();
  if (message.startsWith("{") || message.startsWith("[")) {
    try {
      const body = JSON.parse(message) as ErrorBody;
      reason = asMessage(body.reason);
      const error = asMessage(body.error);
      message = [error, reason].filter(Boolean).join(" · ") || message;
    } catch {
      // Not JSON after all: keep the raw text.
    }
  }
  return new ApiError(
    message || `HTTP ${response.status} ${response.statusText}`.trim(),
    response.status,
    reason,
  );
}

export interface RequestOptions {
  method?: "GET" | "POST" | "DELETE" | "PUT" | "PATCH";
  body?: unknown;
  signal?: AbortSignal;
  /** Headers the caller adds on top of the JSON defaults. */
  headers?: Record<string, string>;
}

async function send(url: string, options: RequestOptions) {
  const { method = options.body === undefined ? "GET" : "POST" } = options;
  const headers: Record<string, string> = { ...options.headers };
  let body: BodyInit | undefined;
  if (options.body instanceof FormData) {
    body = options.body;
  } else if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.body);
  }
  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers,
      body,
      signal: options.signal,
    });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw new ApiError(
      error instanceof Error ? error.message : String(error),
      0,
      "network",
    );
  }
  if (!response.ok) throw await readError(response);
  return response;
}

/** JSON request against a node through the local Client. */
export async function nodeRequest<T>(
  backendId: string,
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const response = await send(nodeUrl(backendId, path), options);
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  if (!text.trim()) return undefined as T;
  return JSON.parse(text) as T;
}

/** JSON request against the local Client itself (backend registry). */
export async function clientRequest<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const response = await send(clientUrl(path), options);
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  if (!text.trim()) return undefined as T;
  return JSON.parse(text) as T;
}

/**
 * Streaming request. Used for inference SSE (`data: {json}`) and control-plane
 * log SSE (`data: raw line`). Callers own the parser choice.
 */
export async function nodeStream(
  backendId: string,
  path: string,
  options: RequestOptions = {},
) {
  return send(nodeUrl(backendId, path), options);
}

export async function clientStream(path: string, options: RequestOptions = {}) {
  return send(clientUrl(path), options);
}

/** Copy helper shared by the log viewers and code blocks. */
export async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.append(area);
  area.select();
  document.execCommand("copy");
  area.remove();
}
