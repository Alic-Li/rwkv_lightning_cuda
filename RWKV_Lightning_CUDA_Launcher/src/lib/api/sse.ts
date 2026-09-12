/** Incremental SSE framing; UTF-8 decoding belongs to the stream reader. */
export class SSEParser {
  private pending = "";
  private data: string[] = [];
  constructor(private emit: (data: string) => void) {}
  push(text: string) {
    this.pending += text;
    let i: number;
    while ((i = this.pending.indexOf("\n")) >= 0) {
      const line = this.pending.slice(0, i).replace(/\r$/, "");
      this.pending = this.pending.slice(i + 1);
      if (!line) {
        if (this.data.length) this.emit(this.data.join("\n"));
        this.data = [];
      } else if (line === "data" || line.startsWith("data:"))
        this.data.push(line.slice(5).replace(/^ /, ""));
    }
  }
}
export interface StreamEvent {
  id?: string;
  error?: string | { message?: string };
  choices?: {
    index?: number;
    delta?: { content?: string };
    finish_reason?: string | null;
  }[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}
export async function readSSE(
  body: ReadableStream<Uint8Array>,
  signal: AbortSignal,
  onEvent: (event: StreamEvent) => void,
) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let done = false;
  const cancel = () => {
    void reader.cancel();
  };
  signal.addEventListener("abort", cancel, { once: true });
  const parser = new SSEParser((data) => {
    if (done) return;
    if (data.trim() === "[DONE]") {
      done = true;
      return;
    }
    let event: StreamEvent;
    try {
      event = JSON.parse(data);
    } catch {
      throw new Error(`Malformed SSE event: ${data.slice(0, 160)}`);
    }
    if (!event || typeof event !== "object")
      throw new Error("Invalid SSE event");
    if (event.error)
      throw new Error(
        typeof event.error === "string"
          ? event.error
          : event.error.message || JSON.stringify(event.error),
      );
    onEvent(event);
  });
  try {
    signal.throwIfAborted();
    while (!done) {
      const part = await reader.read();
      signal.throwIfAborted();
      if (part.done) {
        parser.push(decoder.decode());
        break;
      }
      parser.push(decoder.decode(part.value, { stream: true }));
    }
    if (!done)
      throw new Error(
        "Backend disconnected before [DONE]. Partial output was preserved.",
      );
  } finally {
    signal.removeEventListener("abort", cancel);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
