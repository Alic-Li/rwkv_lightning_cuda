import { describe, it, expect } from "bun:test";
import { SSEParser, readSSE } from "../src/lib/api/sse";
import { chunkText, translationPrompt } from "../src/lib/translate/chunk";
import {
  createTranslationScheduler,
  orderedMerge,
  type TranslateChunk,
} from "../src/lib/translate/scheduler";
import { resolveTheme } from "../src/stores/settings";
const stream = (parts: string[]) =>
  new ReadableStream<Uint8Array>({
    start(c) {
      for (const part of parts) c.enqueue(new TextEncoder().encode(part));
      c.close();
    },
  });
const event = (text: string) =>
  `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: text } }] })}\n\n`;
describe("SSE framing and disconnect semantics", () => {
  it("handles every possible text split with LF and CRLF", () => {
    for (const newline of ["\n", "\r\n"]) {
      const input = [
        ": keep alive",
        "data: one",
        "",
        "data: two",
        "data: three",
        "",
        "data: [DONE]",
        "",
        "",
      ].join(newline);
      for (let split = 0; split <= input.length; split++) {
        const output: string[] = [];
        const p = new SSEParser((s) => output.push(s));
        p.push(input.slice(0, split));
        p.push(input.slice(split));
        expect(output).toEqual(["one", "two\nthree", "[DONE]"]);
      }
    }
  });
  it("handles UTF-8 characters split between network bytes", async () => {
    const bytes = new TextEncoder().encode(
      event("你好🙂") + "data: [DONE]\n\n",
    );
    let output = "";
    await readSSE(
      new ReadableStream({
        start(c) {
          for (const byte of bytes) c.enqueue(new Uint8Array([byte]));
          c.close();
        },
      }),
      new AbortController().signal,
      (e) => (output += e.choices?.[0].delta?.content || ""),
    );
    expect(output).toBe("你好🙂");
  });
  it("reports finish reason and preserves partial output on backend disconnect", async () => {
    let output = "";
    await expect(
      readSSE(
        stream([event("partial")]),
        new AbortController().signal,
        (e) => (output += e.choices?.[0].delta?.content || ""),
      ),
    ).rejects.toThrow("disconnected");
    expect(output).toBe("partial");
    let reason = "";
    await readSSE(
      stream([
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n',
      ]),
      new AbortController().signal,
      (e) => (reason = e.choices?.[0].finish_reason || ""),
    );
    expect(reason).toBe("length");
  });
  it("surfaces malformed and backend error events", async () => {
    await expect(
      readSSE(
        stream(["data: nope\n\n"]),
        new AbortController().signal,
        () => {},
      ),
    ).rejects.toThrow("Malformed");
    await expect(
      readSSE(
        stream(['data: {"error":"CUDA out of memory"}\n\n']),
        new AbortController().signal,
        () => {},
      ),
    ).rejects.toThrow("CUDA out of memory");
  });
  it("cancels a stalled stream when aborted", async () => {
    const controller = new AbortController();
    let canceled = false;
    const body = new ReadableStream<Uint8Array>({
      cancel() {
        canceled = true;
      },
    });
    const task = readSSE(body, controller.signal, () => {});
    controller.abort();
    await expect(task).rejects.toThrow();
    expect(canceled).toBe(true);
  });
});
describe("translation segmentation", () => {
  it("preserves normalized content and sentence boundaries", () => {
    const text =
      "First sentence. Second sentence!\r\n\r\n你好，世界。下一段内容！\n" +
      "A longer sentence about recurrent inference. ".repeat(8);
    const chunks = chunkText(text, 70);
    expect(chunks.join("")).toBe(text.replace(/\r\n/g, "\n").trim());
    expect(chunks.length).toBeGreaterThan(2);
    expect(chunks.every((c) => c.length <= 140)).toBe(true);
  });
  it("preserves long Unicode and punctuation-only input", () => {
    for (const text of ["🙂".repeat(500), "!?。！？\n\n", "x".repeat(1000)])
      expect(chunkText(text, 32).join("")).toBe(text.trim());
    expect(chunkText("")).toEqual([]);
  });
  it("uses exact language-name continuation prompts", () => {
    expect(translationPrompt("Hello", "English", "Chinese")).toBe(
      "English: Hello\n\nChinese:",
    );
    expect(translationPrompt("你好", "Chinese", "Custom Language")).toBe(
      "Chinese: 你好\n\nCustom Language:",
    );
  });
});
const chunks = (n: number): TranslateChunk[] =>
  Array.from({ length: n }, (_, id) => ({
    id,
    source: String(id),
    prompt: "",
    translated: "",
    status: "pending",
  }));
describe("worker scheduler", () => {
  it("enforces hard concurrency and merges out-of-order completions by ID", async () => {
    let active = 0,
      max = 0;
    const list = chunks(12);
    const completions: number[] = [];
    const scheduler = createTranslationScheduler({
      chunks: list,
      concurrency: 3,
      onProgress: () => {},
      execute: async (c) => {
        active++;
        max = Math.max(max, active);
        await new Promise((r) => setTimeout(r, (3 - (c.id % 3)) * 5));
        c.translated = String(c.id);
        completions.push(c.id);
        active--;
      },
    });
    await scheduler.run();
    expect(max).toBe(3);
    expect(completions[0]).not.toBe(0);
    expect(orderedMerge([...list].reverse())).toBe(
      list.map((c) => c.id).join("\n\n"),
    );
    expect(list.every((c) => c.status === "done")).toBe(true);
  });
  it("aborts in-flight tasks and dispatches no more after stop", async () => {
    const list = chunks(20);
    let started = 0;
    const scheduler = createTranslationScheduler({
      chunks: list,
      concurrency: 4,
      onProgress: () => {},
      execute: async (_, signal) => {
        started++;
        await new Promise<void>((_, reject) =>
          signal.addEventListener("abort", () => reject(signal.reason), {
            once: true,
          }),
        );
      },
    });
    const task = scheduler.run();
    expect(started).toBe(4);
    scheduler.stop();
    await task;
    expect(started).toBe(4);
    expect(list.every((c) => c.status === "pending")).toBe(true);
  });
  it("continues after individual errors and supports retrying only failed chunks", async () => {
    const list = chunks(6);
    await createTranslationScheduler({
      chunks: list,
      concurrency: 2,
      onProgress: () => {},
      execute: async (c) => {
        if (c.id === 2) throw new Error("HTTP 500");
        c.translated = "ok";
      },
    }).run();
    expect(list[2].status).toBe("error");
    expect(list[5].status).toBe("done");
    await createTranslationScheduler({
      chunks: list.filter((c) => c.status === "error"),
      concurrency: 1,
      onProgress: () => {},
      execute: async (c) => {
        c.translated = "retry";
      },
    }).run();
    expect(list[2].translated).toBe("retry");
    expect(list[2].status).toBe("done");
  });
  it("rejects invalid concurrency", () => {
    for (const concurrency of [0, -1, 65, 1.5, NaN])
      expect(() =>
        createTranslationScheduler({
          chunks: [],
          concurrency,
          onProgress: () => {},
          execute: async () => {},
        }),
      ).toThrow();
  });
});

describe("theme selection", () => {
  it("supports explicit dark/light and follows the system preference", () => {
    expect(resolveTheme("dark", true)).toBe("dark");
    expect(resolveTheme("light", false)).toBe("light");
    expect(resolveTheme("system", true)).toBe("light");
    expect(resolveTheme("system", false)).toBe("dark");
    expect(resolveTheme("invalid", true)).toBe("dark");
  });
});
