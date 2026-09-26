import { describe, it, expect } from "bun:test";

/**
 * These are pure-logic and server-render tests, but the persisted settings
 * store still touches browser globals. Shim the minimum Bun does not provide.
 */
Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: {
    dispatchEvent: () => true,
    addEventListener: () => {},
    removeEventListener: () => {},
    setTimeout: (handler: () => void, timeout?: number) =>
      setTimeout(handler, timeout),
    clearTimeout: (id: number) => clearTimeout(id),
    setInterval: (handler: () => void, timeout?: number) =>
      setInterval(handler, timeout),
    clearInterval: (id: number) => clearInterval(id),
  },
});
import { SSEParser, readSSE } from "../src/lib/api/sse";
import { chunkText, translationPrompt } from "../src/lib/translate/chunk";
import {
  createTranslationScheduler,
  orderedMerge,
  type TranslateChunk,
} from "../src/lib/translate/scheduler";
import { resolveTheme } from "../src/stores/settings";
import { defaultTuning } from "../src/lib/api/launcher";
import { normalizeLanguage } from "../src/lib/translate/languages";
import {
  buildHTMLPreviewDocument,
  extractHTMLDocuments,
  formatHTMLForMarkdown,
} from "../src/lib/chat/html";
import { suggestedQuantizedPath } from "../src/lib/api/launcher";
import {
  adapterFields,
  adapterIDFromFilename,
  generationBody,
} from "../src/lib/api/client";
it("derives an adapter ID for state-like file upload", () => {
  expect(adapterIDFromFilename("adapter-final.pth")).toBe("adapter-final");
  expect(adapterIDFromFilename("HTML.PTH")).toBe("HTML");
  expect(adapterIDFromFilename(".pth")).toBe("adapter");
});
it("omits unset adapter fields and preserves explicit zero scale", () => {
  expect(
    adapterFields({
      adapter_id: "",
      adapter_version: "stale",
      adapter_scale: "1",
    }),
  ).toEqual({});
  expect(adapterFields({ adapter_id: "html", adapter_scale: "" })).toEqual({
    adapter_id: "html",
  });
  expect(
    adapterFields({
      adapter_id: "html",
      adapter_version: "v1",
      adapter_scale: "0",
    }),
  ).toEqual({ adapter_id: "html", adapter_version: "v1", adapter_scale: 0 });
  expect(() =>
    adapterFields({ adapter_id: "html", adapter_scale: "NaN" }),
  ).toThrow();
  expect(
    generationBody({ adapter_id: "", adapter_scale: "", state_id: "initial" }),
  ).toEqual({ state_id: "initial" });
});
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
describe("translation line segmentation", () => {
  it("creates one task per non-empty normalized line", () => {
    const chunks = chunkText(
      " First sentence. Second sentence!\r\n\r\n你好，世界。下一段内容！\n final line ",
    );
    expect(chunks).toEqual([
      "First sentence. Second sentence!",
      "你好，世界。下一段内容！",
      "final line",
    ]);
  });
  it("keeps long lines intact and ignores blank lines", () => {
    for (const text of ["🙂".repeat(500), "!?。！？\n\n", "x".repeat(1000)])
      expect(chunkText(text).join("")).toBe(text.trim());
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
  it("normalizes removed or unknown saved language values", () => {
    expect(normalizeLanguage("Japanese", "English")).toBe("Japanese");
    expect(normalizeLanguage("Auto", "English")).toBe("English");
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
    for (const concurrency of [0, -1, 129, 1.5, NaN])
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

describe("generated HTML previews", () => {
  it("extracts fenced and complete raw HTML without matching prose", () => {
    expect(
      extractHTMLDocuments(
        "Here you go:\n```html\n<!doctype html><h1>Hello</h1>\n```",
      ),
    ).toEqual(["<!doctype html><h1>Hello</h1>"]);
    expect(extractHTMLDocuments("<html><body>raw</body></html>")).toEqual([
      "<html><body>raw</body></html>",
    ]);
    expect(extractHTMLDocuments("Use an <html> element.")).toEqual([]);
    expect(extractHTMLDocuments("```html\n<div>unfinished</div>")).toEqual([]);
  });
  it("wraps standalone HTML for stable Markdown rendering", () => {
    const raw =
      '<!DOCTYPE html>\n<html lang="zh-CN"><body><script>const value = `ok`</script></body></html>';
    const formatted = formatHTMLForMarkdown(raw);
    expect(formatted).toStartWith("```html\n<!DOCTYPE html>");
    expect(formatted).toEndWith("\n```");
    expect(formatHTMLForMarkdown("Here is `<html>`.")).toBe(
      "Here is `<html>`.",
    );
  });
  it("repairs quoted HTML with an orphan closing fence and keeps the explanation", () => {
    const malformed =
      "><!DOCTYPE html>\n" +
      '<html lang="zh-CN"><body>SVG animation</body></html>\n' +
      "```\n" +
      "这是一个完全用 SVG 手绘的鹈鹕骑自行车动画。";
    expect(extractHTMLDocuments(malformed)).toEqual([
      '<!DOCTYPE html>\n<html lang="zh-CN"><body>SVG animation</body></html>',
    ]);
    const formatted = formatHTMLForMarkdown(malformed);
    expect(formatted).toStartWith("```html\n<!DOCTYPE html>");
    expect(formatted).toContain("</html>\n```\n\n这是一个完全用 SVG");
    expect(formatted.match(/```/g)).toHaveLength(2);
  });
  it("renders generated markup in an isolated sandboxed iframe", () => {
    const preview = buildHTMLPreviewDocument(
      '<script>document.body.textContent="ok"</script>',
    );
    expect(preview).toContain(
      'sandbox="allow-scripts allow-forms allow-modals"',
    );
    expect(preview).not.toContain("allow-same-origin");
    expect(preview).toContain("&lt;script&gt;");
  });
  it("adds the preview action only to assistant HTML messages", async () => {
    const { createElement } = await import("react");
    const { renderToStaticMarkup } = await import("react-dom/server");
    const { MessageBubble } = await import(
      "../src/components/chat/message-bubble"
    );
    const { translate } = await import("../src/lib/i18n");
    const html = renderToStaticMarkup(
      createElement(MessageBubble, {
        message: {
          id: "answer",
          role: "assistant",
          content: "```html\n<!doctype html><h1>Preview</h1>\n```",
        },
      }),
    );
    expect(html).toContain(translate("zh", "chat.preview"));
  });
  it("displays unfenced assistant HTML as one highlighted code block", async () => {
    const { createElement } = await import("react");
    const { renderToStaticMarkup } = await import("react-dom/server");
    const { MessageBubble } = await import(
      "../src/components/chat/message-bubble"
    );
    const { translate } = await import("../src/lib/i18n");
    const html = renderToStaticMarkup(
      createElement(MessageBubble, {
        message: {
          id: "raw-answer",
          role: "assistant",
          content:
            '<!DOCTYPE html>\n<html lang="zh-CN"><head></head><body>ok</body></html>',
        },
      }),
    );
    expect(html).toContain('<code class="hljs language-html">');
    expect(html).toContain("&lt;!DOCTYPE");
    expect(html).toContain('hljs-keyword">html</span>&gt;');
    expect(html).not.toContain("<blockquote>");
    expect(html).toContain(translate("zh", "chat.preview"));
  });
  it("renders the malformed model response as HTML code followed by prose", async () => {
    const { createElement } = await import("react");
    const { renderToStaticMarkup } = await import("react-dom/server");
    const { MessageBubble } = await import(
      "../src/components/chat/message-bubble"
    );
    const { translate } = await import("../src/lib/i18n");
    const html = renderToStaticMarkup(
      createElement(MessageBubble, {
        message: {
          id: "quoted-html-answer",
          role: "assistant",
          content:
            "><!DOCTYPE html>\n<html><body>ok</body></html>\n```\n这是后续说明。",
        },
      }),
    );
    expect(html).toContain('<code class="hljs language-html">');
    expect(html).toContain("<p>这是后续说明。</p>");
    expect(html).not.toContain("<blockquote>");
    expect(html).toContain(translate("zh", "chat.preview"));
  });
});

it("suggests quantized model names for both formats", () => {
  expect(suggestedQuantizedPath("/models/demo.pth", "w4a16")).toBe(
    "/models/demo.w4a16.rwkvq",
  );
  expect(suggestedQuantizedPath("C:\\models\\DEMO.PTH", "w8a16")).toBe(
    "C:\\models\\DEMO.w8a16.rwkvq",
  );
});

it("uses the recommended state tuning defaults", () => {
  expect(defaultTuning).toMatchObject({
    ctx: 512,
    chunk: 128,
    epochs: 1,
    lr: 0.0005,
    lr_final: 0.0001,
    warmup_steps: 10,
    save_every: 100,
    batch_size: 16,
    optimizer: "adam",
    wkv_tape: false,
  });
});

it("shows optimizer and shared WKV tape controls", async () => {
  const { createElement } = await import("react");
  const { renderToStaticMarkup } = await import("react-dom/server");
  const { TrainingPage } = await import("../src/pages/training-page");
  const html = renderToStaticMarkup(createElement(TrainingPage));
  // The optimizer select is a wire-value select, and MiSS pins it to adam.
  expect(html).toContain('<option value="adam" selected="">adam</option>');
  expect(html).toContain('<option value="muon">muon</option>');
  expect(html).toContain("wkv_tape");
  expect(html).toContain("State tuning");
  expect(html).toContain("MiSS");
});

it("allows arbitrary positive learning-rate decimals in the tuning form", async () => {
  const { createElement } = await import("react");
  const { renderToStaticMarkup } = await import("react-dom/server");
  const { TrainingPage } = await import("../src/pages/training-page");
  const html = renderToStaticMarkup(createElement(TrainingPage));
  expect(html.match(/step="any"/g)).toHaveLength(2);
  expect(html).toContain('value="0.0005"');
  expect(html).toContain('value="0.0001"');
});

describe("GPU memory formatting", () => {
  it("spells the unit once so the rail row fits on one line", async () => {
    const { formatGigabytePair } = await import("../src/lib/format");
    const GiB = 1024 ** 3;
    // The 216px rail card leaves ~162px for this row plus the temp/power
    // pair; "61.5 GB / 95.6 GB" overflowed it and wrapped both halves.
    expect(formatGigabytePair(61.5 * GiB, 95.6 * GiB)).toBe("61.5 / 95.6 GB");
    expect(formatGigabytePair(0.6 * GiB, 95.6 * GiB)).toBe("0.6 / 95.6 GB");
  });

  it("reports an em dash when either side is missing", async () => {
    const { formatGigabytePair } = await import("../src/lib/format");
    expect(formatGigabytePair(undefined, 1024 ** 3)).toBe("—");
    expect(formatGigabytePair(1024 ** 3, null)).toBe("—");
    expect(formatGigabytePair(NaN, 1024 ** 3)).toBe("—");
  });
});

describe("GPU memory split", () => {
  it("draws ours and everyone else's along the memory axis", async () => {
    const { memorySplit } = await import("../src/components/node-status");
    const GiB = 1024 ** 3;
    const gpu = (own?: number) => ({
      index: 0,
      name: "NVIDIA RTX PRO 6000",
      memory_total_bytes: 100 * GiB,
      memory_used_bytes: 60 * GiB,
      ...(own === undefined ? {} : { own_memory_bytes: own }),
    });

    // The agent attributed the memory: both owners get their segment.
    expect(memorySplit(gpu(20 * GiB), new Set())).toEqual({
      used: 60,
      own: 20,
      foreign: 40,
    });
    // No attribution and the card is the runtime's: all of it is ours.
    expect(memorySplit(gpu(), new Set([0]))).toEqual({
      used: 60,
      own: 60,
      foreign: 0,
    });
    // No attribution and no runtime holding it: none of it is ours, which is
    // the state a box with the service stopped is always in.
    expect(memorySplit(gpu(), new Set())).toEqual({
      used: 60,
      own: 0,
      foreign: 60,
    });
    // An over-reported own figure is clamped to what the card says is used.
    expect(memorySplit(gpu(90 * GiB), new Set())).toEqual({
      used: 60,
      own: 60,
      foreign: 0,
    });
  });
});

describe("GPU model groups", () => {
  it("names every model on a mixed box, largest group first", async () => {
    const { gpuModelGroups, compactGpuName } = await import(
      "../src/components/node-status"
    );
    const gpu = (index: number, name: string) => ({
      index,
      name,
      memory_total_bytes: 1,
      memory_used_bytes: 0,
    });

    // A uniform box is one group, name intact, vendor prefix gone.
    expect(
      gpuModelGroups([
        gpu(0, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"),
        gpu(1, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"),
      ]),
    ).toEqual([
      { name: "RTX PRO 6000 Blackwell Workstation Edition", count: 2 },
    ]);

    // Mixed: one entry per model, bigger group first; ties by name, so the
    // order does not depend on which card happens to be #0.
    expect(
      gpuModelGroups([
        gpu(0, "NVIDIA RTX A6000"),
        gpu(1, "NVIDIA RTX 4090"),
        gpu(2, "NVIDIA RTX 4090"),
      ]),
    ).toEqual([
      { name: "RTX 4090", count: 2 },
      { name: "RTX A6000", count: 1 },
    ]);
    expect(
      gpuModelGroups([gpu(0, "NVIDIA RTX A6000"), gpu(1, "NVIDIA RTX 4090")]),
    ).toEqual([
      { name: "RTX 4090", count: 1 },
      { name: "RTX A6000", count: 1 },
    ]);

    // Other vendors: the prefix the driver writes is dropped and nothing
    // else is touched.
    expect(compactGpuName("AMD Radeon RX 7900 XTX")).toBe("Radeon RX 7900 XTX");
    expect(compactGpuName("Intel Arc A770")).toBe("Arc A770");
    expect(compactGpuName("")).toBe("");
    expect(gpuModelGroups([gpu(0, "")])).toEqual([{ name: "", count: 1 }]);
  });
});

describe("loaded model reporting", () => {
  it("does not present a configured path as a loaded model", async () => {
    const { modelName } = await import("../src/components/node-status");
    const config = { model_path: "/models/rwkv-g1k-7b.pth" };
    // Offline node: the mockup shows "—", not the path it was asked to load.
    expect(modelName({ status: "offline", config } as never)).toBe("");
    expect(modelName({ status: "ready", config } as never)).toBe(
      "rwkv-g1k-7b.pth",
    );
  });
});

describe("node process rows", () => {
  const rows = async (input: Record<string, unknown>) => {
    const { nodeProcessRows } = await import(
      "../src/components/node-processes"
    );
    const { translate } = await import("../src/lib/i18n");
    type Translate = typeof translate;
    const t = (
      key: Parameters<Translate>[1],
      vars?: Parameters<Translate>[2],
    ) => translate("zh", key, vars);
    return nodeProcessRows(t, {
      backend: { reachable: true } as never,
      hasAgent: true,
      busy: false,
      canStartRuntime: true,
      ...input,
    });
  };

  it("offers Start only for the process whose configuration is one click away", async () => {
    const list = await rows({
      runtime: { status: "offline" },
      jobs: {
        tuning: { running: true, progress: { step: 11, total: 630 } },
        quantization: { running: false, status: "offline" },
      },
    });
    expect(list.map((row) => row.key)).toEqual([
      "runtime",
      "tuning",
      "quantization",
    ]);
    expect(list[0].startDisabled).toBe(false);
    // Starting a multi-hour job from a menu, with a config the reader cannot
    // see, is not a command worth one stray click.
    expect(list[1].startable).toBe(false);
    expect(list[2].startable).toBe(false);
    // Stopping one, however, must never require finding the right page.
    expect(list[1].stopDisabled).toBe(false);
    expect(list[2].stopDisabled).toBe(true);
    expect(list[1].state).toBe("11 / 630");
  });

  it("reports a ready runtime and refuses a second Start", async () => {
    const list = await rows({
      runtime: {
        status: "ready",
        running: true,
        managed: true,
        backend: { model: { name: "rwkv7-g1g-1.5b" } },
      },
      jobs: undefined,
    });
    expect(list[0].tone).toBe("ok");
    expect(list[0].state).toBe("已就绪");
    expect(list[0].detail).toBe("rwkv7-g1g-1.5b");
    expect(list[0].startDisabled).toBe(true);
    expect(list[0].stopDisabled).toBe(false);
    expect(list[0].restartDisabled).toBe(false);
  });

  it("disables every command on a node that cannot be reached", async () => {
    const list = await rows({
      backend: { reachable: false } as never,
      runtime: { status: "offline" },
      jobs: { tuning: { running: true }, quantization: { running: true } },
    });
    for (const row of list) {
      expect(row.tone).toBe("bad");
      expect(row.startDisabled).toBe(true);
      expect(row.stopDisabled).toBe(true);
    }
  });
});

describe("chart axes and downsampling", () => {
  it("puts ticks on round numbers inside the range", async () => {
    const { ticksFor } = await import("../src/components/metric-chart");
    const ticks = ticksFor(0.94, 2.31);
    expect(ticks.length).toBeGreaterThan(1);
    expect(ticks.length).toBeLessThanOrEqual(6);
    expect(ticks[0]).toBeGreaterThanOrEqual(0.94);
    expect(ticks[ticks.length - 1]).toBeLessThanOrEqual(2.31);
    // Round steps, not raw range/4 fractions: 0.3425 would label as 1.2825.
    for (const tick of ticks) expect(Number(tick.toFixed(6)) % 0.25).toBe(0);
    // A learning-rate window spans four orders of magnitude and must not
    // collapse to a single tick.
    expect(ticksFor(1e-5, 5e-4).length).toBeGreaterThan(1);
    // Degenerate ranges (a metric that never moves) stay renderable.
    expect(ticksFor(2, 2)).toEqual([2]);
  });

  it("caps the drawn points but keeps both ends of the run", async () => {
    const { strided } = await import("../src/components/metric-chart");
    const run = Array.from({ length: 4000 }, (_, index) => ({
      step: index,
      value: index,
    }));
    const drawn = strided(run, 900);
    expect(drawn.length).toBeLessThanOrEqual(901);
    expect(drawn[0].step).toBe(0);
    // The newest step is what a live run is watched for; it survives striding.
    expect(drawn[drawn.length - 1].step).toBe(3999);
    // Short runs are passed through untouched.
    expect(strided(run.slice(0, 10), 900)).toHaveLength(10);
  });

  it("renders the empty hint instead of an axis when a run has no points", async () => {
    const { createElement } = await import("react");
    const { renderToStaticMarkup } = await import("react-dom/server");
    const { MetricChart } = await import("../src/components/metric-chart");
    const html = renderToStaticMarkup(
      createElement(MetricChart, {
        points: [],
        label: "loss",
        format: (value: number) => value.toFixed(3),
        emptyLabel: "还没有曲线",
      }),
    );
    expect(html).toContain("还没有曲线");
    expect(html).not.toContain("<svg");
  });
});

describe("log console", () => {
  const render = async (stream: {
    lines: string[];
    status: "idle" | "connecting" | "open" | "error";
    error: string;
  }) => {
    const { createElement } = await import("react");
    const { renderToStaticMarkup } = await import("react-dom/server");
    const { LogConsole } = await import("../src/components/log-console");
    return renderToStaticMarkup(
      createElement(LogConsole, {
        stream,
        endpoint: "/api/v1/runtime/logs",
      }),
    );
  };

  it("marks failure lines apart from ordinary output", async () => {
    const html = await render({
      lines: ["loading model", "CUDA out of memory", "Traceback (most recent)"],
      status: "open",
      error: "",
    });
    expect(html).toContain("loading model");
    // Three lines, two of which must stand out as failures.
    expect(html.split("text-destructive").length - 1).toBe(2);
    expect(html).toContain("/api/v1/runtime/logs");
  });

  it("names the connection state and surfaces the transport error", async () => {
    const { translate } = await import("../src/lib/i18n");
    const open = await render({ lines: ["ready"], status: "open", error: "" });
    expect(open).toContain(translate("zh", "logs.statusOpen"));
    expect(open).toContain(translate("zh", "logs.lineCount", { count: 1 }));

    // A dead stream must say so rather than looking like a quiet one.
    const broken = await render({
      lines: [],
      status: "error",
      error: "dial tcp 127.0.0.1:8000: connection refused",
    });
    expect(broken).toContain(translate("zh", "logs.statusError"));
    expect(broken).toContain("connection refused");
    expect(broken).toContain(translate("zh", "runtime.logsEmpty"));
  });
});

describe("chat session list", () => {
  const session = (title: string, id = title) => ({
    id,
    backendId: "local",
    title,
    messages: [],
    createdAt: 0,
    updatedAt: 0,
  });

  it("searches over the placeholder an unnamed conversation shows", async () => {
    const { filterSessions, titleOf } = await import(
      "../src/components/chat/session-picker"
    );
    const sessions = [session("训练脚本怎么写"), session("", "blank")];
    expect(titleOf(sessions[1], "未命名会话")).toBe("未命名会话");
    expect(filterSessions(sessions, "  ", "未命名会话")).toHaveLength(2);
    expect(filterSessions(sessions, "训练", "未命名会话")).toHaveLength(1);
    expect(filterSessions(sessions, "未命名", "未命名会话")[0].id).toBe(
      "blank",
    );
    expect(filterSessions(sessions, "nothing", "未命名会话")).toHaveLength(0);
  });

  it("offers the history button even before a node has any conversation", async () => {
    const { createElement } = await import("react");
    const { renderToStaticMarkup } = await import("react-dom/server");
    const { SessionPicker } = await import(
      "../src/components/chat/session-picker"
    );
    const { translate } = await import("../src/lib/i18n");
    const html = renderToStaticMarkup(
      createElement(SessionPicker, { backendId: "local" }),
    );
    expect(html).toContain(translate("zh", "chat.conversations"));
    // Live button, not a dead one: history is reachable from the first visit.
    // (`disabled:` also appears inside the class list, hence the attribute.)
    expect(html).not.toContain('disabled=""');
  });
});

describe("training curve smoothing", () => {
  it("debiases the EMA so a run does not appear to start where it did not", async () => {
    const { smoothSeries } = await import("../src/components/metric-chart");
    const flat = Array.from({ length: 20 }, (_, index) => ({
      step: index,
      value: 4,
    }));
    const smoothed = smoothSeries(flat, 0.9);
    // A plain EMA would start near 0.4 on this series and climb for dozens of
    // points; the debiased one sits on the data from the first step.
    expect(smoothed[0].value).toBeCloseTo(4, 6);
    expect(smoothed[smoothed.length - 1].value).toBeCloseTo(4, 6);
  });

  it("follows the trend without reproducing every spike", async () => {
    const { smoothSeries } = await import("../src/components/metric-chart");
    const noisy = Array.from({ length: 50 }, (_, index) => ({
      step: index,
      value: index % 2 === 0 ? 1 : 3,
    }));
    const smoothed = smoothSeries(noisy, 0.8);
    const last = smoothed[smoothed.length - 1].value;
    expect(last).toBeGreaterThan(1.6);
    expect(last).toBeLessThan(2.4);
    // Smoothing 0 is the identity, so the raw curve stays available.
    expect(smoothSeries(noisy, 0)).toBe(noisy);
  });
});

describe("runtime status wording", () => {
  it("names the subject so a stopped runtime does not read as a dead node", async () => {
    const { runtimeLabel, runtimeStateLabel, runtimeHint } = await import(
      "../src/components/node-status"
    );
    const { translate } = await import("../src/lib/i18n");
    type Translate = typeof translate;
    const t = (
      key: Parameters<Translate>[1],
      vars?: Parameters<Translate>[2],
    ) => translate("zh", key, vars);

    // A reachable node with nothing serving: the chip has to say *what* is
    // not started, and the tooltip has to say the node itself is fine.
    expect(runtimeLabel(t, { status: "offline" } as never)).toBe(
      "推理服务未启动",
    );
    expect(runtimeLabel(t, undefined)).toBe("推理服务未启动");
    expect(runtimeHint(t, { status: "offline" } as never)).toBeTruthy();
    expect(runtimeHint(t, { status: "ready" } as never)).toBeUndefined();

    // Rows that already carry a "推理服务" label take the bare state word.
    expect(runtimeStateLabel(t, { status: "offline" } as never)).toBe("未启动");
    expect(runtimeStateLabel(t, { status: "ready" } as never)).toBe("已就绪");
    expect(
      runtimeStateLabel(t, { status: "ready", managed: false } as never),
    ).toBe("已就绪 · 外部进程");
  });
});

describe("runtime control gating", () => {
  const base = {
    reachableAgent: true,
    busy: false,
    managed: false,
    running: false,
    canStart: true,
  };

  it("keeps Stop live while a managed runtime is still loading its model", async () => {
    const { runtimeButtonState } = await import(
      "../src/components/runtime-controls"
    );
    // A cold load sits at status "starting" — managed, not yet running — for
    // as long as the model takes. Gating Stop on `running` made that window
    // uninterruptible and left Start enabled, where it answered 400
    // "backend is already running".
    const loading = runtimeButtonState({ ...base, managed: true });
    expect(loading.stopDisabled).toBe(false);
    expect(loading.restartDisabled).toBe(false);
    expect(loading.startDisabled).toBe(true);
  });

  it("offers Start on an idle node and not on a serving one", async () => {
    const { runtimeButtonState } = await import(
      "../src/components/runtime-controls"
    );
    const offline = runtimeButtonState(base);
    expect(offline.startDisabled).toBe(false);
    expect(offline.stopDisabled).toBe(true);

    const ready = runtimeButtonState({ ...base, managed: true, running: true });
    expect(ready.startDisabled).toBe(true);
    expect(ready.stopDisabled).toBe(false);
  });

  it("never offers Start or Stop for a runtime this launcher does not own", async () => {
    const { runtimeButtonState } = await import(
      "../src/components/runtime-controls"
    );
    // Started by hand or by another launcher: ours to watch, not to control,
    // and starting again would fight over the port.
    const external = runtimeButtonState({ ...base, running: true });
    expect(external.startDisabled).toBe(true);
    expect(external.stopDisabled).toBe(true);
    expect(external.restartDisabled).toBe(true);
  });

  it("blocks everything on an unreachable node, a busy node or a blank model path", async () => {
    const { runtimeButtonState } = await import(
      "../src/components/runtime-controls"
    );
    for (const patch of [{ reachableAgent: false }, { busy: true }]) {
      const state = runtimeButtonState({ ...base, managed: true, ...patch });
      expect(state.startDisabled).toBe(true);
      expect(state.stopDisabled).toBe(true);
      expect(state.restartDisabled).toBe(true);
    }
    // A blank model path stops Start only; a running process stays stoppable.
    const noPath = runtimeButtonState({
      ...base,
      managed: true,
      canStart: false,
    });
    expect(noPath.startDisabled).toBe(true);
    expect(noPath.stopDisabled).toBe(false);
  });
});
