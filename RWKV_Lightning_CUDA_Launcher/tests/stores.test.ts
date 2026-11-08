import { afterAll, afterEach, describe, expect, it, spyOn } from "bun:test";

/* ------------------------------------------------------------------ *
 * Browser shims. Zustand's persist middleware and the hash router both
 * read globals that do not exist in the Bun runtime.
 * ------------------------------------------------------------------ */
const values = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
  },
});
Object.defineProperty(globalThis, "location", {
  configurable: true,
  value: { origin: "http://127.0.0.1:10721", hash: "#/nodes" },
});
// The stores schedule through `window.*` timers; Bun only exposes the bare
// globals.
Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: {
    setTimeout: (handler: () => void, timeout?: number) =>
      setTimeout(handler, timeout),
    clearTimeout: (id: number) => clearTimeout(id),
    setInterval: (handler: () => void, timeout?: number) =>
      setInterval(handler, timeout),
    clearInterval: (id: number) => clearInterval(id),
    dispatchEvent: () => true,
    addEventListener: () => {},
    removeEventListener: () => {},
  },
});

const { useBackends } = await import("../src/stores/backends");
const { useNodes } = await import("../src/stores/nodes");
const { useChat, currentThread } = await import("../src/stores/chat");
const { useTranslate } = await import("../src/stores/translate");
const { useSettings, useSecret } = await import("../src/stores/settings");
const {
  useRuntimeForm,
  useQuantizationForm,
  useTuningForm,
  EMPTY_RUNTIME_FORM,
} = await import("../src/stores/forms");
const { applyDevice, deviceSelection, defaultRuntime } = await import(
  "../src/lib/api/launcher"
);

const fetchTarget = globalThis as unknown as {
  fetch: (...args: Parameters<typeof fetch>) => ReturnType<typeof fetch>;
};
const fetchSpy = spyOn(fetchTarget, "fetch");
afterAll(() => fetchSpy.mockRestore());

const sseStream = (text: string, done = true) =>
  new Response(
    `data: ${JSON.stringify({
      choices: [{ index: 0, delta: { content: text } }],
    })}\n\n${done ? "data: [DONE]\n\n" : ""}`,
    { headers: { "Content-Type": "text/event-stream" } },
  );

const batchResponse = (texts: string[]) =>
  Response.json({
    choices: texts.map((text, index) => ({
      index,
      message: { role: "assistant", content: text },
      finish_reason: "stop",
    })),
  });

const json = (body: unknown) => Response.json(body);

/** Requests captured by the current mock, in call order. */
interface Call {
  url: string;
  method: string;
  body: Record<string, unknown> | null;
}

function capture(): Call[] {
  const calls: Call[] = [];
  fetchSpy.mockImplementation(async (input, init) => {
    const url = String(input);
    const raw = init?.body;
    const parsed =
      typeof raw === "string"
        ? (JSON.parse(raw) as Record<string, unknown>)
        : null;
    calls.push({
      url,
      method: init?.method ?? "GET",
      body: parsed,
    });
    if (url.includes("/v1/chat/completions")) return sseStream("answer");
    if (url.includes("/v1/batch/completions")) {
      // One choice per prompt, indexed to match `contents[]`.
      const contents = (parsed?.contents as string[] | undefined) ?? [];
      return batchResponse(contents.map((_, index) => `译文 ${index + 1}`));
    }
    if (url.endsWith("/api/v1/backends")) return json({ backends: [] });
    return json({});
  });
  return calls;
}

afterEach(() => {
  useChat.getState().clearAll();
  useTranslate.getState().clear();
  useSettings.getState().reset();
  useSecret.getState().setKey("");
  useRuntimeForm.getState().resetAll();
  useQuantizationForm.getState().resetAll();
  useTuningForm.getState().resetAll();
  useBackends.setState({ list: [], currentId: "", error: "", loaded: false });
  useNodes.setState({ snapshots: {}, busy: {} });
  values.clear();
  fetchSpy.mockReset();
});

const backend = {
  id: "a1b2c3",
  name: "gpu-01",
  base_url: "http://10.0.0.21:18766",
  has_token: true,
  kind: "agent" as const,
  capabilities: ["runtime", "tuning_state", "quantization", "metrics", "fs"],
  reachable: true,
  last_probe: 1_758_240_000,
  probe_error: "",
};

describe("backend registry", () => {
  it("lists nodes, keeps the selection and falls back to the first entry", async () => {
    useBackends.setState({ currentId: "gone" });
    fetchSpy.mockImplementation(async () =>
      json({ backends: [backend, { ...backend, id: "local", name: "本机" }] }),
    );
    await useBackends.getState().refresh();
    expect(fetchSpy.mock.calls[0][0]).toBe("/api/v1/backends");
    expect(useBackends.getState().list).toHaveLength(2);
    // The previously selected id no longer exists.
    expect(useBackends.getState().currentId).toBe("a1b2c3");
  });

  it("keeps a still-valid selection across refreshes", async () => {
    fetchSpy.mockImplementation(async () => json({ backends: [backend] }));
    await useBackends.getState().refresh();
    useBackends.getState().select("a1b2c3");
    await useBackends.getState().refresh();
    expect(useBackends.getState().currentId).toBe("a1b2c3");
  });

  it("adds a backend through the Client registry endpoint", async () => {
    fetchSpy.mockImplementation(async () => json(backend));
    const view = await useBackends.getState().add({
      name: "gpu-01",
      base_url: "http://10.0.0.21:18766",
      token: "secret",
    });
    expect(fetchSpy.mock.calls[0][0]).toBe("/api/v1/backends");
    expect(JSON.parse(String(fetchSpy.mock.calls[0][1]?.body))).toEqual({
      name: "gpu-01",
      base_url: "http://10.0.0.21:18766",
      token: "secret",
    });
    expect(view.id).toBe("a1b2c3");
    expect(useBackends.getState().currentId).toBe("a1b2c3");
  });
});

describe("node snapshots", () => {
  it("polls the node through the backend prefix and keeps jobs alongside", async () => {
    useBackends.setState({ list: [backend], currentId: backend.id });
    fetchSpy.mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/api/v1/node"))
        return json({
          status: "ready",
          running: true,
          error: "",
          logs: [],
          elapsed: 12,
          checkpoint: "",
          progress: null,
          losses: [],
          config: defaultRuntime,
          base_url: "http://127.0.0.1:8000",
          translation_adapter: false,
          available: true,
          visible_devices: "0,1",
          role: "agent",
          version: "dev",
          capabilities: ["runtime", "metrics"],
        });
      if (url.endsWith("/api/v1/jobs"))
        return json({
          jobs: {
            tuning: {
              status: "running",
              running: true,
              error: "",
              logs: [],
              elapsed: 3,
              checkpoint: "",
              progress: { step: 5, total: 10 },
              losses: [],
            },
            quantization: {
              status: "offline",
              running: false,
              error: "",
              logs: [],
              elapsed: 0,
              checkpoint: "",
              progress: null,
              losses: [],
            },
          },
        });
      return json({});
    });

    await useNodes.getState().refresh(backend.id);
    const urls = fetchSpy.mock.calls.map((call) => String(call[0]));
    expect(urls).toEqual([
      "/api/v1/backends/a1b2c3/api/v1/node",
      "/api/v1/backends/a1b2c3/api/v1/jobs",
    ]);
    const snapshot = useNodes.getState().snapshots[backend.id];
    expect(snapshot.runtime?.status).toBe("ready");
    expect(snapshot.runtime?.visible_devices).toBe("0,1");
    expect(snapshot.jobs?.tuning.running).toBe(true);
    expect(snapshot.error).toBe("");
  });

  it("does not call the Agent API for an inference-only node", async () => {
    useBackends.setState({
      list: [
        { ...backend, kind: "inference_only", capabilities: ["inference"] },
      ],
      currentId: backend.id,
    });
    fetchSpy.mockImplementation(async () => json({}));
    await useNodes.getState().refresh(backend.id);
    expect(fetchSpy.mock.calls).toHaveLength(0);
  });

  it("records a probe/refresh failure instead of throwing", async () => {
    useBackends.setState({ list: [backend], currentId: backend.id });
    fetchSpy.mockImplementation(
      async () =>
        new Response(JSON.stringify({ error: "dial tcp: refused" }), {
          status: 502,
        }),
    );
    await useNodes.getState().refresh(backend.id);
    expect(useNodes.getState().snapshots[backend.id].error).toContain(
      "dial tcp: refused",
    );
  });
});

describe("runtime control", () => {
  it("starts the runtime with the full config plus the device selection", async () => {
    useBackends.setState({ list: [backend], currentId: backend.id });
    const calls = capture();
    useRuntimeForm
      .getState()
      .set(backend.id, { model_path: "/data/models/m.pth" });
    useRuntimeForm
      .getState()
      .setDevices(backend.id, { mode: "explicit", value: "0,1" });
    await useNodes
      .getState()
      .startRuntime(
        backend.id,
        applyDevice(
          { ...useRuntimeForm.getState().byBackend[backend.id].config },
          { mode: "explicit", value: "0,1" },
        ),
      );
    const start = calls.find((call) => call.url.endsWith("/runtime/start"));
    expect(start?.url).toBe("/api/v1/backends/a1b2c3/api/v1/runtime/start");
    expect(start?.method).toBe("POST");
    expect(start?.body?.model_path).toBe("/data/models/m.pth");
    expect(start?.body?.visible_devices).toBe("0,1");
    // The full config is always submitted: the Agent decodes onto saved state.
    expect(start?.body).toHaveProperty("vocab_path");
    expect(start?.body).toHaveProperty("chunk_size");
    expect(start?.body).toHaveProperty("tune_cache");
  });

  it("omits visible_devices when the selection is 'inherit'", () => {
    const body = applyDevice({ ...defaultRuntime }, { mode: "inherit" });
    expect("visible_devices" in body).toBe(false);
  });

  it("distinguishes the three device-selection states", () => {
    expect(
      applyDevice({ ...defaultRuntime }, { mode: "none" }).visible_devices,
    ).toBe("");
    expect(
      applyDevice({ ...defaultRuntime }, { mode: "explicit", value: " 0 " })
        .visible_devices,
    ).toBe("0");
    expect(deviceSelection(undefined)).toEqual({ mode: "inherit" });
    expect(deviceSelection("")).toEqual({ mode: "none" });
    expect(deviceSelection("0,1")).toEqual({
      mode: "explicit",
      value: "0,1",
    });
  });

  it("accepts an empty body when stopping a job", async () => {
    useBackends.setState({ list: [backend], currentId: backend.id });
    fetchSpy.mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/jobs/tuning/stop"))
        return new Response(null, { status: 200 });
      return json({});
    });
    await useNodes.getState().stopJob(backend.id, "tuning");
    const stopCall = fetchSpy.mock.calls.find((call) =>
      String(call[0]).endsWith("/jobs/tuning/stop"),
    );
    expect(stopCall).toBeDefined();
    expect(String(stopCall?.[0])).toBe(
      "/api/v1/backends/a1b2c3/api/v1/jobs/tuning/stop",
    );
  });
});

describe("chat streaming", () => {
  it("streams through the backend prefix and persists the answer", async () => {
    const calls = capture();
    await useChat.getState().send(backend.id, "Say hello");
    const call = calls.find((entry) =>
      entry.url.includes("/v1/chat/completions"),
    );
    expect(call?.url).toBe("/api/v1/backends/a1b2c3/v1/chat/completions");
    expect(call?.body?.stream).toBe(true);
    expect(call?.body?.stop_tokens).toEqual([0, 261, 24281]);
    expect(call?.body?.messages).toEqual([
      { role: "user", content: "Say hello" },
    ]);
    const thread = currentThread(backend.id);
    expect(thread).toHaveLength(2);
    expect(thread[1].content).toBe("answer");
    expect(useChat.getState().active).toBeNull();
    expect(values.get("rwkv-chat-v3")).toContain("answer");
  });

  it("sends the selected state, thinking mode and MiSS adapter", async () => {
    const calls = capture();
    useSettings.getState().setGeneration({
      state_id: "roleplay.pth",
      think_type: "free",
      adapter_id: "zh-legal",
      adapter_version: "v1",
      adapter_scale: "0",
    });
    await useChat.getState().send(backend.id, "hello");
    const body = calls.find((entry) =>
      entry.url.includes("/v1/chat/completions"),
    )?.body;
    expect(body?.state_id).toBe("roleplay.pth");
    expect(body?.think_type).toBe("free");
    expect(body?.adapter_id).toBe("zh-legal");
    expect(body?.adapter_version).toBe("v1");
    // An explicit "0" scale must survive as the number 0, not be dropped.
    expect(body?.adapter_scale).toBe(0);
  });

  it("regenerates without duplicating the user message", async () => {
    capture();
    await useChat.getState().send(backend.id, "Say hello");
    await useChat.getState().send(backend.id, "", true);
    const thread = currentThread(backend.id);
    expect(thread).toHaveLength(2);
    expect(
      thread.filter((message: { role: string }) => message.role === "user"),
    ).toHaveLength(1);
  });

  it("keeps partial output and reports a stream that ends before [DONE]", async () => {
    fetchSpy.mockImplementation(async () => sseStream("partial", false));
    await useChat.getState().send(backend.id, "hi");
    const reply = currentThread(backend.id)[1];
    expect(reply.content).toBe("partial");
    expect(reply.error).toContain("disconnected");
  });

  it("keeps one thread per backend", async () => {
    capture();
    await useChat.getState().send("a1b2c3", "first");
    await useChat.getState().send("local", "second");
    expect(currentThread("a1b2c3")[0].content).toBe("first");
    expect(currentThread("local")[0].content).toBe("second");
  });
});

describe("log dock", () => {
  it("opens on the stream a page asks for and clamps a dragged height", async () => {
    const { useLogDock, MIN_DOCK_HEIGHT, MAX_DOCK_HEIGHT } = await import(
      "../src/stores/ui"
    );
    useLogDock.getState().show("tuning");
    expect(useLogDock.getState().open).toBe(true);
    expect(useLogDock.getState().kind).toBe("tuning");

    // show() without an argument keeps whichever stream was last read.
    useLogDock.getState().hide();
    useLogDock.getState().show();
    expect(useLogDock.getState().kind).toBe("tuning");

    useLogDock.getState().setHeight(10);
    expect(useLogDock.getState().height).toBe(MIN_DOCK_HEIGHT);
    useLogDock.getState().setHeight(10_000);
    expect(useLogDock.getState().height).toBe(MAX_DOCK_HEIGHT);
    useLogDock.getState().hide();
  });
});

describe("chat history", () => {
  it("keeps earlier conversations when a new one starts", async () => {
    capture();
    await useChat.getState().send(backend.id, "first question");
    const first = useChat.getState().currentId[backend.id];

    useChat.getState().newSession(backend.id);
    expect(currentThread(backend.id)).toHaveLength(0);
    await useChat.getState().send(backend.id, "second question");

    const sessions = useChat
      .getState()
      .sessions.filter((session) => session.backendId === backend.id);
    expect(sessions).toHaveLength(2);
    // The opening message names the conversation, so the list is readable.
    expect(sessions.find((session) => session.id === first)?.title).toBe(
      "first question",
    );
    expect(values.get("rwkv-chat-v3")).toContain("first question");

    // Switching back serves the old transcript, not the new one.
    useChat.getState().selectSession(backend.id, first);
    expect(currentThread(backend.id)[0].content).toBe("first question");
  });

  it("does not stack blank sessions and re-points the selection on delete", async () => {
    capture();
    useChat.getState().newSession(backend.id);
    useChat.getState().newSession(backend.id);
    expect(useChat.getState().sessions).toHaveLength(1);

    await useChat.getState().send(backend.id, "kept");
    const kept = useChat.getState().currentId[backend.id];
    const extra = useChat.getState().newSession(backend.id);
    useChat.getState().deleteSession(extra);
    expect(useChat.getState().currentId[backend.id]).toBe(kept);
  });

  it("carries a v2 thread over as that node's first saved conversation", async () => {
    const { useChat: fresh } = await import("../src/stores/chat");
    const migrate = fresh.persist.getOptions().migrate;
    const migrated = migrate?.(
      { threads: { a1b2c3: [{ id: "m1", role: "user", content: "legacy" }] } },
      2,
    ) as { sessions: { backendId: string; title: string }[] };
    expect(migrated.sessions).toHaveLength(1);
    expect(migrated.sessions[0].backendId).toBe("a1b2c3");
    expect(migrated.sessions[0].title).toBe("legacy");
  });
});

describe("parallel translation", () => {
  it("sends one batch request per group of lines to /v1/batch/completions", async () => {
    const calls = capture();
    useTranslate.getState().set({
      source: "one\ntwo\nthree",
      concurrency: 3,
      from: "English",
      to: "Chinese",
    });
    await useTranslate.getState().run(backend.id);
    const batches = calls.filter((entry) =>
      entry.url.includes("/v1/batch/completions"),
    );
    expect(batches).toHaveLength(1);
    expect(batches[0].url).toBe("/api/v1/backends/a1b2c3/v1/batch/completions");
    expect(batches[0].body?.stream).toBe(false);
    expect(batches[0].body?.stop_tokens).toEqual([0]);
    expect(batches[0].body?.contents).toEqual([
      "English: one\n\nChinese:",
      "English: two\n\nChinese:",
      "English: three\n\nChinese:",
    ]);
    const chunks = useTranslate.getState().chunks;
    expect(chunks).toHaveLength(3);
    expect(chunks.every((chunk) => chunk.status === "done")).toBe(true);
    expect(useTranslate.getState().owner).toBe(backend.id);
  });

  it("caps each native batch at 128 lines", async () => {
    const calls = capture();
    fetchSpy.mockImplementation(async (input, init) => {
      const url = String(input);
      const parsed = JSON.parse(String(init?.body)) as {
        contents: string[];
      };
      calls.push({ url, method: "POST", body: parsed as never });
      return batchResponse(parsed.contents.map(() => "x"));
    });
    const lines = Array.from({ length: 130 }, (_, index) => `line ${index}`);
    useTranslate.getState().set({ source: lines.join("\n"), concurrency: 128 });
    await useTranslate.getState().run(backend.id);
    const sizes = calls
      .filter((entry) => entry.url.includes("/v1/batch/completions"))
      .map((entry) => (entry.body?.contents as string[]).length);
    expect(sizes).toEqual([128, 2]);
  });

  it("keeps results isolated per backend", async () => {
    capture();
    useTranslate.getState().set({ source: "one", concurrency: 1 });
    await useTranslate.getState().run("a1b2c3");
    expect(useTranslate.getState().owner).toBe("a1b2c3");
    const before = fetchSpy.mock.calls.length;
    // Retrying against a different node must not reuse the old results.
    await useTranslate.getState().run("local", [0]);
    expect(fetchSpy.mock.calls.length).toBe(before);
  });
});

describe("theme default", () => {
  it("follows the operating system unless the user picked a theme", async () => {
    const { defaults, resolveTheme, useSettings } = await import(
      "../src/stores/settings"
    );
    expect(defaults.theme).toBe("system");
    expect(resolveTheme("system", true)).toBe("light");
    expect(resolveTheme("system", false)).toBe("dark");
    expect(resolveTheme("dark", true)).toBe("dark");
    expect(resolveTheme("light", false)).toBe("light");
    expect(useSettings.getState().theme).toBe("system");
  });

  it("migrates the old hard-coded dark default to system", async () => {
    const { useSettings } = await import("../src/stores/settings");
    const migrate = useSettings.persist.getOptions().migrate;
    expect(migrate).toBeFunction();
    // v2 stored a flat object; "dark" was the default rather than a choice.
    const fromV2 = migrate!({ theme: "dark", lang: "en" }, 2) as {
      theme: string;
      lang: string;
    };
    expect(fromV2.theme).toBe("system");
    expect(fromV2.lang).toBe("en");
    // An explicit light choice must survive.
    const light = migrate!({ theme: "light" }, 2) as { theme: string };
    expect(light.theme).toBe("light");
    // v1 nested everything under `values`.
    const fromV1 = migrate!({ values: { theme: "dark" } }, 1) as {
      theme: string;
    };
    expect(fromV1.theme).toBe("system");
    // Already-current data is returned untouched.
    const current = migrate!({ theme: "light" }, 3) as { theme: string };
    expect(current.theme).toBe("light");
  });
});

describe("log stream snapshots", () => {
  it("returns one stable reference for every missing stream", async () => {
    const { IDLE_STREAM, selectLogStream, useLogs } = await import(
      "../src/stores/logs"
    );
    // An unstable fallback would make useSyncExternalStore re-render forever,
    // which unmounts the Runtime / Training / Quantization pages.
    const streams = useLogs.getState().streams;
    expect(selectLogStream(streams, "a1b2c3", "runtime")).toBe(IDLE_STREAM);
    expect(selectLogStream(streams, "a1b2c3", "runtime")).toBe(
      selectLogStream(streams, "local", "tuning"),
    );
    // A real stream still wins over the fallback.
    expect(
      selectLogStream(
        { "a1b2c3:runtime": { lines: ["x"], status: "open", error: "" } },
        "a1b2c3",
        "runtime",
      ).lines,
    ).toEqual(["x"]);
  });
});

describe("hash routing", () => {
  it("resolves known routes and aliases, and refuses inherited properties", async () => {
    const { parseRoute } = await import("../src/lib/router");
    expect(parseRoute("#/chat")).toBe("chat");
    expect(parseRoute("#/nodes")).toBe("nodes");
    expect(parseRoute("")).toBe("nodes");
    // Deep links kept from the previous console.
    expect(parseRoute("#/state-tuning")).toBe("training");
    expect(parseRoute("#/quantization")).toBe("quant");
    expect(parseRoute("#/chat?x=1")).toBe("chat");
    expect(parseRoute("#/nope")).toBe("nodes");
    // `raw in ALIASES` walked the prototype chain and handed back a function,
    // which blanked the header title and rendered an empty main pane.
    for (const key of ["toString", "constructor", "__proto__", "valueOf"]) {
      expect(parseRoute(`#/${key}`)).toBe("nodes");
    }
  });
});

describe("backend labels", () => {
  it("names the local node from the catalogue, not from the wire", async () => {
    const { backendLabel } = await import("../src/stores/backends");
    const { translate } = await import("../src/lib/i18n");
    const zh = (key: Parameters<typeof translate>[1]) => translate("zh", key);
    const en = (key: Parameters<typeof translate>[1]) => translate("en", key);

    // The Agent sends no name for the local node precisely so this is
    // localizable rather than hard-coded in Go.
    expect(backendLabel(zh, { id: "local", name: "" })).toBe("本机");
    expect(backendLabel(en, { id: "local", name: "" })).toBe("This machine");
    // Registered backends keep the name their user gave them, in any locale.
    expect(backendLabel(en, { id: "a1b2c3", name: "gpu-01" })).toBe("gpu-01");
    expect(backendLabel(zh, { id: "a1b2c3", name: "gpu-01" })).toBe("gpu-01");
    // An older Agent that still sends one wins over the catalogue.
    expect(backendLabel(en, { id: "local", name: "本机" })).toBe("本机");
  });
});

describe("log stream lifecycle", () => {
  it("never lets a finished task unregister a newer stream", async () => {
    const { useLogs } = await import("../src/stores/logs");
    // Hold the stream open until we say so, the way a live SSE connection
    // does, so close() and open() can race the way they do in StrictMode.
    let release: (() => void) | undefined;
    fetchSpy.mockImplementation(
      async () =>
        new Response(
          new ReadableStream({
            start(controller) {
              release = () => controller.close();
            },
          }),
          { status: 200, headers: { "Content-Type": "text/event-stream" } },
        ),
    );

    useLogs.getState().open("a1b2c3", "runtime");
    await Promise.resolve();
    // React's mount -> unmount -> mount: close and reopen within one tick.
    useLogs.getState().close("a1b2c3", "runtime");
    useLogs.getState().open("a1b2c3", "runtime");
    const openCalls = fetchSpy.mock.calls.length;
    // Let the aborted first task run its finally block.
    await new Promise((resolve) => setTimeout(resolve, 5));

    // If the first task deleted the second task's controller by key, this
    // open() would start a *third* concurrent stream onto the same buffer.
    useLogs.getState().open("a1b2c3", "runtime");
    expect(fetchSpy.mock.calls.length).toBe(openCalls);

    // And close() must still be able to abort the live stream — after which
    // the underlying controller is already closed, hence the guard.
    useLogs.getState().close("a1b2c3", "runtime");
    try {
      release?.();
    } catch {
      /* already closed by the abort, which is the point */
    }
  });
});

describe("credentials", () => {
  it("never writes the runtime password or the Agent token to storage", () => {
    useRuntimeForm.getState().set("n1", { password: "runtime-secret" });
    useSecret.getState().setKey("agent-secret");
    const dump = [...values.values()].join("\n");
    expect(dump).not.toContain("runtime-secret");
    expect(dump).not.toContain("agent-secret");
    // The form itself still holds the password for this session.
    expect(useRuntimeForm.getState().byBackend["n1"].config.password).toBe(
      "runtime-secret",
    );
  });

  it("does not persist the quantization form across resets", () => {
    useQuantizationForm.getState().set("n1", { input_path: "/data/model.pth" });
    expect(
      useQuantizationForm.getState().byBackend["n1"].config.input_path,
    ).toBe("/data/model.pth");
    useQuantizationForm.getState().reset("n1");
    expect(useQuantizationForm.getState().byBackend["n1"]).toBeUndefined();
  });
});

describe("per-backend form isolation", () => {
  it("never carries one node's paths or card selection to another", () => {
    const eightGpu = "node-8gpu";
    const oneGpu = "node-1gpu";

    useRuntimeForm.getState().set(eightGpu, { model_path: "/srv/a/m.pth" });
    useRuntimeForm
      .getState()
      .setDevices(eightGpu, { mode: "explicit", value: "0,5" });

    // The one-GPU node has never been configured: it must read as untouched,
    // not inherit "0,5" from the box next to it.
    const fresh =
      useRuntimeForm.getState().byBackend[oneGpu] ?? EMPTY_RUNTIME_FORM;
    expect(fresh.config.model_path).toBe("");
    expect(fresh.devices).toEqual({ mode: "inherit" });

    // Editing one node leaves the other alone.
    useRuntimeForm.getState().set(oneGpu, { model_path: "/srv/b/m.pth" });
    expect(
      useRuntimeForm.getState().byBackend[eightGpu].config.model_path,
    ).toBe("/srv/a/m.pth");
    expect(useRuntimeForm.getState().byBackend[eightGpu].devices).toEqual({
      mode: "explicit",
      value: "0,5",
    });
  });

  it("returns one stable object for an unconfigured node", () => {
    // A fresh object per read would spin useSyncExternalStore forever.
    const a =
      useRuntimeForm.getState().byBackend["ghost"] ?? EMPTY_RUNTIME_FORM;
    const b =
      useRuntimeForm.getState().byBackend["ghost"] ?? EMPTY_RUNTIME_FORM;
    expect(a).toBe(b);
  });

  it("seeds a node once and then leaves the user's edits alone", () => {
    const id = "node-seed";
    const seeded = {
      config: { ...defaultRuntime, model_path: "/srv/running.pth" },
      devices: { mode: "explicit" as const, value: "2" },
    };
    useRuntimeForm.getState().seed(id, seeded);
    expect(useRuntimeForm.getState().byBackend[id].config.model_path).toBe(
      "/srv/running.pth",
    );

    useRuntimeForm.getState().set(id, { model_path: "/srv/typed.pth" });
    // A later poll must not stomp what the user typed.
    useRuntimeForm.getState().seed(id, seeded);
    expect(useRuntimeForm.getState().byBackend[id].config.model_path).toBe(
      "/srv/typed.pth",
    );
  });

  it("keeps each node's runtime password out of storage", () => {
    useRuntimeForm.getState().set("n-a", { password: "secret-a" });
    useRuntimeForm.getState().set("n-b", { password: "secret-b" });
    const dump = [...values.values()].join("\n");
    expect(dump).not.toContain("secret-a");
    expect(dump).not.toContain("secret-b");
  });
});
