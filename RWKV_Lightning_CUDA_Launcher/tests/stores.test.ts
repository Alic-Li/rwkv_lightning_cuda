import { afterAll, afterEach, expect, it, spyOn } from "bun:test";
const values = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    getItem: (k: string) => values.get(k) || null,
    setItem: (k: string, v: string) => values.set(k, v),
    removeItem: (k: string) => values.delete(k),
  },
});
Object.defineProperty(globalThis, "location", {
  configurable: true,
  value: { origin: "http://127.0.0.1:8088" },
});
const { useChat } = await import("../src/stores/chat");
const { useTranslate } = await import("../src/stores/translate");
const { useSettings, useSecret } = await import("../src/stores/settings");
const { useRuntimeForm } = await import("../src/stores/runtime");
const fetchTarget = globalThis as unknown as {
  fetch: (...args: Parameters<typeof fetch>) => ReturnType<typeof fetch>;
};
const fetchSpy = spyOn(fetchTarget, "fetch");
afterAll(() => fetchSpy.mockRestore());
const output = (text: string, done = true) =>
  new Response(
    `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: text } }] })}\n\n${done ? "data: [DONE]\n\n" : ""}`,
    { headers: { "Content-Type": "text/event-stream" } },
  );
const completion = (texts: string[]) =>
  Response.json({
    choices: texts.map((text, index) => ({
      index,
      message: { role: "assistant", content: text },
      finish_reason: "stop",
    })),
  });
afterEach(() => {
  useChat.getState().clear();
  useTranslate.getState().clear();
  useSettings.getState().reset();
  useSecret.getState().setKey("");
});
it("sends MiSS selection with both chat and translation", async () => {
  const store = useSettings.getState();
  store.set({
    generation: {
      ...store.values.generation,
      adapter_id: "html",
      adapter_version: "v1",
      adapter_scale: "0",
    },
  });
  const sent: Record<string, unknown>[] = [];
  fetchSpy.mockImplementation(async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    sent.push(body);
    return body.stream ? output("ok") : completion(["译文"]);
  });
  await useChat.getState().send("hello");
  useTranslate
    .getState()
    .set({ source: "hello", from: "English", to: "Chinese", concurrency: 1 });
  await useTranslate.getState().run();
  expect(sent).toHaveLength(2);
  for (const body of sent) {
    expect(body.adapter_id).toBe("html");
    expect(body.adapter_version).toBe("v1");
    expect(body.adapter_scale).toBe(0);
  }
});
it("uploads MiSS PTH and legacy JSON with bearer auth and deletes a specific version", async () => {
  const { RWKVClient } = await import("../src/lib/api/client");
  let init: RequestInit | undefined;
  fetchSpy.mockImplementation(async (_url, options) => {
    init = options;
    return Response.json({ adapter_id: "html", version: "v1" });
  });
  const client = new RWKVClient("http://localhost:8000", "secret");
  await client.uploadAdapter(
    "html",
    new File(["pth"], "training.pth"),
    new File(["{}"], "checkpoint.json"),
  );
  expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer secret");
  expect(new Headers(init?.headers).get("Content-Type")).toBeNull();
  expect((init?.body as FormData).get("adapter_id")).toBe("html");
  expect(((init?.body as FormData).get("metadata") as File).name).toBe(
    "checkpoint.json",
  );
  await client.deleteAdapter("html", "v1");
  expect(init?.method).toBe("DELETE");
  expect(JSON.parse(String(init?.body))).toEqual({
    adapter_id: "html",
    adapter_version: "v1",
  });
});
it("saves streamed chat and retries without duplicating user messages", async () => {
  let sent: Record<string, unknown> = {};
  fetchSpy.mockImplementation(async (_url, init) => {
    sent = JSON.parse(String(init?.body));
    return output("hello");
  });
  useChat.getState().newChat();
  await useChat.getState().send("Say hello");
  expect(sent.messages).toEqual([{ role: "user", content: "Say hello" }]);
  let c = useChat.getState().conversations[0];
  expect(c.title).toBe("Say hello");
  expect(c.messages[1].content).toBe("hello");
  await useChat.getState().send("", true);
  c = useChat.getState().conversations[0];
  expect(c.messages).toHaveLength(2);
  expect(values.get("rwkv-conversations-v1")).toContain("hello");
  expect(useChat.getState().active).toBeNull();
});
it("keeps partial chat text and exposes backend disconnect errors", async () => {
  fetchSpy.mockImplementation(async () => output("partial", false));
  await useChat.getState().send("test");
  const c = useChat.getState().conversations[0];
  expect(c.messages[1].content).toBe("partial");
  expect(c.messages[1].error).toContain("disconnected");
});
it("stores credentials only in memory", () => {
  useSecret.getState().setKey("private-api-key");
  useRuntimeForm.getState().set({ password: "private-runtime-password" });
  expect(values.get("rwkv-runtime-form-v1")).not.toContain(
    "private-runtime-password",
  );
  expect([...values.values()].join("")).not.toContain("private-api-key");
});
it("translates each line without streaming and saves ordered results", async () => {
  const urls: string[] = [];
  const prompts: string[] = [];
  fetchSpy.mockImplementation(async (url, init) => {
    urls.push(String(url));
    const body = JSON.parse(String(init?.body));
    prompts.push(...body.contents);
    expect(body.stream).toBe(false);
    return completion(
      body.contents.map(
        (_: string, index: number) => `translated-${index + 1}`,
      ),
    );
  });
  useTranslate
    .getState()
    .set({ source: "First line\nSecond line\nThird line", concurrency: 3 });
  await useTranslate.getState().run();
  expect(urls).toHaveLength(1);
  expect(urls.every((u) => u === "/v1/chat/completions")).toBe(true);
  expect(
    prompts.every(
      (p) => p.startsWith("English: ") && p.endsWith("\n\nChinese:"),
    ),
  ).toBe(true);
  expect(useTranslate.getState().chunks.every((c) => c.status === "done")).toBe(
    true,
  );
  expect(values.get("rwkv-translation-v1")).toContain("translated-1");
});
it("caps each native translation batch at 128 lines", async () => {
  const sizes: number[] = [];
  fetchSpy.mockImplementation(async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    sizes.push(body.contents.length);
    return completion(body.contents.map((text: string) => `done:${text}`));
  });
  useTranslate.getState().set({
    source: Array.from({ length: 130 }, (_, index) => `line ${index}`).join(
      "\n",
    ),
    concurrency: 128,
  });
  await useTranslate.getState().run();
  expect(sizes).toEqual([128, 2]);
  expect(useTranslate.getState().chunks.every((c) => c.status === "done")).toBe(
    true,
  );
});
it("does not silently wrap raw translation with remote native chat templates", async () => {
  useSettings.getState().set({ baseURL: "http://127.0.0.1:8000" });
  useTranslate.getState().set({ source: "Hello" });
  const count = fetchSpy.mock.calls.length;
  await useTranslate.getState().run();
  expect(fetchSpy.mock.calls.length).toBe(count);
  expect(useTranslate.getState().error).toContain("Go Launcher");
});

// Structural render coverage is useful when no interactive browser is attached.
it("renders every workspace route without a client render exception", async () => {
  const { createElement } = await import("react");
  const { renderToString } = await import("react-dom/server");
  const { App } = await import("../src/app/App");
  for (const [route, title] of [
    ["chat", "New conversation"],
    ["translate", "Parallel Translate"],
    ["runtime", "Runtime"],
    ["state-tuning", "State Tuning"],
    ["settings", "Settings"],
  ]) {
    location.hash = "#/" + route;
    const html = renderToString(createElement(App));
    expect(html).toContain(title);
    expect(html).toContain("RWKV");
    expect(html).not.toContain("dangerouslySetInnerHTML");
  }
});

it("sends the selected initial state and explicit thinking mode with chat history", async () => {
  const requests: Record<string, unknown>[] = [];
  fetchSpy.mockImplementation(async (url, init) => {
    expect(String(url)).toBe("/v1/chat/completions");
    requests.push(JSON.parse(String(init?.body)));
    return output("answer");
  });
  useSettings.getState().set({
    generation: {
      ...useSettings.getState().values.generation,
      state_id: "roleplay.pth",
    },
  });
  await useChat.getState().send("Hello");
  expect(requests[0].state_id).toBe("roleplay.pth");
  expect(requests[0].think_type).toBe("fast");
  useSettings.getState().set({
    generation: {
      ...useSettings.getState().values.generation,
      think_type: "free",
    },
  });
  await useChat.getState().send("Continue");
  expect(requests[1].think_type).toBe("free");
  expect(requests[1].messages).toEqual([
    { role: "user", content: "Hello" },
    { role: "assistant", content: "answer" },
    { role: "user", content: "Continue" },
  ]);
});

it("uploads multipart states and authenticates list/delete requests", async () => {
  const { RWKVClient } = await import("../src/lib/api/client");
  const client = new RWKVClient("http://backend/", "secret");
  const calls: { url: string; init?: RequestInit }[] = [];
  fetchSpy.mockImplementation(async (url, init) => {
    calls.push({ url: String(url), init });
    return Response.json({ state_id: "test.pth", data: [], deleted: true });
  });
  await client.uploadState(new File(["checkpoint"], "test.pth"));
  await client.listStates();
  await client.deleteState("test.pth");
  expect(calls.map((c) => c.url)).toEqual([
    "http://backend/v1/state/upload",
    "http://backend/v1/state/list",
    "http://backend/v1/state/delete",
  ]);
  expect(calls[0].init?.body).toBeInstanceOf(FormData);
  expect((calls[0].init?.body as FormData).get("file")).toBeInstanceOf(File);
  expect(new Headers(calls[0].init?.headers).has("Content-Type")).toBe(false);
  for (const call of calls)
    expect(new Headers(call.init?.headers).get("Authorization")).toBe(
      "Bearer secret",
    );
  expect(JSON.parse(String(calls[2].init?.body))).toEqual({
    state_id: "test.pth",
  });
  fetchSpy.mockImplementation(
    async () => new Response("invalid checkpoint", { status: 400 }),
  );
  await expect(
    client.uploadState(new File(["bad"], "bad.pth")),
  ).rejects.toThrow("invalid checkpoint");
});
