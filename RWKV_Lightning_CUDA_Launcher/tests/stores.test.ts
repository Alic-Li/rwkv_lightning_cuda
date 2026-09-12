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
afterEach(() => {
  useChat.getState().clear();
  useTranslate.getState().clear();
  useSettings.getState().reset();
  useSecret.getState().setKey("");
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
it("streams each translation through chat URL and saves ordered results", async () => {
  const urls: string[] = [];
  const prompts: string[] = [];
  fetchSpy.mockImplementation(async (url, init) => {
    urls.push(String(url));
    const body = JSON.parse(String(init?.body));
    prompts.push(body.contents[0]);
    return output(`translated-${prompts.length}`);
  });
  useTranslate
    .getState()
    .set({ source: "Hello world. ".repeat(30), target: 64, concurrency: 3 });
  await useTranslate.getState().run();
  expect(urls.length).toBeGreaterThan(2);
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
