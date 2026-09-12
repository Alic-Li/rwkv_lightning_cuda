import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { chunkText, translationPrompt } from "../lib/translate/chunk";
import {
  createTranslationScheduler,
  type TranslateChunk,
} from "../lib/translate/scheduler";
import { RWKVClient } from "../lib/api/client";
import { useSettings, useSecret, storage } from "./settings";
export const translationBody = (prompt: string) => ({
  contents: [prompt],
  stream: true,
  max_tokens: 2048,
  temperature: 1,
  top_k: 1,
  top_p: 0,
  alpha_presence: 0,
  alpha_frequency: 0,
  alpha_decay: 0.996,
  stop_tokens: [0],
  chunk_size: 8,
});
interface Job {
  source: string;
  chunks: TranslateChunk[];
  busy: boolean;
  error: string;
  elapsed: number;
  from: string;
  to: string;
  concurrency: number;
  target: number;
  set: (
    v: Partial<Pick<Job, "source" | "from" | "to" | "concurrency" | "target">>,
  ) => void;
  run: (ids?: number[]) => Promise<void>;
  stop: () => void;
  clear: () => void;
}
let scheduler: ReturnType<typeof createTranslationScheduler> | undefined;
export const useTranslate = create(
  persist<Job>(
    (set, get) => ({
      source: "",
      chunks: [],
      busy: false,
      error: "",
      elapsed: 0,
      from: "",
      to: "",
      concurrency: 0,
      target: 0,
      set: (v) => set(v),
      stop: () => scheduler?.stop(),
      clear: () => {
        if (!get().busy) set({ source: "", chunks: [], error: "", elapsed: 0 });
      },
      run: async (ids) => {
        if (get().busy) return;
        const v = useSettings.getState().values;
        const state = get();
        // This adapter is a Launcher feature, never silently assume a remote CUDA chat endpoint is raw.
        if (v.baseURL && v.baseURL.replace(/\/$/, "") !== location.origin) {
          set({
            error:
              "Raw translation requires this Go Launcher. Leave API Base URL empty (automatic).",
          });
          return;
        }
        const from = state.from || v.sourceLanguage,
          to = state.to || v.targetLanguage;
        const limit = state.concurrency || v.concurrency,
          target = state.target || v.chunkTarget;
        if (!from.trim() || !to.trim()) {
          set({ error: "Both language names are required." });
          return;
        }
        let chunks: TranslateChunk[];
        try {
          chunks = ids
            ? state.chunks.map((c) => ({ ...c }))
            : chunkText(state.source, target).map((source, id) => ({
                id,
                source,
                prompt: translationPrompt(source, from, to),
                status: "pending",
                translated: "",
              }));
          if (!chunks.length) throw new Error("Paste a document first.");
          const selected = ids
            ? chunks.filter((c) => ids.includes(c.id))
            : chunks;
          const started = performance.now();
          let timer: ReturnType<typeof setTimeout> | undefined;
          const publish = () => {
            timer = undefined;
            set({
              chunks: chunks.map((c) => ({ ...c })),
              elapsed: (performance.now() - started) / 1000,
            });
          };
          const client = new RWKVClient("", useSecret.getState().key);
          scheduler = createTranslationScheduler({
            chunks: selected,
            concurrency: limit,
            onProgress: () => {
              if (!timer) timer = setTimeout(publish, 100);
            },
            execute: async (c, signal) =>
              client.streamChat(translationBody(c.prompt), signal, (event) => {
                for (const choice of event.choices || []) {
                  c.translated += choice.delta?.content || "";
                  if (choice.finish_reason)
                    c.finishReason = choice.finish_reason;
                }
                if (!timer) timer = setTimeout(publish, 100);
              }),
          });
          set({ chunks: chunks.map((c) => ({ ...c })), busy: true, error: "" });
          await scheduler.run();
          clearTimeout(timer);
          publish();
        } catch (e) {
          set({ error: String(e) });
        } finally {
          scheduler = undefined;
          set({ busy: false });
        }
      },
    }),
    {
      name: "rwkv-translation-v1",
      storage: createJSONStorage(() => storage),
      partialize: (s) => ({ ...s, busy: false }),
      onRehydrateStorage: () => (state) => {
        if (state)
          state.chunks = state.chunks.map((c) =>
            c.status === "running"
              ? {
                  ...c,
                  status: "pending",
                  error: "Interrupted by reload. Retry to continue.",
                }
              : c,
          );
      },
    },
  ),
);
