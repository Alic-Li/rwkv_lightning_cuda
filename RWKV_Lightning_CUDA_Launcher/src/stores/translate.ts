import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { chunkText, translationPrompt } from "../lib/translate/chunk";
import { normalizeLanguage } from "../lib/translate/languages";
import type { TranslateChunk } from "../lib/translate/scheduler";
import { RWKVClient, adapterFields } from "../lib/api/client";
import { useSettings, useSecret, storage } from "./settings";
export const translationBody = (prompt: string | string[]) => ({
  contents: Array.isArray(prompt) ? prompt : [prompt],
  stream: false,
  max_tokens: 2048,
  temperature: 1,
  top_k: 1,
  top_p: 0,
  alpha_presence: 0,
  alpha_frequency: 0,
  alpha_decay: 0.996,
  stop_tokens: [0],
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
  set: (
    v: Partial<Pick<Job, "source" | "from" | "to" | "concurrency">>,
  ) => void;
  run: (ids?: number[]) => Promise<void>;
  stop: () => void;
  clear: () => void;
}
let activeController: AbortController | undefined;
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
      set: (v) => set(v),
      stop: () => activeController?.abort(),
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
        const from = normalizeLanguage(
            state.from || v.sourceLanguage,
            "English",
          ),
          to = normalizeLanguage(state.to || v.targetLanguage, "Chinese");
        const limit = state.concurrency || v.concurrency;
        if (!from.trim() || !to.trim()) {
          set({ error: "Both language names are required." });
          return;
        }
        if (from === to) {
          set({ error: "Source and target languages must be different." });
          return;
        }
        let chunks: TranslateChunk[];
        try {
          chunks = ids
            ? state.chunks.map((c) => ({ ...c }))
            : chunkText(state.source).map((source, id) => ({
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
          const publish = () => {
            set({
              chunks: chunks.map((c) => ({ ...c })),
              elapsed: (performance.now() - started) / 1000,
            });
          };
          const client = new RWKVClient("", useSecret.getState().key);
          if (!Number.isInteger(limit) || limit < 1 || limit > 128)
            throw new Error("Batch size must be an integer from 1 to 128");
          const controller = new AbortController();
          activeController = controller;
          set({ chunks: chunks.map((c) => ({ ...c })), busy: true, error: "" });
          for (let offset = 0; offset < selected.length; offset += limit) {
            const batch = selected.slice(offset, offset + limit);
            for (const chunk of batch) {
              chunk.status = "running";
              chunk.error = undefined;
              chunk.translated = "";
            }
            publish();
            const batchStarted = performance.now();
            try {
              const result = await client.completeChat(
                {
                  ...translationBody(batch.map((chunk) => chunk.prompt)),
                  ...adapterFields(v.generation),
                },
                controller.signal,
              );
              const choices = new Map(
                (result.choices || []).map((choice) => [choice.index, choice]),
              );
              for (const [index, chunk] of batch.entries()) {
                const choice = choices.get(index);
                if (!choice?.message?.content)
                  throw new Error(
                    `Backend returned no result for line ${chunk.id + 1}`,
                  );
                chunk.translated = choice.message.content;
                chunk.finishReason = choice.finish_reason;
                chunk.status = "done";
              }
            } catch (e) {
              for (const chunk of batch) {
                if (chunk.status === "done") continue;
                chunk.status = controller.signal.aborted ? "pending" : "error";
                chunk.error = controller.signal.aborted
                  ? "Stopped. Retry to continue."
                  : String(e);
              }
              if (controller.signal.aborted) break;
            } finally {
              const elapsed = (performance.now() - batchStarted) / 1000;
              for (const chunk of batch) chunk.elapsed = elapsed;
              publish();
            }
          }
          publish();
        } catch (e) {
          set({ error: String(e) });
        } finally {
          activeController = undefined;
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
