import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { adapterFields } from "@/lib/api/client";
import { inferenceApi } from "@/lib/api/inference";
import { chunkText, translationPrompt } from "@/lib/translate/chunk";
import { normalizeLanguage } from "@/lib/translate/languages";
import type { TranslateChunk } from "@/lib/translate/scheduler";
import { storage, useSettings } from "./settings";

/**
 * Raw continuation body for `/v1/batch/completions`. `stop_tokens: [0]` and the
 * sampling values are part of the translation contract — do not "tidy" them.
 */
export const translationBody = (contents: string | string[]) => ({
  contents: Array.isArray(contents) ? contents : [contents],
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

interface TranslateState {
  /** Node the current results belong to; results never leak across nodes. */
  owner: string;
  source: string;
  chunks: TranslateChunk[];
  busy: boolean;
  error: string;
  elapsed: number;
  /** Empty string means "follow the saved default". */
  from: string;
  to: string;
  /** 0 means "follow the saved default". */
  concurrency: number;
  set: (
    patch: Partial<
      Pick<TranslateState, "source" | "from" | "to" | "concurrency" | "error">
    >,
  ) => void;
  run: (backendId: string, ids?: number[]) => Promise<void>;
  stop: () => void;
  clear: () => void;
}

let controller: AbortController | null = null;

const MAX_BATCH = 128;

export const useTranslate = create(
  persist<TranslateState>(
    (set, get) => ({
      owner: "",
      source: "",
      chunks: [],
      busy: false,
      error: "",
      elapsed: 0,
      from: "",
      to: "",
      concurrency: 0,

      set: (patch) => set(patch),

      stop: () => controller?.abort(),

      clear: () => {
        if (get().busy) return;
        set({ source: "", chunks: [], error: "", elapsed: 0 });
      },

      run: async (backendId, ids) => {
        const state = get();
        if (state.busy || !backendId) return;
        const settings = useSettings.getState();
        const from = normalizeLanguage(
          state.from || settings.sourceLanguage,
          "English",
        );
        const to = normalizeLanguage(
          state.to || settings.targetLanguage,
          "Chinese",
        );
        const limit = state.concurrency || settings.concurrency;
        if (!from || !to || from === to) {
          set({ error: "invalid-languages" });
          return;
        }
        if (!Number.isInteger(limit) || limit < 1 || limit > MAX_BATCH) {
          set({ error: "invalid-batch" });
          return;
        }

        // Results are scoped to a node: switching nodes starts a fresh job.
        const sameOwner = state.owner === backendId;
        let chunks: TranslateChunk[];
        if (ids) {
          if (!sameOwner) return;
          chunks = state.chunks.map((c) => ({ ...c }));
        } else {
          chunks = chunkText(state.source).map((source, id) => ({
            id,
            source,
            prompt: translationPrompt(source, from, to),
            status: "pending" as const,
            translated: "",
          }));
          if (chunks.length === 0) {
            set({ error: "empty-source" });
            return;
          }
        }

        const selected = ids ? chunks.filter((c) => ids.includes(c.id)) : chunks;
        const started = performance.now();
        controller = new AbortController();
        const signal = controller.signal;

        const publish = () =>
          set({
            chunks,
            owner: backendId,
            elapsed: (performance.now() - started) / 1000,
          });

        set({ busy: true, error: "", chunks, owner: backendId });
        publish();

        try {
          for (let offset = 0; offset < selected.length; offset += limit) {
            if (signal.aborted) break;
            const batch = selected.slice(offset, offset + limit);
            const batchStarted = performance.now();
            for (const chunk of batch) {
              chunk.status = "running";
              chunk.error = undefined;
              chunk.translated = "";
            }
            publish();

            try {
              const response = await inferenceApi.batchCompletions(
                backendId,
                {
                  ...translationBody(batch.map((c) => c.prompt)),
                  ...adapterFields(settings.generation),
                },
                signal,
              );
              const byIndex = new Map(
                (response.choices ?? []).map((choice) => [choice.index, choice]),
              );
              for (const [index, chunk] of batch.entries()) {
                const choice = byIndex.get(index);
                const text = choice?.message?.content;
                if (typeof text !== "string")
                  throw new Error(
                    `Backend returned no result for line ${chunk.id + 1}`,
                  );
                chunk.translated = text;
                chunk.finishReason = choice?.finish_reason;
                chunk.status = "done";
              }
            } catch (error) {
              const aborted = signal.aborted;
              for (const chunk of batch) {
                if (chunk.status === "done") continue;
                chunk.status = aborted ? "pending" : "error";
                chunk.error = aborted
                  ? "Stopped. Retry to continue."
                  : error instanceof Error
                    ? error.message
                    : String(error);
              }
              if (aborted) {
                publish();
                break;
              }
            } finally {
              const seconds = (performance.now() - batchStarted) / 1000;
              for (const chunk of batch) chunk.elapsed = seconds;
              publish();
            }
          }
        } catch (error) {
          if (!signal.aborted)
            set({
              error: error instanceof Error ? error.message : String(error),
            });
        } finally {
          publish();
          controller = null;
          set({ busy: false });
        }
      },
    }),
    {
      name: "rwkv-translation-v2",
      version: 2,
      storage: createJSONStorage(() => storage),
      partialize: (s) =>
        ({
          owner: s.owner,
          source: s.source,
          chunks: s.chunks,
          elapsed: s.elapsed,
          from: s.from,
          to: s.to,
          concurrency: s.concurrency,
        }) as TranslateState,
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        // A reload cannot resume in-flight requests: put them back in the queue.
        state.chunks = state.chunks.map((chunk) =>
          chunk.status === "running"
            ? {
                ...chunk,
                status: "pending",
                error: "Interrupted by reload. Retry to continue.",
              }
            : chunk,
        );
      },
    },
  ),
);

export const translatedOutput = (chunks: TranslateChunk[]) =>
  chunks
    .filter((c) => c.status === "done" && c.translated)
    .map((c) => c.translated)
    .join("\n\n");
