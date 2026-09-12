export interface TranslateChunk {
  id: number;
  source: string;
  prompt: string;
  status: "pending" | "running" | "done" | "error";
  translated: string;
  error?: string;
  elapsed?: number;
  finishReason?: string;
}
export function createTranslationScheduler(options: {
  chunks: TranslateChunk[];
  concurrency: number;
  execute: (chunk: TranslateChunk, signal: AbortSignal) => Promise<void>;
  onProgress: () => void;
}) {
  if (
    !Number.isInteger(options.concurrency) ||
    options.concurrency < 1 ||
    options.concurrency > 64
  )
    throw new Error("Concurrency must be an integer from 1 to 64");
  const controller = new AbortController();
  let cursor = 0;
  const run = async () => {
    const worker = async () => {
      while (!controller.signal.aborted && cursor < options.chunks.length) {
        const c = options.chunks[cursor++];
        const started = performance.now();
        c.status = "running";
        c.error = undefined;
        c.translated = "";
        options.onProgress();
        try {
          await options.execute(c, controller.signal);
          controller.signal.throwIfAborted();
          c.status = "done";
        } catch (e) {
          c.status = controller.signal.aborted ? "pending" : "error";
          c.error = controller.signal.aborted
            ? "Stopped. Retry to continue."
            : String(e);
        } finally {
          c.elapsed = (performance.now() - started) / 1000;
          options.onProgress();
        }
      }
    };
    await Promise.all(
      Array.from(
        { length: Math.min(options.concurrency, options.chunks.length) },
        worker,
      ),
    );
  };
  return { run, stop: () => controller.abort() };
}
export const orderedMerge = (chunks: TranslateChunk[]) =>
  [...chunks]
    .sort((a, b) => a.id - b.id)
    .map((c) => c.translated || "···")
    .join("\n\n");
