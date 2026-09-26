import { create } from "zustand";
import type { JobProgress } from "@/lib/api/types";

/** One poll of the trainer, keyed by the step it reported. */
export interface TrainingPoint {
  step: number;
  loss?: number;
  lr?: number;
  tps?: number;
}

/**
 * The Agent reports `losses[]` as history but everything else — lr, tok/s —
 * only as the current value, so a run's other curves exist only if the console
 * keeps them. Session-scoped on purpose: `losses` comes back from the node on
 * reload, and a stale lr curve from a previous run would be worse than none.
 */
const MAX_POINTS = 4000;

interface TrainingSeriesState {
  series: Record<string, TrainingPoint[]>;
  record: (backendId: string, progress: JobProgress | null | undefined) => void;
  clear: (backendId: string) => void;
}

export const useTrainingSeries = create<TrainingSeriesState>((set) => ({
  series: {},

  record: (backendId, progress) => {
    const step = progress?.step;
    if (!backendId || !step) return;
    set((s) => {
      const previous = s.series[backendId] ?? [];
      const last = previous[previous.length - 1];
      if (last && last.step === step) return s;
      // A step that moves backwards is a new run, not more of this one.
      const base = last && step < last.step ? [] : previous;
      const next = [
        ...base,
        {
          step,
          loss: progress?.loss,
          lr: progress?.lr,
          tps: progress?.tokens_per_second,
        },
      ];
      return {
        series: {
          ...s.series,
          [backendId]:
            next.length > MAX_POINTS
              ? next.slice(next.length - MAX_POINTS)
              : next,
        },
      };
    });
  },

  clear: (backendId) =>
    set((s) => {
      const series = { ...s.series };
      delete series[backendId];
      return { series };
    }),
}));
