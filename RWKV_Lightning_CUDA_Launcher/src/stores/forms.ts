import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import {
  defaultQuantization,
  defaultRuntime,
  defaultTuning,
} from "@/lib/api/launcher";
import type {
  DeviceSelection,
  QuantizationConfig,
  RuntimeConfig,
  TuningConfig,
} from "@/lib/api/types";
import { storage } from "./settings";

/**
 * Runtime / tuning / quantization forms, scoped per backend ID the same way
 * chat threads and log streams are.
 *
 * These used to be single global objects. Switching nodes therefore carried
 * the previous node's paths into the new node's form, and — worse — carried
 * its `visible_devices`: a card selection made against an eight-GPU box would
 * be submitted verbatim to a one-GPU box. Paths are per-machine too, so the
 * model path from node A rarely even exists on node B.
 */

const INHERIT: DeviceSelection = { mode: "inherit" };

export interface FormEntry<C> {
  config: C;
  devices: DeviceSelection;
}

export interface FormState<C> {
  byBackend: Record<string, FormEntry<C>>;
  set: (backendId: string, patch: Partial<C>) => void;
  setDevices: (backendId: string, devices: DeviceSelection) => void;
  /**
   * Populate a node's form from whatever that node is actually running, but
   * only while the user has not touched it. Re-seeding on every poll would
   * fight the person typing.
   */
  seed: (backendId: string, entry: FormEntry<C>) => void;
  reset: (backendId: string) => void;
  resetAll: () => void;
}

/**
 * One helper shared by all three stores: read-or-default, then replace.
 * `empty` must be a stable module-level reference — handing a fresh object to
 * a `useSyncExternalStore` selector re-renders forever.
 */
function updateEntry<C>(
  byBackend: Record<string, FormEntry<C>>,
  backendId: string,
  empty: FormEntry<C>,
  next: (entry: FormEntry<C>) => FormEntry<C>,
): Record<string, FormEntry<C>> {
  return { ...byBackend, [backendId]: next(byBackend[backendId] ?? empty) };
}

function createFormStore<C>(options: {
  name: string;
  version: number;
  empty: FormEntry<C>;
  /** Strip secrets from a config before it reaches localStorage. */
  sanitize?: (config: C) => C;
}) {
  const { empty, sanitize } = options;
  return create(
    persist<FormState<C>>(
      (set) => ({
        byBackend: {},

        set: (backendId, patch) =>
          set((s) =>
            backendId
              ? {
                  byBackend: updateEntry(s.byBackend, backendId, empty, (e) => ({
                    ...e,
                    config: { ...e.config, ...patch },
                  })),
                }
              : s,
          ),

        setDevices: (backendId, devices) =>
          set((s) =>
            backendId
              ? {
                  byBackend: updateEntry(s.byBackend, backendId, empty, (e) => ({
                    ...e,
                    devices,
                  })),
                }
              : s,
          ),

        seed: (backendId, entry) =>
          set((s) =>
            backendId && !s.byBackend[backendId]
              ? { byBackend: { ...s.byBackend, [backendId]: entry } }
              : s,
          ),

        reset: (backendId) =>
          set((s) => {
            if (!(backendId in s.byBackend)) return s;
            const byBackend = { ...s.byBackend };
            delete byBackend[backendId];
            return { byBackend };
          }),

        resetAll: () => set({ byBackend: {} }),
      }),
      {
        name: options.name,
        version: options.version,
        storage: createJSONStorage(() => storage),
        partialize: (s) =>
          ({
            byBackend: sanitize
              ? Object.fromEntries(
                  Object.entries(s.byBackend).map(([id, entry]) => [
                    id,
                    { ...entry, config: sanitize(entry.config) },
                  ]),
                )
              : s.byBackend,
          }) as FormState<C>,
        // The storage key carries the version, so the previous key is simply
        // left behind rather than migrated. That is deliberate: it held one
        // flat {config, devices} with no record of which node it came from,
        // and attributing it to a node would reproduce the exact bug this
        // change fixes. Each node re-seeds from what it reports it is running.
      },
    ),
  );
}

export const EMPTY_RUNTIME_FORM: FormEntry<RuntimeConfig> = {
  config: defaultRuntime,
  devices: INHERIT,
};

export const EMPTY_TUNING_FORM: FormEntry<TuningConfig> = {
  config: defaultTuning,
  devices: INHERIT,
};

export const EMPTY_QUANTIZATION_FORM: FormEntry<QuantizationConfig> = {
  config: defaultQuantization,
  devices: INHERIT,
};

export const useRuntimeForm = createFormStore<RuntimeConfig>({
  name: "rwkv-runtime-form-v3",
  version: 3,
  empty: EMPTY_RUNTIME_FORM,
  // The runtime password never touches disk.
  sanitize: (config) => ({ ...config, password: "" }),
});

export const useTuningForm = createFormStore<TuningConfig>({
  name: "rwkv-tuning-form-v3",
  version: 3,
  empty: EMPTY_TUNING_FORM,
});

export const useQuantizationForm = createFormStore<QuantizationConfig>({
  name: "rwkv-quantization-form-v2",
  version: 2,
  empty: EMPTY_QUANTIZATION_FORM,
});

/** Current node's entry, or a stable default while it has none. */
export function useRuntimeFormEntry(backendId: string) {
  return useRuntimeForm((s) => s.byBackend[backendId] ?? EMPTY_RUNTIME_FORM);
}

export function useTuningFormEntry(backendId: string) {
  return useTuningForm((s) => s.byBackend[backendId] ?? EMPTY_TUNING_FORM);
}

export function useQuantizationFormEntry(backendId: string) {
  return useQuantizationForm(
    (s) => s.byBackend[backendId] ?? EMPTY_QUANTIZATION_FORM,
  );
}

/**
 * Restore every tuning parameter to its default while keeping the paths the
 * user picked on this node (the "恢复默认" button next to the presets).
 */
export function resetTuningParameters(backendId: string) {
  const entry =
    useTuningForm.getState().byBackend[backendId] ?? EMPTY_TUNING_FORM;
  useTuningForm.getState().set(backendId, {
    ...defaultTuning,
    method: entry.config.method,
    model: entry.config.model,
    data: entry.config.data,
    output: entry.config.output,
    vocab: entry.config.vocab,
  });
}
