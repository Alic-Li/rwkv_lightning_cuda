import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import {
  defaultRuntime,
  launcher,
  tuning,
  quantization,
  defaultQuantization,
  type RuntimeState,
  type RuntimeConfig,
  type ProcessStatus,
  type QuantizationStatus,
  type QuantizationConfig,
} from "../lib/api/launcher";
import { storage } from "./settings";
export const useRuntimeForm = create(
  persist<{
    config: RuntimeConfig;
    recent: string[];
    set: (v: Partial<RuntimeConfig>) => void;
    remember: () => void;
  }>(
    (set) => ({
      config: defaultRuntime,
      recent: [],
      set: (v) => set((s) => ({ config: { ...s.config, ...v } })),
      remember: () =>
        set((s) => ({
          recent: [
            s.config.model_path,
            ...s.recent.filter((p) => p !== s.config.model_path),
          ].slice(0, 8),
        })),
    }),
    {
      name: "rwkv-runtime-form-v1",
      storage: createJSONStorage(() => storage),
      partialize: (s) => ({ ...s, config: { ...s.config, password: "" } }),
    },
  ),
);
export const useQuantizationForm = create(
  persist<{
    config: QuantizationConfig;
    set: (v: Partial<QuantizationConfig>) => void;
  }>(
    (set) => ({
      config: defaultQuantization,
      set: (v) => set((s) => ({ config: { ...s.config, ...v } })),
    }),
    {
      name: "rwkv-quantization-form-v1",
      storage: createJSONStorage(() => storage),
    },
  ),
);
let synchronized = false;
const empty: ProcessStatus = {
  status: "offline",
  running: false,
  error: "",
  logs: [],
  elapsed: 0,
};
export const useRuntime = create<{
  runtime: RuntimeState;
  tuning: ProcessStatus;
  quantization: QuantizationStatus;
  connected: boolean;
  busy: boolean;
  quantizationBusy: boolean;
  error: string;
  quantizationError: string;
  refresh: () => Promise<void>;
  action: (action: "start" | "stop" | "restart") => Promise<void>;
  quantizationAction: (action: "start" | "stop") => Promise<void>;
}>((set) => ({
  runtime: empty,
  tuning: empty,
  quantization: empty,
  connected: false,
  busy: false,
  quantizationBusy: false,
  error: "",
  quantizationError: "",
  refresh: async () => {
    try {
      const [runtime, state, quantizationState] = await Promise.all([
        launcher.getStatus(AbortSignal.timeout(4000)),
        tuning.getStatus(AbortSignal.timeout(4000)),
        quantization.getStatus(AbortSignal.timeout(4000)),
      ]);
      if (!synchronized) {
        synchronized = true;
        if (runtime.running && runtime.config)
          useRuntimeForm.getState().set({
            ...runtime.config,
            password: useRuntimeForm.getState().config.password,
          });
      }
      set({
        runtime,
        tuning: state,
        quantization: quantizationState,
        connected: true,
        error: "",
      });
    } catch (e) {
      set({ connected: false, error: String(e) });
    }
  },
  action: async (action) => {
    set({ busy: true, error: "" });
    try {
      if (action === "start") {
        await launcher.start(useRuntimeForm.getState().config);
        useRuntimeForm.getState().remember();
      } else await launcher[action]();
      await useRuntime.getState().refresh();
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ busy: false });
    }
  },
  quantizationAction: async (action) => {
    set({ quantizationBusy: true, quantizationError: "" });
    try {
      if (action === "start")
        await quantization.start(useQuantizationForm.getState().config);
      else await quantization.stop();
      await useRuntime.getState().refresh();
    } catch (e) {
      set({ quantizationError: String(e) });
    } finally {
      set({ quantizationBusy: false });
    }
  },
}));
