import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import {
  defaultRuntime,
  launcher,
  tuning,
  type RuntimeState,
  type RuntimeConfig,
  type ProcessStatus,
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
  connected: boolean;
  busy: boolean;
  error: string;
  refresh: () => Promise<void>;
  action: (action: "start" | "stop" | "restart") => Promise<void>;
}>((set) => ({
  runtime: empty,
  tuning: empty,
  connected: false,
  busy: false,
  error: "",
  refresh: async () => {
    try {
      const [runtime, state] = await Promise.all([
        launcher.getStatus(AbortSignal.timeout(4000)),
        tuning.getStatus(AbortSignal.timeout(4000)),
      ]);
      if (!synchronized) {
        synchronized = true;
        if (runtime.running && runtime.config)
          useRuntimeForm.getState().set({
            ...runtime.config,
            password: useRuntimeForm.getState().config.password,
          });
      }
      set({ runtime, tuning: state, connected: true, error: "" });
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
}));
