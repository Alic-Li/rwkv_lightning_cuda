import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
export const storage = {
  getItem: (name: string) => localStorage.getItem(name),
  setItem: (name: string, value: string) => {
    try {
      localStorage.setItem(name, value);
    } catch {
      window.dispatchEvent(new CustomEvent("storage-error"));
    }
  },
  removeItem: (name: string) => localStorage.removeItem(name),
};
export interface GenerationSettings {
  temperature: number;
  top_p: number;
  top_k: number;
  max_tokens: number;
  alpha_presence: number;
  alpha_frequency: number;
  alpha_decay: number;
  chunk_size: number;
  think_type: string;
  state_id: string;
}
export type ThemeMode = "dark" | "light" | "system";
export const defaultGeneration: GenerationSettings = {
  temperature: 1,
  top_p: 0.3,
  top_k: 20,
  max_tokens: 8192,
  alpha_presence: 2,
  alpha_frequency: 0.2,
  alpha_decay: 0.996,
  chunk_size: 1,
  think_type: "fast",
  state_id: "",
};
export const defaults = {
  theme: "dark" as ThemeMode,
  baseURL: "",
  sourceLanguage: "English",
  targetLanguage: "Chinese",
  concurrency: 8,
  chunkTarget: 800,
  generation: defaultGeneration,
};

export function resolveTheme(theme: ThemeMode | string, systemLight: boolean) {
  if (theme === "system") return systemLight ? "light" : "dark";
  return theme === "light" ? "light" : "dark";
}
export const useSettings = create(
  persist<{
    values: typeof defaults;
    set: (v: Partial<typeof defaults>) => void;
    reset: () => void;
  }>(
    (set) => ({
      values: defaults,
      set: (v) => set((s) => ({ values: { ...s.values, ...v } })),
      reset: () => set({ values: defaults }),
    }),
    { name: "rwkv-settings-v1", storage: createJSONStorage(() => storage) },
  ),
);
// Credentials are deliberately session-only.
export const useSecret = create<{ key: string; setKey: (key: string) => void }>(
  (set) => ({ key: "", setKey: (key) => set({ key }) }),
);
