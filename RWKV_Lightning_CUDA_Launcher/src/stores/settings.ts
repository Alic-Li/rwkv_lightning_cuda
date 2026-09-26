import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { Lang } from "@/lib/i18n";

/** localStorage wrapper that reports quota/availability problems to the UI. */
export const storage = {
  getItem: (name: string) => {
    try {
      return localStorage.getItem(name);
    } catch {
      return null;
    }
  },
  setItem: (name: string, value: string) => {
    try {
      localStorage.setItem(name, value);
    } catch {
      window.dispatchEvent(new CustomEvent("storage-error"));
    }
  },
  removeItem: (name: string) => {
    try {
      localStorage.removeItem(name);
    } catch {
      /* ignore */
    }
  },
};

export interface GenerationSettings {
  adapter_id?: string;
  adapter_version?: string;
  adapter_scale?: string;
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

/** Ordered `think_type` values accepted by the native server. */
export const THINK_TYPES = [
  "none",
  "fast",
  "free",
  "preferChinese",
  "en",
  "enShort",
  "enLong",
] as const;

export const defaultGeneration: GenerationSettings = {
  adapter_id: "",
  adapter_version: "",
  adapter_scale: "",
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

export interface SettingsValues {
  theme: ThemeMode;
  lang: Lang;
  sourceLanguage: string;
  targetLanguage: string;
  concurrency: number;
  generation: GenerationSettings;
  railCollapsed: boolean;
}

export const defaults: SettingsValues = {
  // Follow the OS by default: a hard-coded dark default ignores the desktop
  // preference and is jarring on a light desktop.
  theme: "system",
  lang: "zh",
  sourceLanguage: "English",
  targetLanguage: "Chinese",
  concurrency: 8,
  generation: defaultGeneration,
  railCollapsed: false,
};

export interface SettingsState extends SettingsValues {
  set: (patch: Partial<SettingsValues>) => void;
  setGeneration: (patch: Partial<GenerationSettings>) => void;
  reset: () => void;
}

export function resolveTheme(theme: ThemeMode | string, systemLight: boolean) {
  if (theme === "system") return systemLight ? "light" : "dark";
  return theme === "light" ? "light" : "dark";
}

interface PersistedV1 {
  values?: Partial<SettingsValues> & { baseURL?: string };
}

export const useSettings = create(
  persist<SettingsState>(
    (set) => ({
      ...defaults,
      set: (patch) => set(patch),
      setGeneration: (patch) =>
        set((s) => ({ generation: { ...s.generation, ...patch } })),
      reset: () => set({ ...defaults }),
    }),
    {
      name: "rwkv-settings-v2",
      version: 3,
      storage: createJSONStorage(() => storage),
      migrate: (persisted, version) => {
        if (version >= 3) return persisted as SettingsState;
        // v1 nested everything under `values` and kept a per-install baseURL.
        const legacy =
          version >= 2
            ? (persisted as Partial<SettingsValues> | undefined) ?? {}
            : ((persisted as PersistedV1 | undefined)?.values ?? {});
        return {
          ...defaults,
          ...legacy,
          // Dark used to be the hard-coded default rather than a choice, so a
          // stored "dark" carries no intent: switch those installs to system.
          theme: legacy.theme === "light" ? "light" : "system",
          generation: { ...defaultGeneration, ...legacy.generation },
        } as SettingsState;
      },
    },
  ),
);

/** Credentials are deliberately session-only and never persisted. */
export const useSecret = create<{ key: string; setKey: (key: string) => void }>(
  (set) => ({ key: "", setKey: (key) => set({ key }) }),
);
