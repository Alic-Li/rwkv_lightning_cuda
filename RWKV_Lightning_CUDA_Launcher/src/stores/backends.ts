import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { backendsApi } from "@/lib/api/backends";
import type {
  AddBackendRequest,
  UpdateBackendRequest,
  BackendView,
  Capability,
} from "@/lib/api/types";
import type { MessageKey } from "@/lib/i18n";
import { storage } from "./settings";

interface BackendsState {
  list: BackendView[];
  currentId: string;
  loading: boolean;
  loaded: boolean;
  error: string;
  refresh: (signal?: AbortSignal) => Promise<void>;
  select: (id: string) => void;
  add: (body: AddBackendRequest) => Promise<BackendView>;
  update: (id: string, body: UpdateBackendRequest) => Promise<BackendView>;
  remove: (id: string) => Promise<void>;
  probe: (id: string) => Promise<BackendView>;
  probeAll: () => Promise<void>;
}

export const useBackends = create(
  persist<BackendsState>(
    (set, get) => ({
      list: [],
      currentId: "",
      loading: false,
      loaded: false,
      error: "",

      refresh: async (signal) => {
        set({ loading: true });
        try {
          const { backends } = await backendsApi.list(signal);
          const current = get().currentId;
          const stillThere = backends.some((b) => b.id === current);
          set({
            list: backends,
            loaded: true,
            loading: false,
            error: "",
            // Keep the selection stable; fall back to the first backend.
            currentId: stillThere ? current : (backends[0]?.id ?? ""),
          });
        } catch (error) {
          // Still clear `loading`: the store is a singleton, so leaving it set
          // strands the spinner for every later consumer.
          if (signal?.aborted) {
            set({ loading: false });
            return;
          }
          set({
            loading: false,
            loaded: true,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      },

      select: (id) => set({ currentId: id }),

      add: async (body) => {
        const view = await backendsApi.add(body);
        set((s) => ({
          list: [...s.list.filter((b) => b.id !== view.id), view],
          currentId: view.id,
        }));
        return view;
      },

      update: async (id, body) => {
        const view = await backendsApi.update(id, body);
        set((s) => ({
          list: s.list.map((b) => (b.id === view.id ? view : b)),
        }));
        return view;
      },

      remove: async (id) => {
        await backendsApi.remove(id);
        set((s) => {
          const list = s.list.filter((b) => b.id !== id);
          return {
            list,
            currentId: s.currentId === id ? (list[0]?.id ?? "") : s.currentId,
          };
        });
      },

      probe: async (id) => {
        const view = await backendsApi.probe(id);
        set((s) => ({
          list: s.list.map((b) => (b.id === view.id ? view : b)),
        }));
        return view;
      },

      probeAll: async () => {
        const targets = get().list;
        await Promise.all(
          targets.map(async (backend) => {
            try {
              const view = await backendsApi.probe(backend.id);
              set((s) => ({
                list: s.list.map((b) => (b.id === view.id ? view : b)),
              }));
            } catch (error) {
              set((s) => ({
                list: s.list.map((b) =>
                  b.id === backend.id
                    ? {
                        ...b,
                        reachable: false,
                        probe_error:
                          error instanceof Error
                            ? error.message
                            : String(error),
                      }
                    : b,
                ),
              }));
            }
          }),
        );
      },
    }),
    {
      name: "rwkv-backends-v1",
      storage: createJSONStorage(() => storage),
      partialize: (s) => ({ currentId: s.currentId }) as BackendsState,
    },
  ),
);

export function currentBackend(list: BackendView[], id: string) {
  return list.find((b) => b.id === id);
}

export function hasCapability(
  backend: BackendView | undefined,
  capability: Capability,
) {
  if (!backend) return false;
  if (backend.kind === "inference_only") return capability === "inference";
  return backend.capabilities.includes(capability);
}

/**
 * Display name for a backend. Every registered backend carries the name its
 * user gave it; the local node is the one entry the Client synthesizes, so
 * the Go side leaves `name` empty and the label comes from the message
 * catalogue instead of being hard-coded in one language on the wire.
 */
export function backendLabel(
  t: (key: MessageKey) => string,
  backend: Pick<BackendView, "id" | "name">,
) {
  if (backend.name) return backend.name;
  return backend.id === "local" ? t("backend.local") : backend.id;
}

/** Human-facing label for a probed backend. */
export function backendKindLabel(kind: BackendView["kind"]) {
  if (kind === "inference_only") return "inference";
  if (kind === "agent") return "agent";
  return "";
}

/** A node is "agent-like" when it exposes process control. */
export function isAgentNode(backend: BackendView | undefined) {
  return Boolean(
    backend &&
      backend.kind !== "inference_only" &&
      (backend.kind === "agent" || backend.capabilities.length > 0),
  );
}
