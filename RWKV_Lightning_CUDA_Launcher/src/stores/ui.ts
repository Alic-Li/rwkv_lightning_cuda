import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { storage } from "./settings";

export interface Toast {
  id: string;
  title: string;
  description?: string;
  variant: "default" | "success" | "error";
}

interface ToastState {
  toasts: Toast[];
  push: (toast: Omit<Toast, "id">) => string;
  dismiss: (id: string) => void;
  clear: () => void;
}

export const useToasts = create<ToastState>((set, get) => ({
  toasts: [],
  push: (toast) => {
    const id = crypto.randomUUID();
    set((s) => ({ toasts: [...s.toasts, { ...toast, id }].slice(-4) }));
    window.setTimeout(
      () => get().dismiss(id),
      toast.variant === "error" ? 8000 : 4200,
    );
    return id;
  },
  dismiss: (id) =>
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  clear: () => set({ toasts: [] }),
}));

/** Imperative helper so stores and event handlers can raise toasts. */
export const toast = {
  info: (title: string, description?: string) =>
    useToasts.getState().push({ title, description, variant: "default" }),
  success: (title: string, description?: string) =>
    useToasts.getState().push({ title, description, variant: "success" }),
  error: (title: string, description?: string) =>
    useToasts.getState().push({ title, description, variant: "error" }),
};

export type FsPicker = (path: string) => void;

interface UIState {
  addBackendOpen: boolean;
  /** Set while the shared remote-directory browser is open. */
  fsPicker: FsPicker | null;
  fsInitialPath: string;
  openAddBackend: () => void;
  closeAddBackend: () => void;
  openFsBrowser: (pick: FsPicker, initialPath?: string) => void;
  closeFsBrowser: () => void;
}

export const useUI = create<UIState>((set) => ({
  addBackendOpen: false,
  fsPicker: null,
  fsInitialPath: "",
  openAddBackend: () => set({ addBackendOpen: true }),
  closeAddBackend: () => set({ addBackendOpen: false }),
  openFsBrowser: (fsPicker, fsInitialPath = "") =>
    set({ fsPicker, fsInitialPath }),
  closeFsBrowser: () => set({ fsPicker: null, fsInitialPath: "" }),
}));

/**
 * Whether the page's own <h1> is on screen. The toolbar repeats the view name
 * only once that heading scrolls away (HIG: omit a toolbar title when it is
 * redundant; iOS large titles collapse into the bar the same way), so the name
 * never appears twice at rest.
 */
interface PageTitleState {
  inlineVisible: boolean;
  setInlineVisible: (inlineVisible: boolean) => void;
}

export const usePageTitle = create<PageTitleState>((set) => ({
  inlineVisible: false,
  setInlineVisible: (inlineVisible) => set({ inlineVisible }),
}));

/**
 * The log dock: one console at the bottom of the workspace that any page can
 * open, showing whichever of the node's streams is selected. Logs used to be
 * a card at the bottom of three separate pages, which meant they were only
 * readable on the page that happened to own that stream.
 */
export type LogKind = "runtime" | "tuning" | "quantization";

interface LogDockState {
  open: boolean;
  kind: LogKind;
  /** Dock height in px, dragged by the grip on its top edge. */
  height: number;
  toggle: () => void;
  show: (kind?: LogKind) => void;
  hide: () => void;
  setKind: (kind: LogKind) => void;
  setHeight: (height: number) => void;
}

export const MIN_DOCK_HEIGHT = 140;
export const MAX_DOCK_HEIGHT = 620;

export const useLogDock = create(
  persist<LogDockState>(
    (set) => ({
      open: false,
      kind: "runtime",
      height: 260,
      toggle: () => set((s) => ({ open: !s.open })),
      show: (kind) => set((s) => ({ open: true, kind: kind ?? s.kind })),
      hide: () => set({ open: false }),
      setKind: (kind) => set({ kind }),
      setHeight: (height) =>
        set({
          height: Math.min(MAX_DOCK_HEIGHT, Math.max(MIN_DOCK_HEIGHT, height)),
        }),
    }),
    { name: "rwkv-log-dock-v1", storage: createJSONStorage(() => storage) },
  ),
);

/** Whether the workspace rail is collapsed; kept in localStorage. */
interface RailState {
  collapsed: boolean;
  toggle: () => void;
  setCollapsed: (collapsed: boolean) => void;
}

export const useRail = create(
  persist<RailState>(
    (set) => ({
      collapsed: false,
      toggle: () => set((s) => ({ collapsed: !s.collapsed })),
      setCollapsed: (collapsed) => set({ collapsed }),
    }),
    { name: "rwkv-rail-v1", storage: createJSONStorage(() => storage) },
  ),
);

/**
 * Whether the chat page's generation column is showing. On a wide page it is
 * a docked column and this is remembered; on a narrow one it floats over the
 * transcript and always starts closed, so it is kept separately in the page.
 */
interface ChatPanelState {
  docked: boolean;
  setDocked: (docked: boolean) => void;
}

export const useChatPanel = create(
  persist<ChatPanelState>(
    (set) => ({
      docked: true,
      setDocked: (docked) => set({ docked }),
    }),
    { name: "rwkv-chat-panel-v1", storage: createJSONStorage(() => storage) },
  ),
);
