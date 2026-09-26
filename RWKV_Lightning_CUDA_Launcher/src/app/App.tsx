import { useEffect, useRef } from "react";
import { AddBackendDialog } from "@/components/add-backend-dialog";
import { FsBrowserDialog } from "@/components/fs-browser-dialog";
import { Toaster } from "@/components/ui/toaster";
import { Header } from "@/app/header";
import { LogDock } from "@/app/log-dock";
import { Rail } from "@/app/rail";
import { useCurrent } from "@/app/use-current";
import { tNow, useI18n } from "@/lib/i18n";
import { useRoute } from "@/lib/router";
import { ChatPage } from "@/pages/chat-page";
import { NodesPage } from "@/pages/nodes-page";
import { QuantizationPage } from "@/pages/quantization-page";
import { RuntimePage } from "@/pages/runtime-page";
import { SettingsPage } from "@/pages/settings-page";
import { TrainingPage } from "@/pages/training-page";
import { TranslatePage } from "@/pages/translate-page";
import { useBackends } from "@/stores/backends";
import { useLogs } from "@/stores/logs";
import { useNodes } from "@/stores/nodes";
import { resolveTheme, useSettings } from "@/stores/settings";
import { toast, useLogDock } from "@/stores/ui";

/** Keeps `data-theme`, `color-scheme` and the browser chrome colour in sync. */
function useThemeEffect() {
  const theme = useSettings((s) => s.theme);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: light)");
    const apply = () => {
      const resolved = resolveTheme(theme, media.matches);
      const root = document.documentElement;
      root.dataset.theme = resolved;
      root.style.colorScheme = resolved;
      document
        .querySelector('meta[name="theme-color"]')
        ?.setAttribute("content", resolved === "light" ? "#ffffff" : "#09090b");
    };
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [theme]);

  const lang = useSettings((s) => s.lang);
  useEffect(() => {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  }, [lang]);
}

/** The registry list is cheap; poll it slowly for reachability changes. */
function useRegistryPolling() {
  const refresh = useBackends((s) => s.refresh);
  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    const timer = window.setInterval(() => void refresh(), 8000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [refresh]);
}

/** Runtime + jobs poll fast (design: 1.6s); GPU metrics are slower. */
function useNodePolling(backendId: string) {
  const refresh = useNodes((s) => s.refresh);
  const refreshMetrics = useNodes((s) => s.refreshMetrics);
  useEffect(() => {
    if (!backendId) return;
    const controller = new AbortController();
    void refresh(backendId, controller.signal);
    void refreshMetrics(backendId, controller.signal);
    const fast = window.setInterval(() => void refresh(backendId), 1600);
    const slow = window.setInterval(() => void refreshMetrics(backendId), 6000);
    return () => {
      controller.abort();
      window.clearInterval(fast);
      window.clearInterval(slow);
    };
  }, [backendId, refresh, refreshMetrics]);
}

/** Drop log streams belonging to a node the user just navigated away from. */
function useLogStreamScope(backendId: string) {
  const closeBackend = useLogs((s) => s.closeBackend);
  const previous = useRef(backendId);
  useEffect(() => {
    if (previous.current && previous.current !== backendId)
      closeBackend(previous.current);
    previous.current = backendId;
  }, [backendId, closeBackend]);
}

/** ⌘⇧L / Ctrl+⇧+L toggles the log console from anywhere. */
function useLogShortcut() {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!event.shiftKey || !(event.metaKey || event.ctrlKey)) return;
      if (event.key.toLowerCase() !== "l") return;
      event.preventDefault();
      useLogDock.getState().toggle();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

function useStorageErrorNotice() {
  useEffect(() => {
    const onError = () =>
      toast.error(tNow("common.error"), tNow("error.storageFull"));
    window.addEventListener("storage-error", onError);
    return () => window.removeEventListener("storage-error", onError);
  }, []);
}

export function App() {
  const route = useRoute();
  const backendId = useBackends((s) => s.currentId);
  const { t } = useI18n();
  const { backend, error } = useCurrent();

  useThemeEffect();
  useRegistryPolling();
  useNodePolling(backendId);
  useLogStreamScope(backendId);
  useLogShortcut();
  useStorageErrorNotice();

  return (
    <div className="grid h-full grid-rows-[52px_minmax(0,1fr)] bg-background text-foreground">
      <Header />
      <div className="flex min-h-0">
        <Rail />
        <div className="flex min-w-0 flex-1 flex-col">
          <main className="min-h-0 flex-1 overflow-auto bg-background">
            {route === "nodes" && <NodesPage />}
            {route === "chat" && <ChatPage />}
            {route === "translate" && <TranslatePage />}
            {route === "runtime" && <RuntimePage />}
            {route === "training" && <TrainingPage />}
            {route === "quant" && <QuantizationPage />}
            {route === "settings" && <SettingsPage />}
          </main>
          <LogDock />
        </div>
      </div>

      {backend && !backend.reachable && error && (
        <div className="sr-only" role="status">
          {t("status.unreachable")}: {error}
        </div>
      )}

      <AddBackendDialog />
      <FsBrowserDialog />
      <Toaster />
    </div>
  );
}
