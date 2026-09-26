import { useCallback, useEffect, useRef } from "react";
import { ChevronDown, ScrollText } from "lucide-react";
import { useCurrent } from "@/app/use-current";
import { LogConsole } from "@/components/log-console";
import { StatusDot, type StatusTone } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { jobsApi } from "@/lib/api/jobs";
import { useI18n, type MessageKey } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { selectLogStream, useLogs, type LogKind } from "@/stores/logs";
import { MAX_DOCK_HEIGHT, MIN_DOCK_HEIGHT, useLogDock } from "@/stores/ui";

const STREAMS: { kind: LogKind; label: MessageKey }[] = [
  { kind: "runtime", label: "logs.streamRuntime" },
  { kind: "tuning", label: "logs.streamTuning" },
  { kind: "quantization", label: "logs.streamQuantization" },
];

const endpointFor = (kind: LogKind) =>
  kind === "runtime" ? "/api/v1/runtime/logs" : jobsApi.logsPath(kind);

/**
 * Workspace-wide log console. One SSE connection at a time — the selected
 * stream of the selected node — because the Agent replays its buffer on
 * connect, so switching tabs costs a reconnect rather than losing history.
 */
export function LogDock() {
  const { t } = useI18n();
  const { backendId, hasAgent, jobs, runtime } = useCurrent();
  const open = useLogDock((s) => s.open);
  const kind = useLogDock((s) => s.kind);
  const height = useLogDock((s) => s.height);
  const setKind = useLogDock((s) => s.setKind);
  const hide = useLogDock((s) => s.hide);
  const setHeight = useLogDock((s) => s.setHeight);
  const stream = useLogs((s) => selectLogStream(s.streams, backendId, kind));
  const clear = useLogs((s) => s.clear);

  useEffect(() => {
    if (!open || !backendId || !hasAgent) return;
    const { open: connect, close } = useLogs.getState();
    connect(backendId, kind);
    return () => close(backendId, kind);
  }, [open, backendId, hasAgent, kind]);

  const dragFrom = useRef<{ y: number; height: number } | null>(null);
  const onPointerMove = useCallback(
    (event: PointerEvent) => {
      const from = dragFrom.current;
      if (from) setHeight(from.height + (from.y - event.clientY));
    },
    [setHeight],
  );
  useEffect(() => {
    const stopDrag = () => {
      dragFrom.current = null;
      document.body.style.removeProperty("cursor");
      document.body.style.removeProperty("user-select");
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", stopDrag);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", stopDrag);
      stopDrag();
    };
  }, [onPointerMove]);

  if (!open) return null;

  /**
   * A dot per tab says "this stream is live". It does not pulse: a run lasts
   * hours, and a blinking marker in permanent chrome is an alarm that never
   * resolves. Only the genuinely transient states — a runtime coming up or
   * going down — animate.
   */
  const activity: Record<LogKind, { tone: StatusTone; pulse: boolean } | null> =
    {
      runtime:
        runtime?.status === "starting" || runtime?.status === "stopping"
          ? { tone: "warn", pulse: true }
          : runtime?.running
            ? { tone: "ok", pulse: false }
            : null,
      tuning: jobs?.tuning?.running ? { tone: "ok", pulse: false } : null,
      quantization: jobs?.quantization?.running
        ? { tone: "ok", pulse: false }
        : null,
    };

  return (
    <section
      aria-label={t("logs.title")}
      className="relative flex shrink-0 flex-col border-t border-border bg-card"
      style={{
        height: Math.min(MAX_DOCK_HEIGHT, Math.max(MIN_DOCK_HEIGHT, height)),
      }}
    >
      {/* Grip on the top edge; the dock is a workspace, not a fixed strip. */}
      <div
        role="separator"
        aria-orientation="horizontal"
        title={t("logs.resize")}
        onPointerDown={(event) => {
          dragFrom.current = { y: event.clientY, height };
          document.body.style.cursor = "row-resize";
          document.body.style.userSelect = "none";
        }}
        className="absolute inset-x-0 -top-1 z-10 h-2 cursor-row-resize"
      />

      <header className="flex shrink-0 items-center gap-2 px-3 py-2">
        <ScrollText className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="shrink-0 text-[12.5px] font-semibold">
          {t("logs.title")}
        </span>

        <div className="flex min-w-0 items-center gap-0.5 overflow-x-auto">
          {STREAMS.map((entry) => (
            <button
              key={entry.kind}
              type="button"
              onClick={() => setKind(entry.kind)}
              className={cn(
                "flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1 text-[12px] whitespace-nowrap transition-colors",
                entry.kind === kind
                  ? "bg-accent text-foreground"
                  : "text-muted-foreground hover:bg-muted",
              )}
            >
              {activity[entry.kind] && (
                <StatusDot
                  tone={activity[entry.kind]!.tone}
                  pulse={activity[entry.kind]!.pulse}
                />
              )}
              {t(entry.label)}
            </button>
          ))}
        </div>

        <div className="flex-1" />
        <Button size="xs" title={t("logs.hide")} onClick={hide}>
          <ChevronDown className="size-3.5" />
        </Button>
      </header>

      {hasAgent ? (
        <LogConsole
          stream={stream}
          endpoint={endpointFor(kind)}
          onClear={() => clear(backendId, kind)}
          className="min-h-0 flex-1 border-t border-border"
        />
      ) : (
        <p className="flex flex-1 items-center justify-center px-3 text-[11.5px] text-muted-foreground">
          {t("logs.noAgent")}
        </p>
      )}
    </section>
  );
}

/** Header affordance: the dock is only discoverable if something names it. */
export function LogDockToggle() {
  const { t } = useI18n();
  const open = useLogDock((s) => s.open);
  const toggle = useLogDock((s) => s.toggle);
  return (
    // Same treatment as the sidebar toggle at the other end of the bar: ghost,
    // 36px, 16px icon. Two bordered squares of different sizes at the two ends
    // of the same row read as two different kinds of control.
    <Button
      variant="ghost"
      size="icon-sm"
      className={cn("size-9", open && "bg-accent text-foreground")}
      title={t("logs.toggle")}
      aria-pressed={open}
      onClick={toggle}
    >
      <ScrollText className="size-4" />
    </Button>
  );
}
