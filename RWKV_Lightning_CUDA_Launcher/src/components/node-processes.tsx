import { Play, RotateCcw, ScrollText, Square } from "lucide-react";
import type { ReactNode } from "react";
import { useCurrent } from "@/app/use-current";
import { modelName, runtimeStateLabel } from "@/components/node-status";
import { runtimeButtonState } from "@/components/runtime-controls";
import { StatusDot, type StatusTone } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { applyDevice } from "@/lib/api/launcher";
import type {
  BackendView,
  Jobs,
  ProcessStatus,
  RuntimeState,
} from "@/lib/api/types";
import { useI18n, type MessageKey } from "@/lib/i18n";
import { useAction } from "@/lib/use-action";
import { cn } from "@/lib/utils";
import { useRuntimeFormEntry } from "@/stores/forms";
import { useNodes } from "@/stores/nodes";
import { useLogDock, type LogKind } from "@/stores/ui";

type TFn = (key: MessageKey, vars?: Record<string, string | number>) => string;

/**
 * A node runs three long-lived processes. Each one used to carry its own pair
 * of unlabelled buttons in whatever place happened to need them, which is how
 * the header's runtime Stop ended up sitting directly above the training Stop.
 * One descriptor per process, one row per descriptor, one name per button.
 */
export interface ProcessRow {
  key: LogKind;
  label: MessageKey;
  tone: StatusTone;
  /** Localised state word: 已就绪 / 训练中 step 11/630 / 未运行. */
  state: string;
  /** Second line, when the process has something to say (model, output). */
  detail?: string;
  startDisabled: boolean;
  stopDisabled: boolean;
  restartDisabled: boolean;
  /** Start is only offered where the configuration is in view. */
  startable: boolean;
}

function jobState(
  t: TFn,
  job?: ProcessStatus,
): { tone: StatusTone; state: string } {
  if (job?.running) {
    const step = job.progress?.step;
    return {
      tone: "ok",
      state: step
        ? t("training.steps", { step, total: job.progress?.total ?? 0 })
        : t("status.running"),
    };
  }
  if (job?.status === "error") return { tone: "bad", state: t("status.error") };
  if (job?.status === "completed")
    return { tone: "idle", state: t("status.completed") };
  return { tone: "idle", state: t("process.idle") };
}

/** Pure view model behind the process menu — the part worth pinning in tests. */
export function nodeProcessRows(
  t: TFn,
  input: {
    backend?: BackendView;
    runtime?: RuntimeState;
    jobs?: Jobs;
    hasAgent: boolean;
    busy: boolean;
    /** Whether the runtime form has enough to start (a model path). */
    canStartRuntime: boolean;
  },
): ProcessRow[] {
  const { backend, runtime, jobs, hasAgent, busy, canStartRuntime } = input;
  const reachable = Boolean(backend?.reachable) && hasAgent;
  const controls = runtimeButtonState({
    reachableAgent: reachable,
    busy,
    managed: Boolean(runtime?.managed),
    running: Boolean(runtime?.running),
    canStart: canStartRuntime,
  });

  const rows: ProcessRow[] = [
    {
      key: "runtime",
      // Its own key, not the page's: "推理服务" beside "训练 / 量化" made one
      // label twice as wide as the other two and shoved the states out of
      // line. The page keeps "推理服务"; this list says "推理".
      label: "process.runtime",
      tone: !reachable
        ? "bad"
        : runtime?.status === "ready"
          ? "ok"
          : runtime?.status === "error"
            ? "bad"
            : runtime?.status === "starting" || runtime?.status === "stopping"
              ? "warn"
              : "idle",
      state: runtimeStateLabel(t, runtime),
      detail: modelName(runtime) || undefined,
      startDisabled: controls.startDisabled,
      stopDisabled: controls.stopDisabled,
      restartDisabled: controls.restartDisabled,
      startable: true,
    },
  ];

  for (const [key, label, job] of [
    ["tuning", "logs.streamTuning", jobs?.tuning],
    ["quantization", "logs.streamQuantization", jobs?.quantization],
  ] as const) {
    const { tone, state } = jobState(t, job);
    rows.push({
      key,
      label,
      tone: reachable ? tone : "bad",
      state,
      detail: job?.checkpoint || undefined,
      startDisabled: true,
      // Stop is the one job command worth having everywhere: a run that has
      // to die does not wait for the right page to load.
      stopDisabled: !reachable || busy || !job?.running,
      restartDisabled: true,
      startable: false,
    });
  }

  return rows;
}

/**
 * One action slot. Only used where the four actions get fixed columns: a
 * process that does not offer one leaves its column empty, which keeps the
 * same action under itself down the list. The narrow popover packs to the
 * trailing edge instead — there the empty cell reads as a hole.
 */
function Slot({
  children,
  when,
}: {
  children?: ReactNode;
  /** Fixed-columns layout only; the packed one renders the child as-is. */
  when?: boolean;
}) {
  if (!when) return <>{children}</>;
  return children ? (
    <>{children}</>
  ) : (
    <span aria-hidden="true" className="size-7" />
  );
}

function ProcessActions({
  row,
  columns,
}: {
  row: ProcessRow;
  columns?: boolean;
}) {
  const { t } = useI18n();
  const { backendId } = useCurrent();
  const { config, devices } = useRuntimeFormEntry(backendId);
  const run = useAction();
  const show = useLogDock((s) => s.show);

  const nodes = () => useNodes.getState();
  return (
    // Right-packed, in the order start / restart / stop / logs. The rows offer
    // different sets of actions, and packing to the trailing edge lines a
    // button up by its distance from that edge — so the actions every process
    // shares (stop, logs) sit under their twins, and only the two that the
    // runtime alone offers are ever missing, at the left where the gap reads
    // as "not applicable here" rather than a hole between buttons.
    <span
      className={cn(
        "shrink-0 items-center gap-0.5",
        columns ? "grid grid-cols-4" : "flex",
      )}
    >
      <Slot when={columns}>
        {row.startable && (
        <Button
          size="icon"
          variant="ghost"
          title={t("common.start")}
          aria-label={t("common.start")}
          disabled={row.startDisabled}
          onClick={() =>
            void run(
              () =>
                nodes().startRuntime(
                  backendId,
                  applyDevice({ ...config }, devices),
                ),
              t("runtime.started"),
            )
          }
        >
          <Play className="size-3.5" />
        </Button>
        )}
      </Slot>
      <Slot when={columns}>
      {row.startable && (
        <Button
          size="icon"
          variant="ghost"
          title={t("common.restart")}
          aria-label={t("common.restart")}
          disabled={row.restartDisabled}
          onClick={() =>
            void run(
              () => nodes().restartRuntime(backendId),
              t("runtime.restarted"),
            )
          }
        >
          <RotateCcw className="size-3.5" />
        </Button>
      )}
      </Slot>
      <Slot when={columns}>
      <Button
        size="icon"
        variant="ghost"
        title={t("common.stop")}
        aria-label={t("common.stop")}
        disabled={row.stopDisabled}
        onClick={() =>
          void run(
            () =>
              row.key === "runtime"
                ? nodes().stopRuntime(backendId)
                : nodes().stopJob(backendId, row.key),
            t("runtime.stopped"),
          )
        }
      >
        <Square className="size-3.5" />
      </Button>
      </Slot>
      <Slot when={columns}>
      <Button
        size="icon"
        variant="ghost"
        title={t("logs.open")}
        aria-label={t("logs.open")}
        onClick={() => show(row.key)}
      >
        <ScrollText className="size-3.5" />
      </Button>
      </Slot>
    </span>
  );
}

/** The three process rows, shared by the header menu and the collapsed rail. */
export function NodeProcessList({
  className,
  columns,
}: {
  className?: string;
  /** Fixed columns per action: right for the wide detail panel. */
  columns?: boolean;
}) {
  const { t } = useI18n();
  const { backend, runtime, jobs, hasAgent, busy, backendId } = useCurrent();
  const { config } = useRuntimeFormEntry(backendId);
  const rows = nodeProcessRows(t, {
    backend,
    runtime,
    jobs,
    hasAgent,
    busy,
    canStartRuntime: Boolean(config.model_path.trim()),
  });

  if (!hasAgent)
    return (
      <p
        className={cn(
          "px-3 py-2.5 text-[11.5px] text-muted-foreground",
          className,
        )}
      >
        {t("process.noAgent")}
      </p>
    );

  return (
    <div className={cn("grid gap-0.5 p-2", className)}>
      {rows.map((row) => (
        // Grid, not flex: an auto column sizes to the model name's
        // max-content and pushes the buttons out through the popover's
        // rounded edge. minmax(0,1fr) is the column that gives way.
        <div
          key={row.key}
          className="grid grid-cols-[7px_minmax(0,1fr)_auto] items-center gap-x-2 rounded-lg px-2 py-1.5"
          title={row.startable ? undefined : t("process.jobPage")}
        >
          <StatusDot tone={row.tone} />
          <span className="block min-w-0">
            {/* One line: the state sits to the right of the name, a size down
                and centred against it. The row already carries a detail line
                below, and a second stacked line here only made the card
                busier. */}
            <span className="flex items-center gap-1.5">
              <span className="shrink-0 text-[12.5px] font-medium">
                {t(row.label)}
              </span>
              <span className="min-w-0 truncate text-[11px] text-muted-foreground">
                {row.state}
              </span>
            </span>
            {row.detail && (
              <span className="block truncate font-mono text-[10.5px] text-muted-foreground/80">
                {row.detail}
              </span>
            )}
          </span>
          <ProcessActions row={row} columns={columns} />
        </div>
      ))}
    </div>
  );
}
