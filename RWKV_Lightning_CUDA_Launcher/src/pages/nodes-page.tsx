import { useEffect, useRef, useState } from "react";
import {
  Boxes,
  Loader2,
  MoreHorizontal,
  Pencil,
  Plus,
  RefreshCw,
  Server,
  Trash2,
} from "lucide-react";
import {
  CapabilityBadges,
  GpuHeatGrid,
  gpuModelGroups,
  memorySplit,
  modelName,
  ownedDevices,
  runtimeStateLabel,
  runtimeStatus,
} from "@/components/node-status";
import { NodeAvatar } from "@/app/node-card";
import { NodeProcessList } from "@/components/node-processes";
import { EditBackendDialog } from "@/components/edit-backend-dialog";
import { PageHeader } from "@/components/common";
import { Badge, StatusDot } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
} from "@/components/ui/dialog";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { EmptyState, Notice } from "@/components/ui/primitives";
import { formatGigabytePair, formatRelativeTime } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import type { BackendView, GpuMetric } from "@/lib/api/types";
import { cn } from "@/lib/utils";
import { backendKindLabel, backendLabel, useBackends } from "@/stores/backends";
import { useNodes } from "@/stores/nodes";
import { toast, useUI } from "@/stores/ui";

export function NodesPage() {
  const { t } = useI18n();
  const list = useBackends((s) => s.list);
  const loaded = useBackends((s) => s.loaded);
  const listError = useBackends((s) => s.error);
  const currentId = useBackends((s) => s.currentId);
  const probeAll = useBackends((s) => s.probeAll);
  const refresh = useBackends((s) => s.refresh);
  const refreshAll = useNodes((s) => s.refreshAll);
  const snapshots = useNodes((s) => s.snapshots);
  const openAddBackend = useUI((s) => s.openAddBackend);
  const detailRef = useRef<HTMLDivElement>(null);

  // The overview needs every node's runtime + job state, not just the active one.
  useEffect(() => {
    const controller = new AbortController();
    void refreshAll(controller.signal);
    const timer = window.setInterval(() => void refreshAll(), 5000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [refreshAll, list.length]);

  const current = list.find((backend) => backend.id === currentId);
  const metrics = current ? snapshots[current.id]?.metrics : undefined;
  const gpus = metrics?.available ? metrics.gpus : [];
  const reachable = list.filter((b) => b.reachable).length;
  const jobCount = list.filter(
    (b) =>
      snapshots[b.id]?.jobs?.tuning?.running ||
      snapshots[b.id]?.jobs?.quantization?.running,
  ).length;
  const usedBytes = gpus.reduce(
    (sum, gpu) => sum + (gpu.memory_used_bytes ?? 0),
    0,
  );
  const totalBytes = gpus.reduce(
    (sum, gpu) => sum + (gpu.memory_total_bytes ?? 0),
    0,
  );

  return (
    <div className="w-full max-w-[1720px] px-6 pt-5.5 pb-10">
      <PageHeader
        title={t("nodes.title")}
        description={t("nodes.subtitle")}
        actions={
          <>
            {/* An icon: the registry already re-probes on its own timer, this
                only forces the next round. */}
            <Button
              size="icon-sm"
              variant="outline"
              title={t("backend.probeAll")}
              aria-label={t("backend.probeAll")}
              disabled={list.length === 0}
              onClick={async () => {
                await probeAll();
                await refresh();
                toast.success(t("backend.probeAll"));
              }}
            >
              <RefreshCw className="size-3.5" />
            </Button>
            <Button variant="default" onClick={openAddBackend}>
              <Plus className="size-3.5" />
              {t("nodes.addBackend")}
            </Button>
          </>
        }
      />

      {listError && (
        <Notice tone="danger" className="mt-5">
          {listError}
        </Notice>
      )}

      {loaded && list.length === 0 ? (
        <EmptyState
          className="mt-5"
          icon={<Server className="size-7" strokeWidth={1.5} />}
          title={t("backend.emptyTitle")}
          body={t("backend.emptyBody")}
          action={
            <Button variant="default" onClick={openAddBackend}>
              <Plus className="size-3.5" />
              {t("nodes.addBackend")}
            </Button>
          }
        />
      ) : (
        <>
          {/* The numbers, one line, in foreground: the console has no tone for
              "a number", and colouring totals is how orange crept in. */}
          <Summary
            pieces={[
              { value: `${reachable}/${list.length}`, label: t("nodes.sumReachable") },
              { value: String(gpus.length), label: t("nodes.sumGpus") },
              // The one piece whose label leads: "显存 289 / 765 GB" reads as a
              // quantity of memory, "289 / 765 GB 显存" does not.
              {
                value: totalBytes
                  ? formatGigabytePair(usedBytes, totalBytes, 0)
                  : "—",
                label: t("nodes.sumMemory"),
                labelFirst: true,
              },
              { value: String(jobCount), label: t("nodes.sumJobs") },
            ]}
          />

          {current && (
            <div ref={detailRef}>
              {/* 150ms cross-fade on the swap: keyed so React remounts it. */}
              <div key={current.id} className="animate-fade-in">
                <NodeDetail backend={current} />
              </div>
            </div>
          )}

          {/* One node is the whole registry: the grid would be a copy of the
              panel above it. */}
          {list.length > 1 && (
            <div className="mt-4 grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3.5">
              {list.map((backend) => (
                <NodeTile
                  key={backend.id}
                  backend={backend}
                  current={backend.id === currentId}
                  onPick={() =>
                    detailRef.current?.scrollIntoView({
                      behavior: "smooth",
                      block: "start",
                    })
                  }
                />
              ))}
            </div>
          )}

          {list.some((backend) => !backend.reachable) && (
            <p className="mt-4 flex items-center gap-2 text-[11.5px] text-muted-foreground">
              <Boxes className="size-3.5" />
              {t("nodes.unreachableHint")}
            </p>
          )}
        </>
      )}
    </div>
  );
}

interface SummaryPiece {
  value: string;
  label: string;
  /** Label on the left, for readings that are a quantity of something. */
  labelFirst?: boolean;
}

/** Counters in one line: the value in foreground, what it counts in muted. */
function Summary({ pieces }: { pieces: SummaryPiece[] }) {
  return (
    <p className="mt-4 flex flex-wrap items-center gap-x-2 gap-y-1 text-[12.5px]">
      {pieces.map((piece, index) => (
        <span key={piece.label} className="flex items-center gap-1.5">
          {index > 0 && (
            <span aria-hidden="true" className="text-muted-foreground">
              ·
            </span>
          )}
          {(piece.labelFirst ? ["label", "value"] : ["value", "label"]).map(
            (part) => (
              <span
                key={part}
                className={
                  part === "value"
                    ? "tabular-nums text-foreground"
                    : "text-muted-foreground"
                }
              >
                {part === "value" ? piece.value : piece.label}
              </span>
            ),
          )}
        </span>
      ))}
    </p>
  );
}

/** The active node, full width: services on the left, cards on the right. */
function NodeDetail({ backend }: { backend: BackendView }) {
  const { t } = useI18n();
  const snapshot = useNodes((s) => s.snapshots[backend.id]);
  const runtime = snapshot?.runtime;
  const metrics = snapshot?.metrics;
  const gpus = metrics?.available ? metrics.gpus : [];
  const status = runtimeStatus(
    backend.reachable,
    backend.kind,
    runtime?.status,
  );
  const statusLabel = backend.reachable
    ? status.key === "inferenceOnly"
      ? t("status.inferenceOnly")
      : runtimeStateLabel(t, runtime)
    : t("status.unreachable");
  const owned = ownedDevices(
    runtime,
    gpus.map((gpu) => gpu.index),
  );
  const smoothed = useSmoothedUtilization(metrics?.sampled_at, gpus);

  return (
    <Card
      className={cn("mt-3.5 flex flex-col", !backend.reachable && "opacity-60")}
    >
      <CardHeader>
        <NodeAvatar name={backend.name} />
        <span className="truncate text-sm font-semibold tracking-[-0.01em]">
          {backendLabel(t, backend)}
        </span>
        <StatusDot tone={status.tone} />
        <span className="truncate text-[11.5px] text-muted-foreground">
          {statusLabel}
        </span>
        <div className="flex-1" />
        <NodeMenu backend={backend} />
      </CardHeader>

      <CardContent className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
        {/* Services: the three long-lived processes and their four actions,
            each action in its own column. */}
        <NodeProcessList columns className="p-0" />

        {/* A container, not a viewport breakpoint: what the card rows need is
            the width of this column, and that depends on the panel's own
            layout, not on the window. */}
        <div className="@container min-w-0 lg:border-l lg:border-border lg:pl-4">
          <div className="flex items-baseline gap-2 px-2 pb-1 text-[11px] text-muted-foreground">
            {gpus.length > 0 && (
              <span className="truncate">
                {gpuModelGroups(gpus)
                  .map(({ count, name }) =>
                    t("nodes.gpuHeader", {
                      count,
                      name: name || t("nodes.gpuUnknown"),
                    }),
                  )
                  .join(" · ")}
              </span>
            )}
          </div>
          {gpus.length > 0 ? (
            // Two columns as soon as the column is wide enough: the list is
            // then about as tall as the three services beside it. One card
            // keeps the full width, and a narrow column falls back to one.
            // One rule down the middle, drawn once: a border per row would be
            // a segment per row, with the row gap and the rounding showing
            // through. It lives outside the scroller so it stays put while the
            // list moves under it.
            <span className="relative block">
              {gpus.length > 1 && (
                <span
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-y-0 left-1/2 hidden w-px bg-border @min-[30rem]:block"
                />
              )}
              <div
                className={cn(
                  // Capped at about the height of the three services beside
                  // it: one column then scrolls instead of stretching the
                  // panel, two columns fit eight cards without scrolling at
                  // all.
                  "grid max-h-32 gap-x-3 gap-y-0.5 overflow-y-auto",
                  gpus.length > 1 && "@min-[30rem]:grid-cols-2",
                )}
              >
                {gpus.map((gpu) => (
                  <GpuRow
                    key={gpu.index}
                    gpu={gpu}
                    owned={owned}
                    idle={
                      (smoothed(gpu.index) ??
                        gpu.utilization_percent ??
                        0) < IDLE_BELOW
                    }
                  />
                ))}
              </div>
            </span>
          ) : (
            <p className="px-2 text-[11px] text-muted-foreground">
              {metrics?.reason ?? t("runtime.gpuNoMetrics")}
            </p>
          )}
        </div>
      </CardContent>

      <CardFooter className="mt-auto">
        <CapabilityBadges capabilities={backend.capabilities} />
        <div className="flex-1" />
        <span className="text-[11px] text-muted-foreground">
          {t("backend.lastProbe")}{" "}
          {backend.last_probe
            ? formatRelativeTime(backend.last_probe)
            : t("backend.never")}
        </span>
      </CardFooter>
    </Card>
  );
}

/**
 * Each card's utilization averaged over the last ~30 seconds, built from the
 * samples this page already polls (metrics arrive every 6s, so five of them).
 * One instantaneous reading is a poor "is this card working" test — a card
 * between two batches reads as idle — and the payload carries no history, so
 * the smoothing has to happen here.
 */
function useSmoothedUtilization(
  sampledAt: number | undefined,
  gpus: GpuMetric[],
) {
  const history = useRef(new Map<number, number[]>());
  const lastSample = useRef(0);
  useEffect(() => {
    if (!sampledAt || sampledAt === lastSample.current) return;
    lastSample.current = sampledAt;
    for (const gpu of gpus) {
      const seen = history.current.get(gpu.index) ?? [];
      seen.push(gpu.utilization_percent ?? 0);
      if (seen.length > 5) seen.shift();
      history.current.set(gpu.index, seen);
    }
  }, [sampledAt, gpus]);

  return (index: number) => {
    const seen = history.current.get(index);
    if (!seen?.length) return undefined;
    return seen.reduce((sum, value) => sum + value, 0) / seen.length;
  };
}

/** Below this a card is idle, not merely between two batches. */
const IDLE_BELOW = 10;

/** One card's row in the detail: number, memory, utilization, temperature, power. */
function GpuRow({
  gpu,
  owned,
  idle,
  className,
}: {
  gpu: GpuMetric;
  owned: Set<number>;
  /** Averaged over ~30s: one reading is a poor test of "is this card working". */
  idle: boolean;
  className?: string;
}) {
  const split = memorySplit(gpu, owned);
  const share = split.used > 0 ? split.own / split.used : 0;
  const temp = gpu.temperature_c;
  // Idle rows recede as a whole — no colour needed to tell busy from idle. The
  // one colour allowed is the card's number, and only while it is idle. Blue,
  // not the green the status dots use for "the service is running": one colour
  // should not carry two meanings.
  const number = idle ? "text-info" : "text-muted-foreground";
  const faded = idle ? "text-muted-foreground" : "";
  return (
    <div
      className={cn(
        // Tight fixed columns and no monospace: the numbers set narrower in
        // the sans face, and every pixel they give up goes to the bar, which
        // is the one thing here that has to be readable at a glance.
        "grid grid-cols-[1.25rem_minmax(0,1fr)_2.25rem_2.75rem_3rem] items-center gap-x-1 rounded-lg px-2 py-1",
        className,
      )}
      title={gpu.name}
    >
      <span className={cn("text-[11.5px] tabular-nums", number)}>
        #{gpu.index}
      </span>
      <span className="h-[5px] overflow-hidden rounded-full bg-muted">
        <span
          className={cn(
            "flex h-full",
            split.used > 0 && split.own === 0 && "hatch-foreign",
          )}
          style={{ width: `${split.used}%` }}
        >
          <span className="bg-foreground" style={{ width: `${share * 100}%` }} />
          {share < 1 && <span className="hatch-foreign flex-1" />}
        </span>
      </span>
      <span className={cn("text-right text-[11.5px] tabular-nums", faded)}>
        {gpu.utilization_percent !== undefined
          ? `${gpu.utilization_percent}%`
          : "—"}
      </span>
      <span
        className={cn(
          "text-right text-[11.5px] tabular-nums",
          // Attention outranks the idle fade: a card held hot by someone else
          // is exactly the one worth looking at.
          temp !== undefined && temp > 85 ? "text-warning" : faded,
        )}
      >
        {temp !== undefined ? `${temp}°C` : "—"}
      </span>
      <span
        className={cn(
          "text-right text-[11.5px] tabular-nums",
          faded || "text-muted-foreground",
        )}
      >
        {gpu.power_watts !== undefined ? `${gpu.power_watts}W` : "—"}
      </span>
    </div>
  );
}

/** ⋯ : the actions that are not the card's own click. */
function NodeMenu({ backend }: { backend: BackendView }) {
  const { t } = useI18n();
  const [confirming, setConfirming] = useState(false);
  const [editing, setEditing] = useState(false);
  return (
    <>
      <Popover>
        <PopoverTrigger asChild>
          <Button
            size="icon-sm"
            variant="ghost"
            title={t("nodes.actions")}
            aria-label={t("nodes.actions")}
          >
            <MoreHorizontal className="size-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="end" className="w-[240px] p-1.5">
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-[13px] transition-colors hover:bg-muted"
          >
            <Pencil className="size-3.5" />
            {t("backend.edit")}
          </button>
          <ProbeRow backend={backend} />
          <button
            type="button"
            disabled={backend.id === "local"}
            onClick={() => setConfirming(true)}
            className={cn(
              "flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-[13px] transition-colors",
              backend.id === "local"
                ? "cursor-not-allowed text-muted-foreground"
                : "text-destructive hover:bg-destructive/10",
            )}
          >
            <Trash2 className="size-3.5" />
            {t("backend.remove")}
          </button>
        </PopoverContent>
      </Popover>
      <EditBackendDialog
        backend={backend}
        open={editing}
        onOpenChange={setEditing}
      />
      <RemoveDialog
        backend={backend}
        open={confirming}
        onOpenChange={setConfirming}
      />
    </>
  );
}

function ProbeRow({ backend }: { backend: BackendView }) {
  const { t } = useI18n();
  const probe = useBackends((s) => s.probe);
  const refresh = useNodes((s) => s.refresh);
  const name = backendLabel(t, backend);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          const view = await probe(backend.id);
          await refresh(backend.id);
          if (view.reachable) toast.success(t("backend.probed", { name }));
          else
            toast.error(
              t("backend.probeFailed", {
                error: view.probe_error || t("common.unknown"),
              }),
            );
        } catch (error) {
          toast.error(
            t("toast.failed", {
              error: error instanceof Error ? error.message : String(error),
            }),
          );
        }
      }}
      className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-[13px] transition-colors hover:bg-muted"
    >
      <RefreshCw className="size-3.5" />
      {t("backend.probe")}
    </button>
  );
}

function RemoveDialog({
  backend,
  open,
  onOpenChange,
}: {
  backend: BackendView;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useI18n();
  const remove = useBackends((s) => s.remove);
  const refresh = useBackends((s) => s.refresh);
  const name = backendLabel(t, backend);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent width="420px">
        <DialogHeader
          title={t("backend.remove")}
          description={t("backend.removeConfirm", { name })}
        />
        <div className="flex justify-end gap-2 border-t border-border px-4 py-3">
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            {t("common.cancel")}
          </Button>
          <Button
            variant="destructive"
            onClick={async () => {
              onOpenChange(false);
              try {
                await remove(backend.id);
                await refresh();
                toast.success(t("backend.removed", { name }));
              } catch (error) {
                toast.error(
                  t("toast.failed", {
                    error: error instanceof Error ? error.message : String(error),
                  }),
                );
              }
            }}
          >
            {t("common.delete")}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** One node in the grid. The whole card is the switch; the rest is the ⋯. */
function NodeTile({
  backend,
  current,
  onPick,
}: {
  backend: BackendView;
  current: boolean;
  onPick: () => void;
}) {
  const { t } = useI18n();
  const select = useBackends((s) => s.select);
  const snapshot = useNodes((s) => s.snapshots[backend.id]);
  const runtime = snapshot?.runtime;
  const metrics = snapshot?.metrics;
  const gpus = metrics?.available ? metrics.gpus : [];
  const status = runtimeStatus(
    backend.reachable,
    backend.kind,
    runtime?.status,
  );
  const owned = ownedDevices(
    runtime,
    gpus.map((gpu) => gpu.index),
  );
  const model = modelName(runtime);
  return (
    <button
      type="button"
      onClick={() => {
        if (current) return;
        select(backend.id);
        onPick();
      }}
      className={cn(
        "flex items-center gap-3 rounded-xl border bg-card p-3 text-left transition-colors",
        current ? "border-foreground" : "border-border hover:bg-muted",
        !backend.reachable && "opacity-60",
      )}
    >
      {/* The heat block leads the tile, and the words sit in one column
          beside it: the block is the node at a glance, the text is how to
          read it. A node without metrics keeps its avatar in that slot, so
          every tile starts at the same x. */}
      {gpus.length > 0 ? (
        <GpuHeatGrid gpus={gpus} owned={owned} />
      ) : (
        <NodeAvatar name={backend.name} />
      )}
      <span className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="flex items-center gap-2">
          <span className="min-w-0 flex-1 truncate text-[13px] leading-tight font-semibold tracking-[-0.01em]">
            {backendLabel(t, backend)}
          </span>
          {current && <Badge>{t("nodes.current")}</Badge>}
        </span>
        <span className="truncate font-mono text-[10.5px] leading-tight text-muted-foreground">
          {model || backend.base_url}
        </span>
        <span className="flex items-center gap-1.5 text-[11px] leading-tight text-muted-foreground">
          <StatusDot tone={status.tone} />
          <span className="truncate">
            {backend.reachable
              ? status.key === "inferenceOnly"
                ? t("status.inferenceOnly")
                : runtimeStateLabel(t, runtime)
              : t("status.unreachable")}
          </span>
          <span className="text-border">·</span>
          <span className="truncate">
            {backendKindLabel(backend.kind) || t("backend.kind.unknown")}
          </span>
        </span>
      </span>
    </button>
  );
}

/** Shared "no node selected" placeholder used by the operation pages. */
export function NoNodeNotice() {
  const { t } = useI18n();
  const backendId = useBackends((s) => s.currentId);
  const loading = useBackends((s) => s.loading);
  if (backendId) return null;
  return (
    <Notice tone="info" className="mt-5">
      <span className="inline-flex items-center gap-2">
        {loading && <Loader2 className="size-3.5 animate-spin" />}
        {t("backend.emptyTitle")}
      </span>
    </Notice>
  );
}
