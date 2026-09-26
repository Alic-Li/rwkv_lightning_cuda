import { Check, ChevronsUpDown, PanelLeft, Plus, Server } from "lucide-react";
import { useCurrent } from "@/app/use-current";
import { NodeProcessList } from "@/components/node-processes";
import {
  GpuHeatGrid,
  gpuUnavailableReason,
  ownedDevices,
  runtimeHint,
  runtimeLabel,
  runtimeStatus,
} from "@/components/node-status";
import { StatusDot, StatusPill, type StatusTone } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverClose,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { formatGigabytePair, formatRelativeTime } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { backendKindLabel, backendLabel, useBackends } from "@/stores/backends";
import { useNodes } from "@/stores/nodes";
import { useRail, useUI } from "@/stores/ui";

/**
 * Bottom-of-rail node card: one card at two rail widths.
 *
 * It holds exactly two rows in both states — the node avatar and the GPU heat
 * grid under it — because it hangs off the rail's bottom edge. A row that
 * appeared or disappeared with the collapse would push the settings row and
 * the card itself up and down again, which is the jump this card used to
 * have. Everything that does not fit at 44px wide, held or not — the URL, the
 * kind and runtime chips, the per-GPU rows, the process rows and the node
 * switcher — is in the popover, and the popover is the same one in both
 * states.
 */
export function NodeCard({ collapsed }: { collapsed: boolean }) {
  const { t } = useI18n();
  const { backend, runtime, metrics, metricsError } = useCurrent();
  const list = useBackends((s) => s.list);
  const currentId = useBackends((s) => s.currentId);
  const select = useBackends((s) => s.select);
  const snapshots = useNodes((s) => s.snapshots);
  const openAddBackend = useUI((s) => s.openAddBackend);
  const setCollapsed = useRail((s) => s.setCollapsed);

  const status = backend
    ? runtimeStatus(backend.reachable, backend.kind, runtime?.status)
    : undefined;
  const statusLabel = status
    ? status.key === "unreachable"
      ? t("status.unreachable")
      : status.key === "inferenceOnly"
        ? t("status.inferenceOnly")
        : runtimeLabel(t, runtime)
    : "";
  const kindLabel =
    backendKindLabel(backend?.kind ?? "") || t("backend.kind.unknown");

  // Same trap as the header badge: the status line below the node name
  // describes the inference server, not the box, so the tooltip says which
  // one is down.
  const hint =
    backend && !backend.reachable
      ? backend.probe_error || t("status.unreachable")
      : runtimeHint(t, runtime);

  const gpus = metrics?.available ? metrics.gpus : [];
  const average = gpus.length
    ? Math.round(
        gpus.reduce((sum, gpu) => sum + (gpu.utilization_percent ?? 0), 0) /
          gpus.length,
      )
    : 0;
  const usedBytes = gpus.reduce(
    (sum, gpu) => sum + (gpu.memory_used_bytes ?? 0),
    0,
  );
  const totalBytes = gpus.reduce(
    (sum, gpu) => sum + (gpu.memory_total_bytes ?? 0),
    0,
  );
  // Two readings the grid cannot show: how hard the box is working, and how
  // much of its memory is allocated. Whole gigabytes — eight cards summed to
  // one decimal each is noise.
  const utilizationLine = gpus.length
    ? `${average}%`
    : gpuUnavailableReason(t, metrics, metricsError, backend);
  const memoryLine = totalBytes
    ? formatGigabytePair(usedBytes, totalBytes, 0)
    : "";
  // The tallest card decides: averaging a hot one away is how a box cooks
  // without the console mentioning it.
  const temps = gpus
    .map((gpu) => gpu.temperature_c)
    .filter((c): c is number => c !== undefined);
  const temperatureLine = temps.length ? `${Math.max(...temps)}°C` : "";
  const idleCards = gpus.filter(
    (gpu) => (gpu.utilization_percent ?? 0) <= 5,
  ).length;
  const hottest = temps.length ? Math.max(...temps) : undefined;
  // One line per row of the heat block beside them: a four-card node is two
  // cells deep and gets two lines, an eight-card one four. A single card fills
  // the whole block, and so does the placeholder, so both get all four.
  const lineCount =
    gpus.length > 1 ? Math.min(4, Math.ceil(gpus.length / 2)) : 4;
  const hot = hottest !== undefined && hottest > 85;
  const readings = [
    // Priority order: what gets cut first is at the bottom.
    {
      id: "utilization",
      label: gpus.length ? t("rail.utilization") : "",
      value: utilizationLine,
    },
    {
      id: "memory",
      label: memoryLine ? t("rail.memory") : "",
      value: memoryLine,
    },
    {
      id: "temperature",
      label: temperatureLine ? t("rail.temperature") : "",
      value: temperatureLine,
      warn: hot,
    },
    {
      id: "idle",
      label: gpus.length ? t("rail.idleCards") : "",
      value: gpus.length ? `${idleCards} / ${gpus.length}` : "",
    },
  ]
    // A card over the line is never the one cut: it jumps the queue.
    .sort((a, b) => Number(Boolean(b.warn)) - Number(Boolean(a.warn)))
    .slice(0, lineCount)
    // Longest label first, stable for ties.
    .sort((a, b) => b.label.length - a.label.length);
  const owned = ownedDevices(
    runtime,
    gpus.map((gpu) => gpu.index),
  );

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-background">
      <Popover>
        <PopoverTrigger asChild>
          {/* 8px of padding centres the card's 26px content column — the width
              of the heat block and the node tile — in the 44px card. */}
          <button
            type="button"
            title={hint}
            className="block w-full overflow-hidden px-2 py-2.5 text-left transition-colors hover:bg-muted"
            aria-label={t("backend.registered")}
            aria-haspopup="dialog"
          >
            {/* The identity block: the node's tile spanning the name and the
                service state stacked beside it, tight against each other so
                the tile's height is exactly those two lines. */}
            <span className="flex items-stretch gap-2">
              <NodeAvatar name={backend?.name} tone={status?.tone} />
              <span className="flex min-w-0 flex-1 flex-col justify-center">
                <span
                  className={cn(
                    "truncate text-[13px] leading-tight font-semibold tracking-[-0.01em] transition-opacity",
                    collapsed && "opacity-0",
                  )}
                >
                  {backend?.name ?? t("backend.emptyTitle")}
                </span>
                <span
                  className={cn(
                    // -2px: the leading above the name and below the state is
                    // the only air in the pair, and pulling the two lines
                    // together is what keeps the tile beside them square.
                    "-mt-0.5 truncate text-[11.5px] leading-tight text-muted-foreground transition-opacity",
                    collapsed && "opacity-0",
                  )}
                >
                  {statusLabel}
                </span>
              </span>
              <ChevronsUpDown
                className={cn(
                  "size-3.5 shrink-0 self-center text-muted-foreground transition-opacity",
                  collapsed && "opacity-0",
                )}
              />
            </span>

            {/* The heat block, with the numbers a per-cell colour cannot
                show. Same rows in both states: the collapse cuts the right
                side off, it does not take a row away. */}
            <span className="mt-2 flex items-center gap-2">
              {gpus.length > 0 ? (
                <GpuHeatGrid gpus={gpus} owned={owned} />
              ) : (
                // Keeps the block's footprint on a node whose metrics have not
                // arrived yet, so the card does not resize under the pointer.
                <span className="h-[70px] w-[26px]" />
              )}
              {/* Longest label first: the labels are left-aligned and the
                  values follow them, so ordering by length keeps the eye from
                  stepping in and out on every line. Sorted at render time, so
                  each language gets its own order. */}
              <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                {readings.map((line) => (
                  <span
                    key={line.id}
                    className={cn(
                      "flex min-w-0 items-baseline gap-1.5 text-[10.5px] transition-opacity",
                      collapsed && "opacity-0",
                    )}
                  >
                    {line.label && (
                      <span className="shrink-0 text-muted-foreground">
                        {line.label}
                      </span>
                    )}
                    <span
                      className={cn(
                        "min-w-0 truncate tabular-nums",
                        line.warn && "text-warning",
                      )}
                    >
                      {line.value}
                    </span>
                  </span>
                ))}
              </span>
            </span>
          </button>
        </PopoverTrigger>

        <PopoverContent
          side={collapsed ? "right" : "top"}
          align="start"
          className="w-[320px] p-1.5"
        >
          {/* Flat: the popover is already the container, so the node's
              details, the GPU rows and the process rows sit directly in it,
              separated by rules rather than by a second rounded box. */}
          <div className="px-2.5 pt-2.5 pb-2">
              <div className="flex items-center gap-2">
                <NodeAvatar name={backend?.name} />
                <span className="min-w-0 flex-1 truncate text-[13px] font-semibold tracking-[-0.01em]">
                  {backend?.name ?? t("backend.emptyTitle")}
                </span>
              </div>
              <p className="mt-1.5 truncate font-mono text-[10.5px] text-muted-foreground">
                {backend?.base_url ?? "—"}
              </p>
              <div className="mt-1.5 flex items-center gap-2">
                <span className="inline-flex h-[18px] items-center rounded-md bg-muted px-1.5 text-[10px] text-muted-foreground">
                  {kindLabel}
                </span>
                {status && (
                  <StatusPill
                    tone={status.tone}
                    label={statusLabel}
                    className="min-w-0"
                  />
                )}
              </div>
          </div>

          {/* Node switching and the service commands; the per-card numbers
              moved to the overview's detail panel, which has room to show them
              as a table instead of a scrolling list. */}
          <div className="border-t border-border">
            <NodeProcessList />
          </div>

          <p className="border-t border-border px-2 pt-2 pb-1.5 text-[11px] font-semibold tracking-[0.04em] text-muted-foreground">
            {t("backend.registered")}
          </p>
          {list.map((item) => {
            // The tone the card's own dot uses. A second colour vocabulary for
            // the same node in the same popover was two answers to one
            // question; the current node is marked by a check instead.
            const itemStatus = runtimeStatus(
              item.reachable,
              item.kind,
              snapshots[item.id]?.runtime?.status,
            );
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => select(item.id)}
                className={cn(
                  "grid w-full grid-cols-[8px_1fr_auto] items-center gap-2.5 rounded-lg px-2 py-2 text-left transition-colors hover:bg-muted",
                  item.id === currentId && "bg-accent",
                )}
              >
                <StatusDot tone={itemStatus.tone} />
                <span className="block min-w-0">
                  <span className="block truncate text-[13px] font-medium">
                    {backendLabel(t, item)}
                  </span>
                  <span className="block truncate font-mono text-[11px] text-muted-foreground">
                    {item.base_url}
                  </span>
                </span>
                <span className="flex shrink-0 items-center gap-1.5 text-[11px] text-muted-foreground">
                  {item.id === currentId && (
                    <Check
                      aria-hidden="true"
                      className="size-3 text-foreground"
                    />
                  )}
                  {item.reachable
                    ? formatRelativeTime(item.last_probe)
                    : t("status.unreachable")}
                </span>
              </button>
            );
          })}
          <div className="mx-1 my-1.5 h-px bg-border" />
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start gap-2"
            onClick={openAddBackend}
          >
            <Plus className="size-3.5" />
            {t("backend.add")}
          </Button>

          {collapsed && (
            <div className="mt-1.5 flex border-t border-border p-2">
              <div className="flex-1" />
              <PopoverClose asChild>
                <Button
                  size="xs"
                  title={t("header.expandSidebar")}
                  onClick={() => setCollapsed(false)}
                >
                  <PanelLeft className="size-3" />
                </Button>
              </PopoverClose>
            </div>
          )}
        </PopoverContent>
      </Popover>
    </div>
  );
}

/**
 * The node's identity in one glyph: its initial in a tile. It stretches to
 * the height of the name and the service state beside it, so the tile reads
 * as one block with the two lines; 26px wide against ~29px tall keeps it
 * square-ish, and a 6px radius keeps it a rounded rectangle rather than the
 * tall ellipse a larger radius made of it. The status dot rides this tile's
 * corner in both rail states: it belongs to the node, and while the rail is
 * closed it is the only thing that carries the tone.
 */
export function NodeAvatar({ name, tone }: { name?: string; tone?: StatusTone }) {
  return (
    <span className="relative w-[26px] shrink-0">
      <span className="flex h-full min-h-6 w-full items-center justify-center rounded-[6px] bg-muted text-[12px] font-semibold text-muted-foreground">
        {name ? (
          name.trim().slice(0, 1).toUpperCase()
        ) : (
          <Server className="size-3.5" />
        )}
      </span>
      {tone && (
        <StatusDot
          tone={tone}
          className="absolute -top-1 -right-1 ring-2 ring-background"
        />
      )}
    </span>
  );
}
