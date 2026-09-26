import {
  Archive,
  ChartLine,
  Cpu,
  Languages,
  MessageSquare,
  PanelLeft,
  Server,
  Settings2,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { NodeCard } from "@/app/node-card";
import { useCurrent } from "@/app/use-current";
import { runtimeTone } from "@/components/node-status";
import { StatusDot, type StatusTone } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useI18n, type MessageKey } from "@/lib/i18n";
import { navigate, useRoute, type Route } from "@/lib/router";
import { cn } from "@/lib/utils";
import { useBackends } from "@/stores/backends";
import { useRail } from "@/stores/ui";

interface NavItem {
  route: Route;
  label: MessageKey;
  icon: LucideIcon;
}

const SECTIONS: { title: MessageKey; items: NavItem[] }[] = [
  {
    title: "nav.overview",
    items: [{ route: "nodes", label: "nav.nodes", icon: Server }],
  },
  {
    title: "nav.inference",
    items: [
      { route: "chat", label: "nav.chat", icon: MessageSquare },
      { route: "translate", label: "nav.translate", icon: Languages },
    ],
  },
  {
    title: "nav.operations",
    items: [
      { route: "runtime", label: "nav.runtime", icon: Cpu },
      { route: "training", label: "nav.training", icon: ChartLine },
      { route: "quant", label: "nav.quantization", icon: Archive },
    ],
  },
];

/** The rail's two widths. Icons sit at 24px — centred — in both. */
const RAIL_WIDTH = { open: "w-[216px]", closed: "w-16" } as const;

/**
 * The rail is one layout at two widths: collapsing narrows the aside and
 * clips the overflow, rather than switching to a centred, icon-only variant.
 * Ten px of rail padding plus fourteen of row padding fixes the icon column
 * at 24px — a 16px glyph centred in the 64px collapsed rail — and every state
 * change goes through width, so nothing in that column moves. No
 * `justify-content: center` in the collapsed state either: changing the
 * centring basis is what shifts icons by a few px and reads as "jitter" as
 * the width animates.
 */
export function Rail() {
  const { t } = useI18n();
  const route = useRoute();
  const collapsed = useRail((s) => s.collapsed);
  const backendCount = useBackends((s) => s.list.length);
  const { backend, runtime, jobs } = useCurrent();
  const trainingRunning = Boolean(jobs?.tuning?.running);
  const runtimeState: StatusTone =
    backend?.reachable === false ? "bad" : runtimeTone(runtime);

  return (
    <aside
      className={cn(
        "flex min-h-0 flex-col overflow-hidden border-r border-border bg-card p-2.5 transition-[width] duration-200",
        collapsed ? RAIL_WIDTH.closed : RAIL_WIDTH.open,
      )}
    >
      {/* The node card at the bottom is the one control that must never be
          pushed off a short window, so the nav scrolls and the card stays. */}
      <div className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-x-hidden overflow-y-auto">
        {SECTIONS.map((section) => (
          <div key={section.title} className="contents">
            <SectionTitle title={t(section.title)} collapsed={collapsed} />
            {section.items.map((item) => (
              <NavButton
                key={item.route}
                item={item}
                active={route === item.route}
                collapsed={collapsed}
                count={item.route === "nodes" ? backendCount : undefined}
                tone={
                  item.route === "runtime"
                    ? runtimeState
                    : item.route === "training" && trainingRunning
                      ? // Steady, not pulsing: training runs for hours.
                        "warn"
                      : undefined
                }
              />
            ))}
          </div>
        ))}
      </div>

      {/* flex, like the scrolling nav above: a <button> is a form control, so
          width:auto shrinks it to its label. Only the stretch of a flex
          column made the nav rows full width, which is why Settings came out
          a size smaller once it moved into this block. */}
      <div className="flex shrink-0 flex-col pt-2">
        <NavButton
          item={{ route: "settings", label: "nav.settings", icon: Settings2 }}
          active={route === "settings"}
          collapsed={collapsed}
        />
        <div className="mt-2">
          <NodeCard collapsed={collapsed} />
        </div>
      </div>
    </aside>
  );
}

/**
 * Section headings hold their row in both states. Collapsed, the words fade
 * but keep their line box and a short rule takes their place, so the icons
 * below keep their vertical spacing instead of sliding up. The rule is as
 * wide as the icon column, which keeps the collapsed rail one strip of
 * glyphs.
 */
function SectionTitle({
  title,
  collapsed,
}: {
  title: string;
  collapsed: boolean;
}) {
  return (
    <div className="shrink-0 pr-2.5 pt-3.5 pb-1 pl-[14px]">
      <div className="relative">
        <span
          className={cn(
            "block overflow-hidden text-[10.5px] font-semibold tracking-[0.06em] whitespace-nowrap text-muted-foreground transition-opacity",
            collapsed && "opacity-0",
          )}
        >
          {title}
        </span>
        {collapsed && (
          <span
            aria-hidden="true"
            className="absolute top-1/2 left-0 h-px w-4 -translate-y-1/2 bg-border"
          />
        )}
      </div>
    </div>
  );
}

function NavButton({
  item,
  active,
  collapsed,
  count,
  tone,
}: {
  item: NavItem;
  active: boolean;
  collapsed: boolean;
  count?: number;
  tone?: StatusTone;
}) {
  const { t } = useI18n();
  const Icon = item.icon;
  return (
    <button
      type="button"
      title={collapsed ? t(item.label) : undefined}
      onClick={() => navigate(item.route)}
      className={cn(
        // shrink-0: without it the column squashes the rows the moment the
        // node card grows, which changes each row's height between the two
        // rail states. The nav scrolls instead — that is what the scrolling
        // container above it is for.
        "flex shrink-0 items-center gap-2.5 overflow-hidden rounded-lg py-2 pr-2.5 pl-[14px] text-left text-[13px] font-medium transition-colors",
        active ? "bg-accent text-foreground" : "text-foreground hover:bg-muted",
      )}
    >
      {/* The icon is the row's first item of a left-padded row, so the column
          it sits in is fixed by construction. The count and the status dot
          hang off its corner: at the row's right edge the collapse either
          deleted them or left them to be clipped, and how many nodes are
          registered is not something to lose on the way in. */}
      <span className="relative shrink-0">
        <Icon className="size-4 text-muted-foreground" />
        {count !== undefined ? (
          <span className="absolute -top-1.5 -right-2 flex h-3 min-w-3 items-center justify-center rounded-full border border-border bg-card px-px font-mono text-[8.5px] leading-none text-muted-foreground tabular-nums">
            {count}
          </span>
        ) : tone ? (
          <StatusDot
            tone={tone}
            className="absolute -top-1 -right-1 ring-2 ring-card"
          />
        ) : null}
      </span>
      {/* Stays mounted while collapsed: it fades out and is clipped by the
          row's own width, so no other element re-centres around it. */}
      <span
        className={cn(
          "min-w-0 flex-1 truncate transition-opacity",
          collapsed && "opacity-0",
        )}
      >
        {t(item.label)}
      </span>
    </button>
  );
}

export function RailToggle() {
  const { t } = useI18n();
  const toggle = useRail((s) => s.toggle);
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      className="size-9"
      title={t("header.toggleSidebar")}
      onClick={toggle}
    >
      <PanelLeft className="size-4" />
    </Button>
  );
}
