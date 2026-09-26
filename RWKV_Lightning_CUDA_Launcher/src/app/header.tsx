import { LogDockToggle } from "@/app/log-dock";
import { RailToggle } from "@/app/rail";
import { useI18n, type MessageKey } from "@/lib/i18n";
import { useRoute, type Route } from "@/lib/router";
import { usePageTitle } from "@/stores/ui";

const TITLES: Record<Route, MessageKey> = {
  nodes: "nodes.title",
  chat: "chat.title",
  translate: "translate.title",
  runtime: "runtime.title",
  training: "training.title",
  quant: "quant.title",
  settings: "settings.title",
};

export function Header() {
  const { t } = useI18n();
  const route = useRoute();
  // Pages carry their own <h1>; printing the same words here too was the
  // "节点总览 / 节点总览" stutter. Show the view name in the bar only when the
  // page has no heading of its own (Chat) or has scrolled past it.
  const inlineTitle = usePageTitle((s) => s.inlineVisible);

  return (
    <header className="flex h-13 shrink-0 items-center gap-3 border-b border-border bg-card px-3.5">
      <RailToggle />

      <div className="flex h-7 items-center gap-2">
        {/* Black swan mark; invert keeps the silhouette visible on the dark
            header (the source asset is a black-on-transparent cutout). */}
        <img
          src="/logo.png"
          alt=""
          className="size-[22px] object-contain dark:invert"
        />
        <span className="text-[13.5px] font-semibold tracking-[-0.01em]">
          {t("app.name")}
        </span>
      </div>

      {!inlineTitle && (
        <span className="animate-fade-in truncate text-[13.5px] font-medium">
          {t(TITLES[route])}
        </span>
      )}

      <div className="flex-1" />

      {/* The node's process chip used to sit here. Its commands belong to the
          node — the overview's detail panel and the rail's node card both show
          them next to the state they act on, and the bar keeps only what acts
          on the console itself. */}
      <LogDockToggle />
    </header>
  );
}
