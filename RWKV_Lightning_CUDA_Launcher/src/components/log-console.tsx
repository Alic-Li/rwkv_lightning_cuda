import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDownToLine, Eraser, Search } from "lucide-react";
import { CopyButton } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StatusDot, type StatusTone } from "@/components/ui/badge";
import { useI18n, type MessageKey } from "@/lib/i18n";
import type { LogStream } from "@/stores/logs";
import { cn } from "@/lib/utils";

const ERROR_LINE = /error|failed|exception|out of memory|traceback/i;

const STATUS: Record<
  LogStream["status"],
  { tone: StatusTone; key: MessageKey }
> = {
  open: { tone: "ok", key: "logs.statusOpen" },
  connecting: { tone: "warn", key: "logs.statusConnecting" },
  idle: { tone: "idle", key: "logs.statusIdle" },
  error: { tone: "bad", key: "logs.statusError" },
};

/**
 * Filter, follow-tail and the line list — everything a log pane does once
 * someone else has decided which stream it is showing. The dock owns the
 * stream choice; this owns reading it.
 */
export function LogConsole({
  stream,
  endpoint,
  empty,
  onClear,
  leading,
  className,
}: {
  stream: LogStream;
  endpoint?: string;
  empty?: string;
  onClear?: () => void;
  /** Stream picker or title, rendered at the start of the toolbar. */
  leading?: React.ReactNode;
  className?: string;
}) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [follow, setFollow] = useState(true);
  const bodyRef = useRef<HTMLDivElement>(null);
  const status = STATUS[stream.status];

  const visible = useMemo(() => {
    if (!query.trim()) return stream.lines;
    const needle = query.toLowerCase();
    return stream.lines.filter((line) => line.toLowerCase().includes(needle));
  }, [stream.lines, query]);

  useEffect(() => {
    if (!follow) return;
    const node = bodyRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [visible.length, follow]);

  return (
    <div className={cn("flex min-h-0 flex-col", className)}>
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-3 py-2">
        {leading}
        <span
          className="flex shrink-0 items-center gap-1.5 text-[11px] text-muted-foreground"
          title={stream.error || undefined}
        >
          <StatusDot
            tone={status.tone}
            pulse={stream.status === "connecting"}
          />
          {t(status.key)}
        </span>
        {endpoint && (
          <span className="hidden truncate font-mono text-[11px] text-muted-foreground/80 xl:inline">
            {endpoint}
          </span>
        )}
        <div className="flex-1" />
        <span className="shrink-0 font-mono text-[11px] text-muted-foreground/80">
          {t("logs.lineCount", { count: stream.lines.length })}
        </span>
        <div className="relative w-[168px] shrink-0">
          <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground/70" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t("common.filter")}
            className="h-7 pl-7 text-xs"
          />
        </div>
        <Button
          size="xs"
          title={t("common.follow")}
          aria-pressed={follow}
          onClick={() => setFollow((value) => !value)}
          className={cn(follow && "bg-accent text-foreground")}
        >
          <ArrowDownToLine className="size-3.5" />
        </Button>
        <CopyButton text={visible.join("\n")} size="xs" label="" />
        {onClear && (
          <Button size="xs" title={t("logs.clear")} onClick={onClear}>
            <Eraser className="size-3.5" />
          </Button>
        )}
      </div>

      <div
        ref={bodyRef}
        onScroll={(event) => {
          const node = event.currentTarget;
          setFollow(
            node.scrollHeight - node.scrollTop - node.clientHeight < 40,
          );
        }}
        className="min-h-0 flex-1 overflow-auto bg-background px-3.5 py-2.5 font-mono text-[11.5px] leading-[1.85]"
      >
        {stream.status === "error" && stream.error && (
          <p className="mb-1 break-all text-destructive">{stream.error}</p>
        )}
        {visible.length === 0 ? (
          <p className="text-muted-foreground">
            {empty ?? t("runtime.logsEmpty")}
          </p>
        ) : (
          visible.map((line, index) => (
            <div
              key={`${index}-${line.slice(0, 24)}`}
              className={cn(
                "break-all whitespace-pre-wrap",
                ERROR_LINE.test(line)
                  ? "text-destructive"
                  : "text-muted-foreground",
              )}
            >
              {line || " "}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
