import type { ComponentProps, ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Progress({
  value,
  max = 100,
  tone = "primary",
  className,
}: {
  value: number;
  max?: number;
  tone?: "primary" | "warning" | "success";
  className?: string;
}) {
  const safeMax = max > 0 ? max : 1;
  const pct = Math.min(100, Math.max(0, (value / safeMax) * 100));
  return (
    <div
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      className={cn(
        "h-1.5 w-full overflow-hidden rounded-full bg-muted",
        className,
      )}
    >
      <div
        className={cn(
          "h-full rounded-full transition-[width] duration-300",
          tone === "warning"
            ? "bg-warning"
            : tone === "success"
              ? "bg-success"
              : "bg-foreground",
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function Separator({
  className,
  orientation = "horizontal",
}: {
  className?: string;
  orientation?: "horizontal" | "vertical";
}) {
  return (
    <div
      role="separator"
      aria-orientation={orientation}
      className={cn(
        "shrink-0 bg-border",
        orientation === "horizontal" ? "h-px w-full" : "h-full w-px",
        className,
      )}
    />
  );
}

export function StatCard({
  label,
  value,
  tone = "default",
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  tone?: "default" | "success" | "warning" | "info";
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-card px-3.5 py-3",
        className,
      )}
    >
      <div className="text-[11.5px] text-muted-foreground">{label}</div>
      <div
        className={cn(
          "mt-1 text-[26px] leading-none font-semibold tracking-[-0.02em] tabular-nums",
          tone === "success" && "text-success",
          tone === "warning" && "text-warning",
          tone === "info" && "text-info",
        )}
      >
        {value}
      </div>
    </div>
  );
}

export function Metric({
  label,
  value,
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-background px-2.5 py-2.5",
        className,
      )}
    >
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="mt-1 font-mono text-[17px] leading-none font-semibold tabular-nums">
        {value}
      </div>
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  body,
  action,
  className,
}: {
  icon?: ReactNode;
  title: ReactNode;
  body?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border px-6 py-12 text-center",
        className,
      )}
    >
      {icon && <div className="text-muted-foreground/70">{icon}</div>}
      <div className="space-y-1">
        <p className="text-sm font-semibold">{title}</p>
        {body && (
          <p className="mx-auto max-w-md text-xs leading-relaxed text-muted-foreground">
            {body}
          </p>
        )}
      </div>
      {action}
    </div>
  );
}

/** Inline alert used for capability gaps, validation notes and errors. */
export function Notice({
  tone = "info",
  icon,
  children,
  className,
  action,
}: {
  tone?: "info" | "warning" | "danger" | "success";
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
}) {
  return (
    <div
      role={tone === "danger" ? "alert" : "status"}
      className={cn(
        "flex items-start gap-2.5 rounded-lg border px-3 py-2.5 text-xs leading-relaxed",
        tone === "danger" && "border-destructive/50 text-destructive",
        tone === "warning" && "border-warning/50 text-warning",
        tone === "success" && "border-success/50 text-success",
        tone === "info" && "border-border text-muted-foreground",
        className,
      )}
    >
      {icon && <span className="mt-px shrink-0">{icon}</span>}
      <div className="min-w-0 flex-1">{children}</div>
      {action}
    </div>
  );
}

export function CodeBlock({
  children,
  className,
  ...props
}: ComponentProps<"pre">) {
  return (
    <pre
      className={cn(
        "overflow-auto rounded-lg border border-border bg-muted px-3 py-2.5 font-mono text-[11.5px] leading-[1.75] break-all whitespace-pre-wrap text-muted-foreground",
        className,
      )}
      {...props}
    >
      {children}
    </pre>
  );
}
