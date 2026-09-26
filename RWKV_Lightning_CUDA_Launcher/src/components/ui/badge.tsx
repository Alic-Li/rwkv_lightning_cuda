import { cva, type VariantProps } from "class-variance-authority";
import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex shrink-0 items-center gap-1.5 rounded-full border text-[11px] font-medium whitespace-nowrap",
  {
    variants: {
      variant: {
        default: "border-border bg-muted px-2 py-0.5 text-muted-foreground",
        outline:
          "border-border bg-transparent px-2 py-0.5 text-muted-foreground",
        mono: "border-border bg-muted px-1.5 py-0.5 font-mono text-[10.5px] text-muted-foreground",
        success: "border-transparent bg-success/15 px-2 py-0.5 text-success",
        warning: "border-transparent bg-warning/15 px-2 py-0.5 text-warning",
        danger:
          "border-transparent bg-destructive/15 px-2 py-0.5 text-destructive",
        info: "border-transparent bg-info/15 px-2 py-0.5 text-info",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export function Badge({
  className,
  variant,
  ...props
}: ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export type StatusTone = "ok" | "warn" | "bad" | "info" | "idle";

const toneClass: Record<StatusTone, string> = {
  ok: "bg-success",
  warn: "bg-warning",
  bad: "bg-destructive",
  info: "bg-info",
  idle: "bg-muted-foreground/50",
};

export function StatusDot({
  tone,
  pulse = false,
  className,
}: {
  tone: StatusTone;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-block size-[7px] shrink-0 rounded-full",
        toneClass[tone],
        pulse && "animate-pulse-dot",
        className,
      )}
    />
  );
}

/**
 * Dot + label pair used in headers, node cards and the runtime strip. `size`
 * picks between the quiet inline form and the strip's larger, foreground one;
 * `mono` is for values (paths, device lists) rather than words.
 */
export function StatusPill({
  tone,
  label,
  pulse,
  mono,
  size = "sm",
  className,
  labelClassName,
}: {
  tone: StatusTone;
  label: string;
  pulse?: boolean;
  mono?: boolean;
  size?: "sm" | "md";
  className?: string;
  labelClassName?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center",
        size === "md"
          ? "gap-2 text-[12.5px] font-medium"
          : "gap-1.5 text-[11.5px] text-muted-foreground",
        className,
      )}
    >
      <StatusDot tone={tone} pulse={pulse} />
      <span className={cn(mono && "truncate font-mono", labelClassName)}>
        {label}
      </span>
    </span>
  );
}
