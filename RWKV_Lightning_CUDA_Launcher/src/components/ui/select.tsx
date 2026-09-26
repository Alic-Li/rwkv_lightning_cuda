import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

export function Select({ className, ...props }: ComponentProps<"select">) {
  return (
    <select
      className={cn(
        "h-[34px] w-full min-w-0 appearance-none rounded-lg border border-border bg-background bg-[length:14px] bg-[right_0.55rem_center] bg-no-repeat px-2.5 pr-7 text-[12.5px] text-foreground transition-colors outline-none",
        "focus-visible:border-border-strong focus-visible:ring-2 focus-visible:ring-ring/40",
        "disabled:cursor-not-allowed disabled:opacity-60",
        className,
      )}
      style={{
        backgroundImage:
          "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2371717a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\")",
      }}
      {...props}
    />
  );
}

export function MonoSelect({ className, ...props }: ComponentProps<"select">) {
  return <Select className={cn("font-mono text-xs", className)} {...props} />;
}
