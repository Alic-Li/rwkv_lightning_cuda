import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

export function Input({ className, ...props }: ComponentProps<"input">) {
  return (
    <input
      className={cn(
        "flex h-[34px] w-full min-w-0 rounded-lg border border-border bg-background px-2.5 text-[12.5px] text-foreground transition-colors outline-none",
        "placeholder:text-muted-foreground/70 focus-visible:border-border-strong focus-visible:ring-2 focus-visible:ring-ring/40",
        "disabled:cursor-not-allowed disabled:opacity-60",
        "file:mr-2 file:h-6 file:rounded-md file:border file:border-border file:bg-muted file:px-2 file:text-xs file:text-foreground",
        className,
      )}
      {...props}
    />
  );
}

export function MonoInput({ className, ...props }: ComponentProps<"input">) {
  return <Input className={cn("font-mono text-xs", className)} {...props} />;
}

export function Textarea({ className, ...props }: ComponentProps<"textarea">) {
  return (
    <textarea
      className={cn(
        "flex w-full min-w-0 rounded-lg border border-border bg-background px-2.5 py-2 text-[12.5px] leading-relaxed text-foreground transition-colors outline-none",
        "placeholder:text-muted-foreground/70 focus-visible:border-border-strong focus-visible:ring-2 focus-visible:ring-ring/40",
        "disabled:cursor-not-allowed disabled:opacity-60",
        className,
      )}
      {...props}
    />
  );
}
