import type { ComponentProps, ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Label({ className, ...props }: ComponentProps<"label">) {
  return (
    <label
      className={cn("block text-xs font-medium text-foreground", className)}
      {...props}
    />
  );
}

/**
 * Label + optional hint tooltip + control. Every form row in the console uses
 * this so spacing and hint affordances stay identical.
 */
export function Field({
  label,
  hint,
  htmlFor,
  children,
  className,
}: {
  label: ReactNode;
  hint?: string;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <Label htmlFor={htmlFor} className="mb-1.5 flex items-center gap-1">
        <span className="truncate">{label}</span>
        {hint && (
          <span
            title={hint}
            aria-label={hint}
            className="cursor-help text-[10px] text-muted-foreground/70 select-none"
          >
            ⓘ
          </span>
        )}
      </Label>
      {children}
    </div>
  );
}

export function CheckboxField({
  label,
  hint,
  className,
  ...props
}: ComponentProps<"input"> & { label: ReactNode; hint?: ReactNode }) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-center gap-2 text-[12.5px] text-foreground",
        props.disabled && "cursor-not-allowed opacity-60",
        className,
      )}
    >
      <input
        type="checkbox"
        className="size-[15px] shrink-0 rounded border-border"
        {...props}
      />
      <span className="min-w-0">{label}</span>
      {hint && (
        <span className="truncate text-[11.5px] text-muted-foreground">
          {hint}
        </span>
      )}
    </label>
  );
}

/** Segmented control (theme picker, tabs, think_type toggle). */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  disabled,
  className,
  size = "default",
}: {
  value: T;
  options: { value: T; label: ReactNode; disabled?: boolean }[];
  onChange: (value: T) => void;
  disabled?: boolean;
  className?: string;
  size?: "sm" | "default";
}) {
  // A pill tray with the chosen option filled in the foreground — the same
  // shape as the chat composer's thinking switch, so every "one of these"
  // choice in the console reads alike.
  return (
    <div
      role="radiogroup"
      className={cn(
        "inline-flex items-center gap-0.5 rounded-lg border border-border bg-background p-0.5",
        size === "sm" ? "h-8" : "h-[34px]",
        disabled && "opacity-60",
        className,
      )}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={disabled || option.disabled}
            onClick={() => onChange(option.value)}
            className={cn(
              "h-full flex-1 rounded-md px-3 text-[12.5px] font-medium whitespace-nowrap transition-colors disabled:cursor-not-allowed",
              size === "sm" && "px-2.5 text-xs",
              active
                ? "bg-foreground text-background"
                : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
