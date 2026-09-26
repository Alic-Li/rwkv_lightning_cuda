import type { ComponentProps, ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Card({ className, ...props }: ComponentProps<"section">) {
  return (
    <section
      className={cn(
        // overflow-hidden: a child with its own background (a footer strip,
        // say) paints its square corners over the card's rounded ones. Nothing
        // inside a card needs to escape it — popovers and dialogs portal out.
        "overflow-hidden rounded-xl border border-border bg-card shadow-flat",
        className,
      )}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "flex items-center gap-2.5 border-b border-border px-3.5 py-3",
        className,
      )}
      {...props}
    />
  );
}

export function CardTitle({ className, ...props }: ComponentProps<"h2">) {
  return (
    <h2
      className={cn("text-sm font-semibold tracking-[-0.01em]", className)}
      {...props}
    />
  );
}

export function CardDescription({ className, ...props }: ComponentProps<"p">) {
  return (
    <p
      className={cn("text-xs text-muted-foreground", className)}
      {...props}
    />
  );
}

export function CardContent({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("px-3.5 py-3", className)} {...props} />;
}

export function CardFooter({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 border-t border-border bg-background px-3.5 py-2.5",
        className,
      )}
      {...props}
    />
  );
}

/** Titled panel with an optional right-aligned hint or action. */
export function Panel({
  title,
  hint,
  action,
  children,
  className,
  contentClassName,
}: {
  title?: ReactNode;
  hint?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  return (
    <Card className={className}>
      {(title || action) && (
        <CardHeader className="justify-between">
          <div className="flex min-w-0 items-baseline gap-2">
            {title && <CardTitle className="truncate">{title}</CardTitle>}
            {hint && (
              <span className="truncate text-[11px] text-muted-foreground">
                {hint}
              </span>
            )}
          </div>
          {action}
        </CardHeader>
      )}
      <CardContent className={cn("px-3.5 py-4", contentClassName)}>
        {children}
      </CardContent>
    </Card>
  );
}
