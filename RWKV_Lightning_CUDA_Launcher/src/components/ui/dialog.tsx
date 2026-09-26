import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { ComponentProps, ReactNode } from "react";
import { cn } from "@/lib/utils";

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;
export const DialogClose = DialogPrimitive.Close;

function DialogOverlay({
  className,
  ...props
}: ComponentProps<typeof DialogPrimitive.Overlay>) {
  return (
    <DialogPrimitive.Overlay
      className={cn(
        "fixed inset-0 z-80 bg-black/55 backdrop-blur-[2px]",
        "data-[state=open]:animate-fade-in",
        className,
      )}
      {...props}
    />
  );
}

export function DialogContent({
  className,
  children,
  width = "520px",
  ...props
}: ComponentProps<typeof DialogPrimitive.Content> & { width?: string }) {
  return (
    <DialogPrimitive.Portal>
      <DialogOverlay />
      <DialogPrimitive.Content
        style={{ width }}
        className={cn(
          "fixed top-1/2 left-1/2 z-80 max-h-[86vh] max-w-[calc(100vw-32px)] -translate-x-1/2 -translate-y-1/2",
          "flex flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-modal outline-none",
          className,
        )}
        {...props}
      >
        {children}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}

export function DialogHeader({
  title,
  description,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-start justify-between gap-4 border-b border-border px-4.5 py-4",
        className,
      )}
    >
      <div className="min-w-0">
        <DialogPrimitive.Title className="text-[15px] font-semibold tracking-[-0.01em]">
          {title}
        </DialogPrimitive.Title>
        {description && (
          <DialogPrimitive.Description className="mt-1 text-xs leading-relaxed text-muted-foreground">
            {description}
          </DialogPrimitive.Description>
        )}
      </div>
      <DialogPrimitive.Close
        className="-mt-0.5 -mr-1 rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        aria-label="Close"
      >
        <X className="size-4" />
      </DialogPrimitive.Close>
    </div>
  );
}

export function DialogBody({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn("min-h-0 flex-1 overflow-auto px-4.5 py-4", className)}
      {...props}
    />
  );
}

export function DialogFooter({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "flex items-center justify-end gap-2 border-t border-border bg-background px-4.5 py-3.5",
        className,
      )}
      {...props}
    />
  );
}
