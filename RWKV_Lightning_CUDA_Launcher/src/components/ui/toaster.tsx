import { AlertTriangle, Check, Info } from "lucide-react";
import { useToasts } from "@/stores/ui";
import { cn } from "@/lib/utils";

/** Fixed-position toast stack. Rendered once by the app shell. */
export function Toaster() {
  const toasts = useToasts((s) => s.toasts);
  const dismiss = useToasts((s) => s.dismiss);
  if (toasts.length === 0) return null;
  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-100 flex w-[min(360px,calc(100vw-32px))] flex-col gap-2">
      {toasts.map((toast) => (
        <button
          key={toast.id}
          type="button"
          onClick={() => dismiss(toast.id)}
          className={cn(
            "pointer-events-auto flex animate-fade-in items-start gap-2.5 rounded-xl border bg-card px-3.5 py-3 text-left shadow-lift",
            toast.variant === "error" && "border-destructive/50",
            toast.variant === "success" && "border-success/40",
            toast.variant === "default" && "border-border",
          )}
        >
          <span
            className={cn(
              "mt-px shrink-0",
              toast.variant === "error"
                ? "text-destructive"
                : toast.variant === "success"
                  ? "text-success"
                  : "text-info",
            )}
          >
            {toast.variant === "error" ? (
              <AlertTriangle className="size-4" />
            ) : toast.variant === "success" ? (
              <Check className="size-4" />
            ) : (
              <Info className="size-4" />
            )}
          </span>
          <span className="min-w-0 flex-1">
            <span className="block text-[12.5px] font-medium break-words">
              {toast.title}
            </span>
            {toast.description && (
              <span className="mt-0.5 block font-mono text-[11px] leading-relaxed break-all text-muted-foreground">
                {toast.description}
              </span>
            )}
          </span>
        </button>
      ))}
    </div>
  );
}
