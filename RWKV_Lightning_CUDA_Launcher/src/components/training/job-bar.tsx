import { Play, ScrollText, Square } from "lucide-react";
import { StatusDot } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Segmented } from "@/components/ui/field";
import type { TuningMethod } from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";
import { useLogDock } from "@/stores/ui";

/**
 * The page's only job controls. They live here, away from the app header's
 * runtime Start / Stop, and every label says "训练" so the two pairs can
 * never read as one switch. What is being trained sits on the left, what is
 * happening in the middle, and the actions on the right with the primary one
 * last — where a form's submit button is looked for.
 */
export function TrainingJobBar({
  method,
  onMethod,
  canStart,
  blocker,
  running,
  statusText,
  onStart,
  onStop,
}: {
  method: TuningMethod;
  onMethod: (method: TuningMethod) => void;
  canStart: boolean;
  /** Why Start is unavailable, already localised; empty when it is available. */
  blocker: string;
  running: boolean;
  statusText: string;
  onStart: () => void;
  onStop: () => void;
}) {
  const { t } = useI18n();
  return (
    <div className="mt-5 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-xl border border-border bg-card px-3 py-2.5 shadow-flat">
      <Segmented<TuningMethod>
        size="sm"
        value={method}
        onChange={onMethod}
        options={[
          { value: "state", label: t("training.stateTab") },
          { value: "miss", label: t("training.missTab") },
        ]}
      />
      <span className="flex min-w-0 flex-1 items-center gap-2 text-[12px] text-muted-foreground">
        {running && <StatusDot tone="warn" pulse />}
        <span className="truncate" title={blocker || undefined}>
          {running
            ? t("training.running")
            : blocker
              ? t("training.cannotStart", { reason: blocker })
              : statusText}
        </span>
      </span>
      <div className="flex items-center gap-1.5">
        <Button
          size="sm"
          variant="ghost"
          onClick={() => useLogDock.getState().show("tuning")}
        >
          <ScrollText className="size-3.5" />
          {t("training.logs")}
        </Button>
        {running ? (
          <Button size="sm" onClick={onStop}>
            <Square className="size-3.5" />
            {t("training.stop")}
          </Button>
        ) : (
          <Button
            variant="default"
            size="sm"
            disabled={!canStart}
            title={blocker || undefined}
            onClick={onStart}
          >
            <Play className="size-3.5" />
            {t("training.start")}
          </Button>
        )}
      </div>
    </div>
  );
}
