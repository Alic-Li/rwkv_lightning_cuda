import { useEffect, useMemo, useState } from "react";
import { CopyButton } from "@/components/common";
import { MetricChart, type ChartPoint } from "@/components/metric-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Segmented } from "@/components/ui/field";
import { Metric, Progress } from "@/components/ui/primitives";
import { Slider } from "@/components/ui/slider";
import type { ProcessStatus, TuningConfig } from "@/lib/api/types";
import { formatCount, formatDuration, formatNumber } from "@/lib/format";
import { useI18n, type MessageKey } from "@/lib/i18n";
import { useTrainingSeries } from "@/stores/training-series";

/** One curve at a time: lr and tok/s never share loss's axis. */
type MetricKey = "loss" | "lr" | "tps";

const METRIC_LABELS: Record<MetricKey, MessageKey> = {
  loss: "training.metricLoss",
  lr: "training.metricLr",
  tps: "training.metricTps",
};

const METRIC_FORMAT: Record<MetricKey, (value: number) => string> = {
  loss: (value) => value.toFixed(3),
  lr: (value) => value.toExponential(1),
  tps: (value) => formatCount(value),
};

/** Right column of the training page: how the current run is going. */
export function TrainingProgressCard({
  backendId,
  config,
  tuning,
}: {
  backendId: string;
  config: TuningConfig;
  tuning?: ProcessStatus;
}) {
  const { t } = useI18n();
  const [metric, setMetric] = useState<MetricKey>("loss");
  const [smoothing, setSmoothing] = useState(0.6);

  const progress = tuning?.progress ?? null;
  const losses = useMemo(() => tuning?.losses ?? [], [tuning?.losses]);
  const checkpoint = tuning?.checkpoint ?? "";

  // The Agent only reports loss as history; lr and tok/s arrive one poll at a
  // time, so the console keeps them for the life of the session.
  const recorded = useTrainingSeries((s) => s.series[backendId]);
  const recordPoint = useTrainingSeries((s) => s.record);
  const running = Boolean(tuning?.running);
  useEffect(() => {
    if (running) recordPoint(backendId, progress);
  }, [backendId, progress, recordPoint, running]);

  const series: Record<MetricKey, ChartPoint[]> = useMemo(() => {
    const pick = (
      read: (point: (typeof recorded)[number]) => number | undefined,
    ) =>
      (recorded ?? [])
        .filter((point) => Number.isFinite(read(point)))
        .map((point) => ({ step: point.step, value: read(point) as number }));
    return {
      loss: losses.map((point) => ({ step: point.step, value: point.loss })),
      lr: pick((point) => point.lr),
      tps: pick((point) => point.tps),
    };
  }, [losses, recorded]);

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle>{t("training.progress")}</CardTitle>
        <div className="flex-1" />
        <span className="font-mono text-[11.5px] text-muted-foreground">
          {t("training.epoch", {
            epoch: progress?.epoch ?? 0,
            epochs: progress?.epochs ?? config.epochs,
          })}
        </span>
        <span className="font-mono text-[11.5px] text-muted-foreground">
          {t("training.eta", { eta: formatDuration(progress?.eta) })}
        </span>
      </CardHeader>
      <CardContent className="grid gap-4 px-3.5 py-4">
        <div className="grid gap-1.5">
          <span className="font-mono text-[11.5px] text-muted-foreground">
            {t("training.steps", {
              step: progress?.step ?? 0,
              total: progress?.total ?? 0,
            })}
          </span>
          <Progress value={progress?.step ?? 0} max={progress?.total ?? 0} />
        </div>

        <div className="grid grid-cols-3 gap-2">
          <Metric label="loss" value={formatNumber(progress?.loss, 4)} />
          <Metric label="lr" value={formatNumber(progress?.lr, 4)} />
          <Metric
            label={t("training.tokensPerSecond")}
            value={formatCount(progress?.tokens_per_second)}
          />
        </div>

        <div>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Segmented<MetricKey>
              size="sm"
              value={metric}
              onChange={setMetric}
              options={[
                { value: "loss", label: t("training.metricLoss") },
                { value: "lr", label: t("training.metricLr") },
                { value: "tps", label: t("training.metricTps") },
              ]}
            />
            <div className="flex-1" />
            <div
              className="flex items-center gap-1 text-[11px] text-muted-foreground"
              title={t("training.chartHint")}
            >
              {t("training.smoothing")}
              <Slider
                label={t("training.smoothing")}
                min={0}
                max={0.95}
                step={0.05}
                value={smoothing}
                onChange={setSmoothing}
                className="w-[96px]"
              />
              <span className="w-7 shrink-0 text-right font-mono text-foreground tabular-nums">
                {smoothing.toFixed(2)}
              </span>
            </div>
          </div>
          <MetricChart
            points={series[metric]}
            smoothing={smoothing}
            label={t(METRIC_LABELS[metric])}
            format={METRIC_FORMAT[metric]}
            emptyLabel={t("training.chartEmpty")}
          />
        </div>

        <div className="flex items-center gap-2 border-t border-border pt-3">
          <span className="shrink-0 text-[11px] text-muted-foreground">
            {t("training.checkpoint")}
          </span>
          <span
            className="min-w-0 flex-1 truncate font-mono text-[11.5px]"
            title={checkpoint}
          >
            {checkpoint || t("training.noCheckpoint")}
          </span>
          <CopyButton text={checkpoint} size="xs" />
        </div>
      </CardContent>
    </Card>
  );
}
