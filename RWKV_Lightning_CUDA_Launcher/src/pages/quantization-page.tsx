import { useCallback, useState } from "react";
import { Cpu, Play, ScrollText, Square, TriangleAlert } from "lucide-react";
import { useCurrent } from "@/app/use-current";
import { CopyButton, PageHeader, PathField } from "@/components/common";
import { StatusDot } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Notice, Progress } from "@/components/ui/primitives";
import { Select } from "@/components/ui/select";
import { suggestedQuantizedPath } from "@/lib/api/launcher";
import type { QuantizationConfig, QuantizationFormat } from "@/lib/api/types";
import { formatDuration } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { useAction } from "@/lib/use-action";
import { hasCapability } from "@/stores/backends";
import { useQuantizationForm, useQuantizationFormEntry } from "@/stores/forms";
import { useNodeBusy, useNodes } from "@/stores/nodes";
import { useLogDock } from "@/stores/ui";

export function QuantizationPage() {
  const { t } = useI18n();
  const { backendId, backend, jobs } = useCurrent();
  const nodeBusy = useNodeBusy(backendId);
  const { config } = useQuantizationFormEntry(backendId);
  const setField = useQuantizationForm((s) => s.set);
  const set = useCallback(
    (patch: Partial<QuantizationConfig>) => setField(backendId, patch),
    [backendId, setField],
  );
  /** True while `output_path` is still the value we derived from the input. */
  const [autoOutput, setAutoOutput] = useState(false);
  const run = useAction();

  const quant = jobs?.quantization;
  const progress = quant?.progress ?? null;
  const outputPath = quant?.output_path ?? "";

  const supportsQuant = hasCapability(backend, "quantization");

  const canStart =
    Boolean(backendId) &&
    config.input_path.trim() !== "" &&
    config.output_path.trim() !== "" &&
    supportsQuant &&
    !nodeBusy &&
    !quant?.running;

  const derive = (
    patch: Partial<QuantizationConfig>,
    format: QuantizationFormat,
    input: string,
  ) => {
    if (!config.output_path || autoOutput) {
      patch.output_path = suggestedQuantizedPath(input, format);
      setAutoOutput(true);
    }
    return patch;
  };

  const onInputChange = (value: string) => {
    set(derive({ input_path: value }, config.format, value));
  };

  const onFormatChange = (value: string) => {
    const format: QuantizationFormat = value === "w8a16" ? "w8a16" : "w4a16";
    set(derive({ format }, format, config.input_path));
  };

  const start = () => {
    if (!backendId) return;
    void run(() => useNodes.getState().startQuantization(backendId, config));
  };

  const stop = () => {
    if (!backendId) return;
    void run(() => useNodes.getState().stopJob(backendId, "quantization"));
  };

  return (
    <div className="max-w-[760px] px-6 pt-5.5 pb-10">
      <PageHeader title={t("quant.title")} description={t("quant.subtitle")} />

      {!supportsQuant && (
        <Notice
          tone="warning"
          className="mt-4"
          icon={<TriangleAlert className="size-3.5" />}
        >
          {t("quant.unsupported")}
        </Notice>
      )}

      <Card className="mt-5">
        <CardContent className="grid gap-4 p-4">
          <PathField
            label={t("quant.input")}
            value={config.input_path}
            onChange={onInputChange}
            backendId={backendId}
            hostDialog={hasCapability(backend, "host_dialog")}
          />

          <Field label={t("quant.output")}>
            <Input
              value={config.output_path}
              spellCheck={false}
              onChange={(event) => {
                setAutoOutput(false);
                set({ output_path: event.target.value });
              }}
              className="font-mono text-xs"
            />
          </Field>

          <div className="grid gap-3 sm:grid-cols-2">
            <Field label={t("quant.format")}>
              <Select
                value={config.format}
                onChange={(event) => onFormatChange(event.target.value)}
              >
                <option value="w4a16">{t("quant.formatW4")}</option>
                <option value="w8a16">{t("quant.formatW8")}</option>
              </Select>
            </Field>
            <Field label={t("quant.groupSize")}>
              <Select
                value={config.group_size}
                disabled={config.format === "w8a16"}
                onChange={(event) =>
                  set({
                    group_size: Number(event.target.value) === 32 ? 32 : 128,
                  })
                }
              >
                <option value="128">{t("quant.group128")}</option>
                <option value="32">{t("quant.group32")}</option>
              </Select>
            </Field>
          </div>
        </CardContent>

        {/* Same order as the training bar: what to know on the left, the
            actions on the right with the primary one last. */}
        <CardFooter className="flex-wrap gap-y-2 bg-card px-4 py-3">
          <span className="flex min-w-0 flex-1 items-center gap-2 text-[12px] text-muted-foreground">
            {quant?.running ? (
              <StatusDot tone="warn" pulse />
            ) : (
              <Cpu className="size-3.5 shrink-0" />
            )}
            <span className="truncate">
              {quant?.running ? t("quant.running") : t("quant.footnote")}
            </span>
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => useLogDock.getState().show("quantization")}
          >
            <ScrollText className="size-3.5" />
            {t("logs.open")}
          </Button>
          {quant?.running ? (
            <Button size="sm" onClick={stop}>
              <Square className="size-3.5" />
              {t("common.stop")}
            </Button>
          ) : (
            <Button
              size="sm"
              variant="default"
              disabled={!canStart}
              onClick={start}
            >
              <Play className="size-3.5" />
              {t("quant.start")}
            </Button>
          )}
        </CardFooter>
      </Card>

      {quant && (quant.running || quant.checkpoint) && (
        <Card className="mt-3.5">
          <CardHeader>
            <CardTitle>{t("quant.progress")}</CardTitle>
            <div className="flex-1" />
            {quant.running && <StatusDot tone="warn" pulse />}
            <span className="font-mono text-[11.5px] text-muted-foreground">
              {formatDuration(quant.elapsed)}
            </span>
          </CardHeader>
          <CardContent className="grid gap-3">
            {progress?.total ? (
              <div className="grid gap-1.5">
                <span className="font-mono text-[11.5px] text-muted-foreground">
                  {t("training.steps", {
                    step: progress.step ?? 0,
                    total: progress.total,
                  })}
                </span>
                <Progress value={progress.step ?? 0} max={progress.total} />
              </div>
            ) : null}

            <div className="flex items-center gap-2">
              <span className="shrink-0 text-[11px] text-muted-foreground">
                {t("quant.outputPath")}
              </span>
              <span
                className="min-w-0 flex-1 truncate font-mono text-[11.5px]"
                title={outputPath}
              >
                {outputPath || "—"}
              </span>
              <CopyButton text={outputPath} size="xs" />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
