import { useCallback, useEffect, useState } from "react";
import { TriangleAlert } from "lucide-react";
import { useCurrent } from "@/app/use-current";
import { PageHeader } from "@/components/common";
import { TrainingConfigCard } from "@/components/training/config-card";
import { TrainingJobBar } from "@/components/training/job-bar";
import { TrainingProgressCard } from "@/components/training/progress-card";
import { Notice } from "@/components/ui/primitives";
import {
  applyDevice,
  defaultTuning,
  effectiveOptimizer,
} from "@/lib/api/launcher";
import type {
  DeviceSelection,
  TuningConfig,
  TuningMethod,
} from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";
import { useAction } from "@/lib/use-action";
import { hasCapability } from "@/stores/backends";
import {
  resetTuningParameters,
  useTuningForm,
  useTuningFormEntry,
} from "@/stores/forms";
import { useNodeBusy, useNodes } from "@/stores/nodes";

/**
 * State / MiSS training. The page owns the node wiring and the rules that
 * decide whether a run may start; the four cards below it own their own
 * layout.
 */
export function TrainingPage() {
  const { t } = useI18n();
  const { backendId, backend, runtime, jobs } = useCurrent();
  const nodeBusy = useNodeBusy(backendId);
  const { config, devices } = useTuningFormEntry(backendId);
  const setField = useTuningForm((s) => s.set);
  const setDeviceSelection = useTuningForm((s) => s.setDevices);
  const run = useAction();
  const set = useCallback(
    (patch: Partial<TuningConfig>) => setField(backendId, patch),
    [backendId, setField],
  );
  const setDevices = useCallback(
    (selection: DeviceSelection) => setDeviceSelection(backendId, selection),
    [backendId, setDeviceSelection],
  );
  const resetParameters = useCallback(
    () => resetTuningParameters(backendId),
    [backendId],
  );
  const [samples, setSamples] = useState<number | null>(null);
  const [validating, setValidating] = useState(false);

  const tuning = jobs?.tuning;

  // A stale sample count must never outlive the path it described.
  useEffect(() => {
    setSamples(null);
  }, [config.data]);

  const supportsState = hasCapability(backend, "tuning_state");
  const supportsMiss = hasCapability(backend, "tuning_miss");
  const isMiss = config.method === "miss";

  const numbersValid =
    config.ctx >= 1 &&
    config.chunk >= 1 &&
    config.batch_size >= 1 &&
    config.batch_size <= 128 &&
    config.epochs >= 1 &&
    config.lr >= 0 &&
    config.lr_final >= 0 &&
    config.warmup_steps >= 0 &&
    config.save_every >= 0 &&
    config.seed >= 0 &&
    (!isMiss || (config.rank >= 1 && config.rank <= 1024 && config.alpha >= 0));

  /** Why Start is unavailable, in the order a person would fix it. */
  const startBlocker = !backendId
    ? t("training.needNode")
    : !supportsState || (isMiss && !supportsMiss)
      ? t("training.needCapability")
      : !config.model.trim()
        ? t("training.needModel")
        : !config.data.trim()
          ? t("training.needData")
          : !config.output.trim()
            ? t("training.needOutput")
            : !numbersValid
              ? t("training.needNumbers")
              : nodeBusy
                ? t("training.busy")
                : "";

  const canStart = startBlocker === "" && !tuning?.running;

  const deviceLabel =
    runtime?.visible_devices ||
    (devices.mode === "explicit" ? devices.value : "") ||
    t("common.auto");

  const trainerText = tuning?.running
    ? t("training.trainerRunning", { device: deviceLabel })
    : tuning?.status === "completed"
      ? t("training.trainerCompleted")
      : tuning?.status === "error"
        ? t("training.trainerError")
        : t("training.trainerIdle");

  /** MiSS and state tuning want different learning rates and output dirs. */
  const setMethod = (method: TuningMethod) => {
    if (method === config.method) return;
    if (method === "miss")
      set({
        method,
        optimizer: "adam",
        lr: 0.0001,
        lr_final: 0.00001,
        output: "./miss_output",
      });
    else
      set({
        method,
        lr: defaultTuning.lr,
        lr_final: defaultTuning.lr_final,
        output: "./state_output",
      });
  };

  const start = () => {
    if (!backendId) return;
    void run(() =>
      useNodes
        .getState()
        .startTuning(
          backendId,
          applyDevice(
            { ...config, optimizer: effectiveOptimizer(config) },
            devices,
          ),
        ),
    );
  };

  const stop = () => {
    if (!backendId) return;
    void run(() => useNodes.getState().stopJob(backendId, "tuning"));
  };

  const validate = async () => {
    if (!backendId || !config.data) return;
    setValidating(true);
    const count = await run(() =>
      useNodes.getState().validateDataset(backendId, config.data),
    );
    setSamples(count ?? null);
    setValidating(false);
  };

  return (
    <div className="max-w-[1180px] px-6 pt-5.5 pb-10">
      {/* No job actions in the page header: the app header's Start / Stop
          drive the inference runtime, and a second unlabelled pair right
          under them is what made the two look like one switch. */}
      <PageHeader
        eyebrow={backend?.name}
        title={t("training.title")}
        description={t("training.subtitle")}
      />

      {!supportsState && (
        <Notice
          tone="warning"
          className="mt-4"
          icon={<TriangleAlert className="size-3.5" />}
        >
          {t("training.unsupported")}
        </Notice>
      )}
      {isMiss && !supportsMiss && (
        <Notice
          tone="warning"
          className="mt-3"
          icon={<TriangleAlert className="size-3.5" />}
        >
          {t("training.missUnsupported")}
        </Notice>
      )}

      <TrainingJobBar
        method={config.method}
        onMethod={setMethod}
        canStart={canStart}
        blocker={startBlocker}
        running={Boolean(tuning?.running)}
        statusText={trainerText}
        onStart={start}
        onStop={stop}
      />

      {/* The progress column sticks: on a long MiSS form the curve stays in
          view while the fields under it are edited. */}
      <div className="mt-3.5 grid items-start gap-3.5 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] [&>*:last-child]:lg:sticky [&>*:last-child]:lg:top-4">
        <TrainingConfigCard
          backendId={backendId}
          backend={backend}
          config={config}
          set={set}
          devices={devices}
          setDevices={setDevices}
          samples={samples}
          validating={validating}
          onValidate={validate}
          onReset={resetParameters}
        />
        <TrainingProgressCard
          backendId={backendId}
          config={config}
          tuning={tuning}
        />
      </div>
    </div>
  );
}
