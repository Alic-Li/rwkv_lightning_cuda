import { useCurrent } from "@/app/use-current";
import { compactGpuName } from "@/components/node-status";
import { Field } from "@/components/ui/field";
import { MonoInput } from "@/components/ui/input";
import { MonoSelect, Select } from "@/components/ui/select";
import type { DeviceSelection, GpuMetric } from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";

/**
 * Device picker shared by Runtime and Training — quantization has none, since
 * rwkv_quantize never touches a GPU. The UI offers two modes: 自动 (the
 * Agent's §5.8 chain: --card, inherited env, else free-VRAM placement) and
 * 指定 (a dropdown fed by the node's GPU metrics). The wire keeps the
 * tri-state DeviceSelection (the empty "inject nothing" mode stays API-only).
 *
 * When the node's launcher is pinned by --card (`runtime.card`), 指定 locks to
 * the pinned spec: the Agent refuses anything else with 400, so offering the
 * other cards would only set up a guaranteed failure.
 */
export function DeviceSelector({
  label,
  selection,
  onChange,
  className,
  retuneHint = false,
}: {
  label: string;
  selection: DeviceSelection;
  onChange: (selection: DeviceSelection) => void;
  className?: string;
  /**
   * Runtime only: moving the runtime to a different card changes the W8A16
   * tune-cache path, so the first start there retunes. Training does not use
   * that cache, and quantization does not use a GPU at all.
   */
  retuneHint?: boolean;
}) {
  const { t } = useI18n();
  const { runtime, metrics } = useCurrent();
  const pinned = runtime?.card?.trim() ?? "";
  const gpus = metrics?.available ? metrics.gpus : [];
  const current = selection.mode === "explicit" ? selection.value : "";

  const switchMode = (mode: DeviceSelection["mode"]) => {
    if (mode === "explicit") {
      const fallback =
        pinned ||
        current ||
        (gpus.length > 0 ? String(freestGpu(gpus).index) : "");
      onChange({ mode: "explicit", value: fallback });
    } else {
      onChange({ mode: "inherit" });
    }
  };

  return (
    <Field label={label} hint={t("runtime.deviceHint")} className={className}>
      <div className="grid gap-2">
        <Select
          value={selection.mode === "explicit" ? "explicit" : "inherit"}
          onChange={(event) =>
            switchMode(event.target.value as DeviceSelection["mode"])
          }
        >
          <option value="inherit">{t("runtime.deviceAuto")}</option>
          <option value="explicit">{t("runtime.deviceExplicit")}</option>
        </Select>

        {selection.mode === "explicit" && pinned && (
          <MonoSelect
            value={pinned}
            disabled
            onChange={() => onChange({ mode: "explicit", value: pinned })}
          >
            <option value={pinned}>
              GPU {pinned} · {t("runtime.devicePinned")}
            </option>
          </MonoSelect>
        )}

        {selection.mode === "explicit" && !pinned && gpus.length > 0 && (
          <MonoSelect
            value={current}
            onChange={(event) =>
              onChange({ mode: "explicit", value: event.target.value })
            }
          >
            {!gpus.some((gpu) => String(gpu.index) === current) && current && (
              <option value={current}>{current}</option>
            )}
            {gpus.map((gpu) => (
              <option key={gpu.index} value={String(gpu.index)}>
                GPU {gpu.index} · {compactGpuName(gpu.name)} ·{" "}
                {gigabytesFree(gpu)}G
              </option>
            ))}
          </MonoSelect>
        )}

        {selection.mode === "explicit" && !pinned && gpus.length === 0 && (
          <MonoInput
            value={current}
            placeholder="0,1"
            onChange={(event) =>
              onChange({ mode: "explicit", value: event.target.value })
            }
          />
        )}

        {pinned && (
          <p className="text-[11px] leading-snug text-muted-foreground">
            {t("runtime.devicePinnedHint", { card: pinned })}
          </p>
        )}
        {selection.mode === "explicit" && !pinned && gpus.length === 0 && (
          <p className="text-[11px] leading-snug text-muted-foreground">
            {t("runtime.deviceNoMetricsHint")}
          </p>
        )}
        {retuneHint &&
          selection.mode === "explicit" &&
          current !== "" &&
          current !== (runtime?.visible_devices ?? "") && (
            <p className="text-[11px] leading-snug text-muted-foreground">
              {t("runtime.deviceRetuneHint")}
            </p>
          )}
      </div>
    </Field>
  );
}

/** The emptiest card by free VRAM — the same rule the Agent's auto placement uses. */
function freestGpu(gpus: GpuMetric[]) {
  return gpus.reduce((best, gpu) =>
    gpu.memory_total_bytes - gpu.memory_used_bytes >
    best.memory_total_bytes - best.memory_used_bytes
      ? gpu
      : best,
  );
}

function gigabytesFree(gpu: GpuMetric) {
  const free = gpu.memory_total_bytes - gpu.memory_used_bytes;
  return (free / 2 ** 30).toFixed(1).replace(/\.0$/, "");
}
