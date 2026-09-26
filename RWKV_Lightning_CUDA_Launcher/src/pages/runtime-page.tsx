import { useCallback, useEffect, useState } from "react";
import { Info, Loader2, MonitorPlay, ScrollText } from "lucide-react";
import { useCurrent } from "@/app/use-current";
import { PageHeader, PathField } from "@/components/common";
import { DeviceSelector } from "@/components/device-selector";
import {
  GpuList,
  modelName,
  runtimeLabel,
  runtimeTone,
} from "@/components/node-status";
import { RuntimeControls } from "@/components/runtime-controls";
import { StatusPill } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckboxField, Field } from "@/components/ui/field";
import { Input, MonoInput } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Notice } from "@/components/ui/primitives";
import { inferenceApi } from "@/lib/api/inference";
import type { DeviceSelection, RuntimeConfig } from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";
import { useRuntimeForm, useRuntimeFormEntry } from "@/stores/forms";
import { useNodes } from "@/stores/nodes";
import { toast, useLogDock } from "@/stores/ui";
import { NoNodeNotice } from "@/pages/nodes-page";

/** Spacing for the reachable → runtime → model chain at the top of the page. */
const STRIP_PILL = "px-3.5 first:pl-0";

export function RuntimePage() {
  const { t } = useI18n();
  const {
    backendId,
    backend,
    runtime,
    metrics,
    metricsError,
    error,
    hasAgent,
  } = useCurrent();
  const { config, devices } = useRuntimeFormEntry(backendId);
  const setField = useRuntimeForm((s) => s.set);
  const setDeviceSelection = useRuntimeForm((s) => s.setDevices);
  const seed = useRuntimeForm((s) => s.seed);
  const set = useCallback(
    (patch: Partial<RuntimeConfig>) => setField(backendId, patch),
    [backendId, setField],
  );
  const setDevices = useCallback(
    (selection: DeviceSelection) => setDeviceSelection(backendId, selection),
    [backendId, setDeviceSelection],
  );

  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [loadingModel, setLoadingModel] = useState(false);

  // Seed this node's form from what this node is actually running, so opening
  // a node shows its own configuration rather than whichever node was last
  // visited. `seed` is a no-op once the entry exists, so it never fights the
  // user's edits. The password is never echoed by the Agent.
  useEffect(() => {
    if (!backendId || !runtime?.config?.model_path) return;
    seed(backendId, {
      config: { ...runtime.config, password: "" },
      // An empty `visible_devices` is indistinguishable from "unset" on the
      // wire, so only adopt an explicit selection.
      devices: runtime.visible_devices
        ? { mode: "explicit", value: runtime.visible_devices }
        : { mode: "inherit" },
    });
  }, [backendId, runtime?.config, runtime?.visible_devices, seed]);

  const dynamicReady =
    runtime?.status === "ready" && config.enable_dynamic_loading;

  useEffect(() => {
    if (!dynamicReady || !backendId) return;
    const controller = new AbortController();
    void inferenceApi
      .models(backendId, controller.signal)
      .then((response) => {
        setModels(response.available ?? response.data.map((entry) => entry.id));
        setSelectedModel(response.loaded ?? "");
      })
      .catch(() => setModels([]));
    return () => controller.abort();
  }, [dynamicReady, backendId]);

  const inferenceOnly = backend?.kind === "inference_only";

  const gpus = metrics?.available ? metrics.gpus : [];
  const gpuSummary =
    gpus.length > 0
      ? t("rail.gpuSummary", {
          count: gpus.length,
          avg: Math.round(
            gpus.reduce((sum, gpu) => sum + (gpu.utilization_percent ?? 0), 0) /
              gpus.length,
          ),
        })
      : "";

  return (
    <div className="max-w-[1240px] px-6 pt-5.5 pb-10">
      <PageHeader
        eyebrow={backend?.name ?? "—"}
        title={t("runtime.title")}
        description={t("runtime.subtitle")}
        actions={hasAgent ? <RuntimeControls size="default" /> : undefined}
      />

      <NoNodeNotice />

      {inferenceOnly && (
        <Notice
          tone="info"
          className="mt-5"
          icon={<Info className="size-3.5" />}
        >
          {t("runtime.unsupported")}
        </Notice>
      )}

      {error && (
        <Notice tone="danger" className="mt-5">
          {error}
        </Notice>
      )}

      <div className="mt-4.5 flex flex-wrap items-center gap-x-0 gap-y-2 rounded-xl border border-border bg-card px-3.5 py-3">
        <StatusPill
          size="md"
          className={STRIP_PILL}
          tone={backend?.reachable ? "ok" : "bad"}
          label={
            backend?.reachable ? t("status.reachable") : t("status.unreachable")
          }
        />
        <Connector />
        <StatusPill
          size="md"
          className={STRIP_PILL}
          tone={runtimeTone(runtime)}
          label={runtimeLabel(t, runtime)}
        />
        <Connector />
        <StatusPill
          size="md"
          mono
          className={STRIP_PILL}
          labelClassName="max-w-[280px]"
          tone={runtime?.status === "ready" ? "ok" : "idle"}
          label={modelName(runtime) || t("status.noModel")}
        />
        {runtime?.visible_devices ? (
          <>
            <Connector />
            <StatusPill
              size="md"
              mono
              className={STRIP_PILL}
              labelClassName="max-w-[280px]"
              tone="info"
              label={`${t("runtime.visibleDevices")}: ${
                runtime.visible_devices || '""'
              }`}
            />
          </>
        ) : null}
        <div className="flex-1" />
        {/* The strip itself has no horizontal gap — the connectors provide it
            — so this trailing control brings its own. */}
        <span className="flex items-center pl-3.5">
          <Button
            size="xs"
            onClick={() => useLogDock.getState().show("runtime")}
          >
            <ScrollText className="size-3.5" />
            {t("runtime.logs")}
          </Button>
        </span>
      </div>

      <div className="mt-3.5 grid grid-cols-[repeat(auto-fit,minmax(330px,1fr))] gap-3.5">
        <Card>
          <CardHeader>
            <CardTitle>{t("runtime.model")}</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3.5">
            <PathField
              label={t("runtime.modelPath")}
              value={config.model_path}
              placeholder={
                config.enable_dynamic_loading
                  ? "/data/models"
                  : "/data/models/model.pth"
              }
              backendId={backendId}
              hostDialog={backend?.capabilities.includes("host_dialog")}
              onChange={(model_path) => set({ model_path })}
            />
            <Field label={t("runtime.vocab")}>
              <MonoInput
                value={config.vocab_path}
                onChange={(event) => set({ vocab_path: event.target.value })}
              />
            </Field>
            <CheckboxField
              label={t("runtime.dynamicLoading")}
              checked={config.enable_dynamic_loading}
              onChange={(event) =>
                set({ enable_dynamic_loading: event.target.checked })
              }
            />

            {dynamicReady && (
              <div className="grid gap-2 border-t border-border pt-3.5">
                <Field label={t("runtime.model")}>
                  <div className="flex gap-2">
                    <Select
                      value={selectedModel}
                      onChange={(event) => setSelectedModel(event.target.value)}
                    >
                      <option value="">—</option>
                      {models.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </Select>
                    <Button
                      disabled={!selectedModel || loadingModel}
                      onClick={async () => {
                        setLoadingModel(true);
                        try {
                          await useNodes
                            .getState()
                            .loadModel(backendId, selectedModel);
                          toast.success(t("settings.saved"));
                        } catch (caught) {
                          toast.error(
                            t("toast.failed", {
                              error:
                                caught instanceof Error
                                  ? caught.message
                                  : String(caught),
                            }),
                          );
                        } finally {
                          setLoadingModel(false);
                        }
                      }}
                    >
                      {loadingModel ? (
                        <Loader2 className="size-3.5 animate-spin" />
                      ) : (
                        <MonitorPlay className="size-3.5" />
                      )}
                      {t("runtime.model")}
                    </Button>
                  </div>
                </Field>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t("runtime.device")}</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3.5">
            <DeviceSelector
              label="visible_devices"
              selection={devices}
              onChange={setDevices}
              retuneHint
            />

            <div className="grid grid-cols-2 gap-3">
              <Field label={t("runtime.port")}>
                <MonoInput
                  type="number"
                  min={1}
                  max={65535}
                  value={config.port}
                  onChange={(event) => set({ port: event.target.value })}
                />
              </Field>
              <Field
                label={t("runtime.password")}
                hint={t("runtime.passwordHint")}
              >
                <Input
                  type="password"
                  autoComplete="off"
                  value={config.password}
                  onChange={(event) => set({ password: event.target.value })}
                />
              </Field>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t("runtime.performance")}</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3.5">
            <Field label={t("runtime.chunkSize")}>
              <MonoInput
                type="number"
                min={0}
                value={config.chunk_size}
                onChange={(event) =>
                  set({ chunk_size: Number(event.target.value) })
                }
              />
            </Field>
            <div className="grid gap-2.5">
              <CheckboxField
                label={t("runtime.chunkLoad")}
                hint={t("runtime.chunkLoadHint")}
                checked={config.chunk_load}
                onChange={(event) => set({ chunk_load: event.target.checked })}
              />
              <CheckboxField
                label={t("runtime.wkv32")}
                hint={t("runtime.wkv32Hint")}
                checked={config.use_wkv32}
                onChange={(event) => set({ use_wkv32: event.target.checked })}
              />
            </div>
            <Field label={t("runtime.stateDb")}>
              <MonoInput
                value={config.state_db_path}
                onChange={(event) => set({ state_db_path: event.target.value })}
              />
            </Field>
            <Field label={t("runtime.tuneCache")}>
              <MonoInput
                value={config.tune_cache}
                placeholder={t("runtime.tuneCacheHint")}
                onChange={(event) => set({ tune_cache: event.target.value })}
              />
            </Field>
          </CardContent>
        </Card>
      </div>

      {/* Full width, not one column of the form grid: inside a ~370px column
          the card grid can only ever fit one card, so an 8-GPU box became a
          column eight cards tall. Out here the same grid fits three or four
          per row. */}
      <Card className="mt-3.5">
        <CardHeader>
          <CardTitle>GPU</CardTitle>
          {gpuSummary && (
            <span className="truncate text-[11.5px] text-muted-foreground">
              {gpuSummary}
            </span>
          )}
        </CardHeader>
        <CardContent>
          <GpuList
            metrics={metrics}
            metricsError={metricsError}
            backend={backend}
            runtime={runtime}
            t={t}
          />
        </CardContent>
      </Card>
    </div>
  );
}

function Connector() {
  return <span className="hidden h-px w-5.5 bg-border sm:block" />;
}
