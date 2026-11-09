import { useEffect, useState, type ReactNode } from "react";
import { ChevronRight, RotateCcw } from "lucide-react";
import { AdapterManager } from "@/components/chat/adapter-manager";
import { StateManager } from "@/components/chat/state-manager";
import { Separator } from "@/components/ui/primitives";
import { SliderField } from "@/components/ui/slider";
import { Select } from "@/components/ui/select";
import { inferenceApi } from "@/lib/api/inference";
import type { AdapterEntry, UploadedState } from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";
import { defaultGeneration, useSettings } from "@/stores/settings";

const encodeAdapter = (id: string, version: string) =>
  JSON.stringify([id, version]);

/** `adapter_id + version` is the identity, so the select value carries both. */
function decodeAdapter(value: string): [string, string] {
  try {
    const parsed = JSON.parse(value) as unknown;
    if (
      Array.isArray(parsed) &&
      parsed.length === 2 &&
      typeof parsed[0] === "string" &&
      typeof parsed[1] === "string"
    )
      return [parsed[0], parsed[1]];
  } catch {
    // Not an encoded adapter value: fall through to "no adapter".
  }
  return ["", ""];
}

/** Right-hand column of the chat page: generation settings and node assets. */
export function GenerationPanel({ backendId }: { backendId: string }) {
  const { t } = useI18n();
  const generation = useSettings((s) => s.generation);
  const setGeneration = useSettings((s) => s.setGeneration);
  const [states, setStates] = useState<UploadedState[]>([]);
  const [statesFailed, setStatesFailed] = useState(false);
  const [adapters, setAdapters] = useState<AdapterEntry[]>([]);
  const [adaptersError, setAdaptersError] = useState("");
  const [penaltiesOpen, setPenaltiesOpen] = useState(false);
  const [statesOpen, setStatesOpen] = useState(false);
  const [adaptersOpen, setAdaptersOpen] = useState(false);
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    if (!backendId) {
      setStates([]);
      setStatesFailed(false);
      setAdapters([]);
      setAdaptersError("");
      return;
    }
    const controller = new AbortController();
    inferenceApi
      .listStates(backendId, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setStates(result?.data ?? []);
        setStatesFailed(false);
      })
      .catch(() => {
        if (controller.signal.aborted) return;
        setStates([]);
        setStatesFailed(true);
      });
    inferenceApi
      .listAdapters(backendId, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setAdapters(result?.data ?? []);
        setAdaptersError("");
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setAdapters([]);
        setAdaptersError(
          cause instanceof Error ? cause.message : String(cause),
        );
      });
    return () => controller.abort();
  }, [backendId, revision]);

  /** Re-read both lists once a manager dialog closes; it may have changed them. */
  const closed = (setOpen: (open: boolean) => void) => (open: boolean) => {
    setOpen(open);
    if (!open) setRevision((value) => value + 1);
  };

  const selectedAdapter = encodeAdapter(
    generation.adapter_id ?? "",
    generation.adapter_version ?? "",
  );
  const adapterMissing =
    Boolean(generation.adapter_id) &&
    !adapters.some(
      (entry) =>
        entry.id === generation.adapter_id &&
        entry.version === (generation.adapter_version ?? ""),
    );
  const stateMissing =
    Boolean(generation.state_id) &&
    !states.some((state) => state.state_id === generation.state_id);

  return (
    <div className="flex flex-col gap-5">
      <section className="flex flex-col gap-2.5">
        <div className="flex h-6 items-center gap-2">
          <h2 className="text-[12.5px] font-semibold">
            {t("chat.generation")}
          </h2>
          <div className="flex-1" />
          <button
            type="button"
            title={t("chat.resetGeneration")}
            aria-label={t("chat.resetGeneration")}
            className="-mr-[5px] inline-flex size-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            onClick={() =>
              setGeneration({
                temperature: defaultGeneration.temperature,
                top_p: defaultGeneration.top_p,
                top_k: defaultGeneration.top_k,
                max_tokens: defaultGeneration.max_tokens,
                alpha_presence: defaultGeneration.alpha_presence,
                alpha_frequency: defaultGeneration.alpha_frequency,
                alpha_decay: defaultGeneration.alpha_decay,
              })
            }
          >
            <RotateCcw className="size-3.5" />
          </button>
        </div>
        <SliderField
          label={t("chat.temperature")}
          hint={t("chat.temperatureHint")}
          value={generation.temperature}
          min={0}
          max={2}
          step={0.05}
          onChange={(value) => setGeneration({ temperature: value })}
        />
        <SliderField
          label={t("chat.topP")}
          hint={t("chat.topPHint")}
          value={generation.top_p}
          min={0}
          max={1}
          step={0.05}
          onChange={(value) => setGeneration({ top_p: value })}
        />
        <SliderField
          label={t("chat.topK")}
          hint={t("chat.topKHint")}
          value={generation.top_k}
          min={0}
          max={200}
          step={1}
          onChange={(value) => setGeneration({ top_k: value })}
        />
        {/* The native default is 8192; a 4096 cap silently truncated long
            answers on nodes configured for more. */}
        <SliderField
          label={t("chat.maxTokens")}
          hint={t("chat.maxTokensHint")}
          value={generation.max_tokens}
          min={64}
          max={8192}
          step={64}
          onChange={(value) => setGeneration({ max_tokens: value })}
        />

        {/* Repetition penalties are the parameters people reach for when a
            model loops, but they are three of seven: folded away behind a
            summary that still shows where they sit. */}
        <details
          className="group -mx-1.5 rounded-lg px-1.5"
          open={penaltiesOpen}
          onToggle={(event) =>
            setPenaltiesOpen((event.currentTarget as HTMLDetailsElement).open)
          }
        >
          <summary
            title={t("chat.penaltiesHint")}
            className="-mx-1.5 flex h-8 cursor-pointer list-none items-center gap-1.5 rounded-md px-1.5 text-[12px] text-muted-foreground transition-colors marker:content-none hover:bg-muted hover:text-foreground"
          >
            <ChevronRight className="size-3.5 shrink-0 transition-transform group-open:rotate-90" />
            <span className="min-w-0 flex-1 truncate">
              {t("chat.penalties")}
            </span>
            <span className="shrink-0 font-mono text-[11px] tabular-nums">
              {generation.alpha_presence} · {generation.alpha_frequency} ·{" "}
              {generation.alpha_decay}
            </span>
          </summary>
          <div className="flex flex-col gap-2.5 pt-1 pb-1">
            <SliderField
              label={t("chat.alphaPresence")}
              hint={t("chat.alphaPresenceHint")}
              value={generation.alpha_presence}
              min={0}
              max={4}
              step={0.05}
              onChange={(value) => setGeneration({ alpha_presence: value })}
            />
            <SliderField
              label={t("chat.alphaFrequency")}
              hint={t("chat.alphaFrequencyHint")}
              value={generation.alpha_frequency}
              min={0}
              max={4}
              step={0.05}
              onChange={(value) => setGeneration({ alpha_frequency: value })}
            />
            <SliderField
              label={t("chat.alphaDecay")}
              hint={t("chat.alphaDecayHint")}
              value={generation.alpha_decay}
              min={0.9}
              max={1}
              step={0.001}
              onChange={(value) => setGeneration({ alpha_decay: value })}
            />
          </div>
        </details>
      </section>

      <Separator />

      {/* State and adapter are the same shape: a picker, plus a way into the
          manager that fills it. The manager is a secondary action, so it is
          a quiet link in the heading row rather than a full-width button. */}
      <AssetSection
        title={t("chat.state")}
        manageLabel={t("chat.manage")}
        disabled={!backendId}
        onManage={() => setStatesOpen(true)}
        error={statesFailed ? t("chat.stateUnavailable") : ""}
      >
        <Select
          value={statesFailed ? "" : generation.state_id}
          disabled={!backendId || statesFailed}
          onChange={(event) => setGeneration({ state_id: event.target.value })}
        >
          <option value="">
            {statesFailed ? t("chat.unavailable") : t("chat.stateNone")}
          </option>
          {stateMissing && !statesFailed && (
            <option value={generation.state_id}>{generation.state_id}</option>
          )}
          {states.map((state) => (
            <option key={state.state_id} value={state.state_id}>
              {state.filename}
            </option>
          ))}
        </Select>
      </AssetSection>

      <AssetSection
        title={t("chat.adapter")}
        manageLabel={t("chat.manage")}
        disabled={!backendId}
        onManage={() => setAdaptersOpen(true)}
        error={adaptersError}
      >
        <Select
          value={selectedAdapter}
          disabled={!backendId || Boolean(adaptersError)}
          onChange={(event) => {
            const [id, version] = decodeAdapter(event.target.value);
            setGeneration({
              adapter_id: id,
              adapter_version: version,
              adapter_scale: "",
            });
          }}
        >
          <option value={encodeAdapter("", "")}>
            {adaptersError ? t("chat.unavailable") : t("chat.adapterNone")}
          </option>
          {adapterMissing && (
            <option value={selectedAdapter}>{generation.adapter_id}</option>
          )}
          {adapters.map((entry) => (
            <option
              key={`${entry.id}@${entry.version}`}
              value={encodeAdapter(entry.id, entry.version)}
            >
              {entry.id} · {entry.version.slice(0, 12)} · {t("adapter.rank")}{" "}
              {entry.manifest.rank}
            </option>
          ))}
        </Select>
      </AssetSection>

      <StateManager
        backendId={backendId}
        open={statesOpen}
        onOpenChange={closed(setStatesOpen)}
      />
      <AdapterManager
        backendId={backendId}
        open={adaptersOpen}
        onOpenChange={closed(setAdaptersOpen)}
      />
    </div>
  );
}

/**
 * One node asset picker. A failed list read is one short muted line with the
 * raw error on hover: the runtime's dial error is useful to someone
 * debugging, and noise to someone choosing a state.
 */
function AssetSection({
  title,
  manageLabel,
  disabled,
  onManage,
  error,
  children,
}: {
  title: string;
  manageLabel: string;
  disabled: boolean;
  onManage: () => void;
  error: string;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-2">
      <div className="flex h-6 items-center gap-2">
        <h2 className="min-w-0 flex-1 truncate text-[12.5px] font-semibold">
          {title}
        </h2>
        <button
          type="button"
          disabled={disabled}
          onClick={onManage}
          className="-mr-1.5 inline-flex h-6 shrink-0 items-center rounded-md px-1.5 text-[11.5px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
        >
          {manageLabel}
        </button>
      </div>
      <div title={error || undefined}>{children}</div>
    </section>
  );
}
