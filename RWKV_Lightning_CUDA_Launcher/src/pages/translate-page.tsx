/**
 * Parallel Translate. Every non-empty line of the source document is one
 * independent continuation on `/v1/batch/completions`; the job engine (chunk
 * queue, batching, abort, persistence) lives in `stores/translate`.
 *
 * Laid out like the training and quantization pages: the page header only
 * names the view, one job bar holds the settings and the Start / Stop pair,
 * and the two documents sit side by side in cards of the same height with a
 * footer each, so their edges line up at every width.
 */
import {
  ArrowLeftRight,
  Download,
  Languages,
  Play,
  RotateCcw,
  Square,
  TriangleAlert,
} from "lucide-react";
import { useEffect, type ReactNode } from "react";
import { useCurrent } from "@/app/use-current";
import { CopyButton, PageHeader } from "@/components/common";
import { StatusDot, type StatusTone } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input, Textarea } from "@/components/ui/input";
import { EmptyState, Notice, Progress } from "@/components/ui/primitives";
import { Select } from "@/components/ui/select";
import { formatCount, formatDuration } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { chunkText } from "@/lib/translate/chunk";
import { languages, normalizeLanguage } from "@/lib/translate/languages";
import type { TranslateChunk } from "@/lib/translate/scheduler";
import { cn } from "@/lib/utils";
import { useSettings } from "@/stores/settings";
import { translatedOutput, useTranslate } from "@/stores/translate";
import { useUI } from "@/stores/ui";

const MAX_BATCH = 128;

/**
 * A floor, not a height: both documents flex to fill their card, and the grid
 * stretches the two cards to one height, so they end on the same line even
 * when one card's header or footer wraps.
 */
const DOCUMENT_HEIGHT = "min-h-[440px] flex-1";

/**
 * One footer height for both cards: the source footer holds only a hint and
 * would otherwise come out 8px shorter than the translation footer's buttons,
 * lifting its document's bottom edge off its neighbour's.
 */
const FOOTER = "min-h-[49px] bg-card";

const STATUS_TONE: Record<TranslateChunk["status"], StatusTone> = {
  pending: "idle",
  running: "warn",
  done: "ok",
  error: "bad",
};

/** Browser-side text export; the document never leaves the page. */
function download(text: string, extension: string) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `rwkv-translation.${extension}`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function TranslatePage() {
  const { t } = useI18n();
  // The whole job state is subscribed on purpose: the engine publishes progress
  // by mutating the chunk objects, so derived values must be recomputed on
  // every render instead of being memoised on the stable `chunks` reference.
  const job = useTranslate();
  const sourceLanguage = useSettings((s) => s.sourceLanguage);
  const targetLanguage = useSettings((s) => s.targetLanguage);
  const defaultConcurrency = useSettings((s) => s.concurrency);
  const { backendId, backend, canInfer } = useCurrent();

  const sourceLang = normalizeLanguage(job.from || sourceLanguage, "English");
  const targetLang = normalizeLanguage(job.to || targetLanguage, "Chinese");
  const batchSize = job.concurrency || defaultConcurrency;

  const lines = chunkText(job.source);
  const ordered = [...job.chunks].sort((a, b) => a.id - b.id);
  const output = translatedOutput(job.chunks);
  const total = job.chunks.length;
  const count = (status: TranslateChunk["status"]) =>
    job.chunks.filter((c) => c.status === status).length;
  const done = count("done");
  const running = count("running");
  const chars = job.chunks.reduce((sum, c) => sum + c.translated.length, 0);
  const errorIds = job.chunks
    .filter((c) => c.status === "error")
    .map((c) => c.id);
  const pendingIds = job.chunks
    .filter((c) => c.status === "pending")
    .map((c) => c.id);

  const unsupported = backend?.kind === "inference_only";
  const batchValid =
    Number.isInteger(batchSize) && batchSize >= 1 && batchSize <= MAX_BATCH;
  // First reason wins; the bar says it in words instead of leaving a grey
  // button to be guessed at.
  const blocker = unsupported
    ? t("translate.reasonUnsupported")
    : !canInfer
      ? t("translate.reasonNoRuntime")
      : sourceLang === targetLang
        ? t("translate.reasonSameLanguage")
        : !batchValid
          ? t("translate.reasonBatch", { max: MAX_BATCH })
          : lines.length === 0
            ? t("translate.reasonEmpty")
            : "";
  const canStart = !job.busy && !blocker;

  const start = () => void useTranslate.getState().run(backendId);
  const stop = () => useTranslate.getState().stop();
  const retry = (ids: number[]) =>
    void useTranslate.getState().run(backendId, ids);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Enter" || !(event.ctrlKey || event.metaKey)) return;
      if (!canStart) return;
      const ui = useUI.getState();
      if (ui.addBackendOpen || ui.fsPicker) return;
      if (document.querySelector('[role="dialog"]')) return;
      event.preventDefault();
      void useTranslate.getState().run(backendId);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [backendId, canStart]);

  const errorText = !job.error
    ? ""
    : job.error === "invalid-languages"
      ? t("translate.reasonSameLanguage")
      : job.error === "invalid-batch"
        ? t("translate.reasonBatch", { max: MAX_BATCH })
        : job.error === "empty-source"
          ? t("translate.reasonEmpty")
          : job.error;

  const statusText = job.busy
    ? t("translate.statusRunning", { done, total, running })
    : blocker
      ? t("translate.cannotStart", { reason: blocker })
      : total > 0 && done === total
        ? t("translate.statusDone", {
            total,
            elapsed: formatDuration(job.elapsed),
          })
        : t("translate.statusReady", { count: lines.length });

  const languageOptions = languages.map((language) => (
    <option key={language} value={language}>
      {language}
    </option>
  ));

  return (
    <div className="max-w-[1240px] px-6 pt-5.5 pb-10">
      <PageHeader
        title={t("translate.title")}
        description={t("translate.subtitle")}
      />

      {unsupported && (
        <Notice
          tone="warning"
          className="mt-4"
          icon={<TriangleAlert className="size-3.5" />}
        >
          {t("translate.unsupported")}
        </Notice>
      )}
      {!unsupported && !canInfer && (
        <Notice
          tone="info"
          className="mt-4"
          action={
            <a
              href="#/runtime"
              className="shrink-0 text-info underline underline-offset-2"
            >
              {t("runtime.title")}
            </a>
          }
        >
          {t("translate.noRuntime")}
        </Notice>
      )}
      {errorText && (
        <Notice
          tone="danger"
          className="mt-4"
          icon={<TriangleAlert className="size-3.5" />}
        >
          {errorText}
        </Notice>
      )}

      {/* Same order as the training bar: what the job is on the left, what
          is happening in the middle, the actions on the right with the
          primary one last. */}
      <div className="mt-5 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-xl border border-border bg-card px-3 py-2.5 shadow-flat">
        <div className="flex items-center gap-1.5">
          <Select
            className="h-8 w-[132px]"
            aria-label={t("translate.source")}
            title={t("translate.source")}
            value={sourceLang}
            disabled={job.busy}
            onChange={(event) => job.set({ from: event.target.value })}
          >
            {languageOptions}
          </Select>
          <Button
            size="icon-sm"
            variant="ghost"
            title={t("translate.swap")}
            aria-label={t("translate.swap")}
            disabled={job.busy}
            onClick={() => job.set({ from: targetLang, to: sourceLang })}
          >
            <ArrowLeftRight className="size-3.5" />
          </Button>
          <Select
            className="h-8 w-[132px]"
            aria-label={t("translate.target")}
            title={t("translate.target")}
            value={targetLang}
            disabled={job.busy}
            onChange={(event) => job.set({ to: event.target.value })}
          >
            {languageOptions}
          </Select>
        </div>
        <label
          className="flex items-center gap-1.5 text-[12px] text-muted-foreground"
          title={t("translate.batchHint")}
        >
          {t("translate.batchSize")}
          <Input
            type="number"
            min={1}
            max={MAX_BATCH}
            className={cn(
              "h-8 w-[64px] text-center font-mono text-xs",
              !batchValid &&
                "border-destructive focus-visible:border-destructive",
            )}
            aria-invalid={!batchValid || undefined}
            value={batchSize}
            disabled={job.busy}
            onChange={(event) =>
              job.set({ concurrency: Number(event.target.value) })
            }
          />
        </label>
        <span className="flex min-w-[120px] flex-1 items-center gap-2 text-[12px] text-muted-foreground">
          {job.busy && <StatusDot tone="warn" pulse />}
          <span className="truncate" title={blocker || undefined}>
            {statusText}
          </span>
        </span>
        <div className="flex items-center gap-1.5">
          <Button
            size="sm"
            variant="ghost"
            disabled={job.busy || (!job.source && total === 0)}
            onClick={() => useTranslate.getState().clear()}
          >
            {t("translate.clear")}
          </Button>
          {!job.busy && canInfer && errorIds.length > 0 && (
            <Button
              size="sm"
              variant="danger"
              onClick={() => retry(errorIds)}
            >
              <RotateCcw className="size-3.5" />
              {t("translate.retryFailedCount", { count: errorIds.length })}
            </Button>
          )}
          {!job.busy && canInfer && pendingIds.length > 0 && (
            <Button
              size="sm"
              onClick={() => retry(pendingIds)}
            >
              <Play className="size-3.5" />
              {t("translate.resume", { count: pendingIds.length })}
            </Button>
          )}
          {job.busy ? (
            <Button size="sm" onClick={stop}>
              <Square className="size-3.5" />
              {t("translate.stop")}
            </Button>
          ) : (
            <Button
              size="sm"
              variant="default"
              disabled={!canStart}
              title={blocker || t("translate.shortcut")}
              onClick={start}
            >
              <Play className="size-3.5" />
              {t("translate.start")}
            </Button>
          )}
        </div>
      </div>

      <div className="mt-3.5 grid gap-3.5 lg:grid-cols-2">
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>{t("translate.sourcePanel")}</CardTitle>
            <div className="flex-1" />
            <Meta>
              {t("translate.lines", { count: formatCount(lines.length) })} ·{" "}
              {t("translate.chars", { count: formatCount(job.source.length) })}
            </Meta>
          </CardHeader>
          <CardContent className="flex flex-1 flex-col px-3.5 py-3">
            <Textarea
              className={cn(DOCUMENT_HEIGHT, "resize-none")}
              placeholder={t("translate.placeholder")}
              spellCheck={false}
              value={job.source}
              disabled={job.busy}
              onChange={(event) => job.set({ source: event.target.value })}
            />
          </CardContent>
          <CardFooter className={FOOTER}>
            <span className="min-w-0 flex-1 truncate text-[11.5px] text-muted-foreground">
              {t("translate.sourceFootnote")}
            </span>
            <kbd className="shrink-0 rounded border border-border bg-background px-1.5 py-px font-mono text-[10.5px] text-muted-foreground">
              {t("translate.shortcut")}
            </kbd>
          </CardFooter>
        </Card>

        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>{t("translate.targetPanel")}</CardTitle>
            <div className="flex-1" />
            {total > 0 && (
              <Meta>
                {t("translate.progress", { done, total })} ·{" "}
                {t("translate.chars", { count: formatCount(chars) })}
              </Meta>
            )}
          </CardHeader>
          <CardContent className="flex flex-1 flex-col gap-2.5 px-3.5 py-3">
            {total > 0 && (
              <Progress
                value={done}
                max={total}
                tone={done === total ? "success" : "primary"}
              />
            )}
            {ordered.length > 0 ? (
              <ol
                // Pinned by basis-0 so a long result scrolls inside the list
                // instead of growing the card past its neighbour.
                className="min-h-[424px] flex-1 basis-0 overflow-auto rounded-lg border border-border bg-background py-1"
              >
                {ordered.map((chunk) => (
                  <TranslatedLine key={chunk.id} chunk={chunk} />
                ))}
              </ol>
            ) : (
              <EmptyState
                className={cn(DOCUMENT_HEIGHT, "py-0")}
                icon={<Languages className="size-7" strokeWidth={1.5} />}
                title={t("translate.empty")}
                body={t("translate.emptyBody")}
              />
            )}
          </CardContent>
          <CardFooter className={FOOTER}>
            <div className="flex min-w-0 flex-1 items-center gap-3 overflow-hidden text-[11.5px] whitespace-nowrap text-muted-foreground">
              {total > 0 &&
                (
                  [
                    ["done", "translate.countDone"],
                    ["running", "translate.countRunning"],
                    ["error", "translate.countFailed"],
                    ["pending", "translate.countPending"],
                  ] as const
                ).map(([status, key]) =>
                  count(status) > 0 ? (
                    <span key={status} className="flex items-center gap-1.5">
                      <StatusDot tone={STATUS_TONE[status]} />
                      {t(key, { count: count(status) })}
                    </span>
                  ) : null,
                )}
            </div>
            <CopyButton text={output} disabled={!output} />
            <Button
              size="xs"
              disabled={!output}
              onClick={() => download(output, "txt")}
            >
              <Download className="size-3.5" />
              TXT
            </Button>
            <Button
              size="xs"
              disabled={!output}
              onClick={() => download(output, "md")}
            >
              <Download className="size-3.5" />
              Markdown
            </Button>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}

function Meta({ children }: { children: ReactNode }) {
  return (
    <span className="font-mono text-[11.5px] text-muted-foreground tabular-nums">
      {children}
    </span>
  );
}

/** One source line's result: number, state and the text (or why not). */
function TranslatedLine({ chunk }: { chunk: TranslateChunk }) {
  const { t } = useI18n();
  const failed = chunk.status === "error";
  const placeholder =
    chunk.status === "running"
      ? t("translate.running")
      : chunk.status === "pending"
        ? t("translate.pending")
        : "";
  return (
    <li
      className="flex items-baseline gap-2.5 px-2.5 py-1.5 hover:bg-muted/50"
      title={chunk.source}
    >
      <span className="w-6 shrink-0 text-right font-mono text-[11px] text-muted-foreground tabular-nums">
        {chunk.id + 1}
      </span>
      {/* A 7px dot on a 12.5px line: lift it onto the text's x-height. */}
      <StatusDot
        tone={STATUS_TONE[chunk.status]}
        pulse={chunk.status === "running"}
        className="-translate-y-px"
      />
      <p
        className={cn(
          "min-w-0 flex-1 text-[12.5px] leading-relaxed break-words whitespace-pre-wrap",
          failed && "text-destructive",
          !chunk.translated && !failed && "text-muted-foreground",
        )}
      >
        {chunk.translated ||
          (failed ? chunk.error || t("translate.failed") : placeholder)}
      </p>
    </li>
  );
}
