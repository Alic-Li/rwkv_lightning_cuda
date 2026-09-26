import { useEffect, useState } from "react";
import { HardDrive, Trash2 } from "lucide-react";
import { adapterIDFromFilename } from "@/lib/api/client";
import { inferenceApi } from "@/lib/api/inference";
import type { AdapterEntry, AdapterListResponse } from "@/lib/api/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogFooter,
  DialogHeader,
} from "@/components/ui/dialog";
import { Field } from "@/components/ui/field";
import { FileDrop } from "@/components/ui/file-drop";
import { Input } from "@/components/ui/input";
import { Notice, Separator } from "@/components/ui/primitives";
import { formatBytes } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { useSettings } from "@/stores/settings";
import { toast, useUI } from "@/stores/ui";

const describe = (cause: unknown) =>
  cause instanceof Error ? cause.message : String(cause);

/** Register / upload / select / delete MiSS adapters on the node. */
export function AdapterManager({
  backendId,
  open,
  onOpenChange,
}: {
  backendId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useI18n();
  const generation = useSettings((s) => s.generation);
  const setGeneration = useSettings((s) => s.setGeneration);
  const [listing, setListing] = useState<AdapterListResponse | null>(null);
  const [registerId, setRegisterId] = useState("");
  const [path, setPath] = useState("");
  const [uploadId, setUploadId] = useState("");
  const [metadata, setMetadata] = useState<File>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const openFsBrowser = useUI((s) => s.openFsBrowser);

  const entries = listing?.data ?? [];

  useEffect(() => {
    if (!open || !backendId) {
      setListing(null);
      setError("");
      setBusy(false);
      return;
    }
    const controller = new AbortController();
    setBusy(true);
    setError("");
    inferenceApi
      .listAdapters(backendId, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setListing(result ?? null);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        const message = describe(cause);
        setError(message);
        toast.error(t("toast.failed", { error: message }));
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false);
      });
    return () => controller.abort();
  }, [backendId, open, revision, t]);

  /** Selecting an adapter always drops any stale scale override. */
  const select = (id: string, version: string) =>
    setGeneration({
      adapter_id: id,
      adapter_version: version,
      adapter_scale: "",
    });

  const registered = (id: string, version: string) => {
    select(id, version);
    toast.success(t("toast.adapterRegistered", { id, version }));
    setRevision((value) => value + 1);
  };

  const register = async () => {
    const id = registerId.trim();
    const target = path.trim();
    if (!backendId || !id || !target) return;
    setBusy(true);
    try {
      const result = await inferenceApi.registerAdapter(backendId, id, target);
      registered(result?.adapter_id ?? id, result?.version ?? "");
    } catch (cause) {
      toast.error(t("toast.failed", { error: describe(cause) }));
    } finally {
      setBusy(false);
    }
  };

  const upload = async (file: File, id: string) => {
    if (!backendId) return;
    setBusy(true);
    try {
      const result = await inferenceApi.uploadAdapter(
        backendId,
        id,
        file,
        metadata,
      );
      registered(result?.adapter_id ?? id, result?.version ?? "");
    } catch (cause) {
      toast.error(t("toast.failed", { error: describe(cause) }));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (entry: AdapterEntry) => {
    if (!backendId) return;
    if (
      !window.confirm(
        t("adapter.deleteConfirm", {
          id: entry.id,
          version: entry.version,
        }),
      )
    )
      return;
    setBusy(true);
    try {
      await inferenceApi.deleteAdapter(backendId, entry);
      const current = useSettings.getState().generation;
      if (
        current.adapter_id === entry.id &&
        (current.adapter_version ?? "") === entry.version
      )
        select("", "");
      toast.success(t("toast.adapterDeleted", { id: entry.id }));
      setRevision((value) => value + 1);
    } catch (cause) {
      toast.error(t("toast.failed", { error: describe(cause) }));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent width="680px">
        <DialogHeader
          title={t("adapter.title")}
          description={t("adapter.description")}
        />
        <DialogBody className="space-y-3">
          <Field label={t("adapter.scale")} hint={t("common.optional")}>
            <Input
              type="number"
              step="any"
              className="font-mono"
              disabled={!generation.adapter_id}
              value={generation.adapter_scale ?? ""}
              onChange={(event) =>
                setGeneration({ adapter_scale: event.target.value })
              }
            />
          </Field>

          <Separator />

          <div className="space-y-2">
            {/* The native API registers an adapter straight from a path on
                the node, so browsing the node's filesystem is the direct
                route for a checkpoint the trainer just wrote. */}
            <Field label={t("adapter.registerPath")}>
              <div className="flex gap-2">
                <Input
                  className="min-w-0 flex-1 font-mono text-xs"
                  value={path}
                  placeholder={t("fs.pathPlaceholder")}
                  disabled={!backendId || busy}
                  onChange={(event) => setPath(event.target.value)}
                />
                <Button
                  disabled={!backendId || busy}
                  onClick={() => openFsBrowser(setPath, path)}
                >
                  <HardDrive className="size-3.5" />
                  {t("common.browse")}
                </Button>
              </div>
            </Field>
            <Field label={t("adapter.id")}>
              <Input
                value={registerId}
                disabled={!backendId || busy}
                onChange={(event) => setRegisterId(event.target.value)}
              />
            </Field>
            <Button
              variant="default"
              disabled={
                !backendId || busy || !registerId.trim() || !path.trim()
              }
              onClick={() => void register()}
            >
              {t("common.add")}
            </Button>
          </div>

          <Separator />

          <div className="space-y-2">
            <Field label={t("adapter.id")}>
              <Input
                value={uploadId}
                disabled={!backendId || busy}
                onChange={(event) => setUploadId(event.target.value)}
              />
            </Field>
            {/* Metadata first: it is attached to the upload, so picking the
                .pth must stay the last action — it is what starts it. */}
            <Field label={t("adapter.uploadJson")} hint={t("common.optional")}>
              <FileDrop
                accept=".json"
                label={t("adapter.uploadJson")}
                file={metadata}
                disabled={!backendId || busy}
                onFile={setMetadata}
              />
            </Field>
            <Field label={t("adapter.uploadPth")}>
              <FileDrop
                accept=".pth"
                label={t("adapter.uploadPth")}
                disabled={!backendId || busy}
                onFile={(file) => {
                  if (!file) return;
                  const id =
                    uploadId.trim() || adapterIDFromFilename(file.name);
                  setUploadId(id);
                  void upload(file, id);
                }}
              />
            </Field>
          </div>

          <Separator />

          <p className="font-mono text-[11px] text-muted-foreground">
            {t("adapter.ram")} {formatBytes(listing?.ram_bytes)} ·{" "}
            {t("adapter.gpu")} {formatBytes(listing?.gpu_bytes)} ·{" "}
            {listing?.uploads ?? 0} {t("adapter.uploads")}
          </p>

          {error && <Notice tone="danger">{error}</Notice>}

          {entries.length === 0 ? (
            <p className="text-[12.5px] text-muted-foreground">
              {busy ? t("common.loading") : t("adapter.empty")}
            </p>
          ) : (
            <div className="space-y-2">
              {entries.map((entry) => {
                const current =
                  generation.adapter_id === entry.id &&
                  (generation.adapter_version ?? "") === entry.version;
                return (
                  <div
                    key={`${entry.id}@${entry.version}`}
                    className={cn(
                      "rounded-lg border border-border px-3 py-2.5",
                      current && "bg-accent",
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="truncate text-[12.5px] font-medium">
                          {entry.id}
                        </div>
                        <div
                          className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground"
                          title={entry.version}
                        >
                          {entry.version}
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-1.5">
                        <Button
                          size="xs"
                          disabled={current || busy || !backendId}
                          onClick={() => select(entry.id, entry.version)}
                        >
                          {t("common.select")}
                        </Button>
                        <Button
                          size="xs"
                          variant="danger"
                          disabled={busy || !backendId}
                          title={t("common.delete")}
                          onClick={() => void remove(entry)}
                        >
                          <Trash2 className="size-3.5" />
                        </Button>
                      </div>
                    </div>
                    <div className="mt-1.5 font-mono text-[11px] text-muted-foreground">
                      {t("adapter.rank")} {entry.manifest.rank}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </DialogBody>
        <DialogFooter>
          <Button
            disabled={!backendId || busy}
            onClick={() => setRevision((value) => value + 1)}
          >
            {t("common.refresh")}
          </Button>
          <Button variant="default" onClick={() => onOpenChange(false)}>
            {t("common.close")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
