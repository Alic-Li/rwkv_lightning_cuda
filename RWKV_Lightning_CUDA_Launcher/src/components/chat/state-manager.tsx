import { useEffect, useState } from "react";
import { HardDrive, Trash2 } from "lucide-react";
import { inferenceApi } from "@/lib/api/inference";
import { runtimeApi } from "@/lib/api/runtime";
import type { UploadedState } from "@/lib/api/types";
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
import { basename, formatBytes } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { hasCapability } from "@/stores/backends";
import { useSettings } from "@/stores/settings";
import { toast, useUI } from "@/stores/ui";
import { useCurrent } from "@/app/use-current";

const describe = (cause: unknown) =>
  cause instanceof Error ? cause.message : String(cause);

/** Upload / select / delete the node's recurrent state snapshots. */
export function StateManager({
  backendId,
  open,
  onOpenChange,
}: {
  backendId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useI18n();
  const selected = useSettings((s) => s.generation.state_id);
  const setGeneration = useSettings((s) => s.setGeneration);
  const [states, setStates] = useState<UploadedState[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [nodePath, setNodePath] = useState("");
  const { backend } = useCurrent();
  const openFsBrowser = useUI((s) => s.openFsBrowser);
  // Importing from the node needs the Agent hop; older agents only take
  // uploads from this computer.
  const canImport = hasCapability(backend, "state_import");

  useEffect(() => {
    if (!open || !backendId) {
      setStates([]);
      setError("");
      setBusy(false);
      return;
    }
    const controller = new AbortController();
    setBusy(true);
    setError("");
    inferenceApi
      .listStates(backendId, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setStates(result?.data ?? []);
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

  const upload = async (file: File) => {
    if (!backendId) return;
    setBusy(true);
    try {
      const state = await inferenceApi.uploadState(backendId, file);
      if (state?.state_id) setGeneration({ state_id: state.state_id });
      toast.success(t("toast.stateUploaded", { name: file.name }));
      setRevision((value) => value + 1);
    } catch (cause) {
      toast.error(t("toast.failed", { error: describe(cause) }));
    } finally {
      setBusy(false);
    }
  };

  /** The .pth a tuning run produced lives on the node, not on this laptop. */
  const importFromNode = async () => {
    const path = nodePath.trim();
    if (!backendId || !path) return;
    setBusy(true);
    try {
      const state = await runtimeApi.importState(backendId, path);
      if (state?.state_id) setGeneration({ state_id: state.state_id });
      toast.success(
        t("state.imported", { name: state?.state_id || basename(path) }),
      );
      setNodePath("");
      setRevision((value) => value + 1);
    } catch (cause) {
      toast.error(t("toast.failed", { error: describe(cause) }));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (state: UploadedState) => {
    if (!backendId) return;
    if (!window.confirm(t("state.deleteConfirm", { name: state.filename })))
      return;
    setBusy(true);
    try {
      await inferenceApi.deleteState(backendId, state.state_id);
      if (useSettings.getState().generation.state_id === state.state_id)
        setGeneration({ state_id: "" });
      setStates((items) =>
        items.filter((item) => item.state_id !== state.state_id),
      );
      toast.success(t("toast.stateDeleted", { name: state.filename }));
    } catch (cause) {
      toast.error(t("toast.failed", { error: describe(cause) }));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent width="620px">
        <DialogHeader
          title={t("state.title")}
          description={t("state.description")}
        />
        <DialogBody className="space-y-3">
          <Field label={t("state.sourceLocal")}>
            <FileDrop
              accept=".pth"
              label={t("state.upload")}
              disabled={!backendId || busy}
              onFile={(file) => file && void upload(file)}
            />
          </Field>

          <Separator />

          <Field
            label={t("state.sourceNode")}
            hint={canImport ? t("state.importHint") : undefined}
          >
            {canImport ? (
              <div className="flex gap-2">
                <Input
                  value={nodePath}
                  placeholder={t("fs.pathPlaceholder")}
                  disabled={!backendId || busy}
                  className="min-w-0 flex-1 font-mono text-xs"
                  onChange={(event) => setNodePath(event.target.value)}
                />
                <Button
                  disabled={!backendId || busy}
                  onClick={() => openFsBrowser(setNodePath, nodePath)}
                >
                  <HardDrive className="size-3.5" />
                  {t("common.browse")}
                </Button>
                <Button
                  variant="default"
                  disabled={!backendId || busy || !nodePath.trim()}
                  onClick={() => void importFromNode()}
                >
                  {t("state.import")}
                </Button>
              </div>
            ) : (
              <Notice tone="info">{t("state.importUnsupported")}</Notice>
            )}
          </Field>

          {error && <Notice tone="danger">{error}</Notice>}

          {states.length === 0 ? (
            <p className="text-[12.5px] text-muted-foreground">
              {busy ? t("common.loading") : t("state.empty")}
            </p>
          ) : (
            <div className="space-y-2">
              {states.map((state) => {
                const current = selected === state.state_id;
                return (
                  <div
                    key={state.state_id}
                    className={cn(
                      "rounded-lg border border-border px-3 py-2.5",
                      current && "bg-accent",
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="truncate text-[12.5px] font-medium">
                          {state.filename}
                        </div>
                        <div className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                          {state.state_id}
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-1.5">
                        <Button
                          size="xs"
                          disabled={current || busy || !backendId}
                          onClick={() =>
                            setGeneration({ state_id: state.state_id })
                          }
                        >
                          {t("common.select")}
                        </Button>
                        <Button
                          size="xs"
                          variant="danger"
                          disabled={busy || !backendId}
                          title={t("common.delete")}
                          onClick={() => void remove(state)}
                        >
                          <Trash2 className="size-3.5" />
                        </Button>
                      </div>
                    </div>
                    <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-muted-foreground">
                      <span className="font-mono">
                        {formatBytes(state.size_bytes)}
                      </span>
                      <span>·</span>
                      <span>
                        {t("state.tensors")} {state.tensor_count}
                      </span>
                      <span>·</span>
                      <span>
                        {new Date(state.created * 1000).toLocaleString()}
                      </span>
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
