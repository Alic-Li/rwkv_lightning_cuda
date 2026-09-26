import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import type { BackendView, UpdateBackendRequest } from "@/lib/api/types";
import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogFooter,
  DialogHeader,
} from "@/components/ui/dialog";
import { Field } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Notice } from "@/components/ui/primitives";
import { useI18n } from "@/lib/i18n";
import { backendLabel, useBackends } from "@/stores/backends";
import { useNodes } from "@/stores/nodes";
import { toast } from "@/stores/ui";

/**
 * Edit a registered node. The same three things the add form asks for, because
 * they are the things that can change: a node gets renamed, it moves to another
 * host, its token is rotated. The id is random and is not one of them — that is
 * what keeps the selection pointing at the same node afterwards.
 *
 * The token field starts empty and empty means "keep the stored one": the
 * console never receives a token, so it has nothing to pre-fill.
 */
export function EditBackendDialog({
  backend,
  open,
  onOpenChange,
}: {
  backend: BackendView;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useI18n();
  const update = useBackends((s) => s.update);
  const refresh = useNodes((s) => s.refresh);
  const [name, setName] = useState(backend.name);
  const [baseUrl, setBaseUrl] = useState(backend.base_url);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setName(backend.name);
    setBaseUrl(backend.base_url);
    setToken("");
    setError("");
  }, [open, backend.id, backend.name, backend.base_url]);

  const submit = async () => {
    setBusy(true);
    setError("");
    try {
      const body: UpdateBackendRequest = {
        name: name.trim(),
        base_url: baseUrl.trim(),
      };
      if (token.trim()) body.token = token.trim();
      const view = await update(backend.id, body);
      await refresh(view.id);
      toast.success(t("backend.updated", { name: backendLabel(t, view) }));
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent width="520px">
        <DialogHeader
          title={t("backend.editTitle")}
          description={t("backend.editDescription")}
        />
        <DialogBody className="grid gap-4">
          <Field label={t("backend.name")}>
            <Input
              value={name}
              placeholder={t("backend.namePlaceholder")}
              onChange={(event) => setName(event.target.value)}
            />
          </Field>
          <Field label={t("backend.baseUrl")} hint={t("backend.baseUrlHint")}>
            <Input
              value={baseUrl}
              placeholder={t("backend.baseUrlPlaceholder")}
              autoComplete="off"
              spellCheck={false}
              onChange={(event) => setBaseUrl(event.target.value)}
              className="font-mono text-xs"
            />
          </Field>
          <Field label={t("backend.token")} hint={t("backend.tokenKeep")}>
            <Input
              type="password"
              value={token}
              placeholder={
                backend.has_token
                  ? t("backend.tokenStored")
                  : t("backend.tokenPlaceholder")
              }
              autoComplete="new-password"
              onChange={(event) => setToken(event.target.value)}
            />
          </Field>
          {error && <Notice tone="danger">{error}</Notice>}
        </DialogBody>
        <DialogFooter>
          <Button onClick={() => onOpenChange(false)}>{t("common.cancel")}</Button>
          <Button
            variant="default"
            disabled={busy || !name.trim() || !baseUrl.trim()}
            onClick={submit}
          >
            {busy && <Loader2 className="size-3.5 animate-spin" />}
            {busy ? t("common.loading") : t("common.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
