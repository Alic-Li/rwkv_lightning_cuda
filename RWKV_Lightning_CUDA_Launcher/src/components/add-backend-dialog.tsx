import { useEffect, useState } from "react";
import { Loader2, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogFooter,
  DialogHeader,
} from "@/components/ui/dialog";
import { Field } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Notice } from "@/components/ui/primitives";
import { CapabilityBadges } from "@/components/node-status";
import { StatusDot } from "@/components/ui/badge";
import { useI18n } from "@/lib/i18n";
import type { BackendView } from "@/lib/api/types";
import { useBackends } from "@/stores/backends";
import { toast, useUI } from "@/stores/ui";

/** A registry entry is a root address; `/api` or `/v1` suffixes are rejected. */
function validateBaseUrl(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return "required";
  if (!/^https?:\/\//i.test(trimmed)) return "scheme";
  if (/\/+(api|v1)\/?$/i.test(trimmed)) return "suffix";
  return "";
}

export function AddBackendDialog() {
  const { t } = useI18n();
  const open = useUI((s) => s.addBackendOpen);
  const close = useUI((s) => s.closeAddBackend);
  const add = useBackends((s) => s.add);
  const [name, setName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<BackendView | null>(null);

  useEffect(() => {
    if (!open) return;
    setName("");
    setBaseUrl("");
    setToken("");
    setError("");
    setResult(null);
    setBusy(false);
  }, [open]);

  const urlProblem = validateBaseUrl(baseUrl);

  const submit = async () => {
    if (busy) return;
    if (urlProblem) {
      setError(
        urlProblem === "suffix"
          ? t("backend.baseUrlHint")
          : t("backend.baseUrl"),
      );
      return;
    }
    setBusy(true);
    setError("");
    try {
      const view = await add({
        name: name.trim() || baseUrl.trim(),
        base_url: baseUrl.trim().replace(/\/+$/, ""),
        token: token.trim(),
      });
      setResult(view);
      if (view.reachable) toast.success(t("backend.added", { name: view.name }));
      else
        toast.error(
          t("backend.addedUnreachable", {
            name: view.name,
            error: view.probe_error || t("common.unknown"),
          }),
        );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(next) => !next && close()}>
      <DialogContent width="540px">
        <DialogHeader
          title={t("backend.addTitle")}
          description={t("backend.addDescription")}
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
          <Field label={t("backend.token")} hint={t("backend.tokenHint")}>
            <Input
              type="password"
              value={token}
              placeholder={t("backend.tokenPlaceholder")}
              autoComplete="new-password"
              onChange={(event) => setToken(event.target.value)}
            />
          </Field>

          {error && <Notice tone="danger">{error}</Notice>}

          {result && (
            <div className="rounded-lg border border-border bg-background p-3">
              <div className="flex items-center gap-2 text-[12.5px] font-medium">
                <StatusDot
                  tone={result.reachable ? "ok" : "bad"}
                  pulse={!result.reachable}
                />
                {result.reachable
                  ? `${t("backend.kind.agent")} · ${result.kind || "agent"}`
                  : t("backend.probeFailed", {
                      error: result.probe_error || t("common.unknown"),
                    })}
              </div>
              {result.reachable && (
                <CapabilityBadges
                  capabilities={result.capabilities}
                  className="mt-2.5"
                />
              )}
              {!result.reachable && result.probe_error && (
                <p className="mt-2 font-mono text-[11px] break-all text-muted-foreground">
                  {result.probe_error}
                </p>
              )}
            </div>
          )}
        </DialogBody>
        <DialogFooter>
          <Button onClick={close}>{t("common.close")}</Button>
          {!result && (
            <Button
              variant="default"
              disabled={busy || !baseUrl.trim()}
              onClick={submit}
            >
              {busy ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Plus className="size-3.5" />
              )}
              {busy ? t("common.loading") : t("backend.add")}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
