import { useCallback, useEffect, useState, type ReactNode } from "react";
import { ArrowUp, File, Folder, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogFooter,
  DialogHeader,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Notice } from "@/components/ui/primitives";
import { ApiError } from "@/lib/api/http";
import { nodeApi } from "@/lib/api/node";
import {
  isFsRoots,
  type FsDirectory,
  type FsEntry,
  type FsResponse,
} from "@/lib/api/types";
import { formatBytes } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { useBackends } from "@/stores/backends";
import { useUI } from "@/stores/ui";
import { cn } from "@/lib/utils";

export function FsBrowserDialog() {
  const { t } = useI18n();
  const pick = useUI((s) => s.fsPicker);
  const initialPath = useUI((s) => s.fsInitialPath);
  const close = useUI((s) => s.closeFsBrowser);
  const backendId = useBackends((s) => s.currentId);

  const [path, setPath] = useState("");
  const [input, setInput] = useState("");
  const [listing, setListing] = useState<FsDirectory | null>(null);
  const [roots, setRoots] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(
    async (target?: string): Promise<FsResponse | null> => {
      if (!backendId) return null;
      setBusy(true);
      setError("");
      try {
        const response = await nodeApi.fs(backendId, target);
        if (isFsRoots(response)) {
          setRoots(response.roots);
          setListing(null);
          setPath("");
        } else {
          setListing(response);
          setPath(response.path);
          setInput(response.path);
        }
        return response;
      } catch (caught) {
        // Out-of-whitelist and missing paths both answer 403 without echoing
        // the path, so the message must stay generic.
        setError(
          // Only the whitelist's own refusal is "forbidden"; any other 403
          // (an origin or token check) is shown as the server said it, or it
          // reads as a missing directory and sends the search the wrong way.
          caught instanceof ApiError &&
            caught.status === 403 &&
            caught.message === "forbidden"
            ? t("fs.forbidden")
            : caught instanceof Error
              ? caught.message
              : String(caught),
        );
        return null;
      } finally {
        setBusy(false);
      }
    },
    [backendId, t],
  );

  useEffect(() => {
    if (!pick) return;
    setInput(initialPath);
    void (async () => {
      const response = await load(initialPath || undefined);
      // An empty form field opens straight into the Agent's default
      // directory (the launcher binary's folder) instead of dead-ending on
      // the whitelist-roots list; the up button leads back to that list.
      if (!initialPath && response && isFsRoots(response) && response.default) {
        setInput(response.default);
        await load(response.default);
      }
    })();
  }, [pick, initialPath, load]);

  const choose = (entry?: FsEntry) => {
    if (!pick) return;
    pick(entry ? joinPath(path, entry.name) : path);
    close();
  };

  return (
    <Dialog open={Boolean(pick)} onOpenChange={(next) => !next && close()}>
      <DialogContent width="580px">
        <DialogHeader
          title={t("fs.title")}
          description={t("fs.description")}
        />
        <div className="flex items-center gap-2 border-b border-border bg-background px-4.5 py-2.5">
          <Input
            value={input}
            placeholder={t("fs.pathPlaceholder")}
            spellCheck={false}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void load(input.trim());
            }}
            className="font-mono text-xs"
          />
          <Button
            disabled={busy}
            onClick={() => void load(input.trim())}
          >
            {t("common.refresh")}
          </Button>
          {listing && (
            <Button
              disabled={busy}
              title={listing.parent ? t("fs.up") : t("fs.roots")}
              onClick={() => void load(listing.parent || undefined)}
            >
              <ArrowUp className="size-3.5" />
            </Button>
          )}
        </div>

        <DialogBody className="p-2">
          {error && (
            <Notice tone="warning" className="m-2">
              {error}
            </Notice>
          )}
          {busy && (
            <p className="flex items-center gap-2 p-3 text-xs text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin" />
              {t("fs.loading")}
            </p>
          )}

          {!busy && !listing && roots.length > 0 && (
            <div className="grid gap-0.5">
              <p className="px-3 py-1.5 text-[11px] font-semibold tracking-[0.04em] text-muted-foreground">
                {t("fs.roots")}
              </p>
              {roots.map((root) => (
                <Row
                  key={root}
                  icon={<Folder className="size-3.5" />}
                  label={root}
                  onClick={() => void load(root)}
                />
              ))}
            </div>
          )}

          {!busy && listing && (
            <div className="grid gap-0.5">
              {listing.entries.length === 0 && (
                <p className="px-3 py-6 text-center text-xs text-muted-foreground">
                  {t("fs.empty")}
                </p>
              )}
              {listing.entries.map((entry) => (
                <Row
                  key={entry.name}
                  icon={
                    entry.is_dir ? (
                      <Folder className="size-3.5" />
                    ) : (
                      <File className="size-3.5" />
                    )
                  }
                  label={entry.name}
                  meta={
                    entry.size !== undefined ? formatBytes(entry.size) : undefined
                  }
                  onClick={() => (entry.is_dir ? void load(joinPath(path, entry.name)) : choose(entry))}
                />
              ))}
            </div>
          )}
        </DialogBody>

        <DialogFooter>
          <span className="mr-auto text-[11px] text-muted-foreground">
            {t("fs.manualHint")}
          </span>
          <Button onClick={close}>{t("common.cancel")}</Button>
          <Button
            variant="default"
            disabled={!path}
            onClick={() => choose()}
          >
            {t("common.select")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function Row({
  icon,
  label,
  meta,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  meta?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "grid grid-cols-[auto_1fr_auto] items-center gap-2.5 rounded-lg px-3 py-2 text-left",
        "text-[12.5px] transition-colors hover:bg-muted",
      )}
    >
      <span className="text-muted-foreground">{icon}</span>
      <span className="min-w-0 truncate font-mono">{label}</span>
      {meta && (
        <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
          {meta}
        </span>
      )}
    </button>
  );
}

/** Remote paths are opaque: join with `/` and never normalise the node's path. */
function joinPath(base: string, name: string) {
  if (!base) return name;
  return base.endsWith("/") ? `${base}${name}` : `${base}/${name}`;
}
