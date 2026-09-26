import { useState, type ComponentProps, type ReactNode } from "react";
import { Check, Copy, ExternalLink, Play } from "lucide-react";
import type { ExtraProps } from "react-markdown";
import { Dialog, DialogContent, DialogHeader } from "@/components/ui/dialog";
import { copyText } from "@/lib/api/http";
import { openHTMLPreview } from "@/lib/chat/html";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { toast } from "@/stores/ui";

type HastNode = {
  type: string;
  value?: string;
  tagName?: string;
  properties?: { className?: unknown };
  children?: HastNode[];
};

/** Plain text of a hast subtree: what the fence held, before highlighting. */
function textOf(node: HastNode | undefined): string {
  if (!node) return "";
  if (node.type === "text") return node.value ?? "";
  return (node.children ?? []).map(textOf).join("");
}

/** `language-xxx` on the inner <code>, as remark writes it. */
function languageOf(node: HastNode | undefined) {
  const code = node?.children?.find((child) => child.tagName === "code");
  const classes = code?.properties?.className;
  const list = Array.isArray(classes) ? classes.map(String) : [];
  const match = list.find((name) => name.startsWith("language-"));
  return match ? match.slice("language-".length).toLowerCase() : "";
}

const PREVIEWABLE = new Set(["html", "htm", "xhtml", "svg"]);

const iconButton =
  "inline-flex h-6 items-center gap-1 rounded-md px-1.5 text-[11px] text-muted-foreground transition-colors hover:bg-background hover:text-foreground [&_svg]:size-3.5";

/**
 * A fenced block in an assistant reply: the language on the left of a thin
 * bar, and the two things one does with generated code on the right — run it
 * (HTML only) and copy it. The copy takes the fence's source text, not the
 * highlighted DOM, so it is byte-for-byte what the model wrote.
 */
export function CodeBlock({
  node,
  children,
  ...props
}: ComponentProps<"pre"> & ExtraProps) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const hast = node as HastNode | undefined;
  const source = textOf(hast).replace(/\n$/, "");
  const language = languageOf(hast);
  const previewable = PREVIEWABLE.has(language) && source.trim() !== "";

  const copy = async () => {
    try {
      await copyText(source);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch (error) {
      toast.error(
        t("toast.failed", {
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    }
  };

  return (
    <div className="md-code">
      <div className="flex h-8 items-center gap-1 pr-1.5 pl-3">
        <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-muted-foreground">
          {language || "text"}
        </span>
        {previewable && (
          <button
            type="button"
            className={iconButton}
            onClick={() => setPreviewing(true)}
          >
            <Play />
            {t("chat.preview")}
          </button>
        )}
        <button
          type="button"
          className={iconButton}
          aria-label={t("common.copy")}
          onClick={() => void copy()}
        >
          {copied ? <Check /> : <Copy />}
          {copied ? t("common.copied") : t("common.copy")}
        </button>
      </div>
      <pre {...props}>{children}</pre>
      {previewable && (
        <HTMLPreviewDialog
          html={source}
          open={previewing}
          onOpenChange={setPreviewing}
        />
      )}
    </div>
  );
}

/**
 * Generated HTML runs inside the console, in an iframe with no same-origin
 * rights: scripts work, but they cannot reach the page, its storage or the
 * node token. Opening a new tab used to be the only way to see it, and that
 * is exactly what popup blockers and WebKit's rules on blob URLs stop.
 */
export function HTMLPreviewDialog({
  html,
  open,
  onOpenChange,
  title,
}: {
  html: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title?: ReactNode;
}) {
  const { t } = useI18n();
  const [revision, setRevision] = useState(0);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        width="min(1100px, calc(100vw - 32px))"
        className="h-[86vh]"
        aria-describedby={undefined}
      >
        <DialogHeader
          title={
            <span className="flex items-center gap-3">
              {title ?? t("chat.previewTitle")}
              <span className="flex items-center gap-0.5 font-normal">
                <button
                  type="button"
                  className={cn(iconButton, "hover:bg-muted")}
                  onClick={() => setRevision((value) => value + 1)}
                >
                  <Play />
                  {t("chat.previewRerun")}
                </button>
                <button
                  type="button"
                  className={cn(iconButton, "hover:bg-muted")}
                  onClick={() => openHTMLPreview(html)}
                >
                  <ExternalLink />
                  {t("chat.previewNewTab")}
                </button>
              </span>
            </span>
          }
          className="py-3"
        />
        <iframe
          key={revision}
          title={t("chat.previewTitle")}
          sandbox="allow-scripts allow-forms allow-modals allow-popups"
          referrerPolicy="no-referrer"
          srcDoc={html}
          className="min-h-0 w-full flex-1 border-0 bg-white"
        />
      </DialogContent>
    </Dialog>
  );
}
