import { useRef, useState, type DragEvent } from "react";
import { FileUp, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatBytes } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";

/**
 * A file well, in place of the browser's bare `<input type="file">`. The
 * native control renders its own unstyled button and "未选择任何文件" label,
 * which is the one piece of chrome in this console that belongs to the
 * browser rather than to the app. The real input stays in the DOM (it is what
 * opens the picker and what keyboard users reach) but is visually hidden.
 */
export function FileDrop({
  accept,
  label,
  hint,
  file,
  disabled,
  onFile,
}: {
  accept: string;
  label: string;
  hint?: string;
  /** Selected file, when the caller keeps one (adapter metadata does). */
  file?: File;
  disabled?: boolean;
  onFile: (file: File | undefined) => void;
}) {
  const { t } = useI18n();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const take = (list: FileList | null) => {
    const picked = list?.[0];
    if (picked) onFile(picked);
  };

  const drop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setOver(false);
    if (!disabled) take(event.dataTransfer.files);
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
      className={cn(
        "flex items-center gap-3 rounded-lg border border-dashed px-3 py-2.5 transition-colors",
        over ? "border-info bg-info/5" : "border-border",
        disabled && "opacity-60",
      )}
    >
      <FileUp className="size-4 shrink-0 text-muted-foreground" />
      <div className="min-w-0 flex-1">
        <div className="truncate text-[12.5px] font-medium">
          {file ? file.name : label}
        </div>
        <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
          {file ? formatBytes(file.size) : (hint ?? t("file.dropHint"))}
        </div>
      </div>
      {file && (
        <Button
          size="xs"
          variant="ghost"
          title={t("common.remove")}
          disabled={disabled}
          onClick={() => onFile(undefined)}
        >
          <X className="size-3.5" />
        </Button>
      )}
      <Button
        size="xs"
        disabled={disabled}
        onClick={() => input.current?.click()}
      >
        {t("file.choose")}
      </Button>
      <input
        ref={input}
        type="file"
        accept={accept}
        disabled={disabled}
        className="sr-only"
        onChange={(event) => {
          take(event.target.files);
          event.target.value = "";
        }}
      />
    </div>
  );
}
