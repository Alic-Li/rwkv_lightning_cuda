import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { ArrowUp, Brain, Layers, Puzzle, Square } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { useChat } from "@/stores/chat";
import { useSettings } from "@/stores/settings";

/** The draft grows with its text up to this height, then scrolls. */
const MAX_DRAFT_HEIGHT = 220;

const chipClass =
  "inline-flex h-7 min-w-0 max-w-[180px] items-center gap-1.5 rounded-full px-2.5 text-[11.5px] transition-colors [&_svg]:size-3.5 [&_svg]:shrink-0";

/** A setting that is on for the next message; shown only while it is on. */
function Chip({ icon, children }: { icon: ReactNode; children: string }) {
  return (
    <span
      title={children}
      className={cn(chipClass, "bg-muted text-muted-foreground")}
    >
      {icon}
      <span className="truncate">{children}</span>
    </span>
  );
}

/**
 * The three things the prompt's tail can do, as the server builds it:
 * - none: "Assistant: " and nothing else — the model decides whether to think.
 * - fast: a closed "<think></think" is prefilled, so thinking is skipped.
 * - free: a half-open "<think" is prefilled and the next tokens are masked
 *   against closing it, so the model has to think.
 * Omitting think_type is not "auto": the server then defaults to fast.
 */
const THINK_MODES = [
  { value: "none", label: "chat.thinkAuto", hint: "chat.thinkAutoHint" },
  { value: "fast", label: "chat.thinkOff", hint: "chat.thinkOffHint" },
  { value: "free", label: "chat.thinkOn", hint: "chat.thinkOnHint" },
] as const;

function ThinkSwitch({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  const { t } = useI18n();
  return (
    <div
      role="radiogroup"
      aria-label={t("chat.thinkLabel")}
      className="inline-flex h-7 shrink-0 items-center rounded-full border border-border p-0.5 text-[11.5px]"
    >
      <Brain
        aria-hidden="true"
        className="mr-0.5 ml-1.5 size-3.5 shrink-0 text-muted-foreground"
      />
      {THINK_MODES.map((mode) => {
        const active = value === mode.value;
        return (
          <button
            key={mode.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={t(mode.hint)}
            onClick={() => onChange(mode.value)}
            className={cn(
              "h-full rounded-full px-2 whitespace-nowrap transition-colors",
              active
                ? "bg-foreground text-background"
                : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
          >
            {t(mode.label)}
          </button>
        );
      })}
    </div>
  );
}

/** Draft box pinned to the bottom of the chat column. */
export function Composer({
  backendId,
  disabled,
  streaming,
}: {
  backendId: string;
  disabled: boolean;
  streaming: boolean;
}) {
  const { t } = useI18n();
  const [draft, setDraft] = useState("");
  const generation = useSettings((s) => s.generation);
  const setGeneration = useSettings((s) => s.setGeneration);
  const stop = useChat((s) => s.stop);
  const textRef = useRef<HTMLTextAreaElement>(null);

  // Grow with the draft: one line at rest leaves the transcript the room,
  // and a long prompt is still readable without a scrollbar in a slot.
  useLayoutEffect(() => {
    const node = textRef.current;
    if (!node) return;
    node.style.height = "auto";
    // Empty: the min-height is the answer. Measuring would count a wrapped
    // placeholder, and nothing re-measures when the column later widens.
    if (!draft) return;
    node.style.height = `${Math.min(node.scrollHeight, MAX_DRAFT_HEIGHT)}px`;
  }, [draft]);

  const submit = () => {
    const text = draft.trim();
    if (!text || disabled || streaming) return;
    setDraft("");
    void useChat.getState().send(backendId, text);
  };

  const canSend = Boolean(draft.trim()) && !disabled;
  const stateNeedsAuto =
    Boolean(generation.state_id) && generation.think_type !== "none";

  return (
    <div>
      <div
        className={cn(
          "rounded-2xl border border-border bg-card shadow-flat transition-colors",
          "focus-within:border-border-strong",
        )}
      >
        <textarea
          ref={textRef}
          value={draft}
          rows={1}
          placeholder={t("chat.placeholder")}
          className="block max-h-[220px] min-h-[48px] w-full resize-none bg-transparent px-4 pt-3.5 pb-1 text-[13.5px] leading-relaxed outline-none placeholder:text-muted-foreground/70"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key !== "Enter" || event.shiftKey) return;
            // IME composition commits with Enter too; never send from it.
            if (event.nativeEvent.isComposing) return;
            event.preventDefault();
            submit();
          }}
        />

        <div className="flex items-center gap-1.5 px-2.5 pb-2.5">
          {/* Thinking is a per-message decision, so its switch lives where the
              message is written. State and adapter are chosen in the panel;
              they appear here only while they are in effect. */}
          <ThinkSwitch
            value={generation.think_type}
            onChange={(think_type) => setGeneration({ think_type })}
          />
          <div className="flex min-w-0 flex-1 items-center gap-1.5 overflow-hidden">
            {generation.state_id && (
              <Chip icon={<Layers />}>{generation.state_id}</Chip>
            )}
            {generation.adapter_id && (
              <Chip icon={<Puzzle />}>{generation.adapter_id}</Chip>
            )}
          </div>
          {streaming ? (
            <button
              type="button"
              title={t("chat.stop")}
              aria-label={t("chat.stop")}
              onClick={stop}
              className="inline-flex size-8 shrink-0 items-center justify-center rounded-full bg-foreground text-background transition-opacity hover:opacity-85"
            >
              <Square className="size-3 fill-current" />
            </button>
          ) : (
            <button
              type="button"
              title={t("chat.sendHint")}
              aria-label={t("chat.send")}
              disabled={!canSend}
              onClick={submit}
              className="inline-flex size-8 shrink-0 items-center justify-center rounded-full bg-foreground text-background transition-opacity hover:opacity-85 disabled:bg-muted disabled:text-muted-foreground/60"
            >
              <ArrowUp className="size-4" strokeWidth={2.25} />
            </button>
          )}
        </div>
      </div>

      {disabled && backendId && (
        <p className="mt-2 text-center text-[11.5px] text-muted-foreground">
          {t("chat.runtimeDown")}
          <a
            href="#/runtime"
            className="text-foreground underline decoration-border-strong underline-offset-2 hover:decoration-foreground"
          >
            {t("chat.goStart")}
          </a>
        </p>
      )}
      {/* A state is tuned on bare "User:/Assistant:" turns, and the server
          only falls back to that when think_type is absent — which the
          console never sends. Say so and offer the switch; do not flip it
          behind the user's back. */}
      {!disabled && stateNeedsAuto && (
        <p className="mt-2 flex items-center justify-center gap-1.5 text-center text-[11.5px] text-muted-foreground">
          <span className="size-1.5 shrink-0 rounded-full bg-warning" />
          {t("chat.stateThinkHint")}
          <button
            type="button"
            onClick={() => setGeneration({ think_type: "none" })}
            className="text-foreground underline decoration-border-strong underline-offset-2 hover:decoration-foreground"
          >
            {t("chat.stateThinkSwitch")}
          </button>
        </p>
      )}
    </div>
  );
}
