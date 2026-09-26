import { useEffect, useRef, useState } from "react";
import { PanelRight, SquarePen } from "lucide-react";
import { useCurrent } from "@/app/use-current";
import { Composer } from "@/components/chat/composer";
import { GenerationPanel } from "@/components/chat/generation-panel";
import { MessageBubble } from "@/components/chat/message-bubble";
import { SessionPicker } from "@/components/chat/session-picker";
import { modelName } from "@/components/node-status";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import { useChat, useStreaming, useThread } from "@/stores/chat";
import { useChatPanel } from "@/stores/ui";

/** How close to the bottom the viewport must be to keep following the stream. */
const FOLLOW_THRESHOLD = 80;

/**
 * Below this page width the generation column stops being a column: docked,
 * it left the transcript narrower than the composer's own buttons. It floats
 * over the transcript instead, opened from the header.
 */
const DOCK_MIN_WIDTH = 760;

export function ChatPage() {
  const { t } = useI18n();
  const { backendId, runtime, canInfer } = useCurrent();
  const thread = useThread(backendId);
  const streaming = useStreaming(backendId);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pageRef = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  const docked = useChatPanel((s) => s.docked);
  const setDocked = useChatPanel((s) => s.setDocked);
  const [wide, setWide] = useState(true);
  const [floating, setFloating] = useState(false);

  const disabled = !backendId || !canInfer;
  const model = modelName(runtime);
  const panelOpen = wide ? docked : floating;

  // The page, not the window, decides: the rail and the log dock both take
  // width away without the window changing size.
  useEffect(() => {
    const node = pageRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      const next = entry.contentRect.width >= DOCK_MIN_WIDTH;
      setWide(next);
      if (next) setFloating(false);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  // Follow new tokens only while the reader is already at the bottom.
  useEffect(() => {
    const node = scrollRef.current;
    if (!node || !follow.current) return;
    node.scrollTop = node.scrollHeight;
  }, [thread, streaming]);

  // A different node is a different conversation: start at its latest message.
  useEffect(() => {
    follow.current = true;
  }, [backendId]);

  const togglePanel = () =>
    wide ? setDocked(!docked) : setFloating((open) => !open);

  return (
    <div
      ref={pageRef}
      className={cn(
        "relative grid h-full",
        wide && docked
          ? "grid-cols-[minmax(0,1fr)_280px]"
          : "grid-cols-[minmax(0,1fr)]",
      )}
    >
      <div className="flex min-h-0 min-w-0 flex-col">
        {/* The toolbar above already names the view, so this bar carries only
            what is specific to the conversation: which one, and what answers
            it. Icon buttons keep it one line at any width. */}
        <div className="flex h-12 shrink-0 items-center gap-1.5 px-3">
          <SessionPicker backendId={backendId} />
          <div className="min-w-0 flex-1" />
          <span
            title={model || t("chat.noModel")}
            className="min-w-0 truncate px-1 font-mono text-[11px] text-muted-foreground"
          >
            {model || t("chat.noModel")}
          </span>
          <Button
            size="icon-sm"
            variant="ghost"
            title={t("chat.newChat")}
            aria-label={t("chat.newChat")}
            disabled={!backendId}
            onClick={() => useChat.getState().newSession(backendId)}
          >
            <SquarePen className="size-4" />
          </Button>
          <Button
            size="icon-sm"
            variant="ghost"
            title={t("chat.parameters")}
            aria-label={t("chat.parameters")}
            aria-pressed={panelOpen}
            className={cn(panelOpen && "bg-muted text-foreground")}
            onClick={togglePanel}
          >
            <PanelRight className="size-4" />
          </Button>
        </div>

        <div
          ref={scrollRef}
          onScroll={(event) => {
            const node = event.currentTarget;
            follow.current =
              node.scrollHeight - node.scrollTop - node.clientHeight <
              FOLLOW_THRESHOLD;
          }}
          className="min-h-0 flex-1 overflow-auto px-4 sm:px-6"
        >
          {thread.length === 0 ? (
            // No card, no paragraph: an empty conversation is an empty page
            // with one line of intent, and the composer is the call to action.
            <div className="flex h-full flex-col items-center justify-center gap-1.5 pb-10 text-center">
              <img
                src="/logo.png"
                alt=""
                className="mb-2 size-9 object-contain opacity-80 dark:invert"
              />
              <h2 className="text-[17px] font-semibold tracking-[-0.015em]">
                {t("chat.emptyTitle")}
              </h2>
            </div>
          ) : (
            <div className="mx-auto flex max-w-[760px] flex-col gap-6 pt-2 pb-8">
              {thread.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </div>
          )}
        </div>

        {/* Same column as the transcript above, so the composer lines up with
            the messages instead of spanning the full pane. */}
        <div className="mx-auto w-full max-w-[760px] px-4 pb-4 sm:px-6">
          <Composer
            backendId={backendId}
            disabled={disabled}
            streaming={streaming}
          />
        </div>
      </div>

      {wide && docked && (
        <aside className="min-h-0 overflow-y-auto border-l border-border bg-card px-4 py-3.5">
          <GenerationPanel backendId={backendId} />
        </aside>
      )}

      {!wide && floating && (
        <>
          <div
            aria-hidden="true"
            className="absolute inset-0 z-20 bg-background/40"
            onClick={() => setFloating(false)}
          />
          <aside
            className="absolute inset-y-2 right-2 z-30 w-[min(300px,calc(100%-16px))] overflow-y-auto rounded-xl border border-border bg-card px-4 py-3.5 shadow-lift"
            onKeyDown={(event) => {
              if (event.key === "Escape") setFloating(false);
            }}
          >
            <GenerationPanel backendId={backendId} />
          </aside>
        </>
      )}
    </div>
  );
}
