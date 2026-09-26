import { memo, useMemo } from "react";
import Markdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { CodeBlock } from "@/components/chat/code-block";
import { CopyButton } from "@/components/common";
import { Notice } from "@/components/ui/primitives";
import { formatHTMLForMarkdown } from "@/lib/chat/html";
import { cn } from "@/lib/utils";
import { useChat, type ChatMessage } from "@/stores/chat";

/** Fenced blocks get a bar with copy and, for HTML, a live preview. */
const markdownComponents = { pre: CodeBlock };

/**
 * One conversation turn. User turns are right-aligned bubbles; assistant turns
 * render Markdown (with the loose-HTML fence fix) on a transparent background.
 */
export const MessageBubble = memo(function MessageBubble({
  message,
}: {
  message: ChatMessage;
}) {
  // A reply is only "streaming" while its own thread is producing tokens; an
  // empty streaming reply must not flash the "…" placeholder.
  const streaming = useChat((s) => s.active !== null);
  const isUser = message.role === "user";

  const rendered = useMemo(
    () => (isUser ? message.content : formatHTMLForMarkdown(message.content)),
    [isUser, message.content],
  );

  const empty = message.content.trim().length === 0;
  const unusualFinish =
    Boolean(message.finishReason) && message.finishReason !== "stop";

  return (
    <article
      className={cn(
        // Sides say who is speaking; a name over every turn was a caption
        // repeating what the layout already shows.
        "group/message flex min-w-0 flex-col gap-1.5",
        isUser ? "items-end" : "items-start",
      )}
    >
      {isUser ? (
        <div className="max-w-[86%] rounded-2xl rounded-br-md bg-muted px-4 py-2.5 text-[13.5px] leading-relaxed break-words whitespace-pre-wrap">
          {message.content}
        </div>
      ) : (
        <div className="md-body w-full min-w-0">
          <Markdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={markdownComponents}
          >
            {empty && !streaming ? "…" : rendered}
          </Markdown>
        </div>
      )}

      {message.error && (
        <Notice tone="danger" className="w-full">
          {message.error}
        </Notice>
      )}

      {/* Actions stay out of the reading until the turn is pointed at; a
          finish reason is only news when it is not a plain "stop". */}
      <div
        className={cn(
          "flex flex-wrap items-center gap-1.5 opacity-0 transition-opacity group-focus-within/message:opacity-100 group-hover/message:opacity-100",
          isUser && "justify-end",
          unusualFinish && "opacity-100",
        )}
      >
        <CopyButton text={message.content} />
        {unusualFinish && (
          <span className="font-mono text-[11px] text-muted-foreground">
            finish_reason: {message.finishReason}
          </span>
        )}
      </div>
    </article>
  );
});
