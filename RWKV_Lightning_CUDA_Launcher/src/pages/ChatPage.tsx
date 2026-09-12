import { memo, useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import {
  ArrowUp,
  Square,
  RotateCcw,
  MessageSquare,
  Languages,
  ScanLine,
  Zap,
  SlidersHorizontal,
} from "lucide-react";
import { useChat, type ChatMessage } from "../stores/chat";
import { useRuntime } from "../stores/runtime";
import { useSettings } from "../stores/settings";
import { GenerationSettings } from "../components/GenerationSettings";
import { CopyButton } from "../components/common";
const Message = memo(function Message({ message }: { message: ChatMessage }) {
  return (
    <article className={`message ${message.role}`}>
      <div className="message-label">
        {message.role === "user" ? (
          "YOU"
        ) : (
          <>
            <Zap size={13} /> RWKV
          </>
        )}
      </div>
      <div className="markdown">
        <Markdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
          {message.content || "…"}
        </Markdown>
      </div>
      {message.error && (
        <div className="inline-error" role="alert">
          {message.error}
        </div>
      )}
      <div className="message-tools">
        <CopyButton text={message.content} />
        {message.finishReason && (
          <span>finish_reason: {message.finishReason}</span>
        )}
      </div>
    </article>
  );
});
export function ChatPage() {
  const { conversations, selected, active, send, stop } = useChat();
  const conversation = conversations.find((c) => c.id === selected);
  const [text, setText] = useState("");
  const [settings, setSettings] = useState(false);
  const runtime = useRuntime((s) => s.runtime);
  const baseURL = useSettings((s) => s.values.baseURL);
  const generation = useSettings((s) => s.values.generation);
  const body = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const follow = useRef(true);
  const ready =
    (runtime.status === "ready" && runtime.backend?.model?.loaded !== false) ||
    !!baseURL;
  useEffect(() => {
    if (follow.current && body.current)
      body.current.scrollTop = body.current.scrollHeight;
  }, [conversation?.messages]);
  useEffect(() => {
    follow.current = true;
    input.current?.focus();
  }, [selected]);
  const submit = () => {
    if (!text.trim() || active || !ready) return;
    void send(text);
    setText("");
  };
  return (
    <div className="chat-page">
      <div className="workspace-heading">
        <span>{conversation?.title || "New conversation"}</span>
        <span className="muted">
          Private by default · stored on this device
        </span>
      </div>
      <div
        className="chat-scroll"
        ref={body}
        onScroll={() => {
          const b = body.current!;
          follow.current = b.scrollHeight - b.scrollTop - b.clientHeight < 80;
        }}
      >
        <div className="chat-content">
          {conversation?.messages.length ? (
            conversation.messages.map((m) => <Message key={m.id} message={m} />)
          ) : (
            <div className="chat-empty">
              <div className="brand-mark">
                <Zap size={31} strokeWidth={1.5} />
              </div>
              <div className="eyebrow">YOUR LOCAL WORKSPACE</div>
              <h1>RWKV Lightning</h1>
              <p>
                Local recurrent inference,
                <br />
                without the heavyweight UI.
              </p>
              <div className="suggestions">
                <button
                  onClick={() => {
                    setText("Explain how RWKV recurrent inference works.");
                    input.current?.focus();
                  }}
                >
                  <MessageSquare size={19} />
                  <strong>Ask a question</strong>
                  <span>Think through something</span>
                </button>
                <button onClick={() => (location.hash = "/translate")}>
                  <Languages size={19} />
                  <strong>Translate a document</strong>
                  <span>Short requests. In parallel.</span>
                </button>
                <button onClick={() => (location.hash = "/state-tuning")}>
                  <ScanLine size={19} />
                  <strong>Tune a state</strong>
                  <span>Make the model your own</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="composer-wrap">
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <textarea
            ref={input}
            aria-label="Message RWKV"
            placeholder={
              ready ? "Ask RWKV…" : "Start runtime and load a model to begin…"
            }
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                submit();
              }
            }}
          />
          {settings && <GenerationSettings compact />}
          <div className="composer-tools">
            <span className="model-chip">
              <i />
              {runtime.backend?.model?.name ||
                runtime.backend?.model?.id ||
                "Local model"}
            </span>
            <button
              type="button"
              title="Generation settings"
              onClick={() => setSettings(!settings)}
            >
              <SlidersHorizontal size={14} />
              <span>T {generation.temperature}</span>
            </button>
            <div className="spacer" />
            {conversation?.messages.some((m) => m.role === "user") &&
              !active && (
                <button
                  type="button"
                  disabled={!ready}
                  onClick={() => void send("", true)}
                >
                  <RotateCcw size={14} /> Regenerate / Retry
                </button>
              )}
            {active ? (
              <button type="button" className="stop-generation" onClick={stop}>
                <Square size={14} /> Stop
              </button>
            ) : (
              <button
                className="send-button primary"
                aria-label="Send message"
                disabled={!text.trim() || !ready}
              >
                <ArrowUp size={20} />
              </button>
            )}
          </div>
        </form>
        <div className="composer-caption">
          {ready
            ? "Enter to send · Shift + Enter for a new line"
            : "Runtime or model is not ready."}{" "}
          {!ready && <a href="#/runtime">Configure runtime →</a>}
          <span>RWKV can make mistakes. Check important details.</span>
        </div>
      </div>
    </div>
  );
}
