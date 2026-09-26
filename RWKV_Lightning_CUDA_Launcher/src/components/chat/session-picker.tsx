import { useState } from "react";
import { Check, History, Pencil, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { formatRelativeTime } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/utils";
import {
  useChat,
  useCurrentSession,
  useSessions,
  type ChatSession,
} from "@/stores/chat";

/**
 * Saved conversations for the active node. The transcript pane is already
 * split three ways, so the history lives behind one button instead of a
 * permanent column — the count on the trigger is what makes it discoverable.
 */
export function SessionPicker({ backendId }: { backendId: string }) {
  const { t } = useI18n();
  const sessions = useSessions(backendId);
  const current = useCurrentSession(backendId);
  const select = useChat((s) => s.selectSession);
  const remove = useChat((s) => s.deleteSession);
  const streaming = useChat((s) => s.active !== null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState<string | null>(null);

  const visible = filterSessions(sessions, query, t("chat.untitled"));

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          size="sm"
          variant="ghost"
          disabled={!backendId}
          className="min-w-0 text-foreground"
        >
          <History className="size-3.5 shrink-0" />
          <span className="min-w-0 max-w-[180px] truncate">
            {current
              ? titleOf(current, t("chat.untitled"))
              : t("chat.conversations")}
          </span>
          {sessions.length > 0 && (
            <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
              {sessions.length}
            </span>
          )}
        </Button>
      </PopoverTrigger>

      <PopoverContent align="end" className="w-[320px] p-1.5">
        {sessions.length > 3 && (
          <Input
            value={query}
            autoFocus
            placeholder={t("chat.searchConversations")}
            className="mb-1.5 h-8"
            onChange={(event) => setQuery(event.target.value)}
          />
        )}

        <div className="max-h-[320px] overflow-y-auto">
          {visible.length === 0 ? (
            <p className="px-2 py-3 text-[11.5px] text-muted-foreground">
              {t("chat.noConversations")}
            </p>
          ) : (
            visible.map((session) =>
              editing === session.id ? (
                <RenameRow
                  key={session.id}
                  session={session}
                  placeholder={t("chat.untitled")}
                  onDone={() => setEditing(null)}
                />
              ) : (
                <div
                  key={session.id}
                  className={cn(
                    "group flex items-center gap-1 rounded-lg px-1 transition-colors hover:bg-muted",
                    session.id === current?.id && "bg-accent",
                  )}
                >
                  <button
                    type="button"
                    className="min-w-0 flex-1 px-1 py-2 text-left"
                    onClick={() => {
                      select(backendId, session.id);
                      setOpen(false);
                    }}
                  >
                    <span className="block truncate text-[12.5px] font-medium">
                      {titleOf(session, t("chat.untitled"))}
                    </span>
                    <span className="mt-0.5 block truncate text-[11px] text-muted-foreground">
                      {t("chat.conversationCount", {
                        count: session.messages.length,
                      })}
                      {" · "}
                      {formatRelativeTime(Math.floor(session.updatedAt / 1000))}
                    </span>
                  </button>
                  <button
                    type="button"
                    title={t("chat.renameConversation")}
                    aria-label={t("chat.renameConversation")}
                    className="shrink-0 rounded-md p-1.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:text-foreground focus-visible:opacity-100"
                    onClick={() => setEditing(session.id)}
                  >
                    <Pencil className="size-3.5" />
                  </button>
                  <button
                    type="button"
                    title={t("chat.deleteConversation")}
                    aria-label={t("chat.deleteConversation")}
                    // Deleting the session that is still streaming would strand
                    // the request writing into a row that no longer exists.
                    disabled={streaming && session.id === current?.id}
                    className="shrink-0 rounded-md p-1.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:text-destructive focus-visible:opacity-100 disabled:opacity-30"
                    onClick={() => remove(session.id)}
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </div>
              ),
            )
          )}
        </div>

        <p className="border-t border-border px-2 pt-2 pb-1 text-[11px] leading-relaxed text-muted-foreground">
          {t("chat.historyHint")}
        </p>
      </PopoverContent>
    </Popover>
  );
}

/** A session shows its own title, or the placeholder until it has one. */
export function titleOf(session: ChatSession, fallback: string) {
  return session.title.trim() || fallback;
}

/**
 * Search runs over what the list actually shows, so typing the placeholder
 * finds the conversations that have not been named yet.
 */
export function filterSessions(
  sessions: ChatSession[],
  query: string,
  fallback: string,
) {
  const needle = query.trim().toLowerCase();
  if (!needle) return sessions;
  return sessions.filter((session) =>
    titleOf(session, fallback).toLowerCase().includes(needle),
  );
}

function RenameRow({
  session,
  placeholder,
  onDone,
}: {
  session: ChatSession;
  placeholder: string;
  onDone: () => void;
}) {
  const rename = useChat((s) => s.renameSession);
  const [draft, setDraft] = useState(session.title);
  const commit = () => {
    rename(session.id, draft);
    onDone();
  };
  return (
    <div className="flex items-center gap-1 px-1 py-1">
      <Input
        value={draft}
        autoFocus
        placeholder={placeholder}
        className="h-8"
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") commit();
          if (event.key === "Escape") onDone();
        }}
      />
      <button
        type="button"
        className="shrink-0 rounded-md p-1.5 text-muted-foreground hover:text-foreground"
        onClick={commit}
      >
        <Check className="size-3.5" />
      </button>
      <button
        type="button"
        className="shrink-0 rounded-md p-1.5 text-muted-foreground hover:text-foreground"
        onClick={onDone}
      >
        <X className="size-3.5" />
      </button>
    </div>
  );
}
