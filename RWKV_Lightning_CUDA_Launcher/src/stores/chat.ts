import { useMemo } from "react";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { generationBody } from "@/lib/api/client";
import { inferenceApi } from "@/lib/api/inference";
import type { StreamEvent } from "@/lib/api/sse";
import { tNow } from "@/lib/i18n";
import { storage, useSettings } from "./settings";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  error?: string;
  finishReason?: string;
}

/**
 * One saved conversation. Sessions are scoped to a backend: switching nodes
 * never mixes conversations, and a node's history survives the switch.
 */
export interface ChatSession {
  id: string;
  backendId: string;
  /** Empty until the first user message names it; renameable afterwards. */
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

/**
 * How many conversations a single node keeps. Everything lives in
 * localStorage, which is a few megabytes per origin, so the oldest untouched
 * sessions are dropped rather than letting a quota error take the whole store
 * down with them.
 */
const MAX_SESSIONS_PER_BACKEND = 60;

/** Session title from its opening message: one line, trimmed to fit the list. */
export function deriveTitle(text: string) {
  const line = text.trim().split(/\r?\n/, 1)[0]?.trim() ?? "";
  return line.length > 32 ? `${line.slice(0, 32)}…` : line;
}

interface ChatState {
  sessions: ChatSession[];
  /** Selected session per backend ID. */
  currentId: Record<string, string>;
  /** Backend ID currently streaming, or null. */
  active: string | null;
  send: (backendId: string, text: string, retry?: boolean) => Promise<void>;
  stop: () => void;
  newSession: (backendId: string) => string;
  selectSession: (backendId: string, sessionId: string) => void;
  renameSession: (sessionId: string, title: string) => void;
  deleteSession: (sessionId: string) => void;
  /** Empties the current conversation without leaving a stub in the list. */
  reset: (backendId: string) => void;
  clearAll: () => void;
}

let controller: AbortController | null = null;

const emptySession = (backendId: string): ChatSession => ({
  id: crypto.randomUUID(),
  backendId,
  title: "",
  messages: [],
  createdAt: Date.now(),
  updatedAt: Date.now(),
});

/** Newest first, so a list never has to sort twice. */
const byRecency = (a: ChatSession, b: ChatSession) => b.updatedAt - a.updatedAt;

function pruned(sessions: ChatSession[], backendId: string, keepId: string) {
  const mine = sessions
    .filter((session) => session.backendId === backendId)
    .sort(byRecency);
  if (mine.length <= MAX_SESSIONS_PER_BACKEND) return sessions;
  const dropped = new Set(
    mine
      .slice(MAX_SESSIONS_PER_BACKEND)
      .filter((session) => session.id !== keepId)
      .map((session) => session.id),
  );
  return sessions.filter((session) => !dropped.has(session.id));
}

export const useChat = create(
  persist<ChatState>(
    (set, get) => ({
      sessions: [],
      currentId: {},
      active: null,

      newSession: (backendId) => {
        const session = emptySession(backendId);
        set((s) => ({
          // An untouched blank session is not worth keeping around: reuse it
          // instead of stacking "新会话" entries every time the button is hit.
          sessions: pruned(
            [
              ...s.sessions.filter(
                (entry) =>
                  entry.backendId !== backendId || entry.messages.length > 0,
              ),
              session,
            ],
            backendId,
            session.id,
          ),
          currentId: { ...s.currentId, [backendId]: session.id },
        }));
        return session.id;
      },

      selectSession: (backendId, sessionId) =>
        set((s) => ({ currentId: { ...s.currentId, [backendId]: sessionId } })),

      renameSession: (sessionId, title) =>
        set((s) => ({
          sessions: s.sessions.map((session) =>
            session.id === sessionId
              ? { ...session, title: title.trim().slice(0, 64) }
              : session,
          ),
        })),

      deleteSession: (sessionId) =>
        set((s) => {
          const target = s.sessions.find((session) => session.id === sessionId);
          const sessions = s.sessions.filter(
            (session) => session.id !== sessionId,
          );
          const currentId = { ...s.currentId };
          if (target && currentId[target.backendId] === sessionId) {
            const next = sessions
              .filter((session) => session.backendId === target.backendId)
              .sort(byRecency)[0];
            if (next) currentId[target.backendId] = next.id;
            else delete currentId[target.backendId];
          }
          return { sessions, currentId };
        }),

      reset: (backendId) =>
        set((s) => {
          const id = s.currentId[backendId];
          return {
            sessions: s.sessions.map((session) =>
              session.id === id
                ? { ...session, messages: [], updatedAt: Date.now() }
                : session,
            ),
          };
        }),

      send: async (backendId, text, retry = false) => {
        if (!backendId || get().active) return;

        let sessionId = get().currentId[backendId];
        if (!get().sessions.some((session) => session.id === sessionId))
          sessionId = get().newSession(backendId);

        const session = get().sessions.find(
          (entry) => entry.id === sessionId,
        ) as ChatSession;
        const existing = session.messages;
        if (retry ? existing.length === 0 : !text.trim()) return;

        let history: ChatMessage[];
        if (retry) {
          let last = existing.length - 1;
          while (last >= 0 && existing[last].role !== "user") last--;
          if (last < 0) return;
          history = existing.slice(0, last + 1);
        } else {
          history = [
            ...existing,
            { id: crypto.randomUUID(), role: "user", content: text.trim() },
          ];
        }

        const reply: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "",
        };
        controller = new AbortController();
        const signal = controller.signal;

        const write = (messages: ChatMessage[], title?: string) =>
          set((s) => ({
            sessions: s.sessions.map((entry) =>
              entry.id === sessionId
                ? {
                    ...entry,
                    messages,
                    title: title ?? entry.title,
                    updatedAt: Date.now(),
                  }
                : entry,
            ),
          }));

        write(
          [...history, reply],
          session.title || deriveTitle(history[0]?.content ?? ""),
        );
        set({ active: backendId });

        // Throttle store writes: a token stream would otherwise re-render on
        // every single SSE frame.
        let timer: number | undefined;
        const flush = () => {
          if (timer !== undefined) {
            window.clearTimeout(timer);
            timer = undefined;
          }
          set((s) => ({
            sessions: s.sessions.map((entry) =>
              entry.id === sessionId
                ? {
                    ...entry,
                    messages: entry.messages.map((m) =>
                      m.id === reply.id ? { ...reply } : m,
                    ),
                    updatedAt: Date.now(),
                  }
                : entry,
            ),
          }));
        };

        try {
          const generation = useSettings.getState().generation;
          await inferenceApi.streamChat(
            backendId,
            {
              ...generationBody(generation),
              stream: true,
              stop_tokens: [0, 261, 24281],
              messages: history.map(({ role, content }) => ({ role, content })),
            },
            signal,
            (event: StreamEvent) => {
              for (const choice of event.choices ?? []) {
                reply.content += choice.delta?.content ?? "";
                if (choice.finish_reason)
                  reply.finishReason = choice.finish_reason ?? undefined;
              }
              if (timer === undefined) timer = window.setTimeout(flush, 60);
            },
          );
        } catch (error) {
          reply.error = signal.aborted
            ? tNow("chat.stopped")
            : error instanceof Error
              ? error.message
              : String(error);
        } finally {
          flush();
          controller = null;
          set({ active: null });
        }
      },

      stop: () => controller?.abort(),

      clearAll: () => {
        controller?.abort();
        controller = null;
        set({ sessions: [], currentId: {}, active: null });
      },
    }),
    {
      name: "rwkv-chat-v3",
      version: 3,
      storage: createJSONStorage(() => storage),
      partialize: (s) =>
        ({ sessions: s.sessions, currentId: s.currentId }) as ChatState,
      migrate: (persisted, version) => {
        if (version >= 3) return persisted as ChatState;
        // v2 kept exactly one unnamed thread per backend; carry each one over
        // as that node's first saved session instead of dropping the history.
        const legacy = (
          persisted as { threads?: Record<string, ChatMessage[]> }
        )?.threads;
        const sessions: ChatSession[] = [];
        const currentId: Record<string, string> = {};
        for (const [backendId, messages] of Object.entries(legacy ?? {})) {
          if (!messages?.length) continue;
          const session = {
            ...emptySession(backendId),
            messages,
            title: deriveTitle(
              messages.find((m) => m.role === "user")?.content ?? "",
            ),
          };
          sessions.push(session);
          currentId[backendId] = session.id;
        }
        return { sessions, currentId } as ChatState;
      },
    },
  ),
);

const EMPTY_THREAD: ChatMessage[] = [];

/** The current session's messages, outside React. */
export function currentThread(backendId: string) {
  const { sessions, currentId } = useChat.getState();
  return (
    sessions.find((session) => session.id === currentId[backendId])?.messages ??
    EMPTY_THREAD
  );
}

export function useThread(backendId: string) {
  const sessions = useChat((s) => s.sessions);
  const sessionId = useChat((s) => s.currentId[backendId]);
  return useMemo(
    () =>
      sessions.find((session) => session.id === sessionId)?.messages ??
      EMPTY_THREAD,
    [sessions, sessionId],
  );
}

/** A node's saved conversations, newest first. */
export function useSessions(backendId: string) {
  const sessions = useChat((s) => s.sessions);
  return useMemo(
    () =>
      sessions
        .filter((session) => session.backendId === backendId)
        .sort(byRecency),
    [sessions, backendId],
  );
}

export function useCurrentSession(backendId: string) {
  const sessions = useChat((s) => s.sessions);
  const sessionId = useChat((s) => s.currentId[backendId]);
  return useMemo(
    () => sessions.find((session) => session.id === sessionId),
    [sessions, sessionId],
  );
}

export function useStreaming(backendId: string) {
  return useChat((s) => s.active === backendId);
}
