import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { RWKVClient } from "../lib/api/client";
import {
  useSettings,
  useSecret,
  storage,
  type GenerationSettings,
} from "./settings";
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  error?: string;
  finishReason?: string;
}
export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  settings?: GenerationSettings;
}
interface ChatStore {
  conversations: Conversation[];
  selected: string;
  active: string | null;
  controller: AbortController | null;
  newChat: () => string;
  select: (id: string) => void;
  rename: (id: string, title: string) => void;
  remove: (id: string) => void;
  clear: () => void;
  send: (text: string, retry?: boolean) => Promise<void>;
  stop: () => void;
}
export const useChat = create(
  persist<ChatStore>(
    (set, get) => ({
      conversations: [],
      selected: "",
      active: null,
      controller: null,
      newChat: () => {
        const id = crypto.randomUUID();
        set((s) => ({
          selected: id,
          conversations: [
            {
              id,
              title: "New conversation",
              createdAt: Date.now(),
              updatedAt: Date.now(),
              messages: [],
            },
            ...s.conversations,
          ],
        }));
        return id;
      },
      select: (id) => set({ selected: id }),
      rename: (id, title) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id ? { ...c, title: title.trim() || c.title } : c,
          ),
        })),
      remove: (id) => {
        if (get().active === id) get().stop();
        set((s) => ({
          conversations: s.conversations.filter((c) => c.id !== id),
          selected: s.selected === id ? "" : s.selected,
        }));
      },
      clear: () => {
        get().stop();
        set({ conversations: [], selected: "" });
      },
      stop: () => get().controller?.abort(),
      send: async (text, retry = false) => {
        if (get().active || (!retry && !text.trim())) return;
        const id = get().selected || get().newChat();
        let c = get().conversations.find((c) => c.id === id)!;
        let messages = c.messages;
        if (retry) {
          let last = messages.length - 1;
          while (last >= 0 && messages[last].role !== "user") last--;
          if (last < 0) return;
          messages = messages.slice(0, last + 1);
        } else
          messages = [
            ...messages,
            { id: crypto.randomUUID(), role: "user", content: text.trim() },
          ];
        const reply: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "",
        };
        const controller = new AbortController();
        const settings = { ...useSettings.getState().values.generation };
        c = {
          ...c,
          title: c.messages.length ? c.title : messages[0].content.slice(0, 42),
          updatedAt: Date.now(),
          settings,
          messages: [...messages, reply],
        };
        set((s) => ({
          conversations: s.conversations.map((x) => (x.id === id ? c : x)),
          active: id,
          controller,
        }));
        let timer: ReturnType<typeof setTimeout> | undefined;
        const flush = () => {
          clearTimeout(timer);
          timer = undefined;
          set((s) => ({
            conversations: s.conversations.map((x) =>
              x.id === id
                ? {
                    ...x,
                    updatedAt: Date.now(),
                    messages: x.messages.map((m) =>
                      m.id === reply.id ? { ...reply } : m,
                    ),
                  }
                : x,
            ),
          }));
        };
        try {
          const v = useSettings.getState().values;
          await new RWKVClient(v.baseURL, useSecret.getState().key).streamChat(
            {
              ...settings,
              stream: true,
              stop_tokens: [0, 261, 24281],
              messages: messages.map(({ role, content }) => ({
                role,
                content,
              })),
            },
            controller.signal,
            (event) => {
              for (const choice of event.choices || []) {
                reply.content += choice.delta?.content || "";
                if (choice.finish_reason)
                  reply.finishReason = choice.finish_reason;
              }
              if (!timer) timer = setTimeout(flush, 60);
            },
          );
        } catch (e) {
          reply.error = controller.signal.aborted
            ? "Generation stopped."
            : String(e);
        } finally {
          flush();
          set({ active: null, controller: null });
        }
      },
    }),
    {
      name: "rwkv-conversations-v1",
      storage: createJSONStorage(() => storage),
      partialize: (s) => ({ ...s, active: null, controller: null }),
    },
  ),
);
