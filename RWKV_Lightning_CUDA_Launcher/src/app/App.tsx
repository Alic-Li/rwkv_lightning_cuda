import { useEffect, useMemo, useState } from "react";
import {
  Zap,
  MessageSquare,
  Languages,
  ScanLine,
  Cpu,
  Settings,
  Plus,
  Search,
  PanelLeft,
  Command,
  Pencil,
  Trash2,
} from "lucide-react";
import { ChatPage } from "../pages/ChatPage";
import { TranslatePage } from "../pages/TranslatePage";
import { RuntimePage } from "../pages/RuntimePage";
import { StateTuningPage } from "../pages/StateTuningPage";
import { SettingsPage } from "../pages/SettingsPage";
import { RuntimeBadge, RuntimeControls } from "../components/RuntimeControls";
import { Dialog } from "../components/common";
import { resolveTheme, useSettings } from "../stores/settings";
import { useRuntime } from "../stores/runtime";
import { useChat } from "../stores/chat";
import { useShallow } from "zustand/react/shallow";
import { ThemeControl } from "../components/ThemeControl";
const routes = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "translate", label: "Parallel Translate", icon: Languages },
  { id: "state-tuning", label: "State Tuning", icon: ScanLine },
  { id: "runtime", label: "Runtime", icon: Cpu },
  { id: "settings", label: "Settings", icon: Settings },
];
function route() {
  return location.hash.replace("#/", "") || "chat";
}
export function App() {
  const [page, setPage] = useState(route);
  const [sidebar, setSidebar] = useState(false);
  const [query, setQuery] = useState("");
  const [command, setCommand] = useState(false);
  const [commandQuery, setCommandQuery] = useState("");
  const [edit, setEdit] = useState<{
    id: string;
    title: string;
    remove?: boolean;
  } | null>(null);
  const [storageError, setStorageError] = useState(false);
  // History subscribes only to titles/IDs, not every streamed message token.
  const history = useChat(
    useShallow((s) =>
      s.conversations.map((c) => JSON.stringify([c.id, c.title])),
    ),
  );
  const conversations = useMemo(
    () =>
      history.map((item) => {
        const [id, title] = JSON.parse(item) as [string, string];
        return { id, title };
      }),
    [history],
  );
  const { selected, newChat, select, rename, remove } = useChat(
    useShallow((s) => ({
      selected: s.selected,
      newChat: s.newChat,
      select: s.select,
      rename: s.rename,
      remove: s.remove,
    })),
  );
  const { runtime, refresh } = useRuntime();
  const theme = useSettings((s) => s.values.theme);
  const navigate = (id: string) => {
    location.hash = "/" + id;
    setSidebar(false);
  };
  const createChat = () => {
    newChat();
    navigate("chat");
  };
  useEffect(() => {
    const handler = () => {
      setPage(route());
      setSidebar(false);
    };
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);
  useEffect(() => {
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      await refresh();
      if (live) timer = setTimeout(poll, 1600);
    }
    void poll();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [refresh]);
  useEffect(() => {
    const media = matchMedia("(prefers-color-scheme: light)");
    const apply = () => {
      const resolved = resolveTheme(theme, media.matches);
      document.documentElement.dataset.theme = resolved;
      document.documentElement.style.colorScheme = resolved;
      document
        .querySelector('meta[name="theme-color"]')
        ?.setAttribute("content", resolved === "light" ? "#d5dfe5" : "#171d23");
    };
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [theme]);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCommand((v) => !v);
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "n") {
        e.preventDefault();
        useChat.getState().newChat();
        location.hash = "/chat";
      }
    };
    const storage = () => setStorageError(true);
    window.addEventListener("keydown", handler);
    window.addEventListener("storage-error", storage);
    return () => {
      window.removeEventListener("keydown", handler);
      window.removeEventListener("storage-error", storage);
    };
  }, []);
  const model =
    runtime.backend?.model?.name ||
    runtime.backend?.model?.id ||
    "No model loaded";
  return (
    <div className="window">
      <header className="titlebar">
        <button
          className="sidebar-toggle"
          title="Toggle sidebar"
          onClick={() => setSidebar(!sidebar)}
        >
          <PanelLeft size={17} />
        </button>
        <a className="brand" href="#/chat">
          <Zap size={18} />{" "}
          <strong>
            RWKV <span>Lightning</span>
          </strong>
        </a>
        <span className="titlebar-divider" />
        <span className="titlebar-page">
          {routes.find((r) => r.id === page)?.label || "Chat"}
        </span>
        <div className="spacer" />
        <span className="top-model" title={model}>
          {model}
        </span>
        <ThemeControl compact />
        <RuntimeBadge />
        <RuntimeControls />
      </header>
      <div className="app-body">
        <aside className={sidebar ? "sidebar open" : "sidebar"}>
          <button className="new-chat" onClick={createChat}>
            <Plus size={17} /> New chat <kbd>⌘ N</kbd>
          </button>
          <nav aria-label="Workspace">
            {routes.slice(0, 4).map((r) => (
              <button
                key={r.id}
                className={page === r.id ? "nav-item active" : "nav-item"}
                onClick={() => navigate(r.id)}
              >
                <r.icon size={17} />
                {r.label}
                {r.id === "runtime" && (
                  <span className={`nav-dot ${runtime.status}`} />
                )}
              </button>
            ))}
          </nav>
          <div className="history-heading">
            <span>CONVERSATIONS</span>
            <button title="Command palette" onClick={() => setCommand(true)}>
              <Search size={14} />
            </button>
          </div>
          <div className="history-search">
            <Search size={13} />
            <input
              aria-label="Search conversations"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search conversations…"
            />
          </div>
          <div className="history-list">
            {conversations
              .filter((c) =>
                c.title.toLowerCase().includes(query.toLowerCase()),
              )
              .map((c) => (
                <div
                  key={c.id}
                  className={`history-item ${selected === c.id && page === "chat" ? "selected" : ""}`}
                >
                  <button
                    className="history-title"
                    onClick={() => {
                      select(c.id);
                      navigate("chat");
                    }}
                    title={c.title}
                  >
                    {c.title}
                  </button>
                  <button
                    title={`Rename ${c.title}`}
                    onClick={() => setEdit({ id: c.id, title: c.title })}
                  >
                    <Pencil size={12} />
                  </button>
                  <button
                    title={`Delete ${c.title}`}
                    onClick={() =>
                      setEdit({ id: c.id, title: c.title, remove: true })
                    }
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            {!conversations.length && (
              <p className="history-empty">
                Your conversations,
                <br />
                right here on this device.
              </p>
            )}
          </div>
          <div className="sidebar-bottom">
            <button
              className={page === "settings" ? "nav-item active" : "nav-item"}
              onClick={() => navigate("settings")}
            >
              <Settings size={17} /> Settings
            </button>
            <button
              className="runtime-card"
              onClick={() => navigate("runtime")}
            >
              <RuntimeBadge />
              <strong title={model}>{model}</strong>
              <span>
                {runtime.base_url?.replace("http://", "") || "localhost:8000"}
              </span>
            </button>
            <div className="sidebar-caption">
              <span>LOCAL. RECURRENT. YOURS.</span>
              <button title="Command palette" onClick={() => setCommand(true)}>
                <Command size={12} />
                <span>K</span>
              </button>
            </div>
          </div>
        </aside>
        {sidebar && (
          <button
            className="sidebar-backdrop"
            aria-label="Close sidebar"
            onClick={() => setSidebar(false)}
          />
        )}
        <main className="workspace">
          {storageError && (
            <div className="error-panel" role="alert">
              Browser storage is full or unavailable. Changes remain in memory;
              export your results before closing.{" "}
              <button onClick={() => setStorageError(false)}>Dismiss</button>
            </div>
          )}
          {page === "runtime" ? (
            <RuntimePage />
          ) : page === "translate" ? (
            <TranslatePage />
          ) : page === "state-tuning" ? (
            <StateTuningPage />
          ) : page === "settings" ? (
            <SettingsPage />
          ) : (
            <ChatPage />
          )}
        </main>
      </div>
      <footer className="window-footer">
        <span>
          <i /> Local workspace
        </span>
        <span>RWKV Lightning · Native inference</span>
      </footer>
      {command && (
        <Dialog title="Go to…" onClose={() => setCommand(false)}>
          <input
            autoFocus
            className="command-search"
            aria-label="Search commands"
            placeholder="Search pages or commands…"
            value={commandQuery}
            onChange={(e) => setCommandQuery(e.target.value)}
          />
          <div className="command-list">
            {[{ id: "new", label: "New chat", icon: Plus }, ...routes]
              .filter((r) =>
                r.label.toLowerCase().includes(commandQuery.toLowerCase()),
              )
              .map((r) => (
                <button
                  key={r.id}
                  onClick={() => {
                    if (r.id === "new") createChat();
                    else navigate(r.id);
                    setCommand(false);
                    setCommandQuery("");
                  }}
                >
                  <r.icon size={17} />
                  {r.label}
                  <ArrowKey />
                </button>
              ))}
          </div>
        </Dialog>
      )}
      {edit && (
        <Dialog
          title={edit.remove ? "Delete conversation?" : "Rename conversation"}
          onClose={() => setEdit(null)}
        >
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (edit.remove) remove(edit.id);
              else rename(edit.id, edit.title);
              setEdit(null);
            }}
          >
            {edit.remove ? (
              <p>Delete “{edit.title}” from this device?</p>
            ) : (
              <input
                autoFocus
                aria-label="Conversation title"
                value={edit.title}
                onChange={(e) => setEdit({ ...edit, title: e.target.value })}
              />
            )}
            <div className="toolbar">
              <button type="button" onClick={() => setEdit(null)}>
                Cancel
              </button>
              <button className={edit.remove ? "danger" : "primary"}>
                {edit.remove ? "Delete" : "Save"}
              </button>
            </div>
          </form>
        </Dialog>
      )}
    </div>
  );
}
function ArrowKey() {
  return <span className="spacer key-hint">↵</span>;
}
