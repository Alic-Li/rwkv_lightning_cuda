import { useState } from "react";
import { useSettings, useSecret } from "../stores/settings";
import { useChat } from "../stores/chat";
import { useTranslate } from "../stores/translate";
import { Panel, Field, Dialog } from "../components/common";
import { GenerationSettings } from "../components/GenerationSettings";
import { LanguageInput, languages } from "./TranslatePage";
import { ThemeControl } from "../components/ThemeControl";
export function SettingsPage() {
  const { values, set, reset } = useSettings();
  const { key, setKey } = useSecret();
  const [confirm, setConfirm] = useState("");
  const translationBusy = useTranslate((s) => s.busy);
  return (
    <div className="page settings-page">
      <header className="page-heading">
        <div>
          <div className="eyebrow">MAKE IT YOURS</div>
          <h1>Settings</h1>
          <p>Preferences are saved automatically on this device.</p>
        </div>
      </header>
      <Panel title="Appearance">
        <Field label="Theme">
          <ThemeControl />
        </Field>
        <p className="small muted">
          System follows the operating system and updates immediately when its
          appearance changes.
        </p>
      </Panel>
      <Panel title="API connection">
        <Field
          label="Base URL"
          hint="Empty means the same-origin Go Launcher, which follows the actual runtime port."
        >
          <input
            type="url"
            value={values.baseURL}
            placeholder="Automatic · this Go Launcher"
            onChange={(e) => set({ baseURL: e.target.value.trim() })}
          />
        </Field>
        <Field
          label="API key"
          hint="Session only; not written to local storage."
        >
          <input
            type="password"
            autoComplete="off"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="Optional Bearer token"
          />
        </Field>
        <p className="small muted">
          A custom base URL is for direct native Chat connections, without a
          trailing /v1. Raw parallel translation uses the Launcher adapter.
          Managed runtime passwords are supplied automatically by Go.
        </p>
      </Panel>
      <Panel title="Translation defaults">
        <datalist id="languages">
          {languages.map((l) => (
            <option key={l} value={l} />
          ))}
        </datalist>
        <div className="form-grid">
          <LanguageInput
            label="Source language"
            value={values.sourceLanguage}
            onChange={(sourceLanguage) => set({ sourceLanguage })}
          />
          <LanguageInput
            label="Target language"
            value={values.targetLanguage}
            onChange={(targetLanguage) => set({ targetLanguage })}
          />
          <Field label="Concurrency">
            <input
              type="number"
              min="1"
              max="64"
              value={values.concurrency}
              onChange={(e) => set({ concurrency: Number(e.target.value) })}
            />
          </Field>
          <Field label="Chunk target (characters)">
            <input
              type="number"
              min="32"
              value={values.chunkTarget}
              onChange={(e) => set({ chunkTarget: Number(e.target.value) })}
            />
          </Field>
        </div>
        <p className="small muted">
          Auto is a literal prompt language name; no language detector is run.
        </p>
      </Panel>
      <Panel title="Generation defaults">
        <GenerationSettings />
      </Panel>
      <Panel title="Local storage">
        <p className="muted">
          Conversation text, translation results and preferences stay in this
          browser. Large jobs are saved locally; export important results.
        </p>
        <div className="toolbar">
          <button onClick={() => setConfirm("Clear conversations")}>
            Clear conversations
          </button>
          <button
            disabled={translationBusy}
            onClick={() => setConfirm("Clear translation history")}
          >
            Clear translation history
          </button>
          <button onClick={() => setConfirm("Reset settings")}>
            Reset settings
          </button>
        </div>
      </Panel>
      {confirm && (
        <Dialog title={confirm} onClose={() => setConfirm("")}>
          <p>This removes the selected data from this browser.</p>
          <div className="toolbar">
            <button onClick={() => setConfirm("")}>Cancel</button>
            <button
              className="danger"
              onClick={() => {
                if (confirm === "Clear conversations")
                  useChat.getState().clear();
                else if (confirm === "Clear translation history")
                  useTranslate.getState().clear();
                else reset();
                setConfirm("");
              }}
            >
              {confirm}
            </button>
          </div>
        </Dialog>
      )}
    </div>
  );
}
