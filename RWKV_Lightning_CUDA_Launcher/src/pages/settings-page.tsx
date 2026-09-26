import { useEffect, useState } from "react";
import { Laptop, Moon, Sun, Trash2 } from "lucide-react";
import { useCurrent } from "@/app/use-current";
import { PageHeader } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Field, Segmented } from "@/components/ui/field";
import { Input, MonoInput } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Notice } from "@/components/ui/primitives";
import { useI18n, type Lang } from "@/lib/i18n";
import { languages, normalizeLanguage } from "@/lib/translate/languages";
import { backendKindLabel, useBackends } from "@/stores/backends";
import { useChat } from "@/stores/chat";
import { useRuntimeFormEntry } from "@/stores/forms";
import { useSecret, useSettings, type ThemeMode } from "@/stores/settings";
import { useTranslate } from "@/stores/translate";
import { toast } from "@/stores/ui";

export function SettingsPage() {
  const { t, lang } = useI18n();
  const theme = useSettings((s) => s.theme);
  const sourceLanguage = useSettings((s) => s.sourceLanguage);
  const targetLanguage = useSettings((s) => s.targetLanguage);
  const concurrency = useSettings((s) => s.concurrency);
  const set = useSettings((s) => s.set);
  const reset = useSettings((s) => s.reset);
  const apiKey = useSecret((s) => s.key);
  const setKey = useSecret((s) => s.setKey);
  const { backend, backendId } = useCurrent();
  const list = useBackends((s) => s.list);
  const translationBusy = useTranslate((s) => s.busy);
  const runtimePassword = useRuntimeFormEntry(backendId).config.password;
  const [origin, setOrigin] = useState("");

  useEffect(() => setOrigin(window.location.origin), []);

  return (
    <div className="max-w-[820px] px-6 pt-5.5 pb-10">
      <PageHeader title={t("settings.title")} />

      <Card className="mt-5">
        <CardHeader>
          <CardTitle>{t("settings.appearance")}</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4">
          <p className="text-xs text-muted-foreground">
            {t("settings.appearanceHint")}
          </p>
          <div className="grid max-w-[420px] gap-4">
            <Field label={t("settings.theme")}>
              <Segmented<ThemeMode>
                value={theme}
                onChange={(value) => set({ theme: value })}
                options={[
                  {
                    value: "system",
                    label: (
                      <span className="inline-flex items-center gap-1.5">
                        <Laptop className="size-3.5" />
                        {t("settings.themeSystem")}
                      </span>
                    ),
                  },
                  {
                    value: "dark",
                    label: (
                      <span className="inline-flex items-center gap-1.5">
                        <Moon className="size-3.5" />
                        {t("settings.themeDark")}
                      </span>
                    ),
                  },
                  {
                    value: "light",
                    label: (
                      <span className="inline-flex items-center gap-1.5">
                        <Sun className="size-3.5" />
                        {t("settings.themeLight")}
                      </span>
                    ),
                  },
                ]}
              />
            </Field>
            <Field label={t("settings.language")}>
              <Segmented<Lang>
                value={lang}
                onChange={(value) => set({ lang: value })}
                options={[
                  { value: "zh", label: "中文" },
                  { value: "en", label: "English" },
                ]}
              />
            </Field>
          </div>
        </CardContent>
      </Card>

      <Card className="mt-3.5">
        <CardHeader>
          <CardTitle>{t("settings.client")}</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3.5">
          <p className="text-xs text-muted-foreground">
            {t("settings.clientHint")}
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label={t("settings.clientAddress")}>
              <MonoInput value={origin} readOnly className="bg-muted" />
            </Field>
            <Field label={t("settings.registryFile")}>
              <MonoInput
                value="~/.rwkv_launcher/launcher.json"
                readOnly
                className="bg-muted"
              />
            </Field>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field
              label={t("settings.currentNode")}
              hint={t("settings.nodeHint")}
            >
              <Select
                value={backendId}
                onChange={(event) =>
                  useBackends.getState().select(event.target.value)
                }
              >
                {list.length === 0 && <option value="">—</option>}
                {list.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} ·{" "}
                    {backendKindLabel(item.kind) || t("backend.kind.unknown")}
                  </option>
                ))}
              </Select>
            </Field>
            <Field
              label={t("settings.agentToken")}
              hint={t("backend.tokenHint")}
            >
              <Input
                type="password"
                autoComplete="off"
                value={apiKey}
                placeholder={t("common.optional")}
                onChange={(event) => setKey(event.target.value)}
              />
            </Field>
          </div>
          {backend?.id === "local" && (
            <Badge variant="info">{t("backend.localReserved")}</Badge>
          )}
        </CardContent>
      </Card>

      <Card className="mt-3.5">
        <CardHeader>
          <CardTitle>{t("settings.translateDefaults")}</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-3">
          <Field label={t("settings.sourceLanguage")}>
            <Select
              value={normalizeLanguage(sourceLanguage, "English")}
              onChange={(event) => set({ sourceLanguage: event.target.value })}
            >
              {languages.map((language) => (
                <option key={language} value={language}>
                  {language}
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t("settings.targetLanguage")}>
            <Select
              value={normalizeLanguage(targetLanguage, "Chinese")}
              onChange={(event) => set({ targetLanguage: event.target.value })}
            >
              {languages.map((language) => (
                <option key={language} value={language}>
                  {language}
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t("settings.batchSize")}>
            <MonoInput
              type="number"
              min={1}
              max={128}
              value={concurrency}
              onChange={(event) =>
                set({ concurrency: Number(event.target.value) })
              }
            />
          </Field>
        </CardContent>
      </Card>

      <Card className="mt-3.5">
        <CardHeader>
          <CardTitle>{t("settings.resetTitle")}</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          <Notice tone="warning">{t("settings.storageWarning")}</Notice>
          <div className="flex flex-wrap gap-2">
            <Button
              onClick={() => {
                useChat.getState().clearAll();
                toast.success(t("chat.clear"));
              }}
            >
              <Trash2 className="size-3.5" />
              {t("chat.clear")}
            </Button>
            <Button
              disabled={translationBusy}
              onClick={() => {
                useTranslate.getState().clear();
                toast.success(t("translate.clear"));
              }}
            >
              <Trash2 className="size-3.5" />
              {t("translate.clear")}
            </Button>
            <Button
              onClick={() => {
                reset();
                useSecret.getState().setKey("");
                toast.success(t("settings.saved"));
              }}
            >
              <Trash2 className="size-3.5" />
              {t("common.reset")}
            </Button>
          </div>
          {runtimePassword && (
            <p className="font-mono text-[11px] text-muted-foreground">
              {t("settings.passwordHint")}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
