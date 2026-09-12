import { Laptop, Moon, Sun } from "lucide-react";
import { useSettings, type ThemeMode } from "../stores/settings";

const themes: { value: ThemeMode; label: string; icon: typeof Moon }[] = [
  { value: "dark", label: "Dark", icon: Moon },
  { value: "light", label: "Light", icon: Sun },
  { value: "system", label: "System", icon: Laptop },
];

export function ThemeControl({ compact = false }: { compact?: boolean }) {
  const theme = useSettings((state) => state.values.theme);
  const set = useSettings((state) => state.set);

  if (compact) {
    const selected = themes.find((item) => item.value === theme) ?? themes[0];
    const Icon = selected.icon;
    return (
      <label className="theme-select" title="Color theme">
        <Icon size={14} aria-hidden="true" />
        <span>{selected.label}</span>
        <select
          aria-label="Color theme"
          value={theme}
          onChange={(event) => set({ theme: event.target.value as ThemeMode })}
        >
          {themes.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
      </label>
    );
  }

  return (
    <div className="theme-options" role="radiogroup" aria-label="Color theme">
      {themes.map((item) => {
        const Icon = item.icon;
        return (
          <button
            key={item.value}
            type="button"
            role="radio"
            aria-checked={theme === item.value}
            className={theme === item.value ? "active" : ""}
            onClick={() => set({ theme: item.value })}
          >
            <Icon size={17} aria-hidden="true" />
            <span>{item.label}</span>
          </button>
        );
      })}
    </div>
  );
}
