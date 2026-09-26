import { tNow } from "@/lib/i18n";

const KIB = 1024;
const UNITS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];

/** Human readable byte size; the console shows GPU memory in GiB. */
export function formatBytes(bytes: number | undefined | null, digits = 1) {
  if (bytes === undefined || bytes === null || !Number.isFinite(bytes))
    return "—";
  if (bytes <= 0) return "0 B";
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(KIB)),
    UNITS.length - 1,
  );
  const value = bytes / KIB ** exponent;
  return `${value.toFixed(exponent === 0 ? 0 : digits)} ${UNITS[exponent]}`;
}

/**
 * `61.5 / 95.6 GB` — the unit once, as the mockup's `memText` does. Spelling
 * it twice costs three characters that the 216px rail does not have: the row
 * shares its line with the temperature/power pair, and once the two overflow
 * flexbox shrinks and wraps *both* of them.
 */
export function formatGigabytePair(
  used: number | undefined | null,
  total: number | undefined | null,
  digits = 1,
) {
  const finite = (value: number | undefined | null): value is number =>
    value !== undefined && value !== null && Number.isFinite(value);
  if (!finite(used) || !finite(total)) return "—";
  return `${(used / KIB ** 3).toFixed(digits)} / ${(total / KIB ** 3).toFixed(digits)} GB`;
}

/** `elapsed` from ProcessStatus is in seconds. */
export function formatDuration(seconds: number | undefined | null) {
  if (seconds === undefined || seconds === null || !Number.isFinite(seconds))
    return "—";
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

/** Unix seconds -> coarse "3m ago" style label. */
export function formatRelativeTime(
  unixSeconds: number | undefined | null,
  now = Date.now(),
) {
  if (!unixSeconds) return "—";
  const delta = Math.max(0, Math.floor(now / 1000 - unixSeconds));
  if (delta < 5) return tNow("time.justNow");
  if (delta < 60) return `${delta}s`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h`;
  return `${Math.floor(delta / 86400)}d`;
}

export function formatCount(value: number | undefined | null) {
  if (value === undefined || value === null || !Number.isFinite(value))
    return "—";
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 10_000) return `${(value / 1000).toFixed(1)}k`;
  return value.toLocaleString("en-US");
}

export function formatNumber(value: number | undefined | null, digits = 3) {
  if (value === undefined || value === null || !Number.isFinite(value))
    return "—";
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(digits);
}

export function basename(path: string) {
  const trimmed = path.replace(/[/\\]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return index >= 0 ? trimmed.slice(index + 1) : trimmed;
}

export function dirname(path: string) {
  const trimmed = path.replace(/[/\\]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  if (index <= 0) return index === 0 ? "/" : "";
  return trimmed.slice(0, index);
}

/** Percentage of `used` within `total`, clamped to 0-100. */
export function percent(used: number, total: number) {
  if (!Number.isFinite(used) || !Number.isFinite(total) || total <= 0) return 0;
  return Math.min(100, Math.max(0, Math.round((used / total) * 100)));
}
