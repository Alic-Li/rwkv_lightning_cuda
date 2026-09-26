import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { cn } from "@/lib/utils";

/** Decimals implied by the step, so 0.996 never renders as 0.9960000000001. */
function decimalsOf(step: number) {
  if (!Number.isFinite(step) || step <= 0) return 0;
  const text = String(step);
  const dot = text.indexOf(".");
  return dot < 0 ? 0 : text.length - dot - 1;
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

/** Snap to the step grid anchored at `min`, rounded to the step's decimals. */
function snap(value: number, min: number, max: number, step: number) {
  const snapped = min + Math.round((value - min) / step) * step;
  return Number(clamp(snapped, min, max).toFixed(decimalsOf(step)));
}

/**
 * The rail on its own: a grey track with the travelled part filled in a
 * faint wash of the foreground, and a solid upright bar for the thumb.
 * Hovering shows a hatched ghost bar where a click would land, reported
 * through `onHover` so a caller can print the jump before it is taken.
 */
export function Slider({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
  onHover,
  className,
}: {
  /** Accessible name; the rail has no visible text of its own. */
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (value: number) => void;
  /** The value under the pointer while hovering, null when it leaves. */
  onHover?: (value: number | null) => void;
  className?: string;
}) {
  const [hover, setHoverState] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const trackRef = useRef<HTMLDivElement>(null);
  const setHover = (next: number | null) => {
    setHoverState(next);
    onHover?.(next);
  };

  const span = max - min || 1;
  const percent = ((clamp(value, min, max) - min) / span) * 100;

  /** Pointer x → value on the step grid. */
  const valueAt = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return value;
    const ratio = clamp((clientX - rect.left) / rect.width, 0, 1);
    return snap(min + ratio * span, min, max, step);
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (disabled || event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.focus();
    setDragging(true);
    setHover(null);
    onChange(valueAt(event.clientX));
    event.preventDefault();
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (disabled) return;
    const next = valueAt(event.clientX);
    if (dragging) {
      if (next !== value) onChange(next);
    } else if (event.pointerType === "mouse") {
      setHover(next);
    }
  };

  const endDrag = () => setDragging(false);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    const big = Math.max(step, snap(span / 10, 0, span, step));
    const delta: Record<string, number> = {
      ArrowRight: step,
      ArrowUp: step,
      ArrowLeft: -step,
      ArrowDown: -step,
      PageUp: big,
      PageDown: -big,
    };
    let next: number | undefined;
    if (event.key in delta)
      next = value + delta[event.key] * (event.shiftKey ? 10 : 1);
    else if (event.key === "Home") next = min;
    else if (event.key === "End") next = max;
    if (next === undefined) return;
    event.preventDefault();
    onChange(snap(next, min, max, step));
  };

  const hoverPercent = hover === null ? null : ((hover - min) / span) * 100;

  // 28px of hit area around a 6px rail: easy to grab, quiet to look at. The
  // side padding keeps the thumb's half-width inside the box at both ends.
  return (
    <div
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-valuetext={value.toFixed(decimalsOf(step))}
      aria-disabled={disabled || undefined}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onPointerLeave={() => setHover(null)}
      onKeyDown={onKeyDown}
      className={cn(
        "relative h-7 touch-none rounded-sm px-1 outline-none select-none",
        "focus-visible:ring-2 focus-visible:ring-ring/50",
        disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer",
        className,
      )}
    >
      <div ref={trackRef} className="relative h-full">
        <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-border-strong" />
        <div
          className={cn(
            "absolute top-1/2 left-0 h-1.5 -translate-y-1/2 rounded-l-full bg-foreground opacity-35",
            !dragging && "transition-[width] duration-150 ease-out",
          )}
          style={{ width: `${percent}%` }}
        />
        {hoverPercent !== null && !dragging && (
          <div
            aria-hidden="true"
            className="slider-ghost pointer-events-none absolute top-1/2 h-5 w-1 -translate-x-1/2 -translate-y-1/2 rounded-[2px] opacity-40"
            style={{ left: `${hoverPercent}%` }}
          />
        )}
        <div
          aria-hidden="true"
          className={cn(
            "absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-[3px] bg-foreground",
            // Glides to a clicked spot, but tracks a drag with no lag.
            dragging
              ? "transition-[height,width] duration-100"
              : "transition-[height,width,left] duration-150",
            "ease-out",
            dragging ? "h-7 w-[7px] cursor-grabbing" : "h-5 w-[5px]",
          )}
          style={{ left: `${percent}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Slider plus an editable readout. A slider alone cannot land on an exact
 * value — 0.996 with a 0.001 step is three careful drags — and a bare number
 * field gives no feel for the range, so every tunable parameter offers both.
 * While the rail is hovered the readout turns into "from → to".
 *
 * `hint` is not printed under the row: the panel is a column of seven of
 * these, and seven sentences of help are what made it read as a manual. It
 * rides on the label as a tooltip instead.
 */
export function SliderField({
  label,
  suffix,
  value,
  min,
  max,
  step,
  disabled,
  hint,
  onChange,
  className,
}: {
  label: string;
  /** What the value does; shown as the label's tooltip. */
  hint?: string;
  suffix?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (value: number) => void;
  className?: string;
}) {
  const decimals = decimalsOf(step);
  const [draft, setDraft] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  const commit = (text: string) => {
    setDraft(text);
    const parsed = Number(text);
    if (text.trim() !== "" && Number.isFinite(parsed))
      onChange(clamp(parsed, min, max));
  };

  const format = (n: number) => n.toFixed(decimals);

  return (
    <div className={cn("min-w-0", className)}>
      <div
        className={cn(
          "flex h-6 items-center justify-between gap-2",
          disabled && "opacity-60",
        )}
      >
        <span
          title={hint}
          className={cn(
            "min-w-0 truncate text-[12px] text-muted-foreground",
            hint && "cursor-help",
          )}
        >
          {label}
        </span>
        {/* The readout and the hover preview share one slot: while the
            pointer is over the rail the jump is the more useful number. */}
        <span className="relative flex shrink-0 items-center gap-1">
          {hover !== null && hover !== value ? (
            <span className="flex h-6 items-center gap-1 px-1 font-mono text-[11.5px] tabular-nums">
              <span className="text-muted-foreground">{format(value)}</span>
              <span className="text-muted-foreground/60">→</span>
              <span className="font-semibold text-foreground">
                {format(hover)}
              </span>
            </span>
          ) : (
            <input
              inputMode="decimal"
              disabled={disabled}
              value={draft ?? format(value)}
              aria-label={label}
              onFocus={() => setDraft(String(value))}
              onBlur={() => setDraft(null)}
              onChange={(event) => commit(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
              }}
              className="h-6 w-[64px] rounded-md border border-transparent bg-transparent px-1 text-right font-mono text-[11.5px] font-semibold tabular-nums text-foreground transition-colors outline-none hover:border-border focus-visible:border-border-strong focus-visible:bg-background"
            />
          )}
          {suffix && (
            <span className="font-mono text-[10.5px] text-muted-foreground/70">
              {suffix}
            </span>
          )}
        </span>
      </div>
      <Slider
        label={label}
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={onChange}
        onHover={setHover}
      />
    </div>
  );
}
