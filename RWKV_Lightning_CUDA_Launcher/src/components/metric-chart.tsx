import { useEffect, useMemo, useRef, useState } from "react";

export interface ChartPoint {
  step: number;
  value: number;
}

/**
 * TensorBoard's debiased exponential moving average. The plain recurrence
 * starts at zero and drags the first few hundred points toward it, which on a
 * loss curve reads as a run that began far better than it did; dividing by
 * `1 - smoothing^n` removes that warm-up bias.
 */
export function smoothSeries(points: ChartPoint[], smoothing: number) {
  if (smoothing <= 0) return points;
  const weight = Math.min(smoothing, 0.999);
  let last = 0;
  let debias = 0;
  return points.map((point) => {
    last = last * weight + (1 - weight) * point.value;
    debias = debias * weight + (1 - weight);
    return { step: point.step, value: last / (debias || 1) };
  });
}

/** Four round-ish ticks covering [min, max]. */
export function ticksFor(min: number, max: number, count = 4) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max)
    return [min];
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const factor =
    [1, 2, 2.5, 5, 10].find((candidate) => candidate * magnitude >= raw) ?? 10;
  const unit = factor * magnitude || raw;
  const start = Math.ceil(min / unit) * unit;
  const out: number[] = [];
  for (let value = start; value <= max + unit / 1000; value += unit)
    out.push(value);
  return out.length > 1 ? out : [min, max];
}

/** Drawing a 4000-point run costs more path data than the card has pixels. */
export function strided(points: ChartPoint[], limit: number) {
  if (points.length <= limit) return points;
  const stride = Math.ceil(points.length / limit);
  const out = points.filter((_, index) => index % stride === 0);
  const last = points[points.length - 1];
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) =>
      setWidth(entry.contentRect.width),
    );
    observer.observe(node);
    setWidth(node.getBoundingClientRect().width);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

const PAD = { top: 8, right: 10, bottom: 20, left: 48 };

/**
 * One measure over training steps: the raw series in the background, its
 * smoothed line on top, both on a single axis. Two measures of different scale
 * are never overlaid — the caller switches metric instead of adding a second
 * y-axis.
 */
export function MetricChart({
  points,
  smoothing = 0,
  height = 168,
  label,
  format,
  emptyLabel,
}: {
  points: ChartPoint[];
  smoothing?: number;
  height?: number;
  label: string;
  format: (value: number) => string;
  emptyLabel: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();

  const raw = useMemo(() => strided(points, 900), [points]);
  const smooth = useMemo(
    () => (smoothing > 0 ? smoothSeries(raw, smoothing) : []),
    [raw, smoothing],
  );

  const bounds = useMemo(() => {
    const values = raw.map((point) => point.value).filter(Number.isFinite);
    if (values.length < 2) return null;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const pad = (max - min || Math.abs(max) || 1) * 0.08;
    return {
      minX: raw[0].step,
      maxX: raw[raw.length - 1].step,
      minY: min - pad,
      maxY: max + pad,
    };
  }, [raw]);

  // One stable wrapper in both states: the width observer attaches once and
  // keeps measuring when the first points arrive.
  return (
    <div ref={ref} className="relative" style={{ height }}>
      {bounds && width > 0 ? (
        <Plot
          raw={raw}
          smooth={smooth}
          bounds={bounds}
          width={width}
          height={height}
          label={label}
          format={format}
        />
      ) : bounds ? null : (
        // No points yet. (With points but an unmeasured width, the frame stays
        // blank rather than flashing an "empty" box for one frame.)
        <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-border text-[11.5px] text-muted-foreground">
          {emptyLabel}
        </div>
      )}
    </div>
  );
}

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function Plot({
  raw,
  smooth,
  bounds,
  width,
  height,
  label,
  format,
}: {
  raw: ChartPoint[];
  smooth: ChartPoint[];
  bounds: Bounds;
  width: number;
  height: number;
  label: string;
  format: (value: number) => string;
}) {
  const [hover, setHover] = useState<number | null>(null);

  const plotWidth = Math.max(0, width - PAD.left - PAD.right);
  const plotHeight = height - PAD.top - PAD.bottom;
  const spanX = bounds.maxX - bounds.minX || 1;
  const spanY = bounds.maxY - bounds.minY || 1;
  const x = (step: number) =>
    PAD.left + ((step - bounds.minX) / spanX) * plotWidth;
  const y = (value: number) =>
    PAD.top + (1 - (value - bounds.minY) / spanY) * plotHeight;
  const path = (series: ChartPoint[]) =>
    series
      .map(
        (point, index) =>
          `${index === 0 ? "M" : "L"}${x(point.step).toFixed(1)},${y(
            point.value,
          ).toFixed(1)}`,
      )
      .join(" ");

  const yTicks = ticksFor(bounds.minY, bounds.maxY);
  const xTicks = [raw[0], raw[Math.floor(raw.length / 2)], raw[raw.length - 1]];
  const last = raw[raw.length - 1];
  const active = hover === null ? null : raw[hover];
  const activeSmooth = hover === null ? null : (smooth[hover] ?? null);

  return (
    <>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`${label}: ${format(last.value)} at step ${last.step}`}
        onPointerLeave={() => setHover(null)}
        onPointerMove={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          const ratio =
            (event.clientX - box.left - PAD.left) / (plotWidth || 1);
          const index = Math.round(ratio * (raw.length - 1));
          setHover(Math.min(raw.length - 1, Math.max(0, index)));
        }}
      >
        {yTicks.map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              x2={width - PAD.right}
              y1={y(tick)}
              y2={y(tick)}
              stroke="var(--border)"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 6}
              y={y(tick)}
              textAnchor="end"
              dominantBaseline="middle"
              fill="var(--muted-foreground)"
              className="font-mono"
              fontSize={9.5}
            >
              {format(tick)}
            </text>
          </g>
        ))}

        {xTicks.map((point, index) => (
          <text
            key={`${point.step}-${index}`}
            x={x(point.step)}
            y={height - 6}
            textAnchor={index === 0 ? "start" : index === 2 ? "end" : "middle"}
            fill="var(--muted-foreground)"
            className="font-mono"
            fontSize={9.5}
          >
            {point.step}
          </text>
        ))}

        {/* Raw first and recessive: the smoothed line is the one to read. */}
        <path
          d={path(raw)}
          fill="none"
          stroke="var(--muted-foreground)"
          strokeOpacity={smooth.length > 0 ? 0.3 : 0.85}
          strokeWidth={1}
          strokeLinejoin="round"
        />
        {smooth.length > 0 && (
          <path
            d={path(smooth)}
            fill="none"
            stroke="var(--info)"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {active && (
          <>
            <line
              x1={x(active.step)}
              x2={x(active.step)}
              y1={PAD.top}
              y2={height - PAD.bottom}
              stroke="var(--border-strong)"
              strokeWidth={1}
            />
            <circle
              cx={x(active.step)}
              cy={y((activeSmooth ?? active).value)}
              r={3.5}
              fill="var(--info)"
              stroke="var(--card)"
              strokeWidth={2}
            />
          </>
        )}
      </svg>

      {active && (
        <div
          className="pointer-events-none absolute top-1 rounded-lg border border-border bg-popover px-2 py-1.5 shadow-lg"
          style={{
            left: Math.min(
              Math.max(x(active.step) - 52, 0),
              Math.max(0, width - 116),
            ),
          }}
        >
          <div className="font-mono text-[10px] text-muted-foreground">
            step {active.step}
          </div>
          <div className="mt-0.5 flex items-center gap-1.5 font-mono text-[11px]">
            <span
              aria-hidden
              className="inline-block size-[6px] rounded-full"
              style={{
                background: activeSmooth
                  ? "var(--info)"
                  : "var(--muted-foreground)",
              }}
            />
            {format((activeSmooth ?? active).value)}
          </div>
          {activeSmooth && (
            <div className="mt-0.5 font-mono text-[10px] text-muted-foreground">
              raw {format(active.value)}
            </div>
          )}
        </div>
      )}
    </>
  );
}
