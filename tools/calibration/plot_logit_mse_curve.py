#!/usr/bin/env python3
"""Render the logit MSE curve CSV as a dependency-free SVG."""

import argparse
import csv
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return [(int(row["position"]), float(row["mse"])) for row in csv.DictReader(source)]


def downsample(rows, max_points=2000):
    if len(rows) <= max_points:
        return rows
    last = len(rows) - 1
    indices = [round(index * last / (max_points - 1)) for index in range(max_points)]
    return [rows[index] for index in indices]


def render(rows, output):
    if not rows:
        raise ValueError("curve CSV is empty")
    width, height = 1200, 620
    left, right, top, bottom = 82, 28, 36, 72
    plot_width, plot_height = width - left - right, height - top - bottom
    max_position = rows[-1][0]
    raw_max_mse = max(value for _, value in rows)
    plot_max_mse = max(raw_max_mse * 1.08, 1e-9)

    def point(position, value):
        x = left + (position - 1) * plot_width / max(max_position - 1, 1)
        y = top + (plot_max_mse - value) * plot_height / plot_max_mse
        return x, y

    raw_rows = downsample(rows)
    path = " ".join(
        ("M" if index == 0 else "L") + f" {point(position, value)[0]:.2f},{point(position, value)[1]:.2f}"
        for index, (position, value) in enumerate(raw_rows)
    )
    window = 64
    smooth = []
    running = 0.0
    values = [value for _, value in rows]
    for index, (position, _) in enumerate(rows):
        running += values[index]
        if index >= window:
            running -= values[index - window]
        smooth.append((position, running / min(index + 1, window)))
    smooth_path = " ".join(
        ("M" if index == 0 else "L") + f" {point(position, value)[0]:.2f},{point(position, value)[1]:.2f}"
        for index, (position, value) in enumerate(smooth)
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">FP16 vs INT8 per-position logit MSE</text>',
    ]
    for index in range(5):
        value = raw_max_mse * index / 4.0
        y = top + (plot_max_mse - value) * plot_height / plot_max_mse
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#bbbbbb" stroke-dasharray="4,4"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{value:.6g}</text>'
        )
    for limit in (1024, 4096, 16384):
        if limit <= max_position:
            x, _ = point(limit, 0.0)
            parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#bbbbbb" stroke-dasharray="4,4"/>')
            parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{limit}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
        f'<path d="{path}" fill="none" stroke="#4776b6" stroke-width="0.8" opacity="0.55"/>',
        f'<path d="{smooth_path}" fill="none" stroke="#d14b4b" stroke-width="2"/>',
        f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">MSE</text>',
        f'<text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">Token position (1-based)</text>',
        f'<text x="{left + 12}" y="{top + 18}" font-family="sans-serif" font-size="12" fill="#4776b6">raw MSE</text>',
        f'<text x="{left + 84}" y="{top + 18}" font-family="sans-serif" font-size="12" fill="#d14b4b">64-token moving average</text>',
        f'<text x="{left + plot_width}" y="{top + plot_height + 42}" text-anchor="end" font-family="sans-serif" font-size="11">max {raw_max_mse:.6g}</text>',
        "</svg>",
    ])
    Path(output).write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("svg")
    args = parser.parse_args()
    rows = read_rows(args.csv)
    Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
    render(rows, args.svg)


if __name__ == "__main__":
    main()
