#!/usr/bin/env python3
"""Render per-chunk FP16 and INT8 perplexity as an SVG."""

import argparse
import csv
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        return [
            (int(row["chunk"]), float(row["fp16_ppl"]), float(row["int8_ppl"]))
            for row in csv.DictReader(source)
        ]


def downsample(rows, max_points=2000):
    if len(rows) <= max_points:
        return rows
    last = len(rows) - 1
    indices = [round(index * last / (max_points - 1)) for index in range(max_points)]
    return [rows[index] for index in indices]


def render(rows, output):
    if not rows:
        raise ValueError("PPL CSV is empty")
    width, height = 1200, 620
    left, right, top, bottom = 82, 28, 36, 72
    plot_width, plot_height = width - left - right, height - top - bottom
    max_chunk = rows[-1][0]
    raw_max = max(max(fp16, int8) for _, fp16, int8 in rows)
    raw_min = min(min(fp16, int8) for _, fp16, int8 in rows)
    plot_min = max(0.0, raw_min * 0.92)
    plot_max = max(raw_max * 1.08, plot_min + 1e-9)

    def point(chunk, value):
        x = left + (chunk - 1) * plot_width / max(max_chunk - 1, 1)
        y = top + (plot_max - value) * plot_height / (plot_max - plot_min)
        return x, y

    sampled = downsample(rows)

    def make_path(index):
        return " ".join(
            ("M" if point_index == 0 else "L")
            + f" {point(chunk, values[index])[0]:.2f},{point(chunk, values[index])[1]:.2f}"
            for point_index, (chunk, *values) in enumerate(sampled)
        )

    fp16_path = make_path(0)
    int8_path = make_path(1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">FP16 vs INT8 per-chunk perplexity</text>',
    ]
    for index in range(5):
        value = plot_min + (plot_max - plot_min) * index / 4.0
        y = top + (plot_max - value) * plot_height / (plot_max - plot_min)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#bbbbbb" stroke-dasharray="4,4"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{value:.6g}</text>'
        )
    tick_count = min(max_chunk, 8)
    for index in range(tick_count):
        chunk = 1 if tick_count == 1 else round(1 + index * (max_chunk - 1) / (tick_count - 1))
        x, _ = point(chunk, plot_min)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            'stroke="#bbbbbb" stroke-dasharray="4,4"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{chunk}</text>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
            f'<path d="{fp16_path}" fill="none" stroke="#4776b6" stroke-width="2"/>',
            f'<path d="{int8_path}" fill="none" stroke="#d14b4b" stroke-width="2"/>',
            f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">PPL</text>',
            f'<text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">Chunk (128 tokens)</text>',
            f'<text x="{left + 12}" y="{top + 18}" font-family="sans-serif" font-size="12" fill="#4776b6">FP16</text>',
            f'<text x="{left + 58}" y="{top + 18}" font-family="sans-serif" font-size="12" fill="#d14b4b">INT8</text>',
            "</svg>",
        ]
    )
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
