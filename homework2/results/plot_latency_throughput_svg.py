#!/usr/bin/env python3
"""Render the latency-throughput comparison graph as an SVG (no third-party deps)."""
import csv
from collections import defaultdict
from pathlib import Path

OUTDIR = Path("results")
DATE_TAG = "2026-06-18"
INPUT_CSV = OUTDIR / f"latency_throughput_{DATE_TAG}.csv"
OUTPUT_SVG = OUTDIR / f"latency_throughput_{DATE_TAG}.svg"

series_points = defaultdict(list)
with INPUT_CSV.open() as f:
    for row in csv.DictReader(f):
        series_points[row["series"]].append(
            (float(row["throughput"]), float(row["median_latency_ms"]))
        )

order = ["streams", "queue_1024", "queue_512", "queue_256"]
colors = {
    "streams": "#1b9e77",
    "queue_1024": "#d95f02",
    "queue_512": "#7570b3",
    "queue_256": "#e7298a",
}
labels = {
    "streams": "Streams",
    "queue_1024": "Queue #threads=1024",
    "queue_512": "Queue #threads=512",
    "queue_256": "Queue #threads=256",
}

# Plot geometry
W, H = 1100, 700
ML, MR, MT, MB = 90, 230, 60, 70
PW = W - ML - MR
PH = H - MT - MB

all_x = [p[0] for pts in series_points.values() for p in pts]
all_y = [p[1] for pts in series_points.values() for p in pts]
xmin, xmax = 0, max(all_x) * 1.05
ymin, ymax = 0, max(all_y) * 1.08


def sx(x):
    return ML + (x - xmin) / (xmax - xmin) * PW


def sy(y):
    return MT + PH - (y - ymin) / (ymax - ymin) * PH


def nice_ticks(lo, hi, n=8):
    span = hi - lo
    raw = span / n
    mag = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 1
    step = max(1, round(raw / mag) * mag)
    ticks = []
    t = 0
    while t <= hi:
        ticks.append(t)
        t += step
    return ticks


svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica,Arial,sans-serif">')
svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')
svg.append(
    f'<text x="{ML + PW/2}" y="30" text-anchor="middle" font-size="20" font-weight="bold">'
    "Latency-Throughput Comparison: Streams vs Queue Configurations</text>"
)

# Axes
svg.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="black" stroke-width="1.5"/>')
svg.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="black" stroke-width="1.5"/>')

# Grid + x ticks
for xt in nice_ticks(xmin, xmax):
    X = sx(xt)
    svg.append(f'<line x1="{X}" y1="{MT}" x2="{X}" y2="{MT+PH}" stroke="#e6e6e6" stroke-width="1"/>')
    svg.append(f'<text x="{X}" y="{MT+PH+22}" text-anchor="middle" font-size="12">{int(xt)}</text>')

# Grid + y ticks
for yt in nice_ticks(ymin, ymax):
    Y = sy(yt)
    svg.append(f'<line x1="{ML}" y1="{Y}" x2="{ML+PW}" y2="{Y}" stroke="#e6e6e6" stroke-width="1"/>')
    svg.append(f'<text x="{ML-10}" y="{Y+4}" text-anchor="end" font-size="12">{int(yt)}</text>')

# Axis labels
svg.append(
    f'<text x="{ML+PW/2}" y="{H-20}" text-anchor="middle" font-size="15">Throughput (req/sec)</text>'
)
svg.append(
    f'<text x="25" y="{MT+PH/2}" text-anchor="middle" font-size="15" '
    f'transform="rotate(-90 25 {MT+PH/2})">Median Latency (ms)</text>'
)

# Data series (sorted by throughput so the polyline reads left-to-right)
for s in order:
    pts = sorted(series_points.get(s, []), key=lambda p: p[0])
    if not pts:
        continue
    color = colors[s]
    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
    svg.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    for x, y in pts:
        svg.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')

# Legend
lx = ML + PW + 25
ly = MT + 10
svg.append(f'<rect x="{lx-10}" y="{ly-25}" width="215" height="{len(order)*26+20}" fill="#fafafa" stroke="#cccccc"/>')
svg.append(f'<text x="{lx}" y="{ly-5}" font-size="14" font-weight="bold">Legend</text>')
for i, s in enumerate(order):
    yy = ly + 20 + i * 26
    svg.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+28}" y2="{yy}" stroke="{colors[s]}" stroke-width="3"/>')
    svg.append(f'<circle cx="{lx+14}" cy="{yy}" r="4" fill="{colors[s]}"/>')
    svg.append(f'<text x="{lx+36}" y="{yy+4}" font-size="13">{labels[s]}</text>')

svg.append("</svg>")
OUTPUT_SVG.write_text("\n".join(svg))
print(str(OUTPUT_SVG))
