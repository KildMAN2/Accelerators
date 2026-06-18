#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

OUTDIR = Path("results")
DATE_TAG = "2026-06-18"
INPUT_CSV = OUTDIR / f"latency_throughput_{DATE_TAG}.csv"
OUTPUT_PNG = OUTDIR / f"latency_throughput_{DATE_TAG}.png"

series_points = defaultdict(list)
with INPUT_CSV.open() as f:
    r = csv.DictReader(f)
    for row in r:
        s = row["series"]
        thr = float(row["throughput"])
        lat = float(row["median_latency_ms"])
        series_points[s].append((thr, lat))

plt.figure(figsize=(11, 7))
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

for s in order:
    pts = series_points.get(s, [])
    if not pts:
        continue
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    plt.plot(x, y, marker="o", linewidth=2, markersize=5, color=colors[s], label=labels[s])

plt.xlabel("Throughput (req/sec)")
plt.ylabel("Median Latency (ms)")
plt.title("Latency-Throughput Comparison: Streams vs Queue Configurations")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=160)
print(str(OUTPUT_PNG))
