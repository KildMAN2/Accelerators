#!/usr/bin/env python3
import csv
import subprocess
from pathlib import Path

OUTDIR = Path("results")
DATE_TAG = "2026-06-18"
MASTER_CSV = OUTDIR / f"measurements_{DATE_TAG}.csv"
COMBINED_CSV = OUTDIR / f"latency_throughput_{DATE_TAG}.csv"
SUMMARY_TXT = OUTDIR / f"run_summary_{DATE_TAG}.txt"


def run_and_parse(cmd):
    out = subprocess.check_output(cmd, shell=True, text=True)
    thr = None
    med = None
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "throughput =" in line:
            thr = float(line.split()[2])
        if line.startswith("latency [msec]:") and i + 2 < len(lines):
            med = float(lines[i + 2].split()[2])
    if thr is None or med is None:
        raise RuntimeError(f"Failed to parse output for command: {cmd}\n{out}")
    return thr, med, out


def sweep(config, threads):
    if config == "streams":
        mode_cmd = "./ex2 streams"
        series = "streams"
    else:
        mode_cmd = f"./ex2 queue {threads}"
        series = f"queue_{threads}"

    thr0, med0, out0 = run_and_parse(f"{mode_cmd} 0")
    (OUTDIR / f"{series}_maxload.log").write_text(out0)

    maxload = thr0
    start = maxload / 10.0
    step = ((2.0 * maxload) - start) / 9.0

    rows = []
    combined = []

    for i in range(10):
        load = start + step * i
        thr, med, out = run_and_parse(f"{mode_cmd} {load:.6f}")
        (OUTDIR / f"{series}_point_{i}.log").write_text(out)

        rows.append([config, threads, f"{maxload:.6f}", i, f"{load:.6f}", f"{thr:.4f}", f"{med:.4f}"])
        combined.append([series, f"{thr:.4f}", f"{med:.4f}"])
        print(f"{series}: point {i}/9 done (load={load:.1f}, thr={thr:.1f}, med={med:.2f}ms)", flush=True)

    return series, maxload, rows, combined


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_combined = []
    summary = []

    for config, threads in [("streams", "NA"), ("queue", 1024), ("queue", 512), ("queue", 256)]:
        series, maxload, rows, combined = sweep(config, threads)
        summary.append(f"{series} maxload={maxload:.4f}")
        all_rows.extend(rows)
        all_combined.extend(combined)

    with MASTER_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "threads", "maxload", "point", "load", "throughput", "median_latency_ms"])
        w.writerows(all_rows)

    with COMBINED_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "throughput", "median_latency_ms"])
        w.writerows(all_combined)

    SUMMARY_TXT.write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
