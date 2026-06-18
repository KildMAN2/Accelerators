#!/usr/bin/env bash
set -euo pipefail

OUTDIR="results"
DATE_TAG="2026-06-18"
MASTER_CSV="$OUTDIR/measurements_${DATE_TAG}.csv"
COMBINED_CSV="$OUTDIR/latency_throughput_${DATE_TAG}.csv"
SUMMARY_TXT="$OUTDIR/run_summary_${DATE_TAG}.txt"

echo "config,threads,maxload,point,load,throughput,median_latency_ms" > "$MASTER_CSV"
echo "series,throughput,median_latency_ms" > "$COMBINED_CSV"
: > "$SUMMARY_TXT"

run_case() {
  config="$1"
  threads="$2"

  if [[ "$config" == "streams" ]]; then
    mode_cmd="./ex2 streams"
    series="streams"
  else
    mode_cmd="./ex2 queue $threads"
    series="queue_${threads}"
  fi

  run0_log="$OUTDIR/${series}_maxload.log"
  eval "$mode_cmd 0" > "$run0_log"
  maxload=$(awk '/throughput =/{print $3}' "$run0_log" | tail -1)

  start=$(awk -v m="$maxload" 'BEGIN{printf "%.6f", m/10.0}')
  step=$(awk -v m="$maxload" 'BEGIN{printf "%.6f", ((2.0*m)-(m/10.0))/9.0}')

  for i in $(seq 0 9); do
    load=$(awk -v s="$start" -v st="$step" -v k="$i" 'BEGIN{printf "%.6f", s + st*k}')
    log_file="$OUTDIR/${series}_point_${i}.log"
    eval "$mode_cmd $load" > "$log_file"

    thr=$(awk '/throughput =/{print $3}' "$log_file" | tail -1)
    med=$(awk '/latency \[msec\]/{getline; getline; print $3}' "$log_file" | tail -1)

    echo "$config,$threads,$maxload,$i,$load,$thr,$med" >> "$MASTER_CSV"
    echo "$series,$thr,$med" >> "$COMBINED_CSV"
  done

  echo "$series maxload=$maxload" >> "$SUMMARY_TXT"
}

run_case streams NA
run_case queue 1024
run_case queue 512
run_case queue 256

cat "$SUMMARY_TXT"
