#!/usr/bin/env bash
# Token-generation throughput of one build (llama-bench, target only), printed as one number (median t/s).
#   bench_tg.sh <llama_dir> [model] [n_gen=256] [reps=10]
# THREADS sets llama-bench -t (default: llama-bench's own default).
set -euo pipefail
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
D=$1
M=${2:-$H/models/track_t/LFM2.5-2.6B-Q4_K_M.gguf}
N=${3:-256}
R=${4:-10}
T=()
[ -n "${THREADS:-}" ] && T=(-t "$THREADS")
"$D/build/bin/llama-bench" -m "$M" -ngl 99 -fa 1 -p 0 -n "$N" -r "$R" "${T[@]}" -o json 2>/dev/null \
  | python3 -c 'import json,sys,statistics as s; r=json.load(sys.stdin)[0]; ts=r.get("samples_ts") or [r["avg_ts"]]; print(f"{s.median(ts):.2f}")'
