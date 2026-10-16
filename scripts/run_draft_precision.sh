#!/usr/bin/env bash
# E2 (first pass): fixed target, drafter precision sweep on a probe; prints a summary table.
#   run_draft_precision.sh [target=Q4_K_M] [probe=vl]
set -u
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
PY=$H/.venv/bin/python
L=${LLAMA_DIR:-$H/llama.cpp}
TQ=${1:-Q4_K_M}
PROBE_KIND=${2:-vl}
T=$H/models/target/LFM2.5-VL-3B-$TQ.gguf
MM=(--mmproj $H/models/target/mmproj-LFM2.5-VL-3B-Q8_0.gguf)
PROBE=$H/probe/vl_probe20.jsonl
[ "$PROBE_KIND" = text ] && { PROBE=$H/probe/track_t_probe20.jsonl; MM=(); }
declare -A D=(
  [none]=none
  [F16]=$H/models/draft/LFM2.5-VL-3B-DSpark-F16.gguf
  [Q8_0]=$H/models/quant/LFM2.5-VL-3B-DSpark-Q8_0.gguf
  [P3]=$H/models/quant/LFM2.5-VL-3B-DSpark-P3.gguf
  [Q4_K_M]=$H/models/quant/LFM2.5-VL-3B-DSpark-Q4_K_M.gguf
  [Q4_0]=$H/models/quant/LFM2.5-VL-3B-DSpark-Q4_0.gguf
)
for d in none F16 Q8_0 P3 Q4_K_M Q4_0; do
  $PY "$R/regression/probe_accept.py" "$L" "dp_${PROBE_KIND}_t${TQ}_d$d" "$T" "${D[$d]}" "$PROBE" "${MM[@]}" > /dev/null
done
$PY - "$H" "$PROBE_KIND" "$TQ" <<'PY'
import json, sys
H, kind, tq = sys.argv[1:4]
def load(d):
    return json.load(open(f"{H}/logs/probe/dp_{kind}_t{tq}_d{d}/rows.json"))
def tps(rows):
    n = sum(r["n_pred"] for r in rows); s = sum(r["n_pred"] / r["tps"] for r in rows)
    return n / s
base = tps(load("none"))
print(f"target {tq}, probe {kind} (20 prompts), target-only decode {base:.1f} t/s")
print(f"{'drafter':8s} {'size MiB':>9s} {'accept':>7s} {'mean len':>9s} {'decode t/s':>11s} {'speedup':>8s}")
size = {"F16": 533.1, "Q8_0": 283.2, "P3": 197.9, "Q4_K_M": 156.7, "Q4_0": 150.0}
for d in ("F16", "Q8_0", "P3", "Q4_K_M", "Q4_0"):
    rows = load(d)
    dn = sum(r["draft_n"] or 0 for r in rows); da = sum(r["draft_acc"] or 0 for r in rows)
    ml = [r["mean_len"] for r in rows if r["mean_len"]]
    t = tps(rows)
    print(f"{d:8s} {size[d]:9.1f} {da/dn:7.3f} {sum(ml)/len(ml):9.2f} {t:11.1f} {t/base:7.2f}x")
PY
