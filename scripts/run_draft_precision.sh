#!/usr/bin/env bash
# E2: fixed target, drafter-precision sweep on a probe, repeated with rotated drafter order.
#   run_draft_precision.sh [target=Q4_K_M] [probe=vl|text] [reps=1]
# Runs go to $EDGESPECQ_HOME/logs/probe/dp_<probe>_t<target>_d<drafter>_r<rep>/; summarize with
# analysis/summarize_draft_precision.py.
set -u
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
PY=$H/.venv/bin/python
L=${LLAMA_DIR:-$H/llama.cpp}
TQ=${1:-Q4_K_M}
PROBE_KIND=${2:-vl}
REPS=${3:-1}
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
ORDER=(none F16 Q8_0 P3 Q4_K_M Q4_0)
for rep in $(seq 1 "$REPS"); do
  # rotate the order each repetition so slow drift does not favour one drafter
  k=$(( (rep - 1) % ${#ORDER[@]} ))
  for d in "${ORDER[@]:$k}" "${ORDER[@]:0:$k}"; do
    $PY "$R/regression/probe_accept.py" "$L" "dp_${PROBE_KIND}_t${TQ}_d${d}_r${rep}" "$T" "${D[$d]}" "$PROBE" "${MM[@]}" > /dev/null
  done
done
$PY "$R/analysis/summarize_draft_precision.py" "$PROBE_KIND" "$TQ"
