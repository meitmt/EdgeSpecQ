#!/usr/bin/env bash
# R1: DSpark acceptance on an old commit vs master (text 2.6B and VL-3B, Q4_K_M target, DSpark F16 drafter),
# with and without the drafter, then pairwise output comparison.
#   run_r1.sh [old_commit=38a5b42d9]
set -u
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
PY=$H/.venv/bin/python
OLD=${1:-38a5b42d9}
OLD_DIR=$(bash "$R/scripts/build_llama.sh" "$OLD" | tail -1)
M=$H/models

for d in "$H/llama.cpp:master" "$OLD_DIR:old"; do
  dir=${d%%:*}; tag=${d##*:}
  $PY "$R/regression/probe_accept.py" "$dir" ${tag}_t26_q4km_f16 $M/track_t/LFM2.5-2.6B-Q4_K_M.gguf \
      $M/track_t/LFM2.5-2.6B-DSpark-F16.gguf $H/probe/track_t_probe20.jsonl
  $PY "$R/regression/probe_accept.py" "$dir" ${tag}_t26_q4km_none $M/track_t/LFM2.5-2.6B-Q4_K_M.gguf \
      none $H/probe/track_t_probe20.jsonl
  $PY "$R/regression/probe_accept.py" "$dir" ${tag}_vl_q4km_f16 $M/target/LFM2.5-VL-3B-Q4_K_M.gguf \
      $M/draft/LFM2.5-VL-3B-DSpark-F16.gguf $H/probe/vl_probe20.jsonl --mmproj $M/target/mmproj-LFM2.5-VL-3B-Q8_0.gguf
  $PY "$R/regression/probe_accept.py" "$dir" ${tag}_vl_q4km_none $M/target/LFM2.5-VL-3B-Q4_K_M.gguf \
      none $H/probe/vl_probe20.jsonl --mmproj $M/target/mmproj-LFM2.5-VL-3B-Q8_0.gguf
done
$PY "$R/regression/compare_outputs.py" master_t26_q4km_none master_t26_q4km_f16 old_t26_q4km_none old_t26_q4km_f16
$PY "$R/regression/compare_outputs.py" master_vl_q4km_none master_vl_q4km_f16 old_vl_q4km_none old_vl_q4km_f16
