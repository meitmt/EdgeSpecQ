#!/usr/bin/env bash
# E9: reduced-vocabulary DSpark drafters end to end (llama-server), target-only + full-vocab Q8_0 as references.
#   run_e9_vocab.sh [target=Q4_K_M] [probe=text|ja] [reps=1] [drafters="V16k V32k V48k V64k"]
# Reduced drafters are $EDGESPECQ_HOME/models/e9/LFM2.5-VL-3B-DSpark-Q8_0-<name>.gguf
# (quantization/make_reduced_vocab_drafter.py). Runs go to logs/probe/dp_<probe>_t<target>_d<drafter>_r<rep>/.
set -u
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
PY=$H/.venv/bin/python
L=${LLAMA_DIR:-$H/llama.cpp}
TQ=${1:-Q4_K_M}
PROBE_KIND=${2:-text}
REPS=${3:-1}
read -r -a VS <<< "${4:-V16k V32k V48k V64k}"
T=$H/models/target/LFM2.5-VL-3B-$TQ.gguf
case $PROBE_KIND in
  text) PROBE=$H/probe/track_t_probe20.jsonl ;;
  ja)   PROBE=$H/probe/ja_probe10.jsonl ;;
  *)    echo "unknown probe $PROBE_KIND"; exit 1 ;;
esac
declare -A D=([none]=none [Q8_0]=$H/models/quant/LFM2.5-VL-3B-DSpark-Q8_0.gguf)
for v in "${VS[@]}"; do D[$v]=$H/models/e9/LFM2.5-VL-3B-DSpark-Q8_0-$v.gguf; done
ORDER=(none Q8_0 "${VS[@]}")
for rep in $(seq 1 "$REPS"); do
  # rotate the order each repetition so slow drift does not favour one drafter
  k=$(( (rep - 1) % ${#ORDER[@]} ))
  for d in "${ORDER[@]:$k}" "${ORDER[@]:0:$k}"; do
    $PY "$R/regression/probe_accept.py" "$L" "dp_${PROBE_KIND}_t${TQ}_d${d}_r${rep}" "$T" "${D[$d]}" "$PROBE" > /dev/null
  done
done
$PY "$R/analysis/summarize_draft_precision.py" "$PROBE_KIND" "$TQ" --drafters "$(IFS=,; echo "Q8_0,${VS[*]}")"
