#!/usr/bin/env bash
# S3 greedy-equivalence matrix on the runtime used for the evaluations (default: the edge runtime, C1-C4), then the
# judgment (analysis/judge_greedy.py). Every speculative run is compared with target-only greedy on the same target
# and runtime:
#   text probe (20 prompts, <= 384 tokens): targets F16 / Q8_0 / Q4_K_M / Q4_0 x drafters F16 / Q8_0 / P3-V32k-i32
#                                           x n_max 1 / 8
#   VL probe (20 image+text prompts):       targets F16 / Q4_K_M x drafters F16 / P3-V32k-i32, n_max 8
#   CPU control (10 prompts, 128 tokens):   Q4_K_M target, F16 drafter, n_max 1 / 8
#   run_s3.sh [tag=s3]       env: LLAMA_DIR (default $EDGESPECQ_HOME/llama-edge), TARGETS, DRAFTERS
# Results: $EDGESPECQ_HOME/logs/probe/<tag>*__*/ and results/summary/s3_greedy_equivalence.json
set -u
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
PY=$H/.venv/bin/python
L=${LLAMA_DIR:-$H/llama-edge}
TAG=${1:-s3}
TARGETS=${TARGETS:-F16 Q8_0 Q4_K_M Q4_0}
DRAFTERS=${DRAFTERS:-F16 Q8_0 P3-V32k-i32}
MMPROJ=$H/models/target/mmproj-LFM2.5-VL-3B-Q8_0.gguf
export LOGPROBS=1

target() { echo "$H/models/target/LFM2.5-VL-3B-$1.gguf"; }
drafter() {
  case $1 in
    F16) echo "$H/models/draft/LFM2.5-VL-3B-DSpark-F16.gguf" ;;
    *-V*) echo "$H/models/e9/LFM2.5-VL-3B-DSpark-$1.gguf" ;;
    *) echo "$H/models/quant/LFM2.5-VL-3B-DSpark-$1.gguf" ;;
  esac
}
run() {  # run <run name> <target tag> <drafter tag|none> <probe> [extra server args...]
  local name=$1 t=$2 d=$3 probe=$4; shift 4
  local dp=none; [ "$d" != none ] && dp=$(drafter "$d")
  "$PY" "$R/regression/probe_accept.py" "$L" "$name" "$(target "$t")" "$dp" "$probe" "$@" | head -2
}

for q in $TARGETS; do
  run "${TAG}__${q}__none" "$q" none "$H/probe/track_t_probe20.jsonl"
  for d in $DRAFTERS; do for n in 1 8; do
    N_MAX=$n run "${TAG}__${q}__${d}__n$n" "$q" "$d" "$H/probe/track_t_probe20.jsonl"
  done; done
done

for q in F16 Q4_K_M; do
  run "${TAG}vl__${q}__none" "$q" none "$H/probe/vl_probe20.jsonl" --mmproj "$MMPROJ"
  for d in F16 P3-V32k-i32; do
    N_MAX=8 run "${TAG}vl__${q}__${d}__n8" "$q" "$d" "$H/probe/vl_probe20.jsonl" --mmproj "$MMPROJ"
  done
done

export NGL=0 N_PROMPTS=10 MAX_TOKENS=128
run "${TAG}cpu__Q4_K_M__none" Q4_K_M none "$H/probe/track_t_probe20.jsonl" -t "$(nproc)"
for n in 1 8; do
  N_MAX=$n run "${TAG}cpu__Q4_K_M__F16__n$n" Q4_K_M F16 "$H/probe/track_t_probe20.jsonl" -t "$(nproc)" -td "$(nproc)"
done

"$PY" "$R/analysis/judge_greedy.py" "$TAG" --json "$R/results/summary/s3_greedy_equivalence.json" \
  --runtime "$(git -C "$L" log -1 --format=%h) + $(git -C "$L" diff --stat | tail -1 | sed 's/^ *//')"
