#!/usr/bin/env bash
# Greedy-equivalence matrix on master: VL-3B target precision x DSpark n_max on CUDA, plus a CPU control and
# the no-speculation batch-invariance test. See analysis/divergence.py for how divergences are classified.
#   run_greedy_matrix.sh [precisions="F16 Q8_0 Q4_0 Q4_K_M"] [n_max list="1 4 7 8"]
set -u
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
PY=$H/.venv/bin/python
L=$H/llama.cpp
M=$H/models/target
DR=$H/models/draft/LFM2.5-VL-3B-DSpark-F16.gguf
PROBE=$H/probe/track_t_probe20.jsonl
QS=${1:-F16 Q8_0 Q4_0 Q4_K_M}
NS=${2:-1 4 7 8}
export LOGPROBS=1

for q in $QS; do
  T=$M/LFM2.5-VL-3B-$q.gguf
  $PY "$R/regression/probe_accept.py" $L g_${q}_none $T none $PROBE | head -1
  for n in $NS; do
    N_MAX=$n $PY "$R/regression/probe_accept.py" $L g_${q}_n$n $T $DR $PROBE | sed -n 1,3p
    echo "---- CUDA $q n_max=$n (verify batch $((n + 1)))"
    $PY "$R/analysis/divergence.py" g_${q}_none g_${q}_n$n | tail -2
  done
done

echo "#### CPU control (Q4_K_M, 10 prompts x 128 tokens)"
T=$M/LFM2.5-VL-3B-Q4_K_M.gguf
NGL=0 N_PROMPTS=10 MAX_TOKENS=128 $PY "$R/regression/probe_accept.py" $L c_Q4_K_M_none $T none $PROBE -t "$(nproc)" | head -1
for n in 1 8; do
  NGL=0 N_PROMPTS=10 MAX_TOKENS=128 N_MAX=$n $PY "$R/regression/probe_accept.py" $L c_Q4_K_M_n$n $T $DR $PROBE \
      -t "$(nproc)" -td "$(nproc)" | sed -n 1,3p
  echo "---- CPU Q4_K_M n_max=$n"
  $PY "$R/analysis/divergence.py" c_Q4_K_M_none c_Q4_K_M_n$n | tail -2
done

echo "#### batch invariance without speculation"
for q in $QS; do $PY "$R/analysis/batch_invariance.py" $L $M/LFM2.5-VL-3B-$q.gguf $PROBE; done
$PY "$R/analysis/batch_invariance.py" $L $M/LFM2.5-VL-3B-Q4_K_M.gguf $PROBE 0
