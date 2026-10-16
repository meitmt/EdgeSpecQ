#!/usr/bin/env bash
# E9: export the reduced-vocabulary Q8_0 DSpark drafters used by run_e9_vocab.sh.
# Ranking mix "self+gen": the target's own greedy answers to non-probe prompts (analysis/e9_selfgen.py) with
# generic corpora at lower weight for the tail (analysis/e9_corpus_freq.py).
#   make_e9_drafters.sh [sizes="16 24 32 48"]   (thousands of tokens)
set -eu
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
MIX=self_en=1,self_ja=1,en_chat=.25,en_wiki=.25,code=.25,math=.25,ja_wiki=.5,ja_chat=.5
mkdir -p "$H/models/e9"
for k in ${1:-16 24 32 48}; do
  "$H/.venv/bin/python" "$R/quantization/make_reduced_vocab_drafter.py" \
    "$H/models/quant/LFM2.5-VL-3B-DSpark-Q8_0.gguf" "$H/models/target/LFM2.5-VL-3B-Q4_K_M.gguf" \
    $((k * 1000)) "$H/models/e9/LFM2.5-VL-3B-DSpark-Q8_0-V${k}k.gguf" --mix "$MIX"
done
