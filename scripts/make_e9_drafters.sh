#!/usr/bin/env bash
# E9: export the reduced-vocabulary DSpark drafters used by run_e9_vocab.sh.
# Ranking mix "self+gen": the target's own greedy answers to non-probe prompts (analysis/e9_selfgen.py) with
# generic corpora at lower weight for the tail (analysis/e9_corpus_freq.py).
#   make_e9_drafters.sh [sizes="16 24 32 48"] [bases="Q8_0"]   (sizes in thousands of tokens; bases are
#   drafters under models/quant/, e.g. "Q8_0 Q4_K_M P3")
# Output: $EDGESPECQ_HOME/models/e9/LFM2.5-VL-3B-DSpark-<base>-V<size>k[-i32].gguf
# D2T=i32 writes an I32 d2t (draft-space Markov chain, runtime/llama_cpp_patches/e9-c4-draft-space.patch).
set -eu
D2T=${D2T:-i64}
SUF=; [ "$D2T" = i32 ] && SUF=-i32
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
MIX=self_en=1,self_ja=1,en_chat=.25,en_wiki=.25,code=.25,math=.25,ja_wiki=.5,ja_chat=.5
mkdir -p "$H/models/e9"
for b in ${2:-Q8_0}; do
  for k in ${1:-16 24 32 48}; do
    "$H/.venv/bin/python" "$R/quantization/make_reduced_vocab_drafter.py" \
      "$H/models/quant/LFM2.5-VL-3B-DSpark-$b.gguf" "$H/models/target/LFM2.5-VL-3B-Q4_K_M.gguf" \
      $((k * 1000)) "$H/models/e9/LFM2.5-VL-3B-DSpark-$b-V${k}k$SUF.gguf" --mix "$MIX" --d2t-type "$D2T"
  done
done
