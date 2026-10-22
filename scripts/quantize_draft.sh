#!/usr/bin/env bash
# Quantize the official F16 DSpark drafter into uniform and mixed-precision variants, then write a tensor
# manifest (CSV) and SHA256 for each output.
#   quantize_draft.sh [src.gguf] [out_dir]
# Variants (docs/report.md §3): Q8_0, Q4_K_M, Q4_0 (uniform presets, same as Liquid's official text drafters)
# and P3 = decoder Q4_K, projection (fc) + Markov head Q8_0.
set -euo pipefail
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:-$H/models/draft/LFM2.5-VL-3B-DSpark-F16.gguf}
OUT=${2:-$H/models/quant}
Q=$H/llama.cpp/build/bin/llama-quantize
mkdir -p "$OUT"
base=$(basename "$SRC" -F16.gguf)

run() {  # run <name> <ftype> [llama-quantize options...]
  local name=$1 ftype=$2; shift 2
  local dst=$OUT/$base-$name.gguf
  "$Q" "$@" "$SRC" "$dst" "$ftype" > "$OUT/$base-$name.quantize.log" 2>&1 || { tail -20 "$OUT/$base-$name.quantize.log"; exit 1; }
  "$H/.venv/bin/python" "$R/quantization/inspect_gguf.py" "$dst" --csv "$OUT/$base-$name.manifest.csv" | sed -n '/^group/,/^TOTAL/p'
  sha256sum "$dst" | tee "$dst.sha256"
}

for p in Q8_0 Q4_K_M Q4_0; do
  echo "#### $p"; run "$p" "$p"
done
echo "#### P3 (decoder Q4_K, fc + markov Q8_0)"
run P3 Q4_K_M --tensor-type 'fc=q8_0' --tensor-type 'markov_w1=q8_0' --tensor-type 'markov_w2=q8_0'
