#!/usr/bin/env bash
# E8 micro-benchmark: DSpark Markov-head GEMV shape (K=256 x 128000, N=1) and a long-K reference with the
# same element count (K=2048 x 16000), under each experimental kernel-selection toggle of the E8 timing build.
#   e8_gemv_bench.sh <llama_dir (timing build)> [label=ENV=VAL ...]
set -u
L=$1; shift
B=$L/build/bin/test-backend-ops
PARSE='import re, sys
bpw = {"q8_0": 34/32, "q4_0": 18/32, "q4_K": 144/256, "q6_K": 210/256, "f16": 2.0}
for line in sys.stdin:
    m = re.search(r"type_a=(\w+),type_b=f32,m=(\d+),n=1,k=(\d+).*?([\d.]+) us/run", line)
    if m:
        t, M, K, us = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
        mb = M * K * bpw[t] / 1e6
        print(f"  {t:5s} K={K:<5d} {us:8.1f} us  {mb:6.1f} MB  {mb / us * 1e3:7.1f} GB/s")'
run() {  # run <label> [ENV=VAL ...]
  local label=$1; shift
  echo "#### $label"
  env "$@" $B perf -o MUL_MAT -b CUDA0 -p 'm=(128000|64000|32000|16000),n=1,' 2>&1 | python3 -c "$PARSE"
}
if [ $# -eq 0 ]; then
  run "default"
  run "small_k on"  E8_SMALLK=1
  run "MMQ for all K<=2048" E8_MMQ_K=2048
else
  for spec in "$@"; do run "$spec" ${spec#*=}; done
fi
