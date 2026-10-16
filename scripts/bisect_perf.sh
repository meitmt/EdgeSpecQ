#!/usr/bin/env bash
# Bisect a token-generation throughput regression with `git bisect run`, robust to session-level drift.
#   bisect_perf.sh <good> <bad> <ref_llama_dir> <min_ratio> [model]
# Each step builds the candidate in a dedicated worktree ($EDGESPECQ_HOME/llama-bisect), then benchmarks
# reference / candidate alternately (ROUNDS rounds) and compares median(candidate) / median(reference) with
# min_ratio. A single absolute threshold is not enough: the same build can drift ~3% between sessions
# (unlocked GPU clocks, thermals), which misattributed a regression in an earlier run.
# Commits that fail to build or run are skipped (exit 125). Steps are logged to logs/bisect_perf.tsv.
set -euo pipefail
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
GOOD=$1; BAD=$2; REF=$3; MIN_RATIO=$4
MODEL=${5:-$H/models/track_t/LFM2.5-2.6B-Q4_K_M.gguf}
ROUNDS=${ROUNDS:-3}
W=$H/llama-bisect
mkdir -p "$H/logs"
[ -d "$W" ] || git -C "$H/llama.cpp" worktree add -q --detach "$W" "$BAD"

STEP=$(mktemp)
cat > "$STEP" <<EOF
#!/usr/bin/env bash
[ -f /etc/profile.d/cuda.sh ] && source /etc/profile.d/cuda.sh
cd "$W"
sha=\$(git rev-parse --short=9 HEAD)
log() { echo -e "\$sha\t\$*\t\$(git log -1 --format=%s)" >> "$H/logs/bisect_perf.tsv"; }
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF > build.cmake.log 2>&1 || { log skip cmake; exit 125; }
cmake --build build -j\$(nproc) --target llama-bench > build.log 2>&1 || { log skip build; exit 125; }
ref=(); cand=()
for i in \$(seq $ROUNDS); do
  ref+=(\$(bash "$R/scripts/bench_tg.sh" "$REF" "$MODEL" 256 5)) || { log skip bench-ref; exit 125; }
  cand+=(\$(bash "$R/scripts/bench_tg.sh" "$W" "$MODEL" 256 5)) || { log skip bench; exit 125; }
done
read verdict ratio < <(python3 - "\${ref[*]}" "\${cand[*]}" <<'PY'
import statistics as s, sys
ref = [float(x) for x in sys.argv[1].split()]; cand = [float(x) for x in sys.argv[2].split()]
ratio = s.median(cand) / s.median(ref)
print("good" if ratio > $MIN_RATIO else "bad", f"{ratio:.4f}")
PY
)
log "\$verdict\tratio=\$ratio\tref=\${ref[*]}\tcand=\${cand[*]}"
[ "\$verdict" = good ]
EOF
chmod +x "$STEP"

cd "$W"
git bisect reset 2>/dev/null || true
git bisect start "$BAD" "$GOOD"
git bisect run "$STEP"
git bisect log > "$H/logs/bisect_perf.log"
git bisect reset
rm -f "$STEP"
