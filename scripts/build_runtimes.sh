#!/usr/bin/env bash
# Build the three runtimes of benchmark/configs.json from one llama.cpp commit, as shared-library sets that
# regression/run_eval.py selects with LD_LIBRARY_PATH (the binaries are the same for all three):
#   master : unpatched
#   fixed  : + C3 scheduler fix (0001-...) + C1/C2 kernel selection fix (e8-shortk-igpu.patch)
#   edge   : fixed + C4 draft-space Markov chain (e9-c4-draft-space.patch)
#   build_runtimes.sh [commit=620d2225c] [cuda_arch=120]
# Worktree: $EDGESPECQ_HOME/llama-edge; libraries: $EDGESPECQ_HOME/libs/<platform>/{master,fixed,edge}/
set -euo pipefail
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
R=$(cd "$(dirname "$0")/.." && pwd)
P=$R/runtime/llama_cpp_patches
COMMIT=${1:-620d2225c}
ARCH=${2:-120}
PLAT=${PLATFORM:-5070}
[ -f /etc/profile.d/cuda.sh ] && source /etc/profile.d/cuda.sh
DIR=$H/llama-edge
L=$H/libs/$PLAT
[ -d "$DIR" ] || git -C "$H/llama.cpp" worktree add -q --detach "$DIR" "$COMMIT"
cd "$DIR"
git diff --quiet || { echo "$DIR has local changes; refusing to re-patch" >&2; exit 1; }
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > build.cmake.log 2>&1

stage() {  # build, then save the libraries that the patches touch
  cmake --build build -j"$(nproc)" --target llama-server llama-bench > "build.$1.log" 2>&1 \
    || { grep -i error "build.$1.log" | head -20 >&2; exit 1; }
  mkdir -p "$L/$1"
  cp -a build/bin/libggml-base.so* build/bin/libggml-cuda.so* build/bin/libllama.so* "$L/$1/"
  echo "$1: $(git diff --stat | tail -1)"
}
stage master
git apply "$P/0001-ggml-skip-copying-split-inputs-of-unselected-branches.patch" "$P/e8-shortk-igpu.patch"
stage fixed
git apply "$P/e9-c4-draft-space.patch"
stage edge
