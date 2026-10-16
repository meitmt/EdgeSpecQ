#!/usr/bin/env bash
# Build llama.cpp (llama-server) for one commit in its own git worktree.
#   build_llama.sh <commit|master> [cuda_arch=120]
# The main clone lives at $EDGESPECQ_HOME/llama.cpp (full history, needed for bisect); other commits go to
# $EDGESPECQ_HOME/llama-<shortsha>. Prints the build dir on the last line.
set -euo pipefail
HOME_DIR=${EDGESPECQ_HOME:-$HOME/edgespecq}
COMMIT=$1
ARCH=${2:-120}
[ -f /etc/profile.d/cuda.sh ] && source /etc/profile.d/cuda.sh

MAIN=$HOME_DIR/llama.cpp
[ -d "$MAIN/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp "$MAIN"
if [ "$COMMIT" = master ]; then
  DIR=$MAIN
else
  SHA=$(git -C "$MAIN" rev-parse --short=9 "$COMMIT")
  DIR=$HOME_DIR/llama-$SHA
  [ -d "$DIR" ] || git -C "$MAIN" worktree add -q --detach "$DIR" "$SHA"
fi

cd "$DIR"
echo "### $(git log -1 --format='%h %cd %s' --date=short)" >&2
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > build.cmake.log 2>&1 || { tail -30 build.cmake.log >&2; exit 1; }
cmake --build build -j"$(nproc)" --target llama-server > build.log 2>&1 || { grep -i error build.log | head -20 >&2; exit 1; }
./build/bin/llama-server --version 2>&1 | tail -2 >&2
echo "$DIR"
