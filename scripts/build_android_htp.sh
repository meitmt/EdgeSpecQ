#!/usr/bin/env bash
# Cross-compile llama.cpp for Android with the Hexagon (NPU) and OpenCL (Adreno) backends, inside Qualcomm's
# toolchain image (Android NDK + Hexagon SDK), following docs/backend/snapdragon/README.md.
#   build_android_htp.sh [variant=edge|master]
# Source: edge = $EDGESPECQ_HOME/llama-edge (C3/C4 patches), master = $EDGESPECQ_HOME/llama.cpp.
# Output: $EDGESPECQ_HOME/android/htp-<variant>/{bin,lib}; on the device run with
#   LD_LIBRARY_PATH=<dir>/lib ADSP_LIBRARY_PATH=<dir>/lib <dir>/bin/llama-server --device HTP0 ...
# DOCKER defaults to "docker"; set DOCKER="sudo docker" if the user is not in the docker group.
set -euo pipefail
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
V=${1:-edge}
IMAGE=ghcr.io/snapdragon-toolchain/arm64-android:v0.7
case $V in master) SRC=$H/llama.cpp ;; edge) SRC=$H/llama-edge ;; *) echo "unknown variant $V"; exit 1 ;; esac
O=$H/android/htp-$V
${DOCKER:-docker} run --rm -u "$(id -u):$(id -g)" -v "$SRC:/workspace" -w /workspace --platform linux/amd64 "$IMAGE" bash -lc '
  set -e
  cp docs/backend/snapdragon/CMakeUserPresets.json build-snapdragon.presets.json
  ln -sf build-snapdragon.presets.json CMakeUserPresets.json
  cmake --preset arm64-android-snapdragon-release -B build-snapdragon -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON > build-snapdragon.cmake.log 2>&1
  cmake --build build-snapdragon -j"$(nproc)" > build-snapdragon.build.log 2>&1
  rm -rf pkg-android && cmake --install build-snapdragon --prefix pkg-android > /dev/null
  rm -f CMakeUserPresets.json
' || { grep -iE "error" "$SRC/build-snapdragon.build.log" "$SRC/build-snapdragon.cmake.log" 2>/dev/null | head -20; exit 1; }
rm -rf "$O"; mkdir -p "$O/bin" "$O/lib"
cp "$SRC"/pkg-android/bin/{llama-server,llama-bench,test-backend-ops} "$O/bin/"
cp -a "$SRC"/pkg-android/lib/*.so* "$O/lib/"
echo "htp-$V: $(git -C "$SRC" log -1 --format=%h) $(git -C "$SRC" diff --stat | tail -1) -> $O ($(du -sh "$O" | cut -f1))"
ls "$O/lib"
