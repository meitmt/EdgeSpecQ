#!/usr/bin/env bash
# Cross-compile llama-server / llama-bench for Android (arm64-v8a, CPU backend) from two trees of the same commit:
#   master : $EDGESPECQ_HOME/llama.cpp   (unpatched)
#   edge   : $EDGESPECQ_HOME/llama-edge  (scripts/build_runtimes.sh: C3 scheduler + C4 draft space; C1/C2 are CUDA-only)
# Output: $EDGESPECQ_HOME/android/<variant>/{bin,lib}, ready for `adb push` to /data/local/tmp/edgespecq/.
#   build_android.sh [variants="master edge"] [march=armv8.7-a]
# Needs the NDK at $NDK (default $EDGESPECQ_HOME/tools/android-ndk-r27c).
set -euo pipefail
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
NDK=${NDK:-$H/tools/android-ndk-r27c}
MARCH=${2:-armv8.7-a}  # Snapdragon 8 Elite (Oryon): dotprod, i8mm, bf16
for v in ${1:-master edge}; do
  case $v in master) SRC=$H/llama.cpp ;; edge) SRC=$H/llama-edge ;; *) echo "unknown variant $v"; exit 1 ;; esac
  B=$SRC/build-android
  cmake -S "$SRC" -B "$B" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=$MARCH" -DCMAKE_CXX_FLAGS="-march=$MARCH" \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    > "$B.cmake.log" 2>&1 || { tail -20 "$B.cmake.log"; exit 1; }
  cmake --build "$B" -j"$(nproc)" --target llama-server llama-bench > "$B.build.log" 2>&1 \
    || { grep -iE 'error' "$B.build.log" | head -20; exit 1; }
  O=$H/android/$v
  rm -rf "$O"; mkdir -p "$O/bin" "$O/lib"
  cp "$B"/bin/llama-server "$B"/bin/llama-bench "$O/bin/"
  find "$B/bin" -name '*.so*' -exec cp -a {} "$O/lib/" \;
  cp "$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "$O/lib/"
  find "$O" -type f \( -name 'llama-*' -o -name '*.so*' \) -exec "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded {} \;
  echo "$v: $(git -C "$SRC" log -1 --format=%h) $(git -C "$SRC" diff --stat | tail -1) -> $O ($(du -sh "$O" | cut -f1))"
done
