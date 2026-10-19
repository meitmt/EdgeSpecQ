# C2 on current llama.cpp master (Jetson Orin Nano Super)

llama.cpp `7951f601e` (2026-10-19) vs the same commit + `runtime/llama_cpp_patches/upstream-c2-orin-mmvq-small-k.patch`
(MMVQ small-K layout for every single-column, non-MoE mat-vec on Jetson Orin, `cc == GGML_CUDA_CC_ORIN`).
Only `libggml-cuda.so` differs between the two builds. Measured 2026-10-19; desktop stopped, MAXN_SUPER +
`jetson_clocks` unless noted ("default" = factory 25 W mode), then restored.

- `bench.jsonl`: `llama-bench -p 0 -n 128 -r 5 -ngl 99`, builds interleaved, 2 rounds (MAXN) / 2 rounds (default).
- `gemv_n1_table.md`: `test-backend-ops perf -o MUL_MAT -p 'n=1,'`, mean of 2 runs per build.
- `ppl_*.log`: `llama-perplexity -c 512 -b 512 -ub 1 --chunks 10` on WikiText-2 test (ubatch 1, so every matmul is
  a single-column mat-vec): identical PPL for both builds.
- `test-backend-ops -o MUL_MAT -b CUDA0`: 2445/2445 for both builds, twice each. One earlier C2 run failed a single
  random-data case (q5_1 m=16 n=1 k=4096, NMSE 7.1e-4 > 5e-4) that passed 6/6 in isolation on both builds.
