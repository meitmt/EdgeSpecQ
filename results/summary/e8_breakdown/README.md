# E8 per-round breakdowns (before the runtime fixes)

`analysis/e8_breakdown.py` on the timing build (`runtime/llama_cpp_patches/e8-timing.patch`, llama.cpp `620d2225c`,
no C1–C4), LFM2.5-VL-3B Q4_K_M target, text probe (5 prompts, 256 tokens), ms per speculation round.

- `e8_5070_n8.json`: RTX 5070, `n_max` 8, drafters F16 / Q8_0 / P3 / Q4_K_M / Q4_0
- `e8_jetson_maxn_n{5,6,7,8}.json`: Jetson Orin Nano Super, MAXN_SUPER + `jetson_clocks`, `n_max` 5–8
