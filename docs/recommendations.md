# Deploying LFM2.5-VL-3B + DSpark on edge devices: findings and recommendations

Draft, 2026-10-23. All numbers are measured decode throughput (image encoding and prefill excluded), greedy decoding, `max_tokens` 512, on the evaluation sets described
below; target LFM2.5-VL-3B Q4_K_M unless noted. Each recommendation links to the result files in
`results/summary/`.

## Setup

| Platform | Hardware | Runtime |
|---|---|---|
| Jetson | Orin Nano Super 8 GB (MAXN_SUPER unless noted; desktop off while measuring) | llama.cpp 620d2225c, CUDA, sm_87 |
| Desktop | RTX 5070 12 GB (WSL2) | same commit, CUDA, sm_120 |
| Phone | OnePlus 13 (Snapdragon 8 Elite, 24 GB) | same commit, Hexagon NPU (HTP v79) or CPU |

- **Evaluation set (dev120):** six task categories following MMSpec and Liquid's DSpark-VL speedup table: GQA,
  TextVQA, COCO captions, CharXiv, MMMU-Pro, MM-MT-Bench (multi-turn), 20 seeded samples each
  (`benchmark/build_eval_set.py`). Speedups are paired decode-rate ratios with 95 % bootstrap intervals
  (`analysis/summarize_eval.py`).
- **Runtimes compared:** `master` (unpatched), `fixed` (+ C1–C3 below), `edge` (+ C4), same commit, switched as
  shared-library sets (`scripts/build_runtimes.sh`).
- **Drafters:** the official F16 DSpark drafter, llama.cpp quantizations of it (Q8_0, Q4_K_M, Q4_0, "P3" = decoder
  Q4_K with projections and Markov head kept at Q8_0), and reduced-vocabulary variants exported without retraining
  (`quantization/make_reduced_vocab_drafter.py`).

## Headline results (dev120)

| Platform | Official drafter as shipped | Best configuration | vs. no speculation (same runtime) |
|---|---:|---|---:|
| Jetson MAXN | 50.4 t/s | 86.6 t/s: P3 drafter, 32k vocabulary, `n_max` 6, runtime fixes | 2.77× (3.20× vs unpatched target-only) |
| Jetson 25 W (factory default) | 45.4 t/s | 75.9 t/s, same configuration | 3.07× vs unpatched target-only |
| RTX 5070 | 329 t/s | 408 t/s: P3 drafter, 32k vocabulary (`n_max` 8) | 1.80× (1.85× vs unpatched target-only) |
| RTX 5070, Q4_0 target | — | 428 t/s: Q4_0 target + Q4_0 drafter | 1.76× vs the same Q4_0 target; target quality not evaluated |
| Phone NPU | 29.4 t/s | 31.4 t/s: P3 drafter, 32k vocabulary | 1.29× |
| Phone CPU | slower than no speculation | — | 0.67–0.69× |

In the RTX 5070 greedy-equivalence probe matrix, the target's outputs with every reduced-vocabulary and quantized
drafter were identical to those with the full-vocabulary drafter on the same target; on the 24-sample multimodal
set the official and the edge drafter differed on 1 of 24 samples on each GPU, at near-ties. The draft tokens
themselves can differ. Quantized targets are not bit-for-bit identical to non-speculative greedy
decoding because of batch-shape-dependent kernel numerics (see A6).

## Deployment recommendations

| # | Recommendation | Evidence |
|---|---|---|
| A1 | **Quantize the drafter.** F16 → Q8_0 is the largest single gain on bandwidth-limited devices (+44 % on Jetson, +19 % on the RTX 5070) with unchanged acceptance; P3 or Q4_K_M is slightly faster still. | `dev120_jetson_maxn.csv` A1→A4; `dev120_rtx5070.csv` |
| A2 | **Set the draft length per device:** `n_max` 6 on Jetson, 8 on the RTX 5070 and on the phone NPU. | ablation A4→A5; `phone_breakdown/` |
| A3 | **Speculation pays most where decode is bandwidth-bound**: 2.8–3.2× on Jetson, about 1.8× on a desktop GPU, about 1.3× on the phone NPU. Gains are largest for long outputs (captioning, multi-turn) and smallest for short answers (GQA, TextVQA). | all dev120 CSVs, per-category columns |
| A4 | **Target format is device-dependent:** Q4_0 target + Q4_0 drafter is fastest on the RTX 5070 but fifth on Jetson, where Q4_K_M + a quantized drafter wins. Benchmark on the deployment device. | `dev120_jetson_maxn.csv` B10 vs B11 |
| A5 | **Keep the Markov head at ≥ Q8 if the runtime lacks the short-K kernel fix (C1)**, i.e. use P3. | E8 results |
| A6 | Speculative output with a quantized target differs from non-speculative greedy output at near-ties; this is runtime numerics (F16 targets match exactly except at near-exact ties; in the RTX 5070 probe matrix, changing the drafter did not change the output for a fixed target and `n_max`; on the 24-sample set the official and edge drafters differed on 1 of 24 samples per GPU, at near-ties), not a DSpark issue. Evaluate with near-tie-aware criteria. | `s3_greedy_equivalence.json`, `limitations.md` §1 |
| A8 | **Use a reduced-vocabulary drafter on edge GPUs** (32k for English or mixed workloads): +4.5 % on Jetson on its own, +7.4 % with the draft-space Markov chain (C4); on the desktop GPU it needs C4 to break even. | `dev120_jetson_maxn.csv` A5→A6→B11 |
| A9 | **On 8 GB devices set `--cache-ram 0` (or small)** for llama-server: its 8 GiB default host prompt cache fills the board within one run and leads to swap, CUDA graph instantiation failures and OOM kills. A Q8_0 target with mmproj and a drafter does not fit reliably in 8 GB; use Q4-class targets. | Jetson run logs, 2026-10-17 |
| A10 | **On phones use the Hexagon NPU, and do not enable speculation on the CPU.** NPU decode is 1.5× the CPU's and prompt processing (including image encoding) more than 8× faster; DSpark gives +29 % on the NPU but −31 % on the CPU. Keep target and drafter on the NPU: moving the drafter to the GPU or CPU is slower (cross-device transfers). | `phone_htp_dev120.csv`, `phone_cpu_smoke24.csv`, `phone_breakdown/` |
| A11 | Speculative decoding on the phone NPU slows down short-answer tasks (0.8–1.0× on GQA, TextVQA, multiple choice); enable it per task or only for long generations. | `phone_htp_dev120.csv` per-category columns |
| A12 | **Expect much less from DSpark on Japanese.** On 75 Japanese samples (JMMMU, JA-VLM-Bench-In-the-Wild, JA-VG-VQA-500, ELYZA-tasks-100, Japanese captioning) the best configuration gives 1.30× on the RTX 5070 (vs 1.80× on English dev120) and 1.98–2.02× on Jetson (vs 2.77×); the official F16 drafter gives 1.00× and 1.21×. On the same 15 COCO images, captioning in Japanese has a 15–18 % shorter mean accepted length than in English (llama-server's per-round statistic; 3.36 vs 3.93 on the RTX 5070) and 1.61× instead of 1.94×. The 25 % in B6 is a different measure (accepted draft tokens per round in the round log, 2.34 vs 3.11). | `ja75_rtx5070.csv`, `ja75_jetson_maxn.csv` |
| A13 | **Speculation holds under the model card's sampling settings** (temperature 0.2, top-k 50): the best configuration gives 1.77× vs 1.80× greedy on the RTX 5070 and 2.69× vs 2.77× on Jetson, with the same ranking of drafters and ±4 % across 3 seeds. At temperature 0.7 acceptance drops and the gain falls to 1.5× (RTX 5070) and 2.35× (Jetson, 1 seed). | `sampling_rtx5070.csv`, `sampling_jetson_maxn.csv` |
| A14 | **Tailor a reduced vocabulary to the deployment language.** A 16k vocabulary whose ranking weights Japanese output matches or beats the 32k one on Japanese (RTX 5070 1.30× vs 1.30×, Jetson 2.02× vs 1.98×) while being slower on English (dev120: 1.63× vs 1.80× on the RTX 5070, 2.59× vs 2.77× on Jetson). | `ja75_*.csv`, `dev120_*.csv` |

## Model-side observations for DSpark

| # | Observation | Evidence |
|---|---|---|
| B1 | The Markov head's cost scales with vocabulary × rank × block length and is serial per draft position. On large-cache GPUs it mostly hits cache; on small-cache edge devices it streams from memory and took more than half of the drafter time before the runtime fixes. | E8 per-round breakdowns |
| B2 | A reduced-vocabulary release ("edge" DSpark) is nearly free: the kept rows are copied exactly from the existing drafter, and the target's greedy outputs matched the full drafter's except at near-ties in our tests. On Jetson, P3 + 32k vocabulary + C4 cuts draft time from 13.2 to 6.3 ms per round. | `e9c4_jetson_maxn_breakdown_n6.json` |
| B5 | Rank the reduced vocabulary by the target's own outputs, including multimodal answers, not by generic corpora: LFM2.5's Markdown, LaTeX and special-whitespace tokens are rare in generic text (32k keeps 85 % of English acceptance with generic ranking, 95.5 % with self-generated text), and a text-only ranking loses 10 % of accepted length on multimodal tasks. | `e9_simulate_5070.json`; dev120 B11 vs A5 |
| B6 | **Where Japanese loses acceptance (per-round analysis, Q8_0 drafter, RTX 5070).** (1) The gap depends on the task: on captioning of the same 15 images Japanese accepts 25 % less (2.34 vs 3.11 accepted draft tokens per round); on open-ended text both languages are low (1.40 vs 1.46), but those prompts are different tasks, so they do not show that the gap disappears there. (2) In Japanese captions the weak tokens are kanji: 37.5 % of drafted positions, acceptance 0.60 vs 0.74–0.92 for kana and punctuation. (3) Kanji is also where the target is least confident: its top-1 probability there is 0.57 (kana 0.73–0.76, English words 0.69), and for the character classes analysed acceptance stays within −0.02 to +0.13 of the target's top-1 probability. This is a correlation; under greedy verification a better drafter could still predict the target's top choice where the target is unsure, and no drafter trained on more Japanese was compared. (4) Japanese needs about 3× more tokens for the same text (1.5 vs 4.7 characters per token). Hypothesis to test: if the target's uncertainty is the main limit, more Japanese drafter training helps less than on English; the uncertainty is spread over many candidates (top-2 mass 0.71 on kanji vs 0.83 on English words, both about 0.14 above top-1, so two-candidate tree drafting would likely help kanji no more than English words). LFM2.5-VL also answers Japanese prompts very briefly unless asked for detail. | `analysis/e9_lang_*.py`, `results/summary/e9_lang_analysis_q8.json` |
| C6 | The LFM2-VL vision encoder's `UPSCALE` (position-embedding interpolation) is the only vision op the Hexagon backend does not support; it falls back to the CPU. | phone placement probe |

## Runtime issues found (llama.cpp)

| # | Issue | Effect | Status |
|---|---|---|---|
| C1 | CUDA picks MMVQ for short-K (K ≤ 256), single-column quantized GEMV such as the Markov head, which is slow | Jetson speculation +16–30 %, desktop +4–5 % when fixed | patch `runtime/llama_cpp_patches/e8-shortk-igpu.patch` |
| C2 | Jetson Orin does not use the small-K MMVQ layout for single-column GEMV | Jetson target-only decode +15 % for Q4_K_M; −1 % to +38 % by quantization type (Q4_0 +38 %, Q8_0 +23 %, Q6_K +2 %); the fused gate/up path absorbs most of the gain for some K-quants | same patch; Orin-only version for llama.cpp `7951f601e`: `upstream-c2-orin-mmvq-small-k.patch` |
| C3 | After #29622, LFM2 text decode copies split inputs of unselected graph branches every token | desktop +3.5–4.2 % when fixed | `0001-ggml-skip-copying-split-inputs-of-unselected-branches.patch` |
| C4 | With a reduced draft vocabulary, the DSpark Markov chain still scatters to the full 128k vocabulary at every position | Jetson +3 %, desktop from −2 % to +6 % when fixed | `e9-c4-draft-space.patch` (needs an I32 `d2t`) |
| C5 | Hexagon backend, single-token vs multi-token path: (a) verifying 3 tokens costs 1.3× and 5–9 tokens 1.7–1.8× a single-token decode (HMX path fixed cost; HVX is twice slower); (b) the two paths give clearly different logits: with no speculation, a physical batch of 2 instead of 1 moves next-token logprobs by up to 2.6 (4.3 for larger batches) and flips the top-1 token in 1–3 of 10 prompts (CUDA: ≤ 0.06 F16, ≤ 0.6 quantized) | (a) verification is the largest cost of a DSpark round on the phone NPU (71 of 109 ms), where DSpark gives 1.29×; (b) the NPU's greedy output depends on prompt chunking even without speculation, and speculative output diverges from target-only output at margins up to 0.65 (`s3_batch_invariance_phone.txt`, `s3_smoke24_phone_htp.json`) | to report |
| C7 | llama-server's 8 GiB default `--cache-ram` is unsafe on 8 GB devices | OOM / swap on Jetson | to report |

## Caveats

- 120 samples per platform (6 × 20), 2–3 repetitions; the confidence intervals cover sample variation, not
  run-to-run drift (about 3 % on the unlocked RTX 5070).
- Sampling was checked on the RTX 5070 and Jetson (A13); all other numbers are greedy.
- The Japanese set has 75 samples (15 per category) and 2–3 repetitions; the phone was not measured on Japanese.
- The phone's CPU is capped by the OEM power policy at 1.79 / 1.69 GHz for adb-launched processes (hardware
  3.53 / 4.32 GHz); a foreground app may get more. The phone NPU dev120 run heated the battery from 32 to 51 °C;
  speeds in the second half are indicative (`phone_htp_dev120_thermal.json`).
- Jetson numbers are measured with the desktop stopped; with the desktop running they are about 2–3 % lower.
