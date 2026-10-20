# EdgeSpecQ: quantized DSpark on edge devices — contributions, results and reproduction

Status: 2026-10-19. This document covers everything measured so far on the three platforms. Every number in it
comes from a CSV or JSON file in `results/summary/` (named next to each table) and can be regenerated with the
commands in [Reproduction](#reproduction). Deployment advice derived from these results is in
[`recommendations.md`](recommendations.md).

**Subject.** Liquid AI's LFM2.5-VL-3B with its DSpark speculative-decoding drafter, run through llama.cpp on an
NVIDIA Jetson Orin Nano Super (8 GB), a OnePlus 13 phone (Snapdragon 8 Elite: Hexagon NPU and CPU), and an
RTX 5070 desktop GPU. The official release ships an F16 drafter and was evaluated in 16-bit on large-memory
hardware (H100, Apple M-series). This project asks what DSpark gives when the model is quantized and deployed on
real edge hardware, where the time goes, and what can be changed in the deployment, the runtime and the model.

## Summary

| | Result |
|---|---|
| **Speedup on an 8 GB Jetson** | Unpatched target-only decode 27.1 t/s → official F16 drafter 50.4 t/s → **86.6 t/s** with our drafter and runtime changes: **3.20×** over target-only and **1.72×** over the official drafter as shipped (dev120, 6 multimodal task categories, MAXN_SUPER). 3.07× in the factory-default 25 W mode. |
| **Drafter changes, no retraining** | Quantized (P3: decoder Q4_K, projection and Markov head Q8_0) and reduced-vocabulary (32k of 128k tokens) DSpark drafters, exported from the official F16 GGUF. Outputs are identical to the full-vocabulary drafter's. Regenerating them from this repository gives bit-identical files (checked by SHA256). |
| **Four llama.cpp runtime fixes** | C1/C2: CUDA kernel selection for short-K GEMV and Jetson Orin (target-only decode **+15 %** for Q4_K_M models; on current master −1 % to +38 % depending on the quantization type; DSpark +16–30 %). C3: per-token copies after upstream #29622 (desktop +3.5–4.2 %). C4: DSpark Markov chain kept in the draft vocabulary (Jetson +3 %, desktop −8 % → on par). |
| **Where the time goes** | Per-round breakdowns on all three platforms. On Jetson the Markov head was over half the drafter time, a cost that large-cache hardware hides. On the phone NPU, verifying 2–9 tokens costs about 1.8× a single-token decode, which caps DSpark there at about 1.3×. |
| **Japanese** | DSpark gives about 70 % of its English gain on Japanese (desktop 1.30× vs 1.80×, Jetson 1.98× vs 2.77×). The cause, measured round by round: the **target's own uncertainty on kanji**, amplified by tokenization (1.5 vs 4.7 characters per token). It is not a weaker drafter, and there is no gap on open-ended text. |
| **Robustness** | Holds under the model card's sampling settings (1.77× vs 1.80× greedy on desktop, 2.69× vs 2.77× on Jetson, 3 seeds), and in the default power mode. The drafter choice never changes the output. Against target-only greedy, outputs differ only at near-ties on the RTX 5070 and Jetson (F16 exact except two ties at margins 0.0036 and 0.0011); on the phone NPU and on the phone CPU with long contexts the S3 criteria are not met, which is traced to the backends' batch-shape numerics (`docs/limitations.md` §1). |
| **Measurement pitfalls found** | Seven ways the numbers silently go wrong on edge devices, each found, fixed in the harness and documented (§7). Examples: llama-server's 8 GiB prompt cache on an 8 GB board, an orphaned server answering for later configs, OEM CPU caps on Android. |

## 1. Setup

| Platform | Hardware | Software | Power state while measuring |
|---|---|---|---|
| Jetson | Orin Nano Super Developer Kit, 8 GB unified memory | JetPack 6.2 (L4T 36.4.3), CUDA 12.6, sm_87 | MAXN_SUPER + `jetson_clocks`, desktop stopped (`multi-user.target`); 25 W default for the `jetson_25w` rows |
| Desktop | RTX 5070 12 GB, WSL2 | CUDA 13.2, sm_120 | default (clocks not locked; about 3 % run-to-run drift) |
| Phone | OnePlus 13, Snapdragon 8 Elite, 24 GB | Android, Hexagon HTP v79 + OpenCL (Qualcomm toolchain image), or CPU (NDK r27c) | screen on, USB; cooldown to ≤ 36 °C before each config, paused above 45 °C |

- **llama.cpp** `620d2225c` (v0.7.0), the same commit everywhere. Three runtimes are built from it, with identical
  binaries and different shared libraries: `master` (unpatched), `fixed` (+ C1–C3), and `edge` (+ C4).
- **Models.** Target `LFM2.5-VL-3B` in Q4_K_M (main), Q4_0, Q8_0 and F16, with mmproj Q8_0. Drafter
  `LFM2.5-VL-3B-DSpark-F16` (official) plus the variants of §3. SHA256 of every file: `benchmark/manifests/models.sha256`.
- **Evaluation sets** (`benchmark/build_eval_set.py`, seed 2026).
  - **dev120** has six categories following MMSpec and Liquid's DSpark-VL table, 20 samples each: GQA, TextVQA,
    COCO captioning, CharXiv, MMMU-Pro (step by step) and MM-MT-Bench (multi-turn, model answers every turn).
  - **smoke24** is the first 4 samples of each dev120 category.
  - **ja75** (`benchmark/build_ja_set.py`) has five Japanese categories × 15: JMMMU, JA-VLM-Bench-In-the-Wild,
    JA-VG-VQA-500, ELYZA-tasks-100, and Japanese captioning of the same 15 COCO images as dev120.
  - Sample IDs: `benchmark/manifests/*.tsv`.
- **Metric.** Decode rate (generated tokens / decode time, token-weighted over turns) with greedy decoding and
  `max_tokens` 512. Speedup is the paired ratio against a reference config on the same samples, with a 95 %
  bootstrap interval over samples (`analysis/summarize_eval.py`). Per sample, tokens and time come from the
  repetition with the median decode time. Configs rotate order between repetitions.

### Configurations (`benchmark/configs.json`)

| Name | Target | Drafter | Runtime | `n_max` |
|---|---|---|---|---|
| B0 / A0 | Q4_K_M | none | edge / master | — |
| B1 / A1 (= B1m) | Q4_K_M | official F16 | edge / master | platform / 8 |
| B2, B3 | Q4_K_M | Q8_0, Q4_K_M | edge | platform |
| B4–B7 | Q8_0 | none, F16, Q8_0, Q4_K_M | edge | platform |
| B8–B10 | Q4_0 | none, F16, Q4_0 | edge | platform |
| **B11** | Q4_K_M | **P3, 32k vocabulary, I32 `d2t`** | edge | platform |
| B12 | Q4_K_M | Q4_K_M, 16k vocabulary, I32 `d2t` | edge | platform |
| A2–A7 | Q4_K_M | ablation steps, see §2.2 | | |
| R0 / R1 | F16 | none / F16 | edge | platform |

Platform `n_max`: Jetson 6, desktop 8, phone 8 (each chosen by measurement, §4).

## 2. End-to-end results

### 2.1 Headline (dev120, greedy)

| Platform | Target only, unpatched | Official F16 drafter as shipped | **Best** | vs target-only [95 % CI] | vs official drafter |
|---|---:|---:|---:|---:|---:|
| Jetson MAXN | 27.1 t/s (A0) | 50.4 (A1) | **86.6** (B11) | **3.20×** vs A0; 2.77× [2.52, 3.05] vs B0 | **1.72×** |
| Jetson 25 W | 24.7 (A0) | 45.4 (A1) | **75.9** (B11) | **3.07×** [2.80, 3.36] | 1.67× |
| RTX 5070 | 220.5 (A0) | 329.4 (B1m) | **427.7** (B10: Q4_0 target + Q4_0 drafter) | 1.88× [1.74, 2.05] vs B0 | 1.30× |
| | | | 408.3 (B11) | 1.80× [1.63, 1.99] | 1.24× |
| Phone NPU | 24.4 (B0) | 29.4 (B1, 1.20×) | **31.4** (B11) | **1.29×** [1.15, 1.44] | 1.07× |
| Phone CPU (smoke24) | 16.7 (B0) | 11.6 (B1) | — | 0.67–0.69× (slower) | — |

Files: `dev120_jetson_maxn.csv`, `dev120_jetson_25w.csv`, `dev120_rtx5070.csv`, `phone_htp_dev120.csv`,
`phone_cpu_smoke24.csv`.

Per category (B11 vs B0, Jetson MAXN): multi-turn 3.08×, captioning 2.85×, chart 2.67×, reasoning 2.35×,
TextVQA 2.31×, GQA 2.16×. The gain grows with output length. On the phone NPU, short-answer categories are
0.83–0.99× and multi-turn and chart are 1.36–1.45×.

### 2.2 Ablation: what each step contributes (dev120)

| Step | Change | Jetson MAXN t/s | step | RTX 5070 t/s | step |
|---|---|---:|---:|---:|---:|
| A0 | target only, unpatched runtime | 27.1 | | 220.5 | |
| A1 | + official F16 drafter (`n_max` 8) | 50.4 | +86 % | 330.1 | +50 % |
| A2 | + runtime fixes C1–C3 | 50.4 | +0.1 % | 335.5 | +1.6 % |
| A3 | F16 → **Q8_0 drafter** | 72.4 | **+44 %** | 398.6 | **+19 %** |
| A4 | Q8_0 → **P3** drafter | 76.2 | +5.3 % | 401.6 | +0.7 % |
| A5 | `n_max` 8 → platform value (Jetson 6) | 80.7 | +5.8 % | 413.5 ¹ | — |
| A6 | + **32k reduced vocabulary** | 84.2 | +4.4 % | 378.6 | −8.5 % |
| A7 = B11 | + **C4** draft-space Markov chain (I32 `d2t`) | **86.6** | +2.9 % | **410.6** | +8.5 % |

¹ On the desktop the platform `n_max` is 8, so A4 and A5 are the same configuration. Their 3 % difference is the
desktop's run-to-run drift.

Notes:
- C1–C3 do nothing for an F16 drafter (A1→A2). C1 targets quantized GEMVs, and C2 targets single-token target
  decode, which speculation replaces with multi-token verification. They matter for quantized drafters and
  target-only decode: B0 vs A0 on Jetson is +15.2 %, all from C2.
- The reduced vocabulary needs C4 on the desktop. Without C4 the full-vocabulary work per draft position cancels
  the savings, because the desktop's 48 MB L2 already holds the Markov head.

### 2.3 Drafter precision (20-prompt probes, before dev120)

On both platforms, Q8_0 and P3 keep acceptance within 2 % of F16 at 37–53 % of its size. Below Q8 the saved bytes
buy no further speed. Before C1, Q4_K was slow on sm_87; after C1, Q4_K_M is the fastest drafter on Jetson.
Files: `draft_precision_{5070,jetson_maxn}.csv`, `jetson_maxn_fixval_{MASTER_n8,FIX_n8,FIX_n6}.csv`.

| Drafter | Size | 5070 accept / speedup | Jetson accept / speedup (master runtime) |
|---|---:|---|---|
| F16 (official) | 533 MiB | 0.424 / 1.59× | 0.407 / 1.93× |
| Q8_0 | 283 MiB | 0.424 / 1.82× | 0.409 / 2.40× |
| P3 | 198 MiB | 0.420 / 1.84× | 0.400 / 2.46× |
| Q4_K_M | 157 MiB | 0.421 / 1.78× | 0.398 / 2.24× |
| Q4_0 | 150 MiB | 0.416 / 1.82× | 0.401 / 2.40× |

## 3. Contribution: edge DSpark drafters without retraining

**Quantized drafters** (`scripts/quantize_draft.sh`) are built with stock `llama-quantize` from the official F16
GGUF. The variants are Q8_0, Q4_K_M, Q4_0, and P3, which keeps decoder Q4_K with `fc`, `markov_w1` and `markov_w2`
at Q8_0. The Markov head stays at Q8 so the drafter is safe on runtimes without C1.

**Reduced-vocabulary drafters** (`quantization/make_reduced_vocab_drafter.py`) work as follows:
- They keep the N most useful of 128k tokens. `markov_w2` is sliced to those rows, the drafter gets its own output
  head (the matching rows of the target's tied `token_embd`), and a `d2t` map from draft rows to target ids is
  added. llama.cpp's dflash loader already reads `d2t`.
- Both weights are quantized along the hidden dimension only, so slicing rows is lossless. Every kept token's
  logit is bit-identical to the full drafter's, and on the same target the greedy outputs are identical.
- **Ranking matters more than size.** Generic corpora (chat, Wikipedia, code, math, Japanese) miss LFM2.5's own
  output style: Markdown bold, LaTeX, U+202F and U+2011. Ranking by the target's own answers to 900 non-evaluation
  prompts raises the 32k English retention from 85 % to 95.5 % of the full-vocabulary accepted length
  (`e9_simulate_5070.json`).
- **The ranking can be tailored to a language.** A 16k vocabulary weighted toward Japanese matches or beats the 32k
  one on Japanese (Jetson 2.02× vs 1.98×) and is slower on English (2.59× vs 2.77×).
- With an I32 `d2t` (`--d2t-type i32`) the C4 runtime path keeps the whole Markov chain in draft space.
- **Reproducible bit for bit.** The token-frequency counts used for the measured drafters are committed
  (`quantization/vocab_freq.npz`). Regenerating `P3-V32k-i32` (B11) and `Q4_K_M-V16k-i32` (B12) from them, and
  P3 / Q8_0 with `llama-quantize`, gives files whose SHA256 matches the measured ones (checked 2026-10-19).

**Effect on draft time** (Jetson, per round, English, `n_max` 6):

| Drafter | ms per round |
|---|---:|
| full-vocabulary Q8_0 | 13.2 |
| P3 | 11.5 |
| P3-V32k | 7.4 |
| P3-V32k + C4 | **6.3** (−54 %) |

Accepted length drops 2 %. Files: `e9c4_jetson_maxn_breakdown_n6.json`, `e9c4_jetson_maxn_e2e_n6.csv`.

## 4. Contribution: where the time goes (per-round breakdowns)

These come from timing builds (`runtime/llama_cpp_patches/e8-timing.patch`, `analysis/e8_breakdown.py`). The
setup is target Q4_K_M, text prompts, ms per speculation round.

| Stage | RTX 5070 (Q8_0, `n_max` 8) | Jetson MAXN, unpatched (Q8_0, 8) | Phone NPU (P3-V32k + C4, 8) |
|---|---:|---:|---:|
| draft | 3.5 | **25.3** | 31.6 |
| verify (`n_max` + 1 tokens) | 7.6 | 39.6 | **71.1** |
| inject + sampling | 1.3 | 2.2 | 3.4 + 2.9 |
| total / tokens advanced | 12.5 / 3.6 | 67.2 / 3.5 | 109.0 / 4.06 |
| verify ÷ single-token decode | 1.6× | **1.07×** | **1.8×** |

The tokens-advanced figures are from the timing builds' own prompts. The phone's "3.4 + 2.9" is inject plus the
unattributed remainder of its total.

- **Jetson: the drafter is the bottleneck, and inside it the Markov head.**
  - Nsight shows 8 serial single-column GEMVs over `markov_w2` (256 × 128k) per round, 1.67 ms each. That is
    about 13.5 ms, or 54 % of draft time.
  - The same GEMV takes 88 µs on the desktop, where the 33 MiB matrix sits in the 48 MB L2. The Jetson's 2 MB L2
    sends every read to DRAM.
  - This cost is invisible on the large-cache hardware of the official benchmarks.
  - Verification on Jetson is almost free (1.07× a single token), so speculation pays off most there.
- **Phone NPU: verification is the bottleneck.**
  - Single-token decode is about 40 ms. Verifying 3 tokens takes 52 ms, and 5–9 tokens 67–71 ms.
  - The cost is a fixed one for leaving the single-token path. The default HMX path is already the faster one;
    forcing HVX doubles verify time.
  - Hence `n_max` 8 is best: n_max 2/3/4/6/8 give 35.0/33.2/34.2/36.6/37.2 t/s.
  - Moving the drafter to the Adreno GPU saves 3.6 ms of drafting but adds about 10 ms of cross-device traffic.
    Moving it to the CPU is 6× slower.
  - The reduced vocabulary does not help on the NPU (31.6 vs 32.0 ms of drafting).
- **Phone CPU:** speculation is slower (0.67×). The CPU is compute-bound and capped by the OEM policy (§7), and
  drafter and target compete for the same cores.
- **`n_max`:** 6 on Jetson, 8 on the desktop (after the fixes) and on the phone NPU.

Files: `phone_breakdown/*.json`, `e9c4_jetson_maxn_breakdown_n6.json`. The Nsight and micro-benchmark tools are
`analysis/e8_nsys.sh` and `analysis/e8_gemv_bench.sh`.

## 5. Contribution: llama.cpp runtime fixes

| # | Problem | Fix | Measured effect | Correctness check |
|---|---|---|---|---|
| C1 | CUDA uses MMVQ for single-column quantized GEMV with K ≤ 256, such as the Markov head. At K = 256, MMQ is 1.8–3.0× faster on sm_87. | Route that case to MMQ on Turing+ (`e8-shortk-igpu.patch`) | Jetson DSpark +16.5 / +17.0 / +29.9 / +17.8 % (Q8_0 / P3 / Q4_K_M / Q4_0 drafter); desktop +3.7–5.2 % | `test-backend-ops -o MUL_MAT` same as master; acceptance unchanged |
| C2 | Integrated GPUs do not use the small-K MMVQ layout for N = 1 | Detect integrated GPUs and use small-K (same patch) | **Jetson target-only decode 27.27 → 31.43 t/s (+15.3 %)** with Q4_K_M, also +15.3 % on the 2.6B text model; no change on desktop. Rebased on master (Orin-only, `results/summary/upstream_c2_jetson/`): by quantization type −1 % (Q3_K_S) to +38 % (Q4_0); Q2_K, Q6_K, IQ4_XS only +2 % because the fused gate/up mat-vec path absorbs the gain (with CUDA fusion disabled, Q6_K gains +25 %) | as C1; perplexity identical |
| C3 | After upstream #29622, LFM2 text decode copies the inputs of unselected graph branches every token (CUDA syncs per 64 tokens 1312 → 1767) | Skip copying split inputs that only unselected branches read (`0001-…patch`) | desktop tg256 +3.5 % (2.6B), +4.2 % (VL-3B); sync count back to 1312; Jetson +0.2 % | 80 outputs byte-identical to master; `test-llama-archs` 402 OK |
| C4 | With a reduced draft vocabulary, the DSpark Markov chain still scatters to 128k at every draft position | With I32 `d2t`, add the bias, take argmax and map ids in draft space, and scatter once (`e9-c4-draft-space.patch`, `src/models/dflash.cpp`) | Jetson draft 7.4 → 6.3 ms per round, end-to-end +2.9 %; desktop A6 → A7 +8.5 % | I64 and I32 drafters give the same draft tokens and accepted counts over 3140 rounds |

How the C3 regression was found: `scripts/bisect_perf.sh` bisected the slowdown, judging each build by its ratio
to an interleaved reference build, and `analysis/nsys_decode_overhead.sh` compared the builds' CUDA API calls and
copies.

Two further issues are documented but not fixed:
- **C5:** the Hexagon backend's fixed cost for 2–9-token matrix multiplies (§4).
- **C6:** the LFM2-VL vision encoder's `UPSCALE` (position-embedding interpolation) is the only vision op without a
  Hexagon implementation, so it falls back to the CPU.

## 6. Contribution: Japanese characterization and root cause

### 6.1 Speedup on ja75 (vs B0 on the same platform)

| Config | RTX 5070 ja75 | RTX 5070 dev120 | Jetson ja75 | Jetson dev120 |
|---|---:|---:|---:|---:|
| B1 official F16 | **1.00×** [0.90, 1.11] | 1.52× | 1.21× | 1.80× |
| B2 Q8_0 | 1.19× | 1.78× | 1.70× | 2.52× |
| B11 P3-V32k + C4 | 1.30× [1.18, 1.45] | 1.80× | 1.98× [1.81, 2.21] | 2.77× |
| B12 Q4_K_M-V16k + C4 | 1.30× | 1.63× | **2.02×** [1.86, 2.25] | 2.59× |

Files: `ja75_rtx5070.csv`, `ja75_jetson_maxn.csv`.

On the desktop the official drafter gives no gain on Japanese. On the same 15 COCO images, Japanese captioning
accepts 15–18 % fewer tokens than English (desktop B11: 3.36 vs 3.93). Unless asked for detail, LFM2.5-VL answers
Japanese prompts in 4–14 tokens, so ja75 prompts ask for full answers.

### 6.2 Why: round-by-round analysis

`e9-server-round-log.patch` logs each draft, accepted count and target token. The setup is a Q8_0 drafter with the
full vocabulary on the desktop, and a target-only run for the target's real probabilities.

| Group (15 each) | Accepted draft tokens per round | Characters per token | Target top-1 probability |
|---|---:|---:|---:|
| English captioning (COCO) | 3.11 | 4.71 | 0.689 |
| Japanese captioning (same images) | **2.34 (−25 %)** | 1.52 | 0.677 |
| English open text (UltraChat) | 1.40 | 4.55 | 0.666 |
| Japanese open text (ELYZA) | 1.46 | 1.61 | 0.713 |

| Character class (captioning) | Share of drafted positions | Acceptance | Target top-1 probability |
|---|---:|---:|---:|
| Kanji | 37.5 % | **0.60** | **0.57** |
| Hiragana | 36.4 % | 0.76 | 0.73 |
| Katakana | 10.7 % | 0.74 | 0.76 |
| Japanese punctuation | 10.8 % | 0.92 | 0.79 |
| English words (English captions) | — | 0.77 | 0.69 |

1. **The gap depends on the task.** On open-ended text both languages are equally low. Part of the earlier
   "Japanese text" gap was the task type.
2. **In captions, kanji is the weak class**, and the cause is the target, not the drafter. Acceptance tracks the
   target's confidence the same way in both languages, about 0.03–0.13 above it in every class.
3. **The uncertainty is spread out.** On kanji the target's top-2 mass is only 0.71 (English words 0.83), so
   two-candidate tree drafting would help kanji no more than English.
4. **Tokenization multiplies the effect.** Japanese needs about 3× more tokens per character, so the same caption
   has about 3× more uncertain decisions.

Implication for Liquid: more Japanese drafter training probably helps less than expected, because the ceiling is
the target's uncertainty over kanji word choice. Tokenizer changes (more whole-word kanji tokens) or
retrieval-style drafting for repeated named entities are more promising. Both are hypotheses we have not tested.

Method notes:
- In speculative mode the server reports p ≈ 1 for accepted draft tokens, so probabilities come from target-only
  runs.
- The server merges partial-UTF-8 tokens into the next logprobs entry, so the comparison is per character class,
  not per token.

File: `e9_lang_analysis_q8.json`.

## 7. Contribution: evaluation-design findings (measurement pitfalls)

Each pitfall below produced plausible but wrong numbers before it was caught. The affected data was archived and
rerun, and the harness now guards against it.

| Pitfall | Symptom | Guard in the harness |
|---|---|---|
| llama-server's default `--cache-ram` is 8 GiB | On an 8 GB Jetson: swap, CUDA graph failures and OOM kills within one config | `server_args: --cache-ram 0` for Jetson; memory and swap counters logged per sample |
| A server that missed the readiness timeout kept running | All later configs were silently served by the wrong server | `run_eval.py` refuses a busy port, checks that its own process answered, and always cleans up |
| OEM power policy on Android caps adb-launched processes at 1.79 / 1.69 GHz (hardware 3.53 / 4.32 GHz) | CPU numbers that look like a slow phone | Documented; the NPU is unaffected; CPU numbers are labelled "OEM default policy" |
| Sustained NPU load heats the phone from 32 to 51 °C | Decode drifts downward across a config (B0: 28.5 → 23.1 t/s per 30-sample block) | Cooldown before each config, pause above 45 °C (`phone_htp_dev120_thermal.json`) |
| Silent CPU fallback on the phone | A model "on the NPU" running on the CPU | Verbose placement probe; `require_device: HTP0` refuses to measure otherwise |
| Speculative logprobs and partial UTF-8 | Accepted tokens show p ≈ 1; Japanese entries misaligned | Target-only probabilities; class-level comparison (§6.2) |
| Short answers | Japanese and MMMU-Pro prompts produced 4-token answers, so decode speed was unmeasurable | Instruction placement fixed: MMMU-Pro step-by-step suffix, ja75 "answer in detail" |

Also measured:
- **Greedy equivalence (S3, `docs/limitations.md` §1).** 30 speculative runs on the RTX 5070 (4 target precisions
  × 3 drafters × `n_max` 1/8, plus image prompts and a CPU control), judged by `analysis/judge_greedy.py`:
  - F16 targets match target-only greedy exactly on all 120 text outputs. On the image probe 19/20 match; the one
    divergence is a tie at margin 0.0036, recorded as a known exception.
  - Quantized targets diverge only at near-ties (median top-1/top-2 margin 0.02–0.06, max 0.20).
  - The CPU control with `n_max` 1 is identical.
  - For a given target and `n_max`, all drafters (including the reduced-vocabulary one) give the same outputs.
  - Changing only the physical batch size, with no speculation, moves next-token logprobs by 0.04–0.06 on CUDA
    with F16 and by up to 2 with quantized weights. This is kernel batch-shape numerics, not a DSpark bug.
  - The 24-sample multimodal set on all three platforms: RTX 5070 and Jetson pass (one more F16 tie, margin
    0.0011). On the phone the criteria are not met: the Hexagon NPU diverges at margins up to 0.65 (its logits move
    by up to 4 with batch size even without speculation), and the CPU at `n_max` 1 diverges on long-context image
    prompts (its batch-2 path is bit-exact only for short contexts, on ARM and x86 alike).
- **Sampling** (dev120, speedup vs B0 under the same sampling):

  | Platform | greedy | model card (T 0.2, top-k 50; 3 seeds) | T 0.7 |
  |---|---:|---:|---:|
  | RTX 5070 B11 | 1.80× | 1.77× (1.71–1.85) | 1.50× (3 seeds) |
  | Jetson B11 | 2.77× | 2.69× (2.63–2.72) | 2.35× (1 seed) |

  Files: `sampling_{rtx5070,jetson_maxn}.csv`.

## 8. Limitations

- Sample sizes:
  - 120 samples per platform for English (6 × 20), 2–3 repetitions; the phone NPU had 1.
  - 75 for Japanese (15 per category).
  - 15 + 15 per group in the root-cause analysis.
  - Confidence intervals cover sample variation, not run-to-run drift (about 3 % on the desktop).
- Platform coverage:
  - Q8_0 targets (B4–B7) do not fit reliably in 8 GB with mmproj and a drafter, so they are desktop-only.
  - The phone was not measured on Japanese.
  - The phone NPU dev120 run heated up during measurement (§7), so its second half is indicative.
- Task quality (accuracy) was not scored. Greedy outputs are identical across drafter variants on the same target,
  so drafter choice cannot change quality under greedy decoding. Sampling quality was not evaluated.
- Upstream status: the fixes are validated patches on `620d2225c` and not yet submitted upstream. C1–C4 have not
  been re-validated on newer llama.cpp commits.

## Reproduction

Paths default to `EDGESPECQ_HOME=~/edgespecq` (desktop) and, for the Jetson, `~/mt_test/liquid_ai`, as set in
`benchmark/configs.json` → `platforms.*.home`. Edit `platforms.*` for your own device paths, adb serial and ports.
Shell steps are bash.

### R1. Environment and models (host)

```bash
export EDGESPECQ_HOME=~/edgespecq; H=$EDGESPECQ_HOME
git clone https://github.com/ggml-org/llama.cpp $H/llama.cpp && git -C $H/llama.cpp checkout 620d2225c
python3 -m venv $H/.venv && $H/.venv/bin/pip install numpy pandas pillow datasets huggingface_hub
$H/.venv/bin/pip install -e $H/llama.cpp/gguf-py
TARGETS="llama-server llama-bench llama-quantize" bash scripts/build_llama.sh master 120   # builds the checkout
hf download LiquidAI/LFM2.5-VL-3B-GGUF LFM2.5-VL-3B-Q4_K_M.gguf LFM2.5-VL-3B-Q4_0.gguf LFM2.5-VL-3B-Q8_0.gguf \
  LFM2.5-VL-3B-F16.gguf mmproj-LFM2.5-VL-3B-Q8_0.gguf --local-dir $H/models/target
hf download LiquidAI/LFM2.5-VL-3B-DSpark-GGUF LFM2.5-VL-3B-DSpark-F16.gguf --local-dir $H/models/draft
```

CUDA 13.0 headers conflict with glibc 2.43, so use CUDA ≥ 13.2 on recent distributions.

### R2. Drafters (bit-identical to the measured ones)

```bash
bash scripts/quantize_draft.sh                              # models/quant/*-{Q8_0,Q4_K_M,Q4_0,P3}.gguf (+ .sha256)
D2T=i32 bash scripts/make_e9_drafters.sh 32 P3              # B11: models/e9/*-P3-V32k-i32.gguf
D2T=i32 bash scripts/make_e9_drafters.sh 16 Q4_K_M          # B12: models/e9/*-Q4_K_M-V16k-i32.gguf
bash scripts/make_e9_drafters.sh 32 P3                      # A6:  models/e9/*-P3-V32k.gguf (I64 d2t)
(cd $H/models && sha256sum -c "$OLDPWD/benchmark/manifests/models.sha256")
```

`make_reduced_vocab_drafter.py` reads `$H/results/e9/freq_*.npy` if present and otherwise the committed
`quantization/vocab_freq.npz`. To recompute the counts from scratch, run `analysis/e9_selfgen.py` (the target's
own answers) and then `analysis/e9_corpus_freq.py`.

### R3. Evaluation sets

```bash
$H/.venv/bin/python benchmark/build_eval_set.py      # smoke24, dev120, final600 -> $H/eval (images in $H/eval/img)
$H/.venv/bin/python benchmark/build_ja_set.py        # ja75 (run after dev120: reuses its COCO images)
for s in smoke24 dev120 ja75; do $H/.venv/bin/python -c "import json,sys
for r in map(json.loads, open('$H/eval/$s.jsonl')): print(r['src'], r['id'], len(r['turns']), len(r['images']), sep='\t')" \
  | diff -q - benchmark/manifests/$s.tsv && echo "$s matches"; done
```

Some source datasets require accepting their terms on Hugging Face. ELYZA-tasks-100 (CC-BY-SA-4.0), JA-VG-VQA-500
(CC-BY-4.0) and JMMMU (MIT) are used for local evaluation only, and nothing from them is redistributed here.

### R4. Runtimes

| Platform | Command | Result |
|---|---|---|
| Desktop | `bash scripts/build_runtimes.sh 620d2225c 120` | worktree `$H/llama-edge`; libraries `$H/libs/5070/{master,fixed,edge}` |
| Jetson | `PLATFORM=jetson bash scripts/build_runtimes.sh 620d2225c 87`, then set `platforms.jetson.runtimes` to `libs/jetson/{master,fixed,edge}` ² | same three patch sets for sm_87 |
| Phone NPU | `bash scripts/build_android_htp.sh edge` (and `master`, `timing`); needs Docker | `$H/android/htp-<variant>/{bin,lib}` |
| Phone CPU | `bash scripts/build_runtimes.sh` first (creates `llama-edge`), then `bash scripts/build_android.sh` | `$H/android/{master,edge}/{bin,lib}` |

² The measured Jetson runtimes were built incrementally in one tree as per-library directories:
`libggml-base` (master / C3), `libggml-cuda` (master / C1+C2) and `libllama` (base / C4). The current
`configs.json` lists those directories. `build_runtimes.sh` applies the same patches into combined directories; it
has not been re-run on the Jetson in this form.

Phone layout under `/data/local/tmp/edgespecq` (`adb push`):
- binaries: `htp-edge/{bin,lib}`, plus `master/`, `edge/` and `fixed/lib` for the CPU runtimes (`fixed/lib` holds
  only edge's `libggml-base.so`);
- models: `models/{target,draft,quant,e9}/`, the same file names as on the host.

When adb is the Windows binary used from WSL2, enable mirrored networking so that `adb forward` ports are reachable
from WSL.

### R5. Measurements

The main experiment (A0, A1, B0, B11) runs with one command per device, after checking the llama.cpp commit, the
runtime libraries, the model SHA256 and the evaluation-set manifest, and ends with a comparison against the
published numbers (`analysis/compare_repro.py`):

```bash
bash scripts/run_benchmark.sh --platform rtx5070 [--set smoke24]     # or --platform jetson / jetson_25w on the Jetson
```

Checked on 2026-10-21 with smoke24 on both devices (against the published dev120 runs restricted to the smoke24
samples, `results/summary/repro_ref/`): on the Jetson every decode rate and speedup within 0.1 %, on the RTX 5070
within 4 % (B11 3.9 % slower; the desktop's clocks are not locked).
The full matrix:

```bash
E=$H/eval; RUN="$H/.venv/bin/python regression/run_eval.py"     # resumable: rerun the same command to continue
# Desktop
$RUN rtx5070 $E/dev120.jsonl B0,B1,B1m,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,R0,R1,A0,A1,A2,A3,A4,A5,A6,A7 --reps 3
$RUN rtx5070 $E/ja75.jsonl B0,B1,B2,B11,B12 --reps 3
for p in liquid t07; do for s in 0 1 2; do $RUN rtx5070 $E/dev120.jsonl B0,B1,B2,B11 --sampling $p --seed $s; done; done
# Jetson (B1m = A1 and A7 = B11 there; B4-B7 do not fit in 8 GB)
bash scripts/jetson_measure.sh python3 regression/run_eval.py jetson $E/dev120.jsonl \
  B0,B1,B2,B3,B8,B9,B10,B11,B12,A0,A1,A2,A3,A4,A5,A6 --reps 2
bash scripts/jetson_measure.sh --25w python3 regression/run_eval.py jetson_25w $E/dev120.jsonl A0,A1,B11,B12 --reps 2
bash scripts/jetson_measure.sh python3 regression/run_eval.py jetson $E/ja75.jsonl B0,B1,B2,B11,B12 --reps 2
# Phone (llama-server on the device, harness on the host)
$RUN phone_htp $E/dev120.jsonl B0,B1,B11
$RUN phone_htp $E/smoke24.jsonl B0,B1,B11,B11g,B11c
$RUN phone $E/smoke24.jsonl B0,B1,B11
```

Each sample is written to `logs/eval/<platform>/<set>/<config>[_<sampling>_s<seed>]/r<rep>.jsonl`, together with
the server log, the exact command line and the memory, swap and battery-temperature counters.

`scripts/jetson_measure.sh` stops the desktop, sets MAXN_SUPER and `jetson_clocks`, runs the command, and always
restores the previous mode, clocks and systemd target. It was adapted from the run scripts used for the measurements;
in this form it ran the 2026-10-21 smoke24 reproduction above.

### R6. Summaries

```bash
S="$H/.venv/bin/python analysis/summarize_eval.py"; O=results/summary
$S rtx5070 dev120 --csv $O/dev120_rtx5070.csv;   $S jetson dev120 --csv $O/dev120_jetson_maxn.csv
$S jetson_25w dev120 --ref A0 --csv $O/dev120_jetson_25w.csv
$S rtx5070 ja75 --csv $O/ja75_rtx5070.csv;       $S jetson ja75 --csv $O/ja75_jetson_maxn.csv
$S phone_htp dev120 --csv $O/phone_htp_dev120.csv; $S phone smoke24 --csv $O/phone_cpu_smoke24.csv
$S rtx5070 dev120 --ref B0_liquid_s0 --configs B0_liquid_s0,B1_liquid_s0,B2_liquid_s0,B11_liquid_s0  # one per preset/seed
```

The sampling CSVs concatenate one such summary per preset and seed. The Jetson and phone summaries run on the
machine that holds the logs (`HOME` map in `summarize_eval.py`).

### R7. Analyses

| Result | Build | Command |
|---|---|---|
| Per-round breakdown (§4), desktop or Jetson | timing tree: `620d2225c` + `e8-timing.patch` (+ the runtime patches being studied) | `N_MAX=8 python analysis/e8_breakdown.py <llama_dir> $H/models/target/LFM2.5-VL-3B-Q4_K_M.gguf --drafters none,Q8_0,P3-V32k-i32 --prompts 5 --n 256 --json out.json` |
| Per-round breakdown, phone NPU | `bash scripts/build_android_htp.sh timing` | same, with `ADB`, `SERIAL`, `BIN=<dev>/htp-timing/bin/llama-speculative-simple`, `REMOTE_ENV="LD_LIBRARY_PATH=<dev>/htp-timing/lib ADSP_LIBRARY_PATH=<dev>/htp-timing/lib"`, `EDGESPECQ_HOME=<dev>` and `--extra="--device HTP0 --device-draft HTP0"`; HVX vs HMX via `GGML_HEXAGON_MM_SELECT` in `REMOTE_ENV` |
| Markov-head GEMV and kernel choice | patched tree with `e8-kernel-experiments.patch` | `analysis/e8_gemv_bench.sh`, `analysis/e8_nsys.sh` |
| Reduced-vocabulary acceptance simulation | timing tree | `analysis/e9_collect.py`, then `analysis/e9_simulate.py` |
| Japanese root cause (§6.2) | timing tree + `e9-server-round-log.patch` (llama-server) | `python analysis/e9_lang_rounds.py <timing_dir> <target> <Q8_0 drafter> rounds_q8.jsonl` and the same with drafter `none` → `target_only.jsonl`; then `python analysis/e9_lang_analyze.py rounds_q8.jsonl <target> --target-only target_only.jsonl --json e9_lang_analysis_q8.json` |
| Greedy equivalence and batch invariance (§7) | any | `scripts/run_greedy_matrix.sh`, `analysis/divergence.py`, `analysis/batch_invariance.py` |
| C3 regression bisect | full clone | `scripts/bisect_perf.sh`, `analysis/nsys_decode_overhead.sh` |
| Figures (`results/figures/`, light and dark, plus the plotted numbers in `tables/`) | none | `python analysis/make_figures.py` (reads only `results/summary/`) |

The text probe used by the breakdowns is `$H/probe/track_t_probe20.jsonl` (`benchmark/build_probe_text.py`). The
Japanese analysis used a timing tree with `e8-kernel-experiments.patch` in place of C1–C3. Acceptance and
probabilities do not depend on kernel selection, apart from near-tie numerics.

## Repository map

| Path | Contents |
|---|---|
| `benchmark/` | evaluation-set builders, `configs.json` (platforms, sampling presets, configs), `manifests/` (sample IDs, model SHA256) |
| `scripts/run_benchmark.sh`, `analysis/compare_repro.py` | one-command main experiment with pre-run checks, and comparison with the published numbers |
| `regression/run_eval.py` | resumable multi-turn harness (local and adb), placement probe, thermal and port guards |
| `quantization/` | GGUF inspector, reduced-vocabulary exporter, committed vocabulary counts |
| `runtime/llama_cpp_patches/` | C1/C2 (`e8-shortk-igpu`), C3 (`0001-…`), C4 (`e9-c4-draft-space`), timing and round-log patches |
| `scripts/` | builds (CUDA, Android CPU, Hexagon), drafter export, Jetson power-state wrapper, probes, bisect |
| `analysis/` | summaries, breakdowns, Nsight, vocabulary and Japanese analyses |
| `results/summary/` | every number in this document |
| `docs/recommendations.md` | deployment and model-side recommendations derived from these results |
