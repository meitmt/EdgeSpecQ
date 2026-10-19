# EdgeSpecQ

Quantized DSpark speculative decoding for LFM2.5-VL on real edge devices (phone, Jetson, desktop GPU).

Work in progress. Results, contributions and step-by-step reproduction: [`docs/report.md`](docs/report.md);
deployment recommendations: [`docs/recommendations.md`](docs/recommendations.md).

Headline (dev120, six multimodal task categories): on an 8 GB Jetson Orin Nano Super, LFM2.5-VL-3B decodes at
27.1 t/s without speculation, 50.4 t/s with the official F16 DSpark drafter, and 86.6 t/s (3.20×) with a quantized,
reduced-vocabulary drafter (no retraining, identical outputs) plus four llama.cpp runtime fixes.

## Layout

| Path | What |
|---|---|
| `scripts/build_llama.sh` | Build `llama-server` for any llama.cpp commit in its own worktree (CUDA) |
| `benchmark/build_probe_text.py` | 20-prompt text probe (GSM8K, MATH-500, HumanEval, MBPP, MT-Bench; 4 each) |
| `benchmark/build_probe_vl.py` | 20-sample VL probe (TextVQA, ChartQA, COCO captions) |
| `regression/probe_accept.py` | Run a probe against one build/config: acceptance, mean accepted length, tok/s, outputs, logprobs |
| `regression/compare_outputs.py` | Pairwise exact-match comparison of probe outputs |
| `analysis/divergence.py` | First divergence vs target-only greedy, with the top-1/top-2 logprob margin there |
| `analysis/batch_invariance.py` | Next-token logprobs vs physical batch size (`-ub`), no speculation |
| `scripts/run_r1.sh` | Old commit vs master acceptance comparison |
| `scripts/run_greedy_matrix.sh` | Greedy-equivalence matrix: precision x `n_max`, CPU control, batch invariance |
| `scripts/bench_tg.sh` | Median token-generation t/s of one build (llama-bench, target only) |
| `scripts/bisect_perf.sh` | `git bisect run` for tg regressions, judged by ratio to an interleaved reference build |
| `analysis/nsys_decode_overhead.sh` | Nsight Systems comparison of builds: kernels, CUDA API calls, H2D copy sizes |
| `quantization/inspect_gguf.py` | Tensor manifest of a GGUF (name, shape, dtype, bytes) grouped by DSpark component |
| `scripts/quantize_draft.sh` | Quantize the F16 DSpark drafter to Q8_0 / Q4_K_M / Q4_0 / P3 mixed, with manifests and SHA256 |
| `scripts/run_draft_precision.sh` | Fixed target, drafter-precision sweep: acceptance, mean accepted length, decode t/s |
| `benchmark/build_probe_ja.py` | 10-prompt Japanese pilot probe (every 10th task of ELYZA-tasks-100) |
| `analysis/e9_collect.py` | Per-round drafts and accepted counts from the timing build (`E9R` lines) |
| `analysis/e9_corpus_freq.py`, `analysis/e9_selfgen.py` | Token frequencies of the LFM2.5 vocabulary: generic corpora, and the target's own answers to non-probe prompts |
| `analysis/e9_simulate.py` | Acceptance of a reduced draft vocabulary, simulated from full-vocabulary rounds |
| `quantization/make_reduced_vocab_drafter.py` | Lossless reduced-vocabulary DSpark drafter export (row slices + `d2t`), no retraining |
| `scripts/make_e9_drafters.sh`, `scripts/run_e9_vocab.sh` | Build the 16k–48k drafters; end-to-end acceptance and t/s against the full-vocabulary drafter |
| `quantization/vocab_freq.npz` | Token-frequency counts behind the released reduced vocabularies (regenerates B11/B12 drafters bit for bit) |
| `benchmark/build_eval_set.py`, `benchmark/build_ja_set.py` | Seeded six-category multimodal sets (smoke24 / dev120 / final600) and the Japanese ja75 set |
| `benchmark/configs.json`, `benchmark/manifests/` | Platforms, sampling presets and configurations; sample IDs and model SHA256 |
| `regression/run_eval.py`, `analysis/summarize_eval.py` | Resumable evaluation harness (local and adb) and paired-speedup summaries with bootstrap intervals |
| `scripts/build_runtimes.sh`, `scripts/build_android.sh`, `scripts/build_android_htp.sh` | master / fixed / edge runtimes for CUDA, Android CPU and Hexagon NPU |
| `scripts/jetson_measure.sh` | Run a command on the Jetson with the desktop stopped and MAXN_SUPER, restoring everything afterwards |
| `analysis/e8_breakdown.py`, `analysis/e9_lang_rounds.py`, `analysis/e9_lang_analyze.py` | Per-round time breakdown (local and adb); Japanese acceptance root-cause analysis |

## Setup (WSL2 / Linux, CUDA)

All paths default to `$EDGESPECQ_HOME` (`~/edgespecq`): `llama.cpp/` (full clone), `models/`, `probe/`,
`logs/`, and a Python venv at `.venv/` (`numpy pandas matplotlib pillow datasets huggingface_hub`).

```bash
bash scripts/build_llama.sh master            # sm_120 by default; pass 87 for Jetson Orin
~/edgespecq/.venv/bin/python benchmark/build_probe_text.py
~/edgespecq/.venv/bin/python benchmark/build_probe_vl.py
bash scripts/run_r1.sh 38a5b42d9
bash scripts/run_greedy_matrix.sh
```

Models are expected under `$EDGESPECQ_HOME/models/{target,draft,track_t}/` with the official
LiquidAI GGUF file names.
