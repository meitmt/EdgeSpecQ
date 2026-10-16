# EdgeSpecQ

Quantized DSpark speculative decoding for LFM2.5-VL on real edge devices (phone, Jetson, desktop GPU).

Work in progress.

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
