# EdgeSpecQ

Quantized DSpark speculative decoding for Liquid AI's LFM2.5-VL-3B on edge devices: an 8 GB Jetson Orin Nano
Super, a Snapdragon 8 Elite phone (Hexagon NPU and CPU) and an RTX 5070, all through llama.cpp. Where the time goes
on each device, what fixes it without retraining, and how to deploy it.

An independent project; not affiliated with Liquid AI.

## Results

dev120 (120 multimodal samples in six task categories), greedy decoding, decode tokens per second:

| Platform | Target only, unpatched | Official F16 drafter as shipped | This work | Speedup vs target only | vs official drafter |
|---|---:|---:|---:|---:|---:|
| Jetson Orin Nano Super, MAXN_SUPER | 27.1 | 50.4 | **86.6** | **3.20×** | **1.72×** |
| Jetson Orin Nano Super, 25 W default | 24.7 | 45.4 | 75.9 | 3.07× | 1.67× |
| RTX 5070 | 220.5 | 329.4 | 408.3 (427.7 with a Q4_0 target) | 1.85× (1.94×) | 1.24× (1.30×) |
| OnePlus 13, Hexagon NPU | 24.4 | 29.4 | 31.4 | 1.29× | 1.07× |

Target LFM2.5-VL-3B Q4_K_M. "This work" is the edge drafter (P3 quantization, 32k-token draft vocabulary) on
llama.cpp `620d2225c` with four runtime fixes. On the phone all three columns use the patched runtime. Greedy
outputs do not depend on the drafter variant.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/fig3_platforms_dark.png">
  <img src="results/figures/fig3_platforms.png" alt="Speedup of the official drafter and of the edge configuration on each platform" width="720">
</picture>

- **On an 8 GB Jetson the drafter is the bottleneck, not verification.** Inside the drafter it is the Markov head:
  eight serial GEMVs per round over a 33 MiB matrix that fits in a desktop GPU's 48 MB L2 but not in the Jetson's
  2 MB.
- **Fixed without retraining:** a quantized, reduced-vocabulary drafter exported losslessly from the official F16
  GGUF (regenerated bit for bit from this repository), four llama.cpp fixes (one submitted upstream as
  [ggml-org/llama.cpp#30858](https://github.com/ggml-org/llama.cpp/pull/30858)) and a per-device draft length.
- **The gain follows the device's verification cost:** verifying 7–9 tokens costs 1.07× a single-token decode on
  the Jetson, 1.6× on the RTX 5070 and 1.8× on the phone NPU. On the phone CPU speculation slows decoding down.
- **Japanese gets about 70 % of the English gain**, because the target itself is uncertain over kanji and Japanese
  text needs about three times as many tokens for the same content, not because the drafter is weaker.

| Document | Contents |
|---|---|
| [`docs/tech_note_midterm.md`](docs/tech_note_midterm.md) | Three-page technical note: bottlenecks and deployment recommendations |
| [`docs/report.md`](docs/report.md) | All results, contributions, and step-by-step reproduction |
| [`docs/recommendations.md`](docs/recommendations.md) | Recommendations for the drafter, the runtime and deployment |
| [`docs/limitations.md`](docs/limitations.md) | Greedy-equivalence results, scope and conditions of the measurements |
| [`results/summary/`](results/summary/) | Every number in these documents (CSV / JSON); figures in [`results/figures/`](results/figures/) |

## Reproduce the main experiment

One command per device runs the four configurations behind the table above (A0 target only, unpatched; A1 official
drafter as shipped; B0 target only, patched; B11 edge drafter) and compares the result with the published numbers:

```bash
bash scripts/run_benchmark.sh --platform rtx5070              # on the desktop
bash scripts/run_benchmark.sh --platform jetson               # on the Jetson (jetson_25w: factory-default mode)
bash scripts/run_benchmark.sh --platform jetson --set smoke24 # 24-sample subset of dev120
```

Before measuring, the script stops if the llama.cpp checkout is not at the pinned commit, a runtime library is
missing, a model file does not match [`benchmark/manifests/models.sha256`](benchmark/manifests/models.sha256), or the
evaluation set does not match its manifest (`--check-only` runs just these checks). On the Jetson it measures with
the desktop stopped and MAXN_SUPER + `jetson_clocks`, and restores the board afterwards
([`scripts/jetson_measure.sh`](scripts/jetson_measure.sh); sudo asks once). Runs are resumable. The output ends
with a comparison like this one (Jetson, smoke24, 2026-10-21; on the RTX 5070 the same run was within 4 %):

```
== against the published numbers (results/summary/repro_ref/smoke24_jetson.csv)
              published reproduced    diff
A0 t/s             27.1       27.1   -0.1%
A1 t/s             49.5       49.4   -0.1%
B0 t/s             31.3       31.2   -0.1%
B11 t/s            86.8       86.7   -0.1%
B11/A0            3.20x      3.20x   +0.0%
B11/A1            1.75x      1.75x   +0.1%
B11/B0            2.77x      2.78x   +0.1%
A1/A0             1.82x      1.82x   -0.0%
B0/A0             1.15x      1.15x   -0.0%
all within tolerance
```

smoke24 takes about 25 minutes on the Jetson and 5 on the RTX 5070; dev120, the set behind the table above, about
five times as long.

Setup, once per device (details and the phone runs: [`docs/report.md` § Reproduction](docs/report.md#reproduction)):

```bash
export EDGESPECQ_HOME=~/edgespecq; H=$EDGESPECQ_HOME   # must match platforms.<platform>.home in configs.json
git clone https://github.com/ggml-org/llama.cpp $H/llama.cpp && git -C $H/llama.cpp checkout 620d2225c
python3 -m venv $H/.venv && $H/.venv/bin/pip install numpy pandas pillow datasets huggingface_hub -e $H/llama.cpp/gguf-py
TARGETS=llama-quantize bash scripts/build_llama.sh master 120   # for the drafter export (Jetson: 87)
bash scripts/build_runtimes.sh 620d2225c 120                    # runtimes master / fixed / edge (Jetson: see below)
hf download LiquidAI/LFM2.5-VL-3B-GGUF LFM2.5-VL-3B-Q4_K_M.gguf mmproj-LFM2.5-VL-3B-Q8_0.gguf --local-dir $H/models/target
hf download LiquidAI/LFM2.5-VL-3B-DSpark-GGUF LFM2.5-VL-3B-DSpark-F16.gguf --local-dir $H/models/draft
bash scripts/quantize_draft.sh && D2T=i32 bash scripts/make_e9_drafters.sh 32 P3   # the edge drafter (B11)
$H/.venv/bin/python benchmark/build_eval_set.py                 # smoke24 / dev120 / final600 from public datasets
```

On the Jetson, build with `PLATFORM=jetson bash scripts/build_runtimes.sh 620d2225c 87` and point
`platforms.jetson` in `benchmark/configs.json` at `llama-edge` and `libs/jetson/{master,fixed,edge}`; the published
Jetson runs used the same patches built as separate per-library directories, which is what `configs.json` lists now.
Requirements: CUDA ≥ 12.6 (Jetson, JetPack 6.2) or ≥ 13.2 (desktop), CMake, Ninja, Python 3.10+.

## The edge drafters

Exported from the official `LFM2.5-VL-3B-DSpark-F16.gguf`, no retraining. SHA256 of each file in
[`benchmark/manifests/models.sha256`](benchmark/manifests/models.sha256).

| File | Recipe | Use |
|---|---|---|
| `LFM2.5-VL-3B-DSpark-Q8_0.gguf` | all weights Q8_0 | simplest; acceptance within 1 % of F16 |
| `LFM2.5-VL-3B-DSpark-P3.gguf` | decoder Q4_K; projection and Markov head Q8_0 | |
| `LFM2.5-VL-3B-DSpark-P3-V32k-i32.gguf` | P3 + 32k-token draft vocabulary ranked by the target's own answers, I32 `d2t` | **recommended**, Jetson and phone NPU |
| `LFM2.5-VL-3B-DSpark-Q4_K_M-V16k-i32.gguf` | Q4_K_M + 16k vocabulary with Japanese weight | Japanese |

Recommended draft length (`--spec-draft-n-max`): 6 on the Jetson, 8 on the RTX 5070 and the phone NPU. On 8 GB
boards also pass `--cache-ram 0`.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/run_benchmark.sh`, `analysis/compare_repro.py` | One-command reproduction and comparison with the published numbers |
| `benchmark/` | Evaluation-set builders (smoke24 / dev120 / final600, Japanese ja75), probes, `configs.json` (platforms, configurations B0–B12 and A0–A7, sampling presets), `manifests/` (sample IDs, model SHA256) |
| `regression/run_eval.py`, `analysis/summarize_eval.py` | Resumable evaluation harness (local and over adb) and paired-speedup summaries with bootstrap intervals |
| `quantization/`, `scripts/quantize_draft.sh`, `scripts/make_e9_drafters.sh` | Drafter quantization, lossless reduced-vocabulary export, committed vocabulary counts |
| `runtime/llama_cpp_patches/` | Runtime fixes C1–C4, the upstream C2 patch, timing and round-log patches |
| `scripts/build_*.sh` | Runtimes for CUDA, Android CPU and Hexagon NPU |
| `analysis/e8_*`, `analysis/e9_*` | Per-round time breakdowns, Markov-head kernel study, vocabulary and Japanese analyses |
| `scripts/run_s3.sh`, `analysis/judge_greedy.py`, `tests/` | Greedy-equivalence matrix, its judgment and tests (`python -m unittest discover -s tests`) |
| `analysis/make_figures.py` | All figures, from `results/summary/` only |
