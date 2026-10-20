# Limitations

What the results in [`report.md`](report.md) do and do not show. Numbers refer to files in `results/summary/`.

## 1. Speculative output vs greedy output (S3)

**Mechanism.** DSpark verifies `n_max + 1` tokens in one batch, while target-only decoding evaluates one token at a
time. llama.cpp chooses kernels by batch shape and does not guarantee batch invariance (upstream #25618, #23335,
#16016). So the target's logits during verification can differ slightly from its logits during single-token
decoding, and greedy decoding can pick a different token wherever the top two candidates are almost tied. This is
runtime numerics, not a DSpark logic error: the accepted tokens are still the target's own greedy choices for the
logits it computed.

**Batch invariance without speculation** (`analysis/batch_invariance.py`, `s3_batch_invariance.txt`; same prompts,
only the physical batch size `-ub` changes, reference `-ub 1`):

| Target, backend | max change of a top-5 next-token logprob (ub 2 / 9 / 512) | prompts with bit-identical top-5 |
|---|---|---|
| F16, CUDA | 0.037 / 0.046 / 0.063 | 0/10 |
| Q8_0, CUDA | 0.49 / 0.29 / 0.32 | 0/10 |
| Q4_K_M, CUDA | 0.22 / 0.29 / 0.58 | 0/10 |
| Q4_0, CUDA | 1.91 / 2.05 / 1.49 | 0/10 |
| Q4_K_M, CPU | 0 / 0.34 / 1.23 | 10/10 at ub 2, 0/10 above |

No top-1 token flipped in these tests, but no CUDA configuration is batch-invariant, F16 included.

**Criteria** (defined in proj v1.3, unchanged since): (1) with an F16 target every speculative run must reproduce
the target-only output exactly; (2) with a quantized target, outputs may differ only where the target-only top-1
and top-2 logprobs at the first differing token are within 0.3 (*margin* < 0.3); (3) on the CPU backend with
`n_max` 1 the outputs must be identical. A divergence that cannot be located (same tokens, different length)
always fails. Divergences that break an exact criterion are investigated and listed in
`s3_known_exceptions.json`; the judge accepts such an entry only while the divergence stays below the margin
recorded there, and never counts it as identical.

**Equivalence matrix** (`scripts/run_s3.sh`, judged by `analysis/judge_greedy.py`, recorded in
`s3_greedy_equivalence.json`, checked by `tests/test_greedy_correctness.py`; RTX 5070, edge runtime = llama.cpp
`620d2225c` + C1–C4; each speculative run compared with target-only greedy on the same target):

| Probe | Target | Identical outputs (per configuration) | Margin at divergence, median / max |
|---|---|---|---|
| Text, 20 prompts | F16 | 20/20 in all 6 configurations | — |
| Text | Q8_0 | 9–10/20 | 0.028–0.035 / 0.128 |
| Text | Q4_K_M | 6–10/20 | 0.022–0.028 / 0.165 |
| Text | Q4_0 | 5–7/20 | 0.057–0.062 / 0.204 |
| Image + text, 20 prompts | F16 | 19/20 in both configurations | 0.004 (one prompt) |
| Image + text | Q4_K_M | 11/20 | 0.053 / 0.103 |
| Text, CPU, 10 prompts × 128 tokens | Q4_K_M | 10/10 at `n_max` 1, 5/10 at `n_max` 8 | 0.025 / 0.062 |

Configurations: drafters F16 (official), Q8_0 and P3-V32k-i32 (reduced vocabulary with C4), each at `n_max` 1 and 8
(image probe: `n_max` 8 only).

**Verdict: criteria 2 and 3 hold everywhere; criterion 1 holds on the text probe (120/120) and fails on one image
prompt, recorded as a known exception** (below). All 30 speculative runs pass with that exception documented; the
strict result is "pass with 1 known exception", not "identical".

- Every divergence, with every target, happened where the top two candidates were within 0.21 logprob.
- The CPU control with `n_max` 1 is identical (10/10), as criterion 3 requires.
- For a given target and `n_max`, the three drafters diverge on the same prompts at the same tokens with the same
  margins: DSpark always drafts a full block, so the verification batch shape depends only on `n_max`, not on the
  drafter. The drafter choice (quantization, reduced vocabulary, C4) does not change the target's output.
- **Known exception — the one F16 divergence** (image prompt `coco_3`, token 88, both drafters at `n_max` 8): the
  target-only logprobs of " light" and " pattern" were −1.2433 and −1.2470 (margin 0.0036); the speculative runs
  chose " pattern". This is about a tenth of the F16 CUDA batch-shape variation measured above, so the v1.3 premise
  that F16 targets reproduce target-only output exactly holds only up to such ties; the criterion is kept as
  written and the case documented rather than relaxed after the fact.
- **Coverage gap.** The protocol (proj §4.5) asks for the 24-sample multimodal set (smoke24) on all three
  platforms. This matrix covers the RTX 5070 only, with the 20-prompt text probe and the 20-prompt image probe;
  Jetson and phone have not been run under the S3 judge.

## 2. Scope of the measurements

- **Samples.** English: 120 per platform (6 categories × 20), 2–3 repetitions (phone NPU: 1). Japanese: 75
  (5 × 15). Root-cause analysis: 15 + 15 per group. Confidence intervals cover sample variation, not run-to-run
  drift (about 3 % on the RTX 5070, whose clocks were not locked).
- **Models.** LFM2.5-VL-3B (and the LFM2.5-2.6B text model for some runtime checks) with the DSpark drafter. No MoE
  target was measured (the C2 patch also leaves `MUL_MAT_ID` unchanged).
- **Platforms.** One Jetson Orin Nano Super, one RTX 5070 (WSL2), one OnePlus 13. Q8_0 targets do not fit reliably
  in 8 GB with mmproj and a drafter, so B4–B7 are desktop-only. The phone was not measured on Japanese.
- **Decoding.** Greedy unless noted; the model card's sampling settings and T = 0.7 were checked on the RTX 5070
  and Jetson only.
- **Task quality** (accuracy, judge scores) was not evaluated. Under greedy decoding the drafter choice does not
  change the target's output (section 1), so speed comparisons between drafters are like-for-like; quality under
  sampling was not checked.

## 3. Measurement conditions

- **Jetson.** MAXN_SUPER + `jetson_clocks` with the desktop stopped unless noted (about 2–3 % faster than with the
  desktop running); `--cache-ram 0` for llama-server, whose 8 GiB default host prompt cache does not fit an 8 GB
  board.
- **Phone.** The OEM power policy caps adb-launched processes at 1.79 / 1.69 GHz (hardware 3.53 / 4.32 GHz), so
  CPU numbers describe a background native process, not a foreground app. The NPU dev120 run heated the battery
  from 32 to 51 °C; its second half is indicative (`phone_htp_dev120_thermal.json`).
- **Per-round breakdowns** come from timing builds (`e8-timing.patch`), whose instrumentation adds small overheads;
  throughput claims use the unmodified runtimes.
- **Logprobs in speculative runs.** llama-server reports p ≈ 1 for accepted draft tokens, so target probabilities
  always come from target-only runs; partial UTF-8 tokens are merged into the next logprobs entry, so Japanese
  comparisons are per character class, not per token.

## 4. Runtime versions

All measurements pin llama.cpp `620d2225c`, except the C2 upstream measurements (master `7951f601e`,
`results/summary/upstream_c2_jetson/`). The C1, C3 and C4 patches have not been re-validated on newer commits;
C2 is under review upstream.
