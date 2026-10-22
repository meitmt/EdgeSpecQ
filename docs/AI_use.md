# How AI tools were used

This is a one-person project. I used Claude Code (Anthropic) throughout as a coding and research assistant. This
page says which parts were mine and which were done by the assistant, so that readers can weigh the work accordingly.

## What I did

- Chose the research question, the devices and the evaluation protocol, including the baselines and the
  greedy-equivalence criteria.
- Directed the work: which leads to follow, which fixes to keep, and what to submit upstream.
- Reviewed the code and checked the results and conclusions before they went into the documents.
- Wrote the description of the llama.cpp pull request
  ([ggml-org/llama.cpp#30858](https://github.com/ggml-org/llama.cpp/pull/30858)) myself, as llama.cpp's
  contribution guidelines require.

## What the assistant did, under my direction

- Wrote most of the code in this repository: the evaluation harness, the analysis scripts, the drafter export, the
  llama.cpp patches and the figures.
- Ran the measurements and profiling on the Jetson, the phone and the desktop GPU, over SSH and adb with my
  permission.
- Produced first analyses of the results, for example the per-round time breakdowns, the Japanese acceptance
  analysis and the batch-invariance checks, which I then checked.
- Drafted the documentation: the README, the report, the recommendations, the limitations and the technical note.

## How to check the work

Every number in the documents comes from a file in [`results/summary/`](../results/summary/).
[`scripts/run_benchmark.sh`](../scripts/run_benchmark.sh) reproduces the main experiment with one command and compares
the result with the published numbers, and [`tests/`](../tests/) checks the greedy-equivalence results.
