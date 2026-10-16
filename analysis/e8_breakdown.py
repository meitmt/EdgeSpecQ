#!/usr/bin/env python3
"""E8: per-round time breakdown of DSpark speculative decoding, per drafter precision.

Runs a timing build of llama-speculative-simple (see runtime/llama_cpp_patches/e8-timing.patch) on prompts
from the text probe (LFM2 chat format) and aggregates the "E8 per_round_ms" lines it prints:
draft_decode / draft_sample / draft_other / verify / inject / tgt_sample / other.

usage: e8_breakdown.py <llama_dir> <target.gguf> [--drafters none,F16,Q8_0,...] [--n 256] [--prompts 5] [--extra "..."]
environment: N_MAX (DSpark --spec-draft-n-max, default 8)
Drafter files are looked up under $EDGESPECQ_HOME/models/{draft,quant}/LFM2.5-VL-3B-DSpark-<name>.gguf, and E9
reduced-vocabulary drafters (V16k, ...) under models/e9/.
"""
import argparse, json, os, re, subprocess

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
KEYS = ("draft_decode", "draft_sample", "draft_other", "verify", "inject", "tgt_sample", "other", "total")


def drafter_path(name):
    if name == "F16":
        return f"{HOME}/models/draft/LFM2.5-VL-3B-DSpark-F16.gguf"
    if name.startswith("V"):  # E9 reduced-vocabulary drafters (scripts/make_e9_drafters.sh)
        return f"{HOME}/models/e9/LFM2.5-VL-3B-DSpark-Q8_0-{name}.gguf"
    return f"{HOME}/models/quant/LFM2.5-VL-3B-DSpark-{name}.gguf"


def run(binary, target, drafter, prompt, n, extra):
    # -p, not -f: -f drops the trailing newline after "assistant"; --no-escape keeps backslashes in code prompts
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    args = [binary, "-m", target, "-ngl", "99", "-fa", "on", "-c", "4096", "--temp", "0", "-n", str(n),
            "-p", text, "--no-escape"]
    if drafter == "none":
        args += ["--spec-type", "none"]
    else:
        args += ["-md", drafter_path(drafter), "-ngld", "99", "--spec-type", "draft-dspark",
                 "--spec-draft-n-max", os.environ.get("N_MAX", "8"), "--spec-draft-n-min", "0"]
    args += extra
    out = subprocess.run(args, capture_output=True, text=True).stderr + ""
    m_r = re.search(r"E8 rounds=(\d+) draft_rounds=(\d+) tokens/round=([0-9.]+)", out)
    m_t = re.search(r"E8 per_round_ms (.*)", out)
    m_dec = re.search(r"decoded\s+(\d+) tokens in\s+([0-9.]+) seconds", out)
    if not (m_r and m_t and m_dec):
        raise RuntimeError("could not parse output:\n" + out[-2000:])
    vals = dict(kv.split("=") for kv in m_t.group(1).split())
    return {"rounds": int(m_r.group(1)), "tokens": int(m_dec.group(1)), "secs": float(m_dec.group(2)),
            **{k: float(vals[k]) for k in KEYS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("llama_dir"); ap.add_argument("target")
    ap.add_argument("--drafters", default="F16,Q8_0,P3,Q4_K_M,Q4_0")  # target-only baseline: scripts/bench_tg.sh
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--extra", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    binary = f"{a.llama_dir}/build/bin/llama-speculative-simple"
    prompts = [json.loads(l)["prompt"] for l in open(f"{HOME}/probe/track_t_probe20.jsonl")]
    prompts = prompts[::max(1, len(prompts) // a.prompts)][:a.prompts]
    results = {}
    print(f"target={os.path.basename(a.target)}  prompts={len(prompts)}  n={a.n}  (ms per round, round-weighted)")
    print(f"{'drafter':8s} {'tok/rnd':>7s} {'t/s':>7s} | " + " ".join(f"{k:>12s}" for k in KEYS))
    for d in a.drafters.split(","):
        rs = [run(binary, a.target, d, p, a.n, a.extra.split()) for p in prompts]
        R = sum(r["rounds"] for r in rs)
        agg = {k: sum(r[k] * r["rounds"] for r in rs) / R for k in KEYS}
        tok = sum(r["tokens"] for r in rs); sec = sum(r["secs"] for r in rs)
        results[d] = {"rounds": R, "tokens_per_round": tok / R, "tps": tok / sec, **agg}
        print(f"{d:8s} {tok / R:7.2f} {tok / sec:7.1f} | " + " ".join(f"{agg[k]:12.3f}" for k in KEYS))
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
