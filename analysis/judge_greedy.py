#!/usr/bin/env python3
"""S3 judgment: is speculative greedy decoding equivalent to target-only greedy decoding on the same target?

Criteria (proj v1.11 §4.5, defined in v1.3):
  1. F16 target: every speculative run produces exactly the target-only output, for every prompt.
  2. Quantized target: outputs may differ, but only at numerical near-ties: at the first differing token the
     target-only top-1/top-2 logprob margin must be < 0.3 (quantized kernels are not batch-invariant, so verifying
     n_max + 1 tokens can flip a near-tie; llama.cpp #25618 / #23335 / #16016).
  3. CPU control: CPU backend with n_max = 1 must be exactly identical (CPU numerics do not change for batch <= 2).
A divergence that cannot be located (same tokens, different length or text) always fails.

Divergences that break an exact criterion but were investigated are listed in a known-exceptions file
(results/summary/s3_known_exceptions.json); an entry is honoured only while the divergence is a near-tie below its
max_margin. Such runs are reported as passing with a known exception, never as strictly identical.

Runs come from scripts/run_s3.sh: $EDGESPECQ_HOME/logs/probe/<group>__<target>__none for the reference and
<group>__<target>__<drafter>__n<n_max> for the speculative runs, where <group> starts with the tag and a group
containing "cpu" is the CPU control.

usage: judge_greedy.py <tag> [--json out.json] [--runtime "description"] [--exceptions file.json]
exit status 1 if any run fails
"""
import argparse, glob, json, os, statistics, sys

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARGIN = 0.3


def load_run(path):
    texts = {str(r["id"]): r["text"] for r in map(json.loads, open(f"{path}/outputs.jsonl", encoding="utf-8"))}
    toks = {}
    if os.path.exists(f"{path}/tokens.jsonl"):
        toks = {str(r["id"]): r["tok"] for r in map(json.loads, open(f"{path}/tokens.jsonl", encoding="utf-8"))}
    return texts, toks


def classify(ref_text, ref_tok, spec_text, spec_tok):
    """One prompt -> ("identical", None, None) | ("near_tie" | "large_margin", margin, k) | ("unlocated", None, None)."""
    if ref_text == spec_text:
        return "identical", None, None
    k = next((j for j in range(min(len(ref_tok), len(spec_tok))) if ref_tok[j][0] != spec_tok[j][0]), None)
    if k is None or len(ref_tok[k][2]) < 2:
        return "unlocated", None, None
    top = ref_tok[k][2]
    m = top[0][1] - top[1][1]
    return ("near_tie" if m < MARGIN else "large_margin"), m, k


def criterion(group, target, n_max):
    if "cpu" in group and n_max == 1:
        return "exact (CPU, n_max 1)"
    if target == "F16":
        return "exact (F16 target)"
    return f"near-tie (margin < {MARGIN})"


def judge_pair(ref, spec, exact, exceptions=()):
    """exceptions: {prompt id: max_margin} documented for this run."""
    rt, rk = ref
    st, sk = spec
    exceptions = dict(exceptions)
    counts = {"identical": 0, "near_tie": 0, "large_margin": 0, "unlocated": 0, "missing": 0}
    margins, failures, known = [], [], []
    for i, text in rt.items():
        if i not in st:
            counts["missing"] += 1; failures.append((i, "missing", None, None)); continue
        c, m, k = classify(text, rk.get(i, []), st[i], sk.get(i, []))
        counts[c] += 1
        if m is not None:
            margins.append(m)
        if c == "identical" or (not exact and c == "near_tie"):
            continue
        if c == "near_tie" and i in exceptions and m < exceptions[i]:
            known.append((i, c, m, k))
        else:
            failures.append((i, c, m, k))
    return {"prompts": len(rt), **counts,
            "margin_median": statistics.median(margins) if margins else None,
            "margin_max": max(margins) if margins else None,
            "pass": not failures, "strict_pass": not failures and not known,
            "failures": failures, "known_exceptions": known}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--json"); ap.add_argument("--runtime", default="")
    ap.add_argument("--exceptions", default=f"{ROOT}/results/summary/s3_known_exceptions.json")
    a = ap.parse_args()
    exc = json.load(open(a.exceptions))["exceptions"] if a.exceptions and os.path.exists(a.exceptions) else []
    base = f"{HOME}/logs/probe"
    report = {"tag": a.tag, "runtime": a.runtime, "margin_threshold": MARGIN,
              "exceptions_file": os.path.relpath(a.exceptions, ROOT) if exc else None, "runs": []}
    for path in sorted(glob.glob(f"{base}/{a.tag}*__*__*__n*")):
        group, target, drafter, n = os.path.basename(path).split("__")
        n_max = int(n[1:])
        ref_path = f"{base}/{group}__{target}__none"
        if not os.path.isdir(ref_path):
            print(f"no reference for {path}", file=sys.stderr); continue
        crit = criterion(group, target, n_max)
        mine = {e["id"]: e["max_margin"] for e in exc if e["group"] == group and e["target"] == target}
        r = judge_pair(load_run(ref_path), load_run(path), exact=crit.startswith("exact"), exceptions=mine)
        report["runs"].append({"group": group, "target": target, "drafter": drafter, "n_max": n_max,
                               "criterion": crit, **r})
    runs = report["runs"]
    report["pass"] = bool(runs) and all(r["pass"] for r in runs)
    report["strict_pass"] = bool(runs) and all(r["strict_pass"] for r in runs)
    print(f"{'group':8s} {'target':7s} {'drafter':12s} {'n':>2s} {'same':>5s} {'tie':>4s} {'big':>4s} {'unl':>4s}"
          f" {'margin med/max':>15s}  verdict")
    for r in runs:
        mm = "-" if r["margin_max"] is None else f"{r['margin_median']:.3f}/{r['margin_max']:.3f}"
        v = "PASS" if r["strict_pass"] else ("PASS, known exception" if r["pass"] else "FAIL")
        print(f"{r['group']:8s} {r['target']:7s} {r['drafter']:12s} {r['n_max']:2d} {r['identical']:3d}/{r['prompts']:<2d}"
              f"{r['near_tie']:4d} {r['large_margin']:4d} {r['unlocated']:4d} {mm:>15s}  {v} ({r['criterion']})")
        for tag, items in (("known exception", r["known_exceptions"]), ("FAIL", r["failures"])):
            for i, c, m, k in items:
                print(f"    {tag}: {i} {c}" + ("" if m is None else f" at token {k}, margin {m:.4f}"))
    n_known = sum(len(r["known_exceptions"]) for r in runs)
    print(f"S3 overall: {'PASS' if report['pass'] else 'FAIL'}"
          + (f" with {n_known} documented exception(s) under the exact criteria" if report["pass"] and n_known else "")
          + f" ({len(runs)} speculative runs)")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
