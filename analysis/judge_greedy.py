#!/usr/bin/env python3
"""S3 judgment: is speculative greedy decoding equivalent to target-only greedy decoding on the same target?

Criteria (docs/limitations.md §1):
  1. F16 target: every speculative run produces exactly the target-only output, for every prompt.
  2. Quantized target: outputs may differ, but only at numerical near-ties: at the first differing token the
     target-only top-1/top-2 logprob margin must be < 0.3 (quantized kernels are not batch-invariant, so verifying
     n_max + 1 tokens can flip a near-tie; llama.cpp #25618 / #23335 / #16016).
  3. CPU control: CPU backend with n_max = 1 must be exactly identical (CPU numerics do not change for batch <= 2).
A divergence that cannot be located (same tokens, different length or text) always fails.

Divergences that break an exact criterion but were investigated are listed in a known-exceptions file
(results/summary/s3_known_exceptions.json); an entry is honoured only while the divergence is a near-tie below its
max_margin. Such runs are reported as passing with a known exception, never as strictly identical.

Two sources of runs:
  probe mode   scripts/run_s3.sh: $EDGESPECQ_HOME/logs/probe/<group>__<target>__none for the reference and
               <group>__<target>__<drafter>__n<n_max> for the speculative runs, where <group> starts with the tag
               and a group containing "cpu" is the CPU control.
  eval mode    regression/run_eval.py --logprobs on an evaluation set (e.g. smoke24), any platform:
               <log home>/logs/eval/<platform>/<set>/<config>_lp/{outputs,tokens}_r1.jsonl. Each speculative
               config is paired with the config in the list that has the same target and runtime and no drafter;
               multi-turn samples are compared over all turns in order (a turn ending at a different place fails).

usage: judge_greedy.py <tag> [--json out.json] [--runtime "description"] [--exceptions file.json]
       judge_greedy.py --eval <platform> <set> <config,config,...> [--log-home DIR] [--json out.json]
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


TURN = ["<<turn>>", 0.0, []]  # separator between turns: a divergence there cannot be located, so it fails


def load_eval_run(path):
    """run_eval.py --logprobs output -> (texts, toks) with the turns of each sample joined in order."""
    texts = {r["id"]: "\n<<turn>>\n".join(r["texts"])
             for r in map(json.loads, open(f"{path}/outputs_r1.jsonl", encoding="utf-8"))}
    toks = {}
    for r in map(json.loads, open(f"{path}/tokens_r1.jsonl", encoding="utf-8")):
        seq = []
        for k, t in enumerate(r["turns"]):
            seq += ([TURN] if k else []) + t
        toks[r["id"]] = seq
    return texts, toks


def eval_runs(platform, set_name, names, log_home, exc):
    spec = json.load(open(f"{ROOT}/benchmark/configs.json"))
    plat = spec["platforms"][platform]
    cfgs = {n: spec["configs"][n] for n in names}
    home = os.path.expanduser(log_home or plat.get("log_home", plat["home"]))
    base = f"{home}/logs/eval/{platform}/{set_name}"
    group = platform + ("-cpu" if str(plat.get("ngl", 99)) == "0" else "")
    runs = []
    for n, c in cfgs.items():
        if not c.get("draft"):
            continue
        refs = [m for m, d in cfgs.items() if not d.get("draft") and d["target"] == c["target"]
                and d["runtime"] == c["runtime"]]
        if not refs:
            print(f"no reference for {n}", file=sys.stderr); continue
        n_max = c.get("n_max") or plat["n_max"]
        crit = criterion(group, c["target"], n_max)
        mine = {e["id"]: e["max_margin"] for e in exc if e["group"] == group and e["target"] == c["target"]}
        r = judge_pair(load_eval_run(f"{base}/{refs[0]}_lp"), load_eval_run(f"{base}/{n}_lp"),
                       exact=crit.startswith("exact"), exceptions=mine)
        runs.append({"group": group, "config": n, "reference": refs[0], "target": c["target"], "drafter": c["draft"],
                     "runtime": c["runtime"], "n_max": n_max, "criterion": crit, **r})
    return runs


def probe_runs(tag, exc):
    base = f"{HOME}/logs/probe"
    runs = []
    for path in sorted(glob.glob(f"{base}/{tag}*__*__*__n*")):
        group, target, drafter, n = os.path.basename(path).split("__")
        n_max = int(n[1:])
        ref_path = f"{base}/{group}__{target}__none"
        if not os.path.isdir(ref_path):
            print(f"no reference for {path}", file=sys.stderr); continue
        crit = criterion(group, target, n_max)
        mine = {e["id"]: e["max_margin"] for e in exc if e["group"] == group and e["target"] == target}
        r = judge_pair(load_run(ref_path), load_run(path), exact=crit.startswith("exact"), exceptions=mine)
        runs.append({"group": group, "target": target, "drafter": drafter, "n_max": n_max, "criterion": crit, **r})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?"); ap.add_argument("--json"); ap.add_argument("--runtime", default="")
    ap.add_argument("--eval", nargs=3, metavar=("PLATFORM", "SET", "CONFIGS"))
    ap.add_argument("--log-home")
    ap.add_argument("--exceptions", default=f"{ROOT}/results/summary/s3_known_exceptions.json")
    a = ap.parse_args()
    exc = json.load(open(a.exceptions))["exceptions"] if a.exceptions and os.path.exists(a.exceptions) else []
    report = {"source": "eval " + " ".join(a.eval) if a.eval else f"probe {a.tag}", "runtime": a.runtime,
              "margin_threshold": MARGIN, "exceptions_file": os.path.relpath(a.exceptions, ROOT) if exc else None}
    if a.eval:
        report["runs"] = eval_runs(a.eval[0], a.eval[1], a.eval[2].split(","), a.log_home, exc)
    else:
        report["runs"] = probe_runs(a.tag, exc)
    runs = report["runs"]
    report["pass"] = bool(runs) and all(r["pass"] for r in runs)
    report["strict_pass"] = bool(runs) and all(r["strict_pass"] for r in runs)
    print(f"{'group':10s} {'target':7s} {'drafter':12s} {'n':>2s} {'same':>5s} {'tie':>4s} {'big':>4s} {'unl':>4s}"
          f" {'margin med/max':>15s}  verdict")
    for r in runs:
        mm = "-" if r["margin_max"] is None else f"{r['margin_median']:.3f}/{r['margin_max']:.3f}"
        v = "PASS" if r["strict_pass"] else ("PASS, known exception" if r["pass"] else "FAIL")
        print(f"{r['group']:10s} {r['target']:7s} {r['drafter']:12s} {r['n_max']:2d} {r['identical']:3d}/{r['prompts']:<2d}"
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
