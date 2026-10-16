#!/usr/bin/env python3
"""Where and why does a speculative run diverge from the target-only greedy run?

For each prompt, finds the first differing token and reports the target-only top-1 vs top-2 logprob margin
there. Divergence only at tiny margins indicates numerical near-ties (batch-shape-dependent kernels); large
margins would indicate a logic bug. Both runs must have been recorded with LOGPROBS=1.

usage: divergence.py <target_only_tag> <spec_tag> [--text]   (--text also prints the diverging text)
"""
import json, os, sys

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))


def load(tag, name):
    return {str(x["id"]): x for x in map(json.loads, open(f"{HOME}/logs/probe/{tag}/{name}"))}


def main():
    a_tag, b_tag = sys.argv[1], sys.argv[2]
    show_text = "--text" in sys.argv
    A, B = load(a_tag, "tokens.jsonl"), load(b_tag, "tokens.jsonl")
    margins, same = [], 0
    for i, ra in A.items():
        a, b = ra["tok"], B.get(i, {"tok": []})["tok"]
        k = next((j for j in range(min(len(a), len(b))) if a[j][0] != b[j][0]), None)
        if k is None:
            same += 1
            continue
        ta, tb = a[k][2], b[k][2]
        m = ta[0][1] - ta[1][1] if len(ta) > 1 else float("nan")
        mb = tb[0][1] - tb[1][1] if len(tb) > 1 else float("nan")
        margins.append(m)
        print(f"id={i:28s} tok#{k:4d}  target-only {ta[0][0]!r} {ta[0][1]:.4f} vs {ta[1][0]!r} {ta[1][1]:.4f} "
              f"(margin {m:.4f}) | spec picked {b[k][0]!r} (spec margin {mb:.4f})")
        if show_text:
            ctx = "".join(t[0] for t in a[max(0, k - 20):k])
            print(f"    ...{ctx!r} -> target {a[k][0]!r} / spec {b[k][0]!r}")
    print(f"\nidentical token sequences: {same}/{len(A)}; diverged: {len(margins)}")
    if margins:
        margins.sort()
        print("target-only top1-top2 logprob margin at divergence: min %.4f  median %.4f  max %.4f"
              % (margins[0], margins[len(margins) // 2], margins[-1]))


if __name__ == "__main__":
    main()
