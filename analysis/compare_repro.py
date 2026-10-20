#!/usr/bin/env python3
"""Compare a reproduction summary (analysis/summarize_eval.py --csv) with the published one: decode t/s per config,
and the speedups between configs as ratios of decode rates over the same samples (so the result does not depend on
which reference config either summary was made with).

usage: compare_repro.py <reproduced.csv> <published.csv> [--tol-tps 0.10] [--tol-ratio 0.07]
Exit status 1 if a value is outside its tolerance or no config is in both files.
"""
import argparse, csv, sys

# speedups reported in docs/report.md §2.1: edge drafter vs target only (unpatched and patched) and vs the official
# drafter as shipped, and the official drafter vs target only
PAIRS = [("B11", "A0"), ("B11", "A1"), ("B11", "B0"), ("A1", "A0"), ("B0", "A0")]


def load(path):
    return {r["config"]: r for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reproduced"); ap.add_argument("published")
    ap.add_argument("--tol-tps", type=float, default=0.10)
    ap.add_argument("--tol-ratio", type=float, default=0.07)
    a = ap.parse_args()
    got, ref = load(a.reproduced), load(a.published)
    common = [c for c in got if c in ref]
    if not common:
        print("no config in both files"); return 1
    bad = 0
    print(f"{'':12s} {'published':>10s} {'reproduced':>10s} {'diff':>7s}")
    for c in common:
        n_got, n_ref = int(got[c]["n"]), int(ref[c]["n"])
        p, r = float(ref[c]["tps"]), float(got[c]["tps"])
        d = r / p - 1
        flag = "" if abs(d) <= a.tol_tps else "  <- outside ±%.0f %%" % (100 * a.tol_tps)
        bad += bool(flag)
        note = "" if n_got == n_ref else f"  (samples: {n_ref} published, {n_got} reproduced)"
        print(f"{c + ' t/s':12s} {p:10.1f} {r:10.1f} {100 * d:+6.1f}%{flag}{note}")
    for x, y in PAIRS:
        if x in common and y in common:
            p = float(ref[x]["tps"]) / float(ref[y]["tps"])
            r = float(got[x]["tps"]) / float(got[y]["tps"])
            d = r / p - 1
            flag = "" if abs(d) <= a.tol_ratio else "  <- outside ±%.0f %%" % (100 * a.tol_ratio)
            bad += bool(flag)
            print(f"{x + '/' + y:12s} {p:9.2f}x {r:9.2f}x {100 * d:+6.1f}%{flag}")
    print("all within tolerance" if not bad else f"{bad} value(s) outside tolerance")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
