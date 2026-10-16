#!/usr/bin/env python3
"""Summarize drafter-precision runs: acceptance, mean accepted length, decode t/s (median and range over
repetitions) and speedup vs target-only. Optionally writes a CSV row per (probe, target, drafter).

usage: summarize_draft_precision.py <probe> <target> [--drafters Q8_0,P3,...] [--csv out.csv]
"""
import glob, json, os, re, statistics, sys

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))
DRAFTERS = ("F16", "Q8_0", "P3", "Q4_K_M", "Q4_0")


def runs(probe, tq, d):
    pat = f"{HOME}/logs/probe/dp_{probe}_t{tq}_d{d}_r*/rows.json"
    return [json.load(open(p)) for p in sorted(glob.glob(pat))]


def size_mib(probe, tq, d):
    # drafter file size, from the -md argument recorded in the first run's config.json
    cfg = sorted(glob.glob(f"{HOME}/logs/probe/dp_{probe}_t{tq}_d{d}_r*/config.json"))[0]
    argv = json.load(open(cfg))["argv"]
    return os.path.getsize(argv[argv.index("-md") + 1]) / 2**20


def decode_tps(rows):
    n = sum(r["n_pred"] for r in rows)
    return n / sum(r["n_pred"] / r["tps"] for r in rows)


def main():
    probe, tq = sys.argv[1], sys.argv[2]
    base_runs = runs(probe, tq, "none")
    if not base_runs:
        sys.exit(f"no runs for {probe}/{tq}")
    base = statistics.median(decode_tps(r) for r in base_runs)
    out = []
    print(f"probe {probe}, target {tq}: target-only {base:.1f} t/s (median of {len(base_runs)})")
    print(f"{'drafter':8s} {'MiB':>6s} {'accept':>7s} {'len(tok-w)':>10s} {'t/s median':>11s} {'range':>13s} {'speedup':>8s} {'n':>2s}")
    drafters = sys.argv[sys.argv.index("--drafters") + 1].split(",") if "--drafters" in sys.argv else DRAFTERS
    for d in drafters:
        rs = runs(probe, tq, d)
        if not rs:
            continue
        mib = size_mib(probe, tq, d)
        rows = rs[0]  # acceptance is deterministic across repetitions; use the first
        dn = sum(r["draft_n"] or 0 for r in rows)
        da = sum(r["draft_acc"] or 0 for r in rows)
        w = [(r["mean_len"], r["n_pred"]) for r in rows if r["mean_len"]]
        mlen = sum(x * n for x, n in w) / sum(n for _, n in w)
        t = [decode_tps(r) for r in rs]
        tm = statistics.median(t)
        print(f"{d:8s} {mib:6.0f} {da/dn:7.3f} {mlen:10.2f} {tm:11.1f} {min(t):6.1f}-{max(t):<6.1f} {tm/base:7.2f}x {len(t):2d}")
        out.append({"probe": probe, "target": tq, "drafter": d, "size_mib": round(mib, 1), "accept": round(da / dn, 4),
                    "mean_len_tokw": round(mlen, 3), "tps_median": round(tm, 2), "tps_min": round(min(t), 2),
                    "tps_max": round(max(t), 2), "target_only_tps": round(base, 2), "speedup": round(tm / base, 3),
                    "reps": len(t)})
    if "--csv" in sys.argv:
        import csv
        path = sys.argv[sys.argv.index("--csv") + 1]
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            if new:
                w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
