#!/usr/bin/env python3
"""Summarize regression/run_eval.py results: per config and task category, decode t/s (token-weighted over all
turns, median over repetitions), mean accepted length, acceptance, and the paired decode speedup over a reference
config with a 95% bootstrap interval over samples (samples resampled within the set; per-sample decode time is the
median over repetitions).

usage: summarize_eval.py <platform> <set> [--ref B0] [--configs B1,B2,...] [--csv out.csv]
"""
import argparse, csv, glob, json, os, random, statistics
from collections import defaultdict

HOME = {"jetson": "~/mt_test/liquid_ai", "jetson_25w": "~/mt_test/liquid_ai", "rtx5070": "~/edgespecq"}


def load(base, cfg):
    """{rep: {id: row}} for one config."""
    reps = {}
    for p in sorted(glob.glob(f"{base}/{cfg}/r*.jsonl")):
        reps[os.path.basename(p)[:-6]] = {r["id"]: r for r in map(json.loads, open(p))}
    return reps


def tot(row, key):
    return sum(t[key] or 0 for t in row["turns"])


def per_sample(reps):
    """id -> (src, n_pred, median decode seconds, accepted, drafted, mean_len weighted)"""
    out = {}
    ids = set.intersection(*(set(r) for r in reps.values())) if reps else set()
    for i in ids:
        rows = [r[i] for r in reps.values()]
        r0 = rows[0]
        secs = statistics.median(tot(r, "pred_ms") / 1000 for r in rows)
        w = [(t["mean_len"], t["n_pred"] or 0) for t in r0["turns"] if t["mean_len"]]
        ml = sum(m * n for m, n in w) / sum(n for _, n in w) if w and sum(n for _, n in w) else None
        out[i] = (r0["src"], tot(r0, "n_pred"), secs, tot(r0, "draft_acc"), tot(r0, "draft_n"), ml)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("platform"); ap.add_argument("set")
    ap.add_argument("--ref", default="B0")
    ap.add_argument("--configs")
    ap.add_argument("--csv")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    base = os.path.expanduser(f"{HOME[a.platform]}/logs/eval/{a.platform}/{a.set}")
    cfgs = a.configs.split(",") if a.configs else sorted(d for d in os.listdir(base) if os.path.isdir(f"{base}/{d}"))
    ref = per_sample(load(base, a.ref))
    cats = sorted({v[0] for v in ref.values()})
    rows = []
    hdr = f"{'config':8s} {'reps':>4s} {'n':>4s} {'t/s':>7s} {'speedup [95% CI]':>22s} {'len':>5s} {'acc':>5s} | " + \
          " ".join(f"{c[:10]:>10s}" for c in cats)
    print(f"{a.platform} / {a.set}: decode t/s, paired speedup vs {a.ref}; per-category columns = speedup")
    print(hdr)
    rng = random.Random(0)
    for cfg in cfgs:
        reps = load(base, cfg)
        if not reps:
            continue
        ps = per_sample(reps)
        ids = sorted(set(ps) & set(ref))
        if not ids:
            continue
        n_tok = sum(ps[i][1] for i in ids); secs = sum(ps[i][2] for i in ids)
        # speedup = ref decode time / config decode time over the same samples (token counts can differ a little
        # because quantized targets are not batch-invariant, so compare rates: tokens per second)
        def sp(sample):
            t = sum(ps[i][1] for i in sample) / sum(ps[i][2] for i in sample)
            r = sum(ref[i][1] for i in sample) / sum(ref[i][2] for i in sample)
            return t / r
        point = sp(ids)
        boots = sorted(sp([rng.choice(ids) for _ in ids]) for _ in range(a.boot))
        lo, hi = boots[int(0.025 * a.boot)], boots[int(0.975 * a.boot)]
        acc = sum(ps[i][3] for i in ids); drf = sum(ps[i][4] for i in ids)
        w = [(ps[i][5], ps[i][1]) for i in ids if ps[i][5]]
        ml = sum(m * n for m, n in w) / sum(n for _, n in w) if w else float("nan")
        per_cat = {}
        for c in cats:
            cid = [i for i in ids if ps[i][0] == c]
            per_cat[c] = sp(cid) if cid else float("nan")
        print(f"{cfg:8s} {len(reps):4d} {len(ids):4d} {n_tok / secs:7.1f} {point:9.3f} [{lo:.3f}, {hi:.3f}] "
              f"{ml:5.2f} {acc / drf if drf else float('nan'):5.3f} | " + " ".join(f"{per_cat[c]:10.3f}" for c in cats))
        rows.append({"platform": a.platform, "set": a.set, "config": cfg, "reps": len(reps), "n": len(ids),
                     "tps": round(n_tok / secs, 2), "speedup": round(point, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "mean_len": round(ml, 3), "accept": round(acc / drf, 4) if drf else None,
                     **{f"speedup_{c}": round(per_cat[c], 4) for c in cats}})
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
