#!/usr/bin/env python3
"""E9: acceptance of a reduced-vocabulary DSpark drafter, simulated from full-vocab rounds (analysis/e9_collect.py).

Greedy drafting with a draft vocabulary V keeps the drafter's argmax whenever that argmax is in V (kept rows are
bit-identical, see quantization/make_reduced_vocab_drafter.py), so a round's accepted run is cut at its first accepted
token outside V and is otherwise unchanged. This is exact per round; it ignores the shift of later round boundaries,
which the end-to-end runs with real reduced-vocab GGUFs measure.

usage: e9_simulate.py <rounds.jsonl> <target.gguf> [--n 8000,16000,...] [--mixes "en:en_chat=1,...;all:..."] [--json]
"""
import argparse, importlib.util, json, os
from collections import defaultdict
import numpy as np
import gguf

_spec = importlib.util.spec_from_file_location(
    "mrv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quantization", "make_reduced_vocab_drafter.py"))
mrv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mrv)

MIXES = "en:en_chat=1,en_wiki=1,code=1,math=1;en+ja:en_chat=1,en_wiki=1,code=1,math=1,ja_wiki=1,ja_chat=1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds"); ap.add_argument("target")
    ap.add_argument("--n", default="8000,16000,24000,32000,48000,64000,96000")
    ap.add_argument("--mixes", default=MIXES)
    ap.add_argument("--json")
    a = ap.parse_args()
    rounds = [json.loads(l) for l in open(a.rounds)]
    by_probe = defaultdict(list)
    for r in rounds:
        by_probe[r["probe"]].append(r["draft"][:r["n_acc"]])
    tr = gguf.GGUFReader(a.target)
    n_vocab = len(tr.get_field("tokenizer.ggml.tokens").data)

    res = {}
    print(f"tokens/round = accepted + 1 (bonus); full vocab = {n_vocab}")
    for probe, acc in by_probe.items():
        full = np.mean([len(x) + 1 for x in acc])
        print(f"\n{probe}: rounds={len(acc)} full-vocab tokens/round={full:.3f}")
        print(f"{'N':>7s} " + " ".join(f"{m.split(':')[0]:>16s}" for m in a.mixes.split(";")))
        for n in map(int, a.n.split(",")):
            cells = []
            for spec in a.mixes.split(";"):
                name, mix = spec.split(":")
                ids, _ = mrv.draft_vocab(tr, n, {k: float(v) for k, v in (kv.split("=") for kv in mix.split(","))})
                keep = np.zeros(n_vocab, bool); keep[ids] = True
                tpr = np.mean([next((i for i, t in enumerate(x) if not keep[t]), len(x)) + 1 for x in acc])
                cells.append(f"{tpr:7.3f} ({tpr / full:6.1%})")
                res.setdefault(probe, {"full": full, "rounds": len(acc)}).setdefault(name, {})[n] = tpr
            print(f"{n:7d} " + " ".join(f"{c:>16s}" for c in cells))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
