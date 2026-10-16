#!/usr/bin/env python3
"""Pairwise exact-match comparison of probe outputs.

usage: compare_outputs.py <tag> <tag> [<tag> ...]   (tags under $EDGESPECQ_HOME/logs/probe/)
"""
import itertools, json, os, sys

HOME = os.path.expanduser(os.environ.get("EDGESPECQ_HOME", "~/edgespecq"))


def load(tag):
    return {str(x["id"]): x["text"] for x in map(json.loads, open(f"{HOME}/logs/probe/{tag}/outputs.jsonl"))}


def main():
    outs = {t: load(t) for t in sys.argv[1:]}
    for a, b in itertools.combinations(outs, 2):
        ids = sorted(outs[a].keys() & outs[b].keys())
        diff = [i for i in ids if outs[a][i] != outs[b][i]]
        print(f"{a} vs {b}: {len(ids) - len(diff)}/{len(ids)} identical" + (f"  differ: {diff[:6]}" if diff else ""))


if __name__ == "__main__":
    main()
