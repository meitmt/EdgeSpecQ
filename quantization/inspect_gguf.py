#!/usr/bin/env python3
"""Tensor manifest of a GGUF file: name, shape, dtype, bytes; plus per-group totals and key metadata.

usage: inspect_gguf.py <model.gguf> [--csv out.csv]
Groups follow the DSpark drafter layout (decoder layers, projection, Markov head, norms/confidence, other).
Needs llama.cpp's gguf-py (pip install -e <llama.cpp>/gguf-py).
"""
import collections, csv, re, sys
from gguf import GGUFReader


def group_of(name):
    if re.match(r"blk\.\d+\.", name):
        return "decoder"
    if "markov" in name:
        return "markov_head"
    if "proj" in name or "fc" in name:
        return "projection"
    if "norm" in name or "conf" in name:
        return "norm/confidence"
    return "other"


def main():
    path = sys.argv[1]
    r = GGUFReader(path)
    meta_keys = [k for k in r.fields if k.startswith(("general.architecture", "general.name", "general.file_type"))
                 or ".block_size" in k or "dspark" in k or ".embedding_length" in k or ".block_count" in k]
    for k in meta_keys:
        f = r.fields[k]
        v = f.parts[f.data[0]] if f.data else None
        if v is not None and getattr(v, "dtype", None) is not None and v.dtype.kind == "u" and v.dtype.itemsize == 1:
            v = bytes(v).decode(errors="replace")
        elif v is not None:
            v = v.tolist() if hasattr(v, "tolist") else v
            v = v[0] if isinstance(v, list) and len(v) == 1 else v
        print(f"meta {k} = {v}")

    rows = []
    for t in r.tensors:
        rows.append({"name": t.name, "shape": "x".join(str(int(d)) for d in t.shape), "dtype": t.tensor_type.name,
                     "n_elements": int(t.n_elements), "bytes": int(t.n_bytes), "group": group_of(t.name)})

    by = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for x in rows:
        g = by[x["group"]]
        g[0] += x["n_elements"]; g[1] += x["bytes"]; g[2][x["dtype"]] += 1
    print(f"\n{'group':18s} {'params':>12s} {'MiB':>9s}  dtypes")
    for gname, (n, b, c) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"{gname:18s} {n/1e6:11.1f}M {b/1048576:9.1f}  {dict(c)}")
    tot_n = sum(x["n_elements"] for x in rows); tot_b = sum(x["bytes"] for x in rows)
    print(f"{'TOTAL':18s} {tot_n/1e6:11.1f}M {tot_b/1048576:9.1f}  ({len(rows)} tensors)")

    if "--csv" in sys.argv:
        out = sys.argv[sys.argv.index("--csv") + 1]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("manifest ->", out)
    else:
        print()
        for x in rows:
            print(f"  {x['name']:45s} {x['shape']:>16s} {x['dtype']:>6s} {x['bytes']/1048576:8.2f} MiB  [{x['group']}]")


if __name__ == "__main__":
    main()
