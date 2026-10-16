#!/usr/bin/env python3
"""Self-time by function for one thread from `perf script -F tid,ip,dso` output, symbolized with `nm`.
Needed because some perf builds (e.g. the WSL2 kernel's) lack libelf and print raw addresses only.

usage: perf script -i perf.data -F tid,ip,dsoff -G | perf_symbolize.py [--tid N] [--top 40]
(dsoff gives the file offset inside the DSO; for GNU ld shared objects text vaddr == file offset)
"""
import argparse, bisect, collections, os, re, subprocess, sys

_cache = {}


def symtab(path):
    if path not in _cache:
        addrs, names = [], []
        try:
            out = subprocess.run(["nm", "-C", "--defined-only", "-n", path], capture_output=True, text=True).stdout
            for line in out.splitlines():
                parts = line.split(" ", 2)
                if len(parts) == 3 and parts[1] in "tTwWiI":
                    addrs.append(int(parts[0], 16)); names.append(parts[2])
        except FileNotFoundError:
            pass
        _cache[path] = (addrs, names)
    return _cache[path]


def resolve(dso, ip):
    addrs, names = symtab(dso) if os.path.exists(dso) else ([], [])
    i = bisect.bisect_right(addrs, ip) - 1
    return names[i] if i >= 0 else f"0x{ip:x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tid", type=int)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    cnt, tids = collections.Counter(), collections.Counter()
    pat = re.compile(r"^\s*(\d+)\s+[0-9a-f]+\s+\((.*)\+0x([0-9a-f]+)\)")
    for line in sys.stdin:
        m = pat.match(line)
        if not m:
            continue
        tid, dso, off = int(m.group(1)), m.group(2), int(m.group(3), 16)
        tids[tid] += 1
        if a.tid is not None and tid != a.tid:
            continue
        cnt[(os.path.basename(dso), resolve(dso, off))] += 1
    total = sum(cnt.values())
    print("threads by samples:", ", ".join(f"{t}:{n}" for t, n in tids.most_common(6)))
    print(f"{total} samples for tid {a.tid}")
    for (dso, sym), n in cnt.most_common(a.top):
        print(f"{100 * n / total:6.2f}%  {dso:22s} {sym[:110]}")


if __name__ == "__main__":
    main()
