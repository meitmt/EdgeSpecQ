#!/usr/bin/env bash
# Where does per-token decode time go? Profile llama-bench tg64 under Nsight Systems for several builds and
# compare GPU kernel launches/time, CUDA runtime API calls, and the host->device memcpy size histogram.
#   nsys_decode_overhead.sh <name>=<llama_dir> [<name>=<llama_dir> ...]
# Reports go to $EDGESPECQ_HOME/logs/nsys/<name>.{nsys-rep,sqlite}. Requires nsys (CUDA toolkit).
set -euo pipefail
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
M=${MODEL:-$H/models/track_t/LFM2.5-2.6B-Q4_K_M.gguf}
O=$H/logs/nsys
mkdir -p "$O"
[ -f /etc/profile.d/cuda.sh ] && source /etc/profile.d/cuda.sh
NAMES=()
for arg in "$@"; do
  name=${arg%%=*}; dir=${arg#*=}; NAMES+=("$name")
  rm -f "$O/$name".nsys-rep "$O/$name".sqlite "$O/$name"_*.csv
  # llama.cpp launches decode as CUDA graphs: trace graph nodes to see the individual kernels
  nsys profile -t cuda --cuda-graph-trace=node -o "$O/$name" --force-overwrite true \
    "$dir/build/bin/llama-bench" -m "$M" -ngl 99 -fa 1 -p 0 -n 64 -r 1 > /dev/null 2>&1
  nsys export -t sqlite -o "$O/$name.sqlite" "$O/$name.nsys-rep" > /dev/null 2>&1
done

python3 - "$O" "${NAMES[@]}" <<'PY'
import collections, sqlite3, sys
o, names = sys.argv[1], sys.argv[2:]
def q(name, sql):
    return sqlite3.connect(f"{o}/{name}.sqlite").execute(sql).fetchall()
print(f"{'':28s}" + "".join(f"{n:>14s}" for n in names))
row = lambda label, vals: print(f"{label:28s}" + "".join(f"{v:>14}" for v in vals))
row("GPU kernel launches", [q(n, "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")[0][0] for n in names])
row("GPU kernel time (ms)", [f"{q(n, 'SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL')[0][0] / 1e6:.1f}" for n in names])
api = {n: dict(q(n, "SELECT s.value, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id GROUP BY s.value")) for n in names}
for call in ("cudaStreamSynchronize", "cudaMemcpyAsync", "cudaGraphLaunch", "cudaLaunchKernel"):
    keys = {k for a in api.values() for k in a if k.startswith(call)}
    row(call + " calls", [sum(api[n].get(k, 0) for k in keys) for n in names])
h2d = {n: collections.Counter(b for (b,) in q(n, "SELECT bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 1")) for n in names}
print("\nhost->device memcpy count by size:")
for s in sorted(set().union(*h2d.values())):
    c = [h2d[n].get(s, 0) for n in names]
    row(f"  {s} B", c + (["<- differs"] if len(set(c)) > 1 else []))
PY
