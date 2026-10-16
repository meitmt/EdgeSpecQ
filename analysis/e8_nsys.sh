#!/usr/bin/env bash
# E8: Nsight Systems view of one speculative run: GPU kernel time per CUDA stream (drafter vs target),
# top kernels per stream, and memcpy volume, for each drafter precision.
#   e8_nsys.sh <llama_dir (timing build)> <target.gguf> [drafters="F16 Q4_0"] [n=128]
set -u
H=${EDGESPECQ_HOME:-$HOME/edgespecq}
L=$1; T=$2; DRS=${3:-F16 Q4_0}; N=${4:-128}
O=$H/logs/nsys_e8; mkdir -p $O
[ -f /etc/profile.d/cuda.sh ] && source /etc/profile.d/cuda.sh
P=$(mktemp)
printf '<|im_start|>user\nJanet'"'"'s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers'"'"' market daily for $2 per fresh duck egg. How much in dollars does she make every day? Solve step by step and explain each step in detail.<|im_end|>\n<|im_start|>assistant\n' > $P
for d in $DRS; do
  if [ $d = F16 ]; then DR=$H/models/draft/LFM2.5-VL-3B-DSpark-F16.gguf; else DR=$H/models/quant/LFM2.5-VL-3B-DSpark-$d.gguf; fi
  rm -f $O/$d.nsys-rep $O/$d.sqlite
  nsys profile -t cuda --cuda-graph-trace=node -o $O/$d --force-overwrite true \
    $L/build/bin/llama-speculative-simple -m $T -md $DR --spec-type draft-dspark --spec-draft-n-max 8 --spec-draft-n-min 0 \
    -ngl 99 -ngld 99 -fa on -c 4096 --temp 0 -n $N -f $P > $O/$d.log 2>&1
  nsys export -t sqlite -o $O/$d.sqlite $O/$d.nsys-rep > /dev/null 2>&1
  echo "######## drafter $d: $(grep -E 'E8 (rounds|per_round)' $O/$d.log | tr '\n' ' ')"
  python3 - $O/$d.sqlite <<'PY'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
q = lambda s: db.execute(s).fetchall()
streams = q("""SELECT streamId, COUNT(*), SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY streamId ORDER BY 3 DESC""")
for sid, n, ms in streams:
    print(f"  stream {sid}: {n} kernels, {ms:.1f} ms GPU time")
    top = q(f"""SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
               WHERE k.streamId={sid} GROUP BY s.value ORDER BY 3 DESC LIMIT 8""")
    for name, c, t in top:
        print(f"      {t:8.1f} ms {c:7d}x  {name[:70]}")
for kind, label in ((1, "H2D"), (2, "D2H")):
    r = q(f"SELECT COUNT(*), SUM(bytes)/1e6, SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind={kind}")[0]
    print(f"  memcpy {label}: {r[0]} copies, {r[1] or 0:.1f} MB, {r[2] or 0:.1f} ms")
PY
done
rm -f $P
