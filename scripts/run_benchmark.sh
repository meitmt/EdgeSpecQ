#!/usr/bin/env bash
# One command for the main experiment (docs/report.md §2.1) on this machine: target only on the unpatched runtime
# (A0), the official F16 drafter as shipped (A1), target only on the patched runtime (B0) and the edge drafter (B11),
# on the same evaluation set, then a comparison with the published numbers.
#   run_benchmark.sh --platform {jetson,jetson_25w,rtx5070} [--set dev120|smoke24] [--configs A0,A1,B0,B11]
#                    [--reps N] [--tag repro] [--check-only]
# Before measuring it checks, and stops on any mismatch:
#   - the platform's llama.cpp checkout is at the pinned commit and the runtime libraries of every config exist
#   - every model file the configs use matches benchmark/manifests/models.sha256
#   - the evaluation set matches benchmark/manifests/<set>.tsv
# Paths come from benchmark/configs.json -> platforms.<platform>. Run it on the device itself (on the Jetson from a
# copy of this repository there); the Jetson measurement goes through scripts/jetson_measure.sh (desktop stopped,
# MAXN_SUPER + jetson_clocks, or the 25 W default for jetson_25w; restored afterwards; sudo asks once).
# Results: <home>/logs/eval/<platform>/<set>_<tag>/ (resumable: the same command continues an interrupted run, a new
# --tag starts over); summary results/repro/<platform>_<set>_<tag>.csv.
# Time: smoke24 about 25 min on the Jetson and 5 min on the RTX 5070; dev120 about five times as long.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_COMMIT=620d2225c
PLAT= SET=dev120 CONFIGS=A0,A1,B0,B11 REPS= TAG=repro CHECK_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in
    --platform) PLAT=$2; shift 2 ;;
    --set) SET=$2; shift 2 ;;
    --configs) CONFIGS=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    *) sed -n '2,15p' "$0"; exit 2 ;;
  esac
done
case $PLAT in
  jetson|jetson_25w) [ -e /etc/nv_tegra_release ] || { echo "run --platform $PLAT on the Jetson" >&2; exit 2; }
                     REPS=${REPS:-2} ;;
  rtx5070) REPS=${REPS:-3} ;;
  *) echo "--platform jetson, jetson_25w or rtx5070 (phones: see docs/report.md R5)" >&2; exit 2 ;;
esac
cd "$R"

# platform paths and the files each config needs, from benchmark/configs.json (same rules as regression/run_eval.py)
eval "$(python3 - "$PLAT" "$CONFIGS" <<'EOF'
import json, os, shlex, sys
sys.path.insert(0, "regression")
from run_eval import model_path
spec = json.load(open("benchmark/configs.json"))
plat = spec["platforms"][sys.argv[1]]
home = os.path.expanduser(plat["home"])
models, libs = {"target/mmproj-LFM2.5-VL-3B-Q8_0.gguf"}, set()
for name in sys.argv[2].split(","):
    cfg = spec["configs"][name]
    models.add(os.path.relpath(model_path(home, cfg["target"], "target"), f"{home}/models"))
    if cfg["draft"]:
        models.add(os.path.relpath(model_path(home, cfg["draft"], "draft"), f"{home}/models"))
    libs.update(plat["runtimes"][cfg["runtime"]])
print(f"HOME_DIR={shlex.quote(home)}")
print(f"LLAMA_DIR={shlex.quote(os.path.expanduser(plat['llama_dir']))}")
print(f"MODELS=({' '.join(sorted(models))})")
print(f"LIBS=({' '.join(sorted(libs))})")
EOF
)"
PY=python3; [ -x "$HOME_DIR/.venv/bin/python" ] && PY=$HOME_DIR/.venv/bin/python
fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== $PLAT: $SET, configs $CONFIGS, $REPS repetitions"
head=$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null) || fail "no llama.cpp checkout at $LLAMA_DIR"
case $head in $LLAMA_COMMIT*) ;; *) fail "$LLAMA_DIR is at ${head:0:9}, expected $LLAMA_COMMIT" ;; esac
[ -x "$LLAMA_DIR/build/bin/llama-server" ] || fail "$LLAMA_DIR/build/bin/llama-server not built"
echo "llama.cpp $LLAMA_COMMIT ok"
for d in "${LIBS[@]}"; do
  compgen -G "$HOME_DIR/$d/*.so*" > /dev/null || fail "runtime libraries missing: $HOME_DIR/$d"
done
echo "runtime libraries ok: ${LIBS[*]}"
SUMS=$(grep -F -f <(printf '  %s\n' "${MODELS[@]}") benchmark/manifests/models.sha256)
[ "$(wc -l <<< "$SUMS")" = ${#MODELS[@]} ] || fail "not every model is in benchmark/manifests/models.sha256"
echo "hashing ${#MODELS[@]} model files..."
( cd "$HOME_DIR/models" && sha256sum -c --quiet <<< "$SUMS" ) || fail "model SHA256 mismatch"
echo "models ok: ${MODELS[*]}"
EVAL=$HOME_DIR/eval/$SET.jsonl
[ -f "$EVAL" ] || fail "$EVAL missing: run benchmark/build_eval_set.py (docs/report.md R3)"
"$PY" -c "import json, sys
for r in map(json.loads, open(sys.argv[1])): print(r['src'], r['id'], len(r['turns']), len(r['images']), sep='\t')" \
  "$EVAL" | diff -q - "benchmark/manifests/$SET.tsv" > /dev/null || fail "$EVAL does not match benchmark/manifests/$SET.tsv"
echo "evaluation set ok: $SET"
[ $CHECK_ONLY = 1 ] && exit 0

RUN=("$PY" regression/run_eval.py "$PLAT" "$EVAL" "$CONFIGS" --reps "$REPS" --tag "$TAG")
case $PLAT in
  jetson) bash scripts/jetson_measure.sh "${RUN[@]}" ;;
  jetson_25w) bash scripts/jetson_measure.sh --25w "${RUN[@]}" ;;
  *) "${RUN[@]}" ;;
esac

mkdir -p results/repro
OUT=results/repro/${PLAT}_${SET}_${TAG}.csv
REF=A0; case ,$CONFIGS, in *,A0,*) ;; *) REF=${CONFIGS%%,*} ;; esac
echo
"$PY" analysis/summarize_eval.py "$PLAT" "${SET}_$TAG" --ref "$REF" --configs "$CONFIGS" --csv "$OUT"
# published summaries: dev120 as in docs/report.md §2.1; smoke24 = the same published runs restricted to the smoke24
# samples (summarize_eval.py <platform> dev120 --ids benchmark/manifests/smoke24.tsv)
case $SET:$PLAT in
  dev120:jetson) PUB=results/summary/dev120_jetson_maxn.csv ;;
  dev120:*) PUB=results/summary/dev120_$PLAT.csv ;;
  *) PUB=results/summary/repro_ref/${SET}_$PLAT.csv ;;
esac
echo; echo "== against the published numbers ($PUB)"
[ -f "$PUB" ] || { echo "no published summary for $PLAT / $SET"; exit 0; }
"$PY" analysis/compare_repro.py "$OUT" "$PUB"
