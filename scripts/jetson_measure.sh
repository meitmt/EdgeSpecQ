#!/usr/bin/env bash
# Run one measurement command on the Jetson in the state used for the published numbers, then put the board back:
#   - desktop stopped (systemctl isolate multi-user.target; about 2-3 % faster and ~1 GB more free memory)
#   - MAXN_SUPER (nvpmodel -m 2) + jetson_clocks, unless --25w (factory-default mode, clocks untouched)
# On exit, also on Ctrl-C / SIGTERM, the previous power mode, clocks (jetson_clocks --store/--restore) and default
# systemd target are restored and the sudo credential cache is dropped.
#   jetson_measure.sh [--25w] <command...>
#   e.g. jetson_measure.sh python3 regression/run_eval.py jetson ~/mt_test/liquid_ai/eval/dev120.jsonl B0,B11 --reps 2
# sudo asks for the password once (or uses $SUDO_ASKPASS if set); a background `sudo -v` keeps it valid.
set -u
MAXN=1
[ "${1:-}" = --25w ] && { MAXN=0; shift; }
[ $# -gt 0 ] || { sed -n '2,9p' "$0"; exit 2; }
SUDO=sudo; [ -n "${SUDO_ASKPASS:-}" ] && SUDO="sudo -A"
ORIG_MODE=$(nvpmodel -q 2>/dev/null | sed -n 2p)
ORIG_TARGET=$(systemctl get-default)
SNAP=$(mktemp)
$SUDO -v || exit 1
( while sleep 120; do sudo -n -v 2>/dev/null || exit; done ) & KEEP=$!
CHILD=
restored=0
restore() {
  [ $restored = 1 ] && return; restored=1
  if [ $MAXN = 1 ]; then
    $SUDO nvpmodel -m "$ORIG_MODE" < /dev/null > /dev/null 2>&1
    $SUDO jetson_clocks --restore "$SNAP" > /dev/null 2>&1
  fi
  $SUDO systemctl isolate "$ORIG_TARGET" < /dev/null
  echo "jetson_measure: restored $(nvpmodel -q 2>/dev/null | head -1), target $ORIG_TARGET" >&2
  kill $KEEP 2>/dev/null; rm -f "$SNAP"; sudo -k
}
trap restore EXIT
trap '[ -n "$CHILD" ] && kill $CHILD 2>/dev/null; wait; exit 130' HUP INT TERM

$SUDO systemctl isolate multi-user.target < /dev/null; sleep 10
if [ $MAXN = 1 ]; then
  $SUDO jetson_clocks --store "$SNAP" > /dev/null
  $SUDO nvpmodel -m 2 < /dev/null > /dev/null 2>&1
  $SUDO jetson_clocks > /dev/null 2>&1
fi
echo "jetson_measure: $(nvpmodel -q | head -1); $(free -m | awk '/Mem:/{print $7}') MiB available" >&2
"$@" & CHILD=$!
wait $CHILD
