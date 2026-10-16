#!/bin/bash
# The MXU packing study's out-of-context runs (docs/mxu_packing.md 5), in order, one at a time:
# one packed physical column and one unpacked (tools/mxu_pk_ooc.tcl) at 5.0 ns (200 MHz) and
# 4.0 ns (250 MHz) on the F2's xcvu47p. On a build host with Vivado (tools/tourney/remote.py's
# opentpu or omarchy: ~/.local/bin/vivado; never on the Mac), from the tree's root:
#   tools/mxu_pk_ooc.sh <outdir> [step ...]
# steps (default: all four): pk5 pk4 un5 un4 (packed / unpacked at 5.0 / 4.0 ns). Each step's
# reports go to <outdir>/<step>/ (util.txt, timing.txt, timing_summary.txt, routed.dcp), its
# Vivado log to <outdir>/<step>.log, its OTPU_* summary lines to <outdir>/progress.txt. A step
# whose progress line says DONE rc=0 is skipped, so the script can be rerun.
#   tools/tourney/synth.parse_ooc(log, util, period) reads a step's vivado.log and util.txt.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
O=${1:?outdir}; shift
STEPS=${*:-pk5 pk4 un5 un4}
mkdir -p "$O"
export PATH=$HOME/.local/bin:/opt/Xilinx/2026.1/Vivado/bin:$PATH
viv() {   # step pack period
  local n=$1 pack=$2 period=$3
  if grep -q "^DONE $n rc=0" "$O/progress.txt" 2>/dev/null; then return; fi
  echo "START $n $(date '+%F %T')" >> "$O/progress.txt"
  local t0=$(date +%s)
  mkdir -p "$O/$n"
  (cd "$O/$n" && timeout 10800 vivado -mode batch -nojournal -log "$O/$n/vivado.log" \
     -source "$ROOT/tools/mxu_pk_ooc.tcl" -tclargs "$O/$n" "$pack" "$period" > "$O/$n.out" 2>&1) \
    && rc=0 || rc=$?
  echo "DONE $n rc=$rc $(( $(date +%s) - t0 ))s" >> "$O/progress.txt"
  grep -h "^OTPU_" "$O/$n.out" >> "$O/progress.txt" || true
}
for s in $STEPS; do
  case $s in
    pk5) viv pk5 1 5.0 ;;
    pk4) viv pk4 1 4.0 ;;
    un5) viv un5 0 5.0 ;;
    un4) viv un4 0 4.0 ;;
    *) echo "unknown step $s" >> "$O/progress.txt" ;;
  esac
done
echo "EXIT $(date '+%F %T')" >> "$O/progress.txt"
