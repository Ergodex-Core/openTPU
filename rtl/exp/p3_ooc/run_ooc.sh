#!/bin/bash
# v4 P3: the OOC runs of the MXU on the F2's part, one at a time (rtl/exp/p3_ooc/ooc_mxu.tcl).
#   rtl/exp/p3_ooc/run_ooc.sh <outdir> <name>:<period>:<GEN=V,GEN=V...> ...
# e.g. lane-pk:5.0:C=1,PACK=1,E=1 lane-up:5.0:C=1,PACK=0,E=1. A step whose progress line says
# DONE <name> rc=0 is skipped.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
O=${1:?outdir}; shift
mkdir -p "$O"
source /opt/Xilinx/2025.2/Vivado/settings64.sh
PART=${PART:-xcvu47p-fsvh2892-2-e}
for s in "$@"; do
  n=${s%%:*}; r=${s#*:}; per=${r%%:*}; g=${r#*:}
  if grep -q "^DONE $n rc=0" "$O/progress.txt" 2>/dev/null; then continue; fi
  echo "START $n $PART $per $g $(date -u '+%F %T') MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB" >> "$O/progress.txt"
  t0=$(date +%s)
  (cd "$O" && timeout 14400 vivado -mode batch -nojournal -log "$O/$n.log" \
     -source "$ROOT/rtl/exp/p3_ooc/ooc_mxu.tcl" -tclargs "$O/$n" "$PART" "$per" ${g//,/ } > "$O/$n.out" 2>&1)
  echo "DONE $n rc=$? $(( $(date +%s) - t0 ))s" >> "$O/progress.txt"
  grep -h "^OTPU_" "$O/$n.out" >> "$O/progress.txt"
done
echo "EXIT $(date -u '+%F %T')" >> "$O/progress.txt"
