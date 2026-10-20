#!/bin/bash
# sim/tech/tb_tech_pk.sv under Vivado's simulator with a cell family (generic | xilinxus: the
# unisim DSP48E2), one run a seed; the trace lines (cycle P PCOUT ...) on stdout.
#   sim/tech/run_xsim_pk.sh <family> <seed> [<seed> ...]
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
fam=${1:?family}; shift
[ -n "$XILINX_VIVADO" ] || source /opt/Xilinx/2025.2/Vivado/settings64.sh >/dev/null
W=$(mktemp -d)
cd "$W"
xvlog -sv "$ROOT/sim/tech/tb_tech_pk.sv" "$ROOT/rtl/tech/$fam/otpu_tech_dsp.sv" \
  "$XILINX_VIVADO/data/verilog/src/glbl.v" > xvlog.log 2>&1 || { cat xvlog.log >&2; exit 1; }
xelab -L unisims_ver tb_tech_pk glbl -s snap > xelab.log 2>&1 || { cat xelab.log >&2; exit 1; }
for s in "$@"; do
  echo "SEED $s"
  xsim snap -R -testplusarg "seed=$s" 2>&1 | grep -E '^[0-9]+ '
done
cd / && rm -rf "$W"
