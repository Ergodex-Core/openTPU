# C2 F2 synthesis and benchmark gates

The user approved this hardware continuation. The first gate is isolated synthesis
of `otpu_c2_mxu`, followed by optional placement/routing. It creates no AFI, changes
no FPGA slot, and produces no hardware performance measurements.

## Frozen out-of-context build

From this worktree, after the desired RTL passes its simulation tests:

```sh
python3 tools/c2_ooc.py --out build/c2-ooc/f2-200 --prepare-only
```

Copy that entire output directory to an existing Vivado 2025.2 build host. In the
copied directory:

```sh
source /opt/Xilinx/2025.2/Vivado/settings64.sh
python3 run.py --out . --run-prepared
python3 run.py --out . --run-prepared --stage route
```

The second command synthesizes; the third reuses the verified synthesis checkpoint
and routes it. A fresh `--stage route` snapshot can also run both stages at once.
Every attempt has its own directory; a failed attempt is retained rather than
overwritten. The default is four Vivado threads. No HDK checkout, shell checkpoint,
AWS access, or FPGA device is needed.

The top is `otpu_c2_mxu`, with `D=128`, `C=4`, `DEPTH=256`, `RF=8`, on
`xcvu47p-fsvh2892-2-e`. The source list includes the repository FP32 pipeline,
MXU/skew dependencies, and **Xilinx UltraScale+ DSP48E2 implementation**:

- `rtl/vpu/otpu_fp.sv`
- `rtl/vpu/otpu_fpipe.sv`
- `rtl/top/otpu_pkg.sv`
- `rtl/tech/xilinxus/otpu_tech_dsp.sv`
- `rtl/mxu/otpu_mxu.sv`
- `rtl/mxu/otpu_c2_mxu.sv`

`inputs.json` records exact SHA256 hashes, repository commit/dirty status, part,
tool version, parameters, thread limit, and constraints. The source and flow files
are copied, so moving the worktree cannot change a prepared build. The runner
verifies hashes before and after Vivado and refuses checkpoint reuse if its bytes
or input manifest changed. Vivado must report version 2025.2.

The explicit interface budget is a 5 ns clock, 0.5 ns maximum / 0 ns minimum input
and output delays, and 0.1 ns setup / 0.05 ns hold uncertainty. There are no false
paths or multicycle exceptions. These are standalone assumptions, not the future
HBM adapter or shell timing constraints.

Each stage saves a DCP; flat/hierarchical utilization; global and register-to-register
setup/hold paths; timing summary including unconstrained paths; clocks; DRC;
methodology; and `check_timing`. Machine-readable `metrics.tsv` and `outputs-*.json`
include DSP/BRAM/LUT/FF counts, slack and artifact hashes. Synthesis slack is an
estimate. Route returns failure if reported setup/hold slack misses the 200 MHz
target, preserving the reports. A nonnegative slack flag alone is insufficient:
review unconstrained paths, DRC and routing status before claiming an OOC pass.
Even an OOC pass does not establish integrated shell timing or an achieved F2 clock.

## Native HBM microbenchmark scope

The planned isolated benchmark CL uses the frozen F2 shell/PCIS/native HBM seam
from `tpubench` commit `67dcf0c8`. It has its own identity and CSR ABI; it must not
impersonate the production TPU. It compares matched M=1 legacy and C2 decode runs
over identical int8 weight bytes and original `Sw` / `dw` metadata. ACT and scales
are resident before the measured window. Both paths retain bit-exact logits in a
local result buffer for host validation. This is an isolated matrix benchmark,
not full-token inference, canonical-image compatibility, or a generic HBM test.

The agreed v1 layout assigns row `r` to PC pair `2*(r mod 4)` and the following PC.
The lower/upper 64 bytes of each 128-byte block use local native beat
`BASE_BEAT + floor(r/4)*KB + k`. Host PCIS byte offsets are
`PC*512 MiB + 64*local_beat`. Legacy selects each row's actual pair and feeds core
column 0; decode groups up to four rows. Thus both paths read exactly
`ROWS*KB*128` weight bytes from the same physical addresses. V1 supports
`ROWS=1..512`, `KB=1..255`, one activation snapshot and shared `Sa` / block `da`;
the core's M>1 tests remain separate from this M=1 benchmark.

START-to-final-commit cycles and native per-PC accepted/returned beat counters are
device measurements. Host preload and result readback remain outside that window.
The host must check identity/version, completion/error state, result count, every
output bit, and expected byte counts before exporting openTPU Lens evidence.
Counter-derived profiles must say that they contain measured aggregate counters,
not an invented event timeline. Hardware evidence remains pending until the actual
benchmark image is built, loaded, and run; no synthetic results are substituted.
