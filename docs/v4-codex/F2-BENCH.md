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

## Host invocation and evidence

The host client is `opentpu/c2_f2.py`; the isolated Linux transport is
`opentpu/host/c2_f2.py`. The transport locks the same canonical application-PF BDF
as the production F2 transport before opening any BAR. It verifies vendor/device,
benchmark ID, ABI and expected build ID before any write or BAR4 mapping. BAR4
uses a native WC memcpy, MFENCE and uncached read completion. An additional full
readback compares all uploaded HBM bytes before the measured run. This driver
does not load images, unbind drivers or change PCI configuration.

Generate inputs without accessing hardware:

```sh
PYTHONPATH=. python3 tools/c2_f2_bench.py prepare \
  --fixture build/c2-f2/rows31-kb9.npz --rows 31 --kb 9 --seed 20261019
```

Once the separate benchmark build has passed its physical checks and the intended
image is already loaded, use the actual application-PF BDF, `inputs.json` from
that build, and its eight-digit hexadecimal build ID:

```sh
PYTHONPATH=. python3 tools/c2_f2_bench.py run \
  --fixture build/c2-f2/rows31-kb9.npz \
  --bdf 0000:34:00.0 --expected-build-id 0xBUILD_ID \
  --build-manifest /path/to/qualified-build/inputs.json \
  --out build/lens/c2/f2-rows31-kb9 --repeats 3
```

The BDF and `BUILD_ID` above are placeholders, not an instruction to target a
particular running FPGA. Every run uses a new output directory. The script checks
the manifest identity against the requested device build ID and records the full
manifest and its hash; it does not establish that an AFI was correctly built or
loaded. Root's build/deployment checks provide that separate evidence.

Each repetition runs both modes, reversing order on alternating repetitions.
It checks every fp32 result and signed48 integer diagnostic against canonical MM,
all group counts and per-PC accepted/returned beat counts, the native credit
bound, stable DONE state and no device error. Timeout requests ABORT and waits
for native response drain before returning an error. A run at least 2^32 cycles
is rejected because its 32-bit stall counters could wrap. Input bytes cannot be
changed after preload. Failed checks never become Lens board profiles.

Successful runs retain `result-<repeat>-<mode>.json`, host/build provenance,
`c2-f2.otpuprof`, and standalone `c2-f2.html`. The profile's `board` field contains
the full per-PC counters and exact result bits. The native-interface bandwidth
floor uses maximum per-PC returned beats at at most one 64-byte beat per core
cycle; it is not an HBM peak claim. Stall counters can overlap and cannot be
summed to obtain a critical path. There is no fabricated instruction timeline.

## CSR ABI 1

All accesses are aligned 32-bit words with full write strobes. Unsupported accesses
return SLVERR. MEM_ADDR increments only after an accepted MEM_WDATA write or
successful MEM_RDATA read. Configuration and preload writes require idle state.

| Byte offset | Register |
| --- | --- |
| 000 / 004 / 008 | ID `0x43324d42` / ABI `1` / BUILD_ID |
| 00c | STATUS: busy, done, error, aborted, draining in bits 0..4 |
| 010 | CTRL: START=1, ABORT=2, CLEAR=4 |
| 014 / 018 / 01c / 020 | MODE 0 legacy M1 or 1 decode / KB / ROWS / BASE_BEAT |
| 024 / 028 / 02c | MEM_ADDR word index / MEM_WDATA / MEM_RDATA |
| 030 / 034 / 038 / 03c | ERROR / RESULT_COUNT / CYCLES low / high |
| 040 / 044 / 048 / 04c | Accepted groups / committed groups / issued / returned native beats |
| 050 / 054 / 058 / 05c / 060 / 064 | Core advances / native blocked / weight wait / scale wait / ACT wait / output wait cycles |
| 068 / 06c / 070 / 074 / 078 / 07c | Peak outstanding / CORE_KHZ / MAX_ROWS / resident depth / credits per PC / HBM ready |
| 100 + PC*20 | Per-PC block: issued, returned, peak outstanding, blocked cycles, pending-without-response cycles, credit-wait cycles at offsets 0,4,8,c,10,14 |

Addresses in the table are hexadecimal. ERROR bits 0..3 mean invalid START config,
native fabric error, unexpected response and internal reservation overflow. Native
fabric errors can remain sticky; CLEAR is not a substitute for image recovery.
Result data is indexed by global logical row, so completion grouping does not
change host order. Counters freeze at DONE; cycles are 64-bit and other counters
are 32-bit.

MEM_ADDR write windows: ACT `0x00000 + block*32 + word` (32 packed int8x4 words per
128-byte block), DA `0x10000 + block` (four original uint8 offsets), SA `0x11000`
(original fp32 bits), DW `0x20000 + row*256 + block` (low byte), SW `0x40000 + row`
(original fp32 bits). Read windows: RESULT `0x50000 + row`, signed48 integer low32
`0x60000 + row`, high16 `0x70000 + row`. Preload windows are write-only; results
are read-only. There is no requantization or numerical format conversion.

Offline tests exercise packing including tails and KB 1/9/54/255, reference equality,
fixture hashes, identity rejection, repeated mode changes, corrupted results and
counters, timeout abort and counter-wrap rejection. These tests use an injected
ABI fake and cannot be exported as hardware evidence. Actual F2 host execution,
OCL/PCIS behavior and integrated throughput remain unverified until the FPGA run.
