# Reproducible OpenTPU HBM2 baseline

The local simulator implements OpenTPU instruction timing with a real pinned Ramulator HBM2
backend. It reuses the pinned ISA/compiler and preserves the original RTL, LPN and goldens.
The supported baseline is **one slice, program-resident execution, replicated TMEM
(`TMEM_IMPL=0`), MCOLS4/LANES8, and either two or four MXU epilogue lanes**. WAITC,
DSTEP/STREAM, quantization, vector/reduction/nonlinear pipelines and finite-buffer memory
backpressure are modeled. No sparsity speedup is assumed.

This is a compute-RTL baseline with an explicitly shared native memory controller, not an
FPGA shell. [Consolidated coverage](../../tools/perfsim/receipts/2026-10-19-round6/coverage/coverage.md)
separates current cumulative observations, fresh RTL runs and pytest replays. The machine-readable
matrix includes source hashes and retains numerical failures.

## Run locally

Use Python with NumPy/pytest, CMake, a C++20 compiler and Verilator. Build the pinned Ramulator
library using [the existing instructions](README.md); omit `--fetch` after populating the exact
source commits for an offline build. No paid resources, credentials or model downloads are used.

```sh
# Use .so on Linux; choose a new output directory for each invocation.
BRIDGE=/absolute/path/to/libopentpu_ramulator.dylib
python tools/perfsim/run_baseline.py --bridge "$BRIDGE" \
  --out /tmp/opentpu-hbm-baseline --build-rtl
OTPU_RAMULATOR_BRIDGE="$BRIDGE" python -m pytest tests/perfsim -q
```

The runner compiles a seeded complete tiny Qwen3.5 hybrid topology, builds the explicit RTL
configuration, and checks two-token batched prefill plus two persistent decode launches.
`report.json` is the short result; `commands.json`, `manifest.json`, `rtl-check/` and `metrics/`
contain reproduction inputs, instruction/memory traces and separate numerical/timing verdicts.
Use `--engine dstep` for DSTEP, `--epilogue 4` for E4, or `--rtl-executable PATH` to reuse a
parameter-checked persistent build. `--model-only` is explicitly labeled an unvalidated prediction.

The E2 STREAM baseline matches **25,392 / 16,632 / 16,806 cycles** and independent NumPy
logits. At the stated 200 MHz core, its device TTFT is **126.96 μs**, with **11,956.72 conditional
decode tokens/s** for this tiny random workload and eight-cycle inter-launch gaps. These are
RTL-simulation/model results, not physical hardware performance or full-model throughput.

## Source-backed boundaries

| Path | Required here? | Evidence and treatment |
|---|---|---|
| WAITC, DSTEP/STREAM | Yes | Pinned Qwen3.5 compiler emits them; all are implemented and compared to RTL. |
| Program loader | No, excluded by the declared boundary | `rtl/top/otpu_slice.sv:20` loads DRAM instructions while reset is held. `tools/perfsim/rtl/persistent_body.sv:19` writes IMEM before deasserting reset. Loading is outside core cycles; no zero-cost loader claim is made. |
| HALT CHAIN | No | `rtl/top/otpu_slice.sv:305` drains, resets and reloads through port B. Chosen bundles end with ordinary HALT. CHAIN is explicitly rejected; the harness's host reset gap is not a CHAIN model. |
| WAITW | No | `rtl/dma/otpu_dma.sv:690` polls a word against a reference. The chosen bounded bundles contain no WAITW and have no asynchronous host writer. It is explicitly rejected, not assigned a latency. |
| Pumped TMEM | No | `sim/verilator/tb_top.sv:30` defaults to replicated `TMEM_IMPL=0`; the card selects IMPL3 and pumping in `targets/ypcb-00338.toml`. HBM2 does not mandate a TMEM topology. IMPL1/2/3 are explicitly rejected by this timing model. |
| MXU E4 | Optional supported compute variant | `rtl/mxu/otpu_mxu.sv:107` independently selects epilogue lanes. Default E0 means MCOLS/2; explicit E4 now passes hybrid and three/four-row drain comparisons. This does not validate its composition with pumped TMEM. |
| DDR3/LiteDRAM shell | No | A separate card subsystem; no new DDR3 equivalence work or claim is included. Historical DDR4 evidence remains visible. |

`opentpu/perfsim/baseline.py` explicitly supplies every compute parameter to the derived RTL
build. Ambient `OTPU_UARCH`, `OTPU_TMEM_IMPL`, `OTPU_MXU_E` and related settings cannot silently
change this contract. Both comparators reject mismatches, including topology, epilogue width,
quantizer buffer, composite lanes and AXI mode. Older receipts resolve omitted parameters to
the pinned testbench defaults; they are not assumed to use the card target.

Full-row epilogue tests make the E2/E4 distinction observable: compiled shapes
3×128×19, 4×128×67 and 4×256×33 measure **296/457/577 cycles at E2**, versus
**277/390/507 at E4**. Every instruction and native memory event matches, and independent
NumPy numerical checks pass. Hybrid programs use at most two rows, so unchanged hybrid totals
alone would not have established E4's four-lane drain behavior.

## Limits and preserved failures

The two original int4 float-quality checks still fail: **5.77% at D32 and 7.95% at D128**.
Independent packed arithmetic matches RTL bitwise. Neither verdict replaces the other; see
[the diagnosis](validation-round5.md#independent-int4-diagnosis).

A full pretrained-model run remains a next phase. The pinned baseline lacks Gemma3/Bonsai
frontends; the deliverable uses no checkpoint and claims no completed pretrained inference.
Loading, sampling, communication and host time are excluded. S>1/GATHER, pumped TMEM and
production shell integration remain unsupported. Fmax and resource fit require physical
implementation. Existing LPN errors apply only to its five registered configurations.
