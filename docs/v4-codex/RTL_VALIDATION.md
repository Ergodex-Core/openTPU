# C3 pipelined RTL milestone

The user approved continued implementation of C1, C2 and C3 and required openTPU
Lens for performance measurement and visualization. This milestone adds an isolated
SystemVerilog vocabulary engine and cluster. It does not advertise a runtime capability,
change greedy generation, or deploy anything to F2.

## Implemented boundary

The [engine](../../rtl/c3/otpu_c3_engine.sv) accepts a stream of complete, ordered rows,
each containing `cmd_kb` blocks of 128 signed int8 weights. Its stages are:

1. 128 registered signed products, followed by five balanced reduction levels to four
   32-element group sums.
2. The exact `16 - dw - da` group shift, dropping negative shifts; a balanced sum and
   signed 48-bit row accumulator. New blocks can enter every enabled cycle.
3. The [epilogue](../../rtl/c3/otpu_c3_epilogue.sv): the existing MXU's four-stage
   integer-to-fp32 conversion and two repository `otpu_fmul` stages. Original `Sa` is
   multiplied by `2^-16` in parallel using the same fp32 implementation. Rounding,
   underflow, signed zero and NaN behavior are unchanged.
4. Local greedy reduction using the ISA's flushed-bit `fkey` ordering. Equal keys choose
   the lowest global token ID. An independent diagnostic logit stream exposes every row.

The [cluster](../../rtl/c3/otpu_c3_cluster.sv) supports one or four instantiated engines.
The [global reducer](../../rtl/c3/otpu_c3_reduce.sv) accepts independent or simultaneous
shard completions, masks empty shards, accepts each expected shard once, and retains the
final result until its consumer accepts it. The small-vocabulary validation layout now
accepts one or more rows so empty-shard behavior can be exercised; production eligibility
still requires the original Gemma 3 dimensions.

Diagnostic backpressure freezes all arithmetic and metadata registers together, including
the epilogue. It propagates to the input stream without losing a row or block. A final
blocked result cannot be overwritten by a new command. Consecutive commands need no reset.

## Interface contract and integration work

- Load every activation block and its four five-bit `da` fields while idle before accepting
  a command. `cmd_sa` is the original fp32 activation scale. The activation snapshot uses
  inferred arrays in this prototype; BRAM mapping and routing have not been established.
- Each engine receives contiguous global IDs beginning at `cmd_base`, exactly `cmd_rows`
  rows, and exactly `cmd_kb` ordered blocks per row. `Sw` is sampled on the first block;
  each block carries its original five-bit `dw`. Legal IDs fit the existing 24-bit token
  domain. There is no sampling, penalty application, ACC, or quantization change.
- The future sidecar adapter owns PC addresses, header reads, outstanding requests,
  burst/page splitting, reordering and reassembly. The existing Python layout remains its
  reference. This RTL milestone does not implement an HBM/AXI controller.
- Reset discards every in-flight valid, partial dot, winner and completion mask. Data RAMs
  need not be cleared, but activations must be reloaded. **The upstream adapter must flush
  old responses on reset**: this ordered block stream has no generation tag with which to
  reject stale responses arriving after a new command.
- In the eventual greedy path, the diagnostic sink can stay ready and discard logits;
  full-logit runtime requests still use the existing fallback. No capability is advertised
  and no production caller dispatches this standalone RTL.

The next integration milestone is the actual sidecar memory adapter and command/capability
handoff, followed by greedy-loop EOS, cancellation, reset, bucket chaining and baseline
fallback tests. Synthesis, placement, timing, shared-memory contention, full-checkpoint and
whole-token measurements remain outstanding.

## Validation

The [RTL tests](../../tests/test_vocab_rtl.py) compare every diagnostic logit's fp32 bits
against executed canonical ISA MM programs. They cover one/four engines, KB=1/9/28,
independent input and output stalls, positive/negative scales, extreme offsets, dropped groups,
uneven tails, empty shards, repeated commands, reset during input/arithmetic/blocked-output
phases, and stable backpressured final results. A slow shard is rotated across all four
engines to vary actual completion order and verify cross-engine ties and signed zero.

The standalone epilogue test exercises signed48 endpoint values and RNE ties at every
relevant exponent, random 48-bit sums, random fp32 scales, signed zeros, denormals,
infinities and NaNs. The standalone reducer test exercises every one of the 24 completion
orders, simultaneous completions, every nonempty shard mask, nonmonotonic global IDs,
NaN payload ordering and FTZ ties. The reference tests continue to verify dispatch fallback
for sampling, penalties, missing/wrong capabilities, full logits and unsupported layouts.

```sh
PYTHONPATH=. python -m pytest -q tests/test_vocab_engine.py tests/test_vocab_rtl.py
PYTHONPATH=. python tools/c3_rtl_check.py --output build/lens/c3-rtl
python -m opentpu.lens open build/lens/c3-rtl/c3-head-rtl.otpuprof
```

These are candidate synthesizable SystemVerilog datapaths with Verilator execution evidence,
not FPGA synthesis evidence. A local sv2v/Yosys `proc; opt; check -assert` attempt reports
undriven intermediate function-return bits. The same failure reproduces with the unchanged
repository `otpu_fmul` primitive alone (`build/c3-elaboration/baseline-fmul.log`), so this
converter path does not establish a clean synthesis result. Vivado and a Yosys SystemVerilog
frontend supporting the original source were not available. The working arithmetic was
preserved; there is no achieved clock or resource-fit claim.

Validation result: **24 new RTL tests and 42 existing reference tests passed**. The standalone
epilogue covers 1,757 vectors; the reducer covers 915 completion-order/mask cases. The Lens
serialization/render test was rerun after the final interval-accounting update. Verilator
also elaborates the standalone cluster in lint mode; remaining diagnostics are bounded array
index width casts and unused/package naming warnings, with no reported latch or combinational
loop warning. The simulation validates the documented KB bound before accepting commands.

## Executed Lens evidence

The [export tool](../../tools/c3_rtl_check.py) executes a synthetic 1,024 x 1,152 head,
seed 3. Every logit matches the canonical ISA; all four scenarios choose token **73**
with logit bits **0x4004fce0**. Each run consumes 1,024 headers and 9,216 weight blocks,
executes 9,216 signed48 block accumulations and 1,024 fp32 epilogues, and retires exactly
1,024 logits. The sidecar service accounts for 1,212,416 bytes.

| Standalone RTL scenario | Simulated cycles | Accepted input beats overlapping dot retirement, summed across engines |
|---|---:|---:|
| One engine, unstalled service | 10,300 | 8,184 |
| Four engines, unstalled service | 2,629 | 8,160 |
| Four engines, 20% random service stalls | 3,262 | 6,594 |
| Four engines, service stalls plus 35% diagnostic stalls | 3,395 | 6,628 |

The four-engine unstalled run also accepts **8,132 input beats while an earlier row's fp32
epilogue is active**. These counts come from simulated handshakes and stage events. They
demonstrate pipeline overlap that the earlier serial accounting model could not represent.
The last scenario records 457 input-backpressure cycles and 549 blocked diagnostic-output
cycles, summed across engines; those counts overlap and must not be added as latency savings.

The synthetic service has a one-cycle 32-byte header, then 128-byte weight blocks, initial
latency 30 cycles and a fixed 0/3/6/9-cycle engine start skew. It does not model HBM queues,
PC arbitration, physical burst widths or request/response latency. Activation snapshot
loading, actual pipeline fill/drain and global reduction are included; QACT, transformer
layers, embedding lookup and TMEM logit stores are outside the boundary. Consumer hold
cycles are separately reported; the table's result consumer is always ready.

The bundle `build/lens/c3-rtl/c3-head-rtl.otpuprof` contains these four **RTL** profiles and
the old **model** profile, visibly labeled. The HTML shows concurrent row-service, integer,
fp32, local-reduction and global-completion intervals derived from events. Intervals include
bubbles/backpressure, not only busy arithmetic clocks. The generic Lens `S` denotes head
engines, not complete TPU slices or a physical floorplan. The 200 MHz clock is only a display
assumption. Comparing these isolated head runs does not establish a full-token or F2 speedup,
and comparing the old model's cycles to RTL does not establish an architectural speedup.

`validation.json` records settings, exactness, tool version and source hashes. Compressed
`events-0.json.gz` through `events-3.json.gz` preserve all observed service/stage/completion
events; `input.npz` preserves the exact synthetic input for reruns. All generated files are
under the ignored build directory.
