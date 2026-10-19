# openTPU v5: the full HBM on the F2 (docs/full_hbm.md)

Status: branch `full-hbm` (worktree openTPU-fullhbm from wide-port-alexshell 71919ae2), design
`docs/superpowers/specs/2026-10-20-opentpu-design-v5.md`, Verilator only: no synthesis, no
board. Labels: **measured** = a Verilator run of this tree's RTL, bit-exact with the ISA
simulator (`--check`); **lpn-estimate** = `tools/wideport_lpn.py`; **estimated** = arithmetic
from the v4 synthesis figures. int8 first, int4 secondary.

## What changed

v4 (`docs/wide_port.md`, `docs/f2_hbm.md`) streams one slice's port B at 512 B a cycle over 8 of
the 32 HBM pseudo-channels through per-channel SmartConnect chains, and measured 83.5 tok/s int8
on the RTL (2,394,257 cycles a token) at 0.865 beats a cycle a channel. v5 uses all 32 channels:

- **S = 4 slices** (`f2-hbm:v5`; `f2-hbm:v5s3` three), each the v4 slice at PORTB 4 / four int8
  row lanes / MCOLS 4 / ALINE 64 / LANES 8 on its own 8 pseudo-channels, one slice per SLR and
  two in SLR2; 2,048 B a cycle = 409.6 GB/s at 200 MHz (96% of AWS's measured 426 GB/s read),
  2,048 int8 MACs a cycle. The slice's BRAM is trimmed so two fit an SLR: the MXU's chunk FIFO
  256 entries of 4 chunks (was 1024), ACT RAM 8 rows (16), the trace 4,096 records (16,384).
- **Gemma 3 sharded over the slices** (opentpu/llm/gemma4.py; the spec's section 4): the query
  heads, the K / V projection rows (all-gathered before the norm and RoPE, the cache
  replicated), wo, gate / up, the down projection and the LM head by rows; six GATHERs a layer
  (k, v, the heads' outputs, wo's result, the MLP's activation chunks, the down's result).
  The MLP's width is padded to a multiple of S x D (6912 -> 7168 on four slices, +3.7% of the
  MLP) and chunked in the largest chunk that divides a slice's share (896 on four, 1152 on two
  or three: `_mlp_chunk`), since every chunk is six MMs and a gather on every slice. The
  programs are the same on every slice but for the heads a slice attends with and its LM-head
  columns; the images hold each slice's rows at the same addresses.
- **A 450 MHz engine per HBM port** (rtl/boards/f2/otpu_hbm_ch.sv, otpu_f2_hbm_eng.sv; the
  spec's section 3): the native port's commands into an asynchronous FIFO, a registered AR / AW
  generator coalescing consecutive beats into bursts of up to 8 (512 B, never across a 512 B
  group; the `cont` bit computed at the core clock), 64 bursts outstanding, direction barriers,
  the read data through a 256-beat BRAM asynchronous FIFO to the core clock; the adapter's reads
  in flight raised to 128 a channel (`NRD`). The host keeps AWS's cl_axi_sc_1x1 chain: on its own
  port at S = 3, through a registered 2 x 1 AXI3 mux in front of port 31 at S = 4. Per slice the
  channels' write beats share one 580-bit bus across the SLRs (the SLL budget, section
  "Utilization").
- **The receipt's memory model** (sim/verilator/otpu_hbmeng_mem.sv, `OTPU_HBMENG=1`): the
  engine's behaviour at the core clock behind the three-stage SLR chains, the HBM at 280 ns
  (`--lat 56` at 200 MHz) with 64 bursts outstanding a channel and the port's rate.

## The numbers

### ISA simulator: the token-exact gate

`tests/test_full_hbm.py`: a tiny random Gemma 3 on 2 and 4 slices against one (bit-identical
logits where the MLP's chunking is the same, int8-level agreement and the same token where it
is not), and the real Gemma-3-1B int8 decode step at position 9 on 4 slices against one (the
same token; the logits within the activation quantization's noise). The S = 1 programs of this
tree are word-identical to the baseline tree's (the receipt's step: 678 instructions, the same
sha256) so every v4 number is reproducible here.

### LPN (lpn-estimate; tools/wideport_lpn.py with --slices, the collective timed from otpu_coll's structure)

Gemma-3-1B int8 / int8 head, 26 layers + head, PORTB 4 / MCOLS 4 / RL 4 / ALINE 64, NCH 8 a
slice, --lat 60, 200 MHz, the native path's 14 cycles of adapter latency, RD 128 beats in
flight a channel (the v4 point at RD 64 for reference):

| configuration | cycles (lpn-estimate) | tok/s | binds | MLP / head / attention |
|---|---|---|---|---|
| v4: S 1, RD 64 (docs/wide_port.md; measured 2,394,257) | 2,441,652 | 81.9 | channel at 0.865 | 1,462,050 / 723,866 / 255,736 |
| S 1, RD 128 | 2,156,871 | 92.7 | channel at 1.0 | 1,271,202 / 630,947 / 254,722 |
| S 2, RD 128 | 1,159,284 | 172.5 | channel | 657,137 / 315,571 / 186,576 |
| **S 3, RD 128 (`v5s3`)** | **883,739** | **226.3** | channel | 451,502 / 210,456 / 221,780 |
| **S 4, RD 128 (`v5`)** | **673,252** | **297.1** | channel (508,631) | 366,337 / 157,883 / 149,032 |

What the model says about S = 4: the port-bound token is 506K cycles (the slice's 1,957,551 chunks
at 4 a cycle); the MLP phase sits 15% over its roofline (the two gathers a layer of 3,584 words
and the twelve MMs' tails), the head 6% over, and the **attention phase is 149K cycles against
a 40K roofline**: per layer 5.7K cycles of which 1.5K stream weights; the rest is the VPU's
softmax and norms (replicated on every slice: they do not shrink with S), the small MMs' tails
and the four gathers (k, v, o, y: ~460 cycles). It is 22% of the token where it was 12% of
v4's: the next lever after the memory path is the attention's VPU work (ULANES 16, or the
composite functions pipelined), not the port. The 4-layer + head point at S = 4: 264,302.

### RTL (measured): to be filled from the Verilator receipts

(The configurations table with RTL cycles / tok/s / LPN / Lens roofline %, from
`tools/perf_qwen.py --slices S` on the engine model and on the native model, with the v4
numbers 949,550 / 2,394,257 beside them.)

## Utilization (estimated per SLR)

(The table from the spec's section 6, refined.)

## What still blocks an FPGA build

(To be filled.)
