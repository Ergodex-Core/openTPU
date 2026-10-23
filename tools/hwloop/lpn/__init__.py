"""The LPN (labelled Petri net) tier of the hardware loop: MJChku/LPNDialect's examples/opentpu
model of this slice, run on the program and DRAM image the frozen export's compiler hands the
RTL testbench, graded against the Verilator receipts (tools/hwloop/lpn_run.py, docs/hwloop.md
"The LPN stage", tools/hwloop/lpn/README.md).

Only the standard library here; the two helper scripts (inputs.py, rtl_pin.py) run under the
venv python inside an openTPU tree's context and are started as subprocesses.
"""

# upstream's pin of this repository: examples/opentpu/__init__.py REVISION (FeSens/openTPU main
# 2026-10-14; 60 commits past this Mac's main 7e42a626, not an ancestor of wide-port a9acb2b0)
LPN_RTL_PIN = "f237dab2f196855a59a083b1454786df8f01291f"
LPN_UPSTREAM = "https://github.com/MJChku/LPNDialect"
LPN_BRANCH = "qwen-validation"
LPN_LABEL = "lpn-estimate"                 # every number an LPN run produces
LPN_GATE = 0.05                            # upstream's rule: a replay over 5% error fails

# the configuration upstream validated the model on (examples/opentpu/README.md "Validated
# configuration"): anything else is outside the model's evidence until replayed
VALIDATED = {"S": 1, "D": 128, "MCOLS": 4, "LANES": 8, "epilogue_lanes": 2, "PORTB": 1,
             "window": 32, "fifo_chunks": 128, "dram_latency": 8, "dram_bytes_per_cycle": 128,
             "memory": "behavioral DRAM (otpu_dram.sv), one port, <= 128 B/cycle, 8-cycle read latency",
             "uarch": "the RTL testbench's default (tb_top), replicated TMEM, range sequencer"}

# what upstream's timing backend rejects (workload.execute / model.py)
REJECTED = ("WAITW", "HALT CHAIN", "DSTEP", "STREAM", "WAITC", "multi-slice BAR / GATHER")
