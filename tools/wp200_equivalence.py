#!/usr/bin/env python3
"""Compare the WP200 candidate with the qualified source at transaction-cycle precision.

Run this file in separate processes with --root BASE and --root CANDIDATE, then
compare their result.json files with --compare BASE_JSON CANDIDATE_JSON.
Only the generated simulation harness is instrumented; production RTL is unchanged.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


MONITOR = r'''
  integer wp_fd = 0;
  longint wp_tick = 0;
  longint wp_astall = 0, wp_bstall = 0, wp_wstall = 0, wp_replay = 0;
  string wp_filename;
  initial if ($value$plusargs("wp_trace=%s", wp_filename)) begin
    wp_fd = $fopen(wp_filename, "w");
    if (!wp_fd) $fatal(1, "cannot open WP transaction trace");
  end
  always @(posedge clk) begin
    wp_tick <= wp_tick + 1;
    if (wp_fd) begin
      $fdisplay(wp_fd, "C %0d %b %b %h %h", wp_tick, sys_rst, rst,
                dut.g_slice[0].u_slice.gnt, dut.g_slice[0].u_slice.tm_frz);
      if (dut.g_slice[0].u_slice.a_req) begin
        $fdisplay(wp_fd, "A %b %h", dut.g_slice[0].u_slice.a_rdy,
                  dut.g_slice[0].u_slice.a_addr);
        if (!dut.g_slice[0].u_slice.a_rdy) wp_astall <= wp_astall + 1;
      end
      if (dut.g_slice[0].u_slice.b_req) begin
        $fdisplay(wp_fd, "B %b %b %b %h %h %h", dut.g_slice[0].u_slice.b_rdy,
                  dut.g_slice[0].u_slice.b_tag, dut.g_slice[0].u_slice.b_we,
                  dut.g_slice[0].u_slice.b_addr, dut.g_slice[0].u_slice.b_par,
                  dut.g_slice[0].u_slice.b_n);
        if (dut.g_slice[0].u_slice.b_we)
          $fdisplay(wp_fd, "D %h %h", dut.g_slice[0].u_slice.b_wmask,
                    dut.g_slice[0].u_slice.b_wdata);
        if (!dut.g_slice[0].u_slice.b_rdy) wp_bstall <= wp_bstall + 1;
      end
      for (int k = 0; k < ULANES; k++) begin
        if (dut.g_slice[0].u_slice.mxu_wen[k])
          $fdisplay(wp_fd, "W %0d %h %h", k, dut.g_slice[0].u_slice.mxu_waddr[k],
                    dut.g_slice[0].u_slice.mxu_wdata[k]);
        if (dut.g_slice[0].u_slice.mxu_ren[k])
          $fdisplay(wp_fd, "R %0d %h", k, dut.g_slice[0].u_slice.mxu_raddr[k]);
      end
      if ((|dut.g_slice[0].u_slice.mxu_wen) && !dut.g_slice[0].u_slice.u_mxu.t_gnt)
        wp_wstall <= wp_wstall + 1;
      if (|dut.g_slice[0].u_slice.tm_frz) wp_replay <= wp_replay + 1;
    end
  end
  final begin
    if (wp_fd) $fclose(wp_fd);
    $display("WP_COVER astall=%0d bstall=%0d wstall=%0d replay=%0d abort=%0d",
             wp_astall, wp_bstall, wp_wstall, wp_replay, wp_did_abort);
  end
'''

ABORT = r'''
      if (!wp_did_abort && ((wp_abort == 1 && dut.g_slice[0].u_slice.b_req &&
                            !dut.g_slice[0].u_slice.b_rdy) ||
          (wp_abort == 2 && dut.g_slice[0].u_slice.u_mxu.lo.v &&
                           !dut.g_slice[0].u_slice.u_mxu.c_acc) ||
          (wp_abort == 3 && dut.g_slice[0].u_slice.u_mxu.rw.v))) begin
        wp_did_abort = 1;
        rst = 1;
        repeat (32) @(negedge clk);
        // Match otpu_board.go_ok: stopped reads/writes must drain before restart.
        // A fixed delay alone is invalid when memory latency/backpressure varies.
        while (!(dut.g_slice[0].u_slice.rd_idle_i &&
                 dut.g_slice[0].u_slice.wr_idle_i)) begin
          @(negedge clk);
          wp_drain_wait++;
          if (wp_drain_wait > 100000) $fatal(1, "reset drain timeout");
        end
        repeat (4) @(negedge clk);
        rst = 0;
      end
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(root: Path, out: Path, a_throttle: bool) -> None:
    root, out = root.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "tests")]
    os.environ.update(OTPU_TARGET="f2-hbm:p4rl8a64", OTPU_NCH="8", OTPU_NAXI="1",
                      OTPU_NAXI_XR="3", OTPU_NAXI_BURST="8", OTPU_NAXI_WAIT="1",
                      OTPU_NAXI_OT="32", OTPU_NAXI_IDLE="1", OTPU_IMAGE_CACHE="0")
    import numpy as np
    from opentpu import Config, isa as I, rtlsim
    from opentpu.isasim import Machine
    from test_wide_port_rtl import _rl_matrix, _rl_prog, _walk_prog, _xrows

    rtlsim.VFLAGS[rtlsim.VFLAGS.index("-j") + 1] = "4"
    harness = (rtlsim.TB / "tb_top.sv").read_text()
    marker = "    void'($value$plusargs(\"max_cycles=%d\", max_cycles));"
    assert harness.count(marker) == 1
    harness = harness.replace("  longint cycles =", "  int wp_abort = 0;\n  bit wp_did_abort = 0;\n  longint wp_drain_wait = 0;\n  longint cycles =")
    harness = harness.replace(marker, marker + '\n    void\'($value$plusargs("wp_abort=%d", wp_abort));')
    marker = "      cycles++;\n    end"
    assert harness.count(marker) == 1
    harness = harness.replace(marker, "      cycles++;\n" + ABORT + "    end")
    harness = harness.replace("endmodule", MONITOR + "\nendmodule")
    tb = out / "tb_top.sv"
    tb.write_text(harness)
    # A protocol-preserving test-only gate: both ends agree on each accepted
    # request, while the slice must hold a valid A payload for up to 15 cycles.
    top = (rtlsim.RTL / "top/otpu_top.sv").read_text()
    if a_throttle:
        marker = "    logic [31:0]   sw_addr, sw_wdata;"
        assert top.count(marker) == 1
        top = top.replace(marker, marker + '''
    logic [3:0] wp_a_phase = 0;
    always @(posedge clk) if (sys_rst) wp_a_phase <= 0; else wp_a_phase <= wp_a_phase + 1'b1;
    wire wp_a_allow = wp_a_phase == 0;
''')
        marker = ".a_rdy_x(a_rdy), .a_req_x(a_req)"
        assert top.count(marker) == 1
        top = top.replace(marker, ".a_rdy_x(a_rdy), .a_req_x(a_req && wp_a_allow)")
        marker = "      .a_rdy, .b_rdy, .sw_rdy, .wr_idle, .rd_idle,"
        assert top.count(marker) == 1
        top = top.replace(marker, "      .a_rdy(a_rdy && wp_a_allow), .b_rdy, .sw_rdy, .wr_idle, .rd_idle,")
    top_path = out / "otpu_top.sv"
    top_path.write_text(top)
    original_build, original_run_sim = rtlsim.build, rtlsim.run_sim
    capture: dict = {}

    def build(top, sources, params=None):
        if top == "tb_top":
            sources = [tb if Path(s) == rtlsim.TB / "tb_top.sv" else s for s in sources]
            sources = [top_path if Path(s) == rtlsim.RTL / "top/otpu_top.sv" else s for s in sources]
        return original_build(top, sources, params)

    def run_sim(cmd, timeout=None):
        result = original_run_sim(cmd, timeout)
        capture["stdout"] = result.stdout + result.stderr
        capture["command"] = list(map(str, cmd))
        (capture["case"] / "simulation.log").write_text(capture["stdout"])
        return result

    rtlsim.build, rtlsim.run_sim = build, run_sim
    cfg = Config(S=1, D=128, MCOLS=8, ACT_ROWS=16, PAIR=True, T2=True, PORTB=4,
                 ROWLANES=True, ALINE=64, DRAM_BYTES=1 << 20)
    uarch = {**rtlsim.BOARD_UARCH, "COLL_GATHER": 1}
    cases = [(kind, stall, seed, 0) for kind in ("rl", "walk")
             for stall, seed in ((0, 1), (40, 7), (70, 2026))]
    cases += [("rl", 70, 42, abort) for abort in (1, 2, 3)]
    results = []
    for kind, stall, seed, abort in cases:
        label = f"{kind}-stall{stall}-seed{seed}-abort{abort}"
        case = out / label
        case.mkdir()
        rng = np.random.default_rng(200)
        fmt, L, M, N, KB, k0 = ("int8", 4, 1, 13, 9, 0) if kind == "rl" else ("int4", 4, 16, 11, 9, 0)
        rows, scales = _rl_matrix(rng, fmt, N, (k0 + KB) * cfg.D)
        x = _xrows(rng, M, KB * cfg.D, cfg.D // 4)
        if kind == "rl":
            prog, dram = _rl_prog(cfg, fmt, L, N, KB, k0, x, rows, scales)
        else:
            prog, dram = _walk_prog(cfg, fmt, L, M, N, KB, k0, x, rows, scales)
        # Independent DMA writes contend for TMEM banks while the first MM drains.
        first_mm = 3
        prog[first_mm + 1:first_mm + 1] = [I.ld(0, 0x4000, 8192)]
        # The DMA also exercises port-B write payload and byte-mask storage.
        prog[-1:-1] = [I.st(0xD0000, 0x8000, 256), I.st(0xD1000, 0x9000, 256)]
        reference = Machine(cfg, [prog], [dram.copy()]).run()
        trace = case / "transactions.txt"
        capture["case"] = case
        actual_d, actual_t, stats = rtlsim.run(
            cfg, [prog], [dram.copy()], axi=True, boot=False, stall=stall, seed=seed,
            lat=49, uarch=uarch, plusargs=[f"+wp_trace={trace}", f"+wp_abort={abort}"],
            keep=str(case / "state"))
        (case / "simulation.log").write_text(capture["stdout"])
        assert np.array_equal(actual_d[0], reference.slices[0].dram), f"{label}: DRAM != ISA"
        assert np.array_equal(actual_t[0], reference.slices[0].tmem), f"{label}: TMEM != ISA"
        match = re.search(r"WP_COVER (.*)", capture["stdout"])
        assert match, f"{label}: missing coverage"
        coverage = {k: int(v) for k, v in (kv.split("=") for kv in match[1].split())}
        assert coverage["abort"] == bool(abort), f"{label}: abort point not reached"
        row = {"case": label, "cycles": stats["cycles"], "isa_exact": True,
               "dram_sha256": digest(actual_d[0].tobytes()),
               "tmem_sha256": digest(actual_t[0].tobytes()),
               "transactions_sha256": digest(trace.read_bytes()),
               "trace_lines": sum(1 for _ in trace.open()), "coverage": coverage}
        results.append(row)
        print(json.dumps(row), flush=True)
    result = {"root": str(root), "revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "config": dataclasses.asdict(cfg),
        "uarch": uarch, "harness_sha256": digest(harness.encode()),
        "top_harness_sha256": digest(top.encode()), "a_throttle": a_throttle, "cases": results}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--a-throttle", action="store_true",
                        help="test-only port-A gate admits a request once per 16 cycles")
    args = parser.parse_args()
    if args.compare:
        baseline, candidate = [json.loads(p.read_text()) for p in args.compare]
        for key in ("config", "uarch", "harness_sha256", "cases"):
            assert baseline[key] == candidate[key], f"baseline/candidate mismatch: {key}"
        for key in ("top_harness_sha256", "a_throttle"):
            assert baseline.get(key) == candidate.get(key), f"baseline/candidate mismatch: {key}"
        assert any(c["coverage"]["bstall"] for c in candidate["cases"]), "B stalls not exercised"
        assert any(c["coverage"]["wstall"] for c in candidate["cases"]), "MXU write stalls not exercised"
        assert any(c["coverage"]["replay"] for c in candidate["cases"]), "TMEM replay not exercised"
        if candidate.get("a_throttle"):
            assert any(c["coverage"]["astall"] for c in candidate["cases"]), "A stalls not exercised"
        print(json.dumps({"pass": True, "cases": len(candidate["cases"]),
                          "exact": ["transaction cycles and payloads", "grant/replay/reset cycles",
                                    "DRAM", "TMEM", "ISA reference"]}))
    else:
        if args.root is None or args.out is None:
            parser.error("--root and --out are required for a run")
        run(args.root, args.out, args.a_throttle)


if __name__ == "__main__":
    main()
