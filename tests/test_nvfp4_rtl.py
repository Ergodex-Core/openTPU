"""Standalone RTL scoreboard: no model weights, device, licenses or inference."""
from pathlib import Path
import random
import shutil
import subprocess

import pytest

from opentpu import nvfp4 as N

ROOT = Path(__file__).resolve().parents[1]


def vectors():
    rng = random.Random(20261022)
    v = []
    # Every BF16 exponent crossed with every FP4 encoding, alternating signs.
    for e in range(255):
        for c in range(16):
            a = [(e << 7) | rng.randrange(128) | (rng.randrange(2) << 15) for _ in range(16)]
            v.append((a, [c] * 16, 0xffff, rng.randrange(127)))
    # Extremal width, cancellation, both zero encodings, tie-even, masked nonfinite.
    for c in (7, 15):
        for a in (0x7f7f, 0xff7f, 1, 0x807f, 0, 0x8000):
            v.append(([a] * 16, [c] * 16, 0xffff, 0x7e))
    for m in (128, 129, 130, 131, 255):
        # E-star 10, lower exponent -15 (gap 25): m / 2 RNE.
        v.append(([144 << 7, ((119 << 7) | (m & 127))] + [0] * 14,
                  [0, 2] + [0] * 14, 0xffff, 0x38))
    for bad in (0x7f80, 0xff80, 0x7fc1):
        for mask in (0xffff, 0xfffe, 0):
            v.append(([bad] + [0x3f80] * 15, [7] * 16, mask, 0x38))
    for scale in range(127, 256):
        v.append(([0x3f80] * 16, [2] * 16, 0xffff, scale))
    for _ in range(1600):
        a = [rng.randrange(1 << 16) for _ in range(16)]
        c = [rng.randrange(16) for _ in range(16)]
        v.append((a, c, rng.randrange(1 << 16), rng.randrange(127)))
    return v


TB = r"""
module tb;
  logic clk=0;
  always #5 clk=~clk;
  logic rst=1, in_valid=0, in_ready, out_valid, out_ready=0;
  logic [255:0] in_a;
  logic [63:0] in_w;
  logic [15:0] in_mask;
  logic [7:0] in_scale, out_scale;
  logic [31:0] in_tag, out_tag;
  logic signed [40:0] out_dot;
  logic signed [8:0] out_e_star;
  logic out_discarded, out_invalid;
  otpu_nvfp4_dot16 dut(.*);
  localparam integer COUNT = @COUNT@;
  logic [427:0] vec [0:COUNT-1];
  integer q [0:COUNT-1];
  integer head=0, tail=0, sent=0, checked=0, flushed=0, cycle=0;
  integer stalled=0, max_backlog=0, throughput=0;
  logic [31:0] lfsr=32'h93426781;
  logic held=0, accepted=0, done=0;
  logic [91:0] previous;
  logic [427:0] expected;
  initial begin
    $readmemh("vectors.hex",vec);
    while (cycle < 50000 && !done) begin
      @(negedge clk);
      // Reset with work in flight and while the receiver can be stalled.
      rst = cycle < 2 || cycle == 88 || cycle == 89 || cycle == 330;
      lfsr = {lfsr[30:0],lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
      // The first and final regions saturate the pipe; middle region has bubbles/stalls.
      out_ready = (cycle < 50 || cycle > 8000) ? 1'b1 :
                  ((cycle % 257 < 73) ? 1'b0 : lfsr[3] | lfsr[9]);
      if (rst) in_valid = 0;
      else if (!in_valid || accepted) begin
        in_valid = sent < COUNT && (cycle < 50 || cycle > 8000 || lfsr[0] || lfsr[2]);
        if (sent < COUNT) begin
          in_a = vec[sent][427:172];
          in_w = vec[sent][171:108];
          in_mask = vec[sent][107:92];
          in_scale = vec[sent][91:84];
          in_tag = vec[sent][83:52];
        end
      end
      @(posedge clk);
      accepted = in_valid && in_ready;
      if (rst) begin
        flushed = flushed + tail-head;
        head = tail;
        held = 0;
        if (in_ready || out_valid) $fatal(1,"handshake during reset");
      end else begin
        if (held && (!out_valid || previous !==
            {out_dot,out_e_star,out_discarded,out_invalid,out_scale,out_tag}))
          $fatal(1,"payload changed while stalled cycle=%0d",cycle);
        held = out_valid && !out_ready;
        previous = {out_dot,out_e_star,out_discarded,out_invalid,out_scale,out_tag};
        if (out_valid && !out_ready) stalled = stalled + 1;
        if (out_valid && out_ready) begin
          if (head == tail) $fatal(1,"unexpected/duplicate output");
          expected = vec[q[head]];
          if ({out_dot,out_e_star,out_discarded,out_invalid} !== expected[51:0] ||
               out_tag !== expected[83:52] || out_scale !== expected[91:84])
            $fatal(1,"mismatch tag=%0d expected=%0d dot=%0d want=%0d exp=%0d want=%0d flags=%b%b want=%b",
              out_tag,q[head],out_dot,$signed(expected[51:11]),out_e_star,
              $signed(expected[10:2]),out_discarded,out_invalid,expected[1:0]);
          head = head + 1;
          checked = checked + 1;
        end
        if (in_valid && in_ready) begin
          q[tail] = sent;
          tail = tail+1;
          sent = sent+1;
          if (out_valid && out_ready) throughput = throughput+1;
        end
        if (tail-head > max_backlog) max_backlog = tail-head;
      end
      #1;
      cycle = cycle+1;
      if (sent == COUNT && head == tail) begin
        if (checked+flushed != COUNT || stalled < 100 || flushed == 0 ||
            throughput < 100 || max_backlog != 7)
          $fatal(1,"coverage gate checked=%0d flushed=%0d stalled=%0d throughput=%0d depth=%0d",
                    checked,flushed,stalled,throughput,max_backlog);
        done = 1;
      end
    end
    if (!done) $fatal(1,"watchdog sent=%0d checked=%0d",sent,checked);
    else begin
      $display("TEST_PASS vectors=%0d checked=%0d reset_flushed=%0d stalls=%0d II1=%0d depth=%0d cycles=%0d",
                COUNT,checked,flushed,stalled,throughput,max_backlog,cycle);
      $finish;
    end
  end
endmodule
"""


def write_vectors(path):
    rows = vectors()
    lines = []
    for tag, (a, c, mask, scale) in enumerate(rows):
        invalid = scale > 126 or any(mask >> i & 1 and (x >> 7) & 255 == 255
                                      for i, x in enumerate(a))
        if invalid:
            dot, e, discarded = 0, 0, False
        else:
            r = N.aligned_dot16(a, c, mask=mask)
            dot, e, discarded = r.integer, r.e_star, r.discarded
        fields = [(sum(x << (16*i) for i, x in enumerate(a)), 256),
                  (sum(x << (4*i) for i, x in enumerate(c)), 64), (mask, 16), (scale, 8),
                  (tag, 32), (dot & ((1 << 41)-1), 41), (e & 511, 9),
                  (int(discarded), 1), (int(invalid), 1)]
        word = 0
        for x, width in fields:
            word = (word << width) | x
        lines.append(f"{word:0107x}")
    (path / "vectors.hex").write_text("\n".join(lines) + "\n")
    (path / "tb.sv").write_text(TB.replace("@COUNT@", str(len(rows))))


@pytest.mark.parametrize("backend", ["iverilog", "verilator"])
def test_f24_pipeline_against_rational_vectors(tmp_path, backend):
    if shutil.which(backend) is None:
        pytest.skip(backend + " unavailable")
    write_vectors(tmp_path)
    rtl = ROOT / "rtl/mxu/otpu_nvfp4_dot16.sv"
    if backend == "iverilog":
        compile_cmd = [backend, "-g2012", "-s", "tb", "-o", "sim", str(rtl), "tb.sv"]
        run_cmd = ["vvp", "sim"]
    else:
        compile_cmd = [backend, "--binary", "--timing", "--top-module", "tb", "-j", "2",
                       str(rtl), "tb.sv"]
        run_cmd = [str(tmp_path / "obj_dir/Vtb")]
    build = subprocess.run(compile_cmd, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    (tmp_path / "compile.log").write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(run_cmd, cwd=tmp_path, capture_output=True, text=True, timeout=45)
    (tmp_path / "simulation.log").write_text(run.stdout + run.stderr)
    print(run.stdout)
    transcript = run.stdout + run.stderr
    assert (run.returncode == 0 and "TEST_PASS" in transcript and
            not any(s in transcript.lower() for s in ("fatal", "error", "watchdog"))), transcript
