"""Strict folded K16 RTL checked ONLY against strict per-op software numerics."""
import random
import pytest

from opentpu import nvfp4 as N
from test_nvfp4_scale_acc_rtl import simulate,packed,ROOT,FP

STRICT=ROOT / 'rtl/mxu/otpu_nvfp4_strict_dot16.sv'


def strict_vectors():
    rng=random.Random(336)
    v=[]
    for ftz in (0,1):
        for exponent in range(255):
            for sign in (0,1):
                a=[(exponent<<7)|rng.randrange(128)|(sign<<15) for _ in range(16)]
                v.append((a,list(range(16)),0xffff,ftz,rng.randrange(127)))
        for a,c,mask in [([0x4c00,0x3f80,0xcc00,0x3f80]+[0]*12,[2]*16,0xffff),
                          ([0x8000]*16,[2]*16,0xffff),([0x7f7f]*16,[7]*16,0xffff),
                          ([1]*16,[1]*16,0xffff),([0x0080]*16,[1]*16,0xffff),
                          ([0x7f80]*16,[0]*16,0xffff),([0x7fc1]*16,[7]*16,0),
                          ([0xffff]+[0x3f80]*15,[2]*16,0xfffe)]:
            v.append((a,c,mask,ftz,0x38))
        for _ in range(240):
            a=[rng.randrange(1<<16) for _ in range(16)]
            c=[rng.randrange(16) for _ in range(16)]
            v.append((a,c,rng.randrange(1<<16),ftz,rng.randrange(256)))
    return v


TB=r"""
module tb;
  logic clk=0;always #5 clk=~clk;
  logic rst=1,in_valid=0,in_ready,out_valid,out_ready=0,in_ftz,out_ftz,out_fault;
  logic [255:0] in_a;
  logic [63:0] in_w;
  logic [15:0] in_mask;
  logic [7:0] in_scale,out_scale;
  logic [31:0] in_tag,out_tag,out_bits;
  localparam integer COUNT=@COUNT@;
  logic [409:0] vec[0:COUNT-1];
  integer q[0:COUNT-1],birth[0:COUNT-1];
  integer sent=0,head=0,tail=0,cycles=0,stalls=0,blocked=0;
  logic [31:0] rng=32'h95418726;
  logic accepted=0,held=0,done=0;
  logic [73:0] previous;
  logic [409:0] expected;
  otpu_nvfp4_strict_dot16 dut(.*);
  initial begin
    $readmemh("vectors.hex",vec);
    repeat(2) @(negedge clk);
    for(integer phase=0;phase<34;phase=phase+1) begin
      rst=0;in_valid=1;in_a={16{16'h3f80}};in_w={16{4'd2}};
      in_mask=16'hffff;in_ftz=0;in_scale=8'h38;in_tag=32'hdead;out_ready=0;
      @(posedge clk);if(!in_ready) $fatal(1,"not ready after reset");
      @(negedge clk);in_valid=0;
      repeat(phase) @(negedge clk);
      rst=1;#1;if(in_ready || out_valid) $fatal(1,"reset handshake");
      @(negedge clk);
    end
    rst=0;
    while(cycles<200000 && !done) begin
      @(negedge clk);
      rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
      out_ready=cycles%127>=43 && (rng[1] || rng[2]);
      if(!in_valid || accepted) begin
        in_valid=sent<COUNT && (rng[3] || rng[8]);
        if(sent<COUNT) {in_a,in_w,in_mask,in_ftz,in_scale,in_tag}=vec[sent][409:33];
      end
      @(posedge clk);
      accepted=in_valid && in_ready;
      if(held && (!out_valid || previous !== {out_bits,out_fault,out_ftz,out_scale,out_tag}))
        $fatal(1,"strict output changed while stalled");
      held=out_valid && !out_ready;
      previous={out_bits,out_fault,out_ftz,out_scale,out_tag};
      if(held) stalls=stalls+1;
      if(in_valid && !in_ready) blocked=blocked+1;
      if(out_valid && out_ready) begin
        if(head==tail || cycles-birth[head]<32) $fatal(1,"early/duplicate strict output");
        expected=vec[q[head]];
        if({out_bits,out_fault} !== expected[32:0] || out_ftz !== expected[73] ||
            out_scale !== expected[72:65] || out_tag !== expected[64:33])
          $fatal(1,"strict mismatch row=%0d bits=%h fault=%b want=%h",head,out_bits,out_fault,expected[32:0]);
        head=head+1;
      end
      if(accepted) begin q[tail]=sent;birth[tail]=cycles;tail=tail+1;sent=sent+1;end
      #1;cycles=cycles+1;
      done=sent==COUNT && head==tail;
    end
    if(!done || stalls<100 || blocked<100) $fatal(1,"watchdog/coverage");
    else begin
      $display("TEST_PASS strict_vectors=%0d reset_phases=34 stalls=%0d blocked=%0d cycles=%0d",COUNT,stalls,blocked,cycles);
      $finish;
    end
  end
endmodule
"""


@pytest.mark.parametrize('backend',['iverilog','verilator'])
def test_strict_tree(tmp_path,backend):
    rows=strict_vectors();lines=[]
    for tag,(a,c,mask,ftz,scale) in enumerate(rows):
        aa=[x if mask>>i&1 else 0 for i,x in enumerate(a)]
        cc=[x if mask>>i&1 else 0 for i,x in enumerate(c)]
        try: bits=N.strict_dot16(aa,cc,ftz=bool(ftz));fault=0
        except N.NumericalFault: bits=0;fault=1
        word=packed([(sum(x<<(16*i) for i,x in enumerate(a)),256),
                     (sum(x<<(4*i) for i,x in enumerate(c)),64),(mask,16),(ftz,1),
                     (scale,8),(tag,32),(bits,32),(fault,1)])
        lines.append(f'{word:0103x}')
    (tmp_path/'vectors.hex').write_text('\n'.join(lines)+'\n')
    simulate(tmp_path,backend,[FP,STRICT],TB.replace('@COUNT@',str(len(rows))))
