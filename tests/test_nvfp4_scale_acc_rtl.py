"""Portable arithmetic and ordered-row RTL against exact rational golden bits."""
from pathlib import Path
import random
import shutil
import subprocess

import pytest

from opentpu import nvfp4 as N

ROOT = Path(__file__).resolve().parents[1]
FP = ROOT / "rtl/mxu/otpu_nvfp4_fp.sv"
ACC = ROOT / "rtl/mxu/otpu_nvfp4_scale_acc.sv"


def simulate(path, backend, sources, tb):
    if not shutil.which(backend):
        pytest.skip(backend + " unavailable")
    (path / "tb.sv").write_text(tb)
    if backend == "iverilog":
        cmd = [backend,"-g2012","-s","tb","-o","sim",*map(str,sources),"tb.sv"]
        run = ["vvp","sim"]
    else:
        cmd = [backend,"--binary","--timing","--top-module","tb","-j","2",
               *map(str,sources),"tb.sv"]
        run = [str(path / "obj_dir/Vtb")]
    p = subprocess.run(cmd,cwd=path,capture_output=True,text=True,timeout=120)
    (path / "compile.log").write_text(p.stdout+p.stderr)
    assert p.returncode == 0, p.stdout+p.stderr
    p = subprocess.run(run,cwd=path,capture_output=True,text=True,timeout=90)
    text = p.stdout+p.stderr
    (path / "simulation.log").write_text(text)
    print(text)
    assert p.returncode == 0 and "TEST_PASS" in text, text
    assert not any(s in text.lower() for s in ("fatal","error","watchdog")), text


def packed(fields):
    word = 0
    for value, width in fields:
        word = (word << width) | (value & ((1 << width)-1))
    return word


def primitive_vectors():
    rng = random.Random(333)
    rows = []
    for scale in range(256):
        rows.append((3,scale,0,0,0,0))
    for e in range(-256,256):
        for z in (0,1,-1,(1 << 40)-1,-(1 << 40),(1 << 24)+1,(1 << 24)+3):
            rows.append((0,0,0,z,e,0))
    for sign in (-1,1):
        for z in ((1 << 33)-257,(1 << 33)-256):
            rows.append((0,0,0,sign*z,120,0))
    corners = [0,0x80000000,1,2,0x007fffff,0x00800000,0x00800001,0x3f000000,
               0x3f7fffff,0x3f800000,0x3f800001,0x40000000,0x7f7fffff,
               0x7f800000,0xff800000,0x7fc00001,0xbf800000,0xff7fffff,0x807fffff]
    for ftz in (0,1):
        for a in corners:
            for b in corners:
                rows.extend([(1,a,b,0,0,ftz),(2,a,b,0,0,ftz)])
        for _ in range(2200):
            a,b = rng.randrange(1 << 32),rng.randrange(1 << 32)
            rows.extend([(1,a,b,0,0,ftz),(2,a,b,0,0,ftz)])
    return rows


PRIMITIVE_TB = r"""
module tb;
  import otpu_nvfp4_fp::*;
  logic [149:0] vec [0:@LAST@];
  logic [1:0] op;
  logic [31:0] a,b;
  logic signed [40:0] dot;
  logic signed [8:0] e;
  logic ftz;
  result_t r;
  initial begin
    $readmemh("vectors.hex",vec);
    for (integer i=0;i<=@LAST@;i=i+1) begin
      {op,a,b,dot,e,ftz} = vec[i][149:33];
      case(op)
        0: r=from_dot(dot,e);
        1: r=mul32(a,b,ftz);
        2: r=add32(a,b,ftz);
        3: r=scale32(a[7:0]);
      endcase
      #1;
      if (r !== vec[i][32:0])
        $fatal(1,"primitive mismatch i=%0d op=%0d a=%h b=%h dot=%0d e=%0d ftz=%b got=%h want=%h",
                    i,op,a,b,dot,e,ftz,r,vec[i][32:0]);
    end
    $display("TEST_PASS primitive_vectors=%0d",@COUNT@);
    $finish;
  end
endmodule
"""


@pytest.mark.parametrize("backend",["iverilog","verilator"])
def test_fp32_primitives(tmp_path, backend):
    rows = primitive_vectors()
    lines = []
    for op,a,b,z,e,ftz in rows:
        try:
            if op == 0: bits = N.convert_subtotal(z,e)
            elif op == 1: bits = N.mul32(a,b,ftz=bool(ftz))
            elif op == 2: bits = N.add32(a,b,ftz=bool(ftz))
            else: bits = N.round32(N.e4m3_value(a))
            fault = 0
        except N.NumericalFault:
            bits,fault = 0,1
        word = packed([(op,2),(a,32),(b,32),(z,41),(e,9),(ftz,1),(fault,1),(bits,32)])
        lines.append(f"{word:038x}")
    (tmp_path / "vectors.hex").write_text("\n".join(lines)+"\n")
    tb = PRIMITIVE_TB.replace("@LAST@",str(len(rows)-1)).replace("@COUNT@",str(len(rows)))
    simulate(tmp_path,backend,[FP],tb)


def block(mode=0, dot=0, e=0, strict=0x3f800000, scale=0x38, global_=0x3f800000,
          first=1, last=1, invalid=0, discarded=0, index=0, tag=0,group=5,epoch=7):
    return dict(mode=mode,dot=dot,e=e,strict=strict,scale=scale,global_=global_,first=first,
                last=last,invalid=invalid,discarded=discarded,index=index,tag=tag,
                group=group,epoch=epoch,scale_group=group,scale_epoch=epoch,scale_index=index)


def stream_rows():
    rng = random.Random(334)
    rows = []
    for mode in (0,1,2):
        for s in range(256):
            rows.append([block(mode=mode,dot=(1 << 24)+1,e=25,scale=s)])
    for mode in (0,1):
        for dots in ([0x4c000000,0x3f800000,0xcc000000],
                     [0x4c000000,0xcc000000,0x3f800000],
                     [0x7f7fffff,0x7f7fffff], [0x00800000], [1,0x80000001],
                     [0x80000000], [0x7f800000,0], [0x7fc00001,0]):
            rows.append([block(mode=mode,strict=d,first=int(i==0),last=int(i==len(dots)-1),
                               index=i,scale=0x30 if d==0x00800000 else 0x38,
                               global_=0x40000000) for i,d in enumerate(dots)])
    for z,e in [(-(1 << 40),-133),((1 << 40)-1,120),(1,-125),(3,-125),
                ((1 << 24)+1,25),((1 << 24)+3,25),(0,-134),(0,121)]:
        rows.append([block(mode=2,dot=z,e=e)])
    # Every immutable/protocol field mutation poisons the existing row identity.
    for change in [dict(first=1),dict(mode=1),dict(tag=99),dict(global_=0x40000000),
                   dict(index=0),dict(index=2),dict(index=0xffffffff,last=0),
                   dict(group=6),dict(epoch=8),dict(scale_group=6),dict(scale_epoch=8),
                   dict(scale_index=999),
                   dict(group=6,scale_group=6),dict(epoch=8,scale_epoch=8),
                   dict(index=0,scale_index=0),dict(index=2,scale_index=2),
                   dict(index=0xffffffff,scale_index=0xffffffff,last=0),
                   dict(invalid=1),dict(scale=127),dict(discarded=1,mode=2)]:
        row = [block(first=1,last=0),block(first=0,last=0,index=1),
               block(first=0,last=1,index=2,strict=0)]
        row[1].update(change)
        rows.append(row)
    rows.extend([[block(first=0)],[block(mode=3)], [block(global_=0x80000000)],
                 [block(global_=0x7f800000)],[block(strict=0x7f7fffff,scale=0x7e)],
                 [block(strict=0x7f7fffff,global_=0x40000000)],
                 [block(mode=2,dot=(1 << 40)-1,e=120,scale=0,global_=0)],
                 [block(mode=2,discarded=1)]])
    for _ in range(600):
        mode = rng.randrange(3)
        length = rng.randrange(1,6)
        g = rng.randrange(0x7f800000)  # nonnegative finite, including subnormals
        row = []
        for i in range(length):
            row.append(block(mode=mode,dot=rng.randrange(-(1 << 40),1 << 40),
                             e=rng.randrange(-133,121),strict=rng.randrange(1 << 32),
                             scale=rng.randrange(127),global_=g,
                             first=int(i==0),last=int(i==length-1),index=i))
        rows.append(row)
    # Ordinary next row after faults must recover cleanly.
    rows.append([block(strict=0x40000000)])
    # Same owner is reused only after the terminal output drains, including wrap.
    rows.extend([[block(group=9,epoch=e,strict=0x40000000)] for e in (0xfffffffe,0xffffffff,0,1)])
    for tag,row in enumerate(rows):
        for b in row:
            if b["tag"] != 99: b["tag"] = tag
    return rows


def expected_row(row):
    mode,tag,g = row[0]["mode"],row[0]["tag"],row[0]["global_"]
    group,epoch = row[0]["group"],row[0]["epoch"]
    faults,acc,next_index = 0,0,0
    for i,b in enumerate(row):
        bad = (b["invalid"] or b["mode"]==3 or b["scale"]>126 or g>>31 or
               (b["global_"]>>31) or (b["global_"]>>23)&255==255 or
               (b["mode"]==2 and b["discarded"]) or
               b["scale_group"]!=b["group"] or b["scale_epoch"]!=b["epoch"] or
               b["scale_index"]!=b["index"] or
               (b["index"]==0xffffffff and not b["last"]) or
               (not b["first"] or b["index"]!=0 if i==0 else
                b["first"] or b["mode"]!=mode or b["tag"]!=tag or
                b["global_"]!=g or b["index"]!=next_index or
                b["group"]!=group or b["epoch"]!=epoch))
        faults |= int(bool(bad))
        next_index = (b["index"]+1)&0xffffffff
        try:
            dot = N.convert_subtotal(b["dot"],b["e"]) if mode==2 else b["strict"]
            N.fp32_value(dot)
        except N.NumericalFault:
            faults |= 2; dot = 0
        try:
            scale = N.round32(N.e4m3_value(b["scale"]))
            scale_fault = False
        except N.NumericalFault:
            scale_fault = True; scale = 0
        try: term = N.mul32(dot,scale,ftz=mode==1)
        except N.NumericalFault: faults |= 4; term = 0
        if scale_fault: faults |= 4
        try: acc = N.add32(acc,term,ftz=mode==1)
        except N.NumericalFault: faults |= 8; acc = 0
    try: result = N.mul32(acc,g,ftz=mode==1)
    except N.NumericalFault: faults |= 16; result = 0
    return 0 if faults else result,faults,mode,tag


STREAM_TB = r"""
module tb;
  logic clk=0;
  always #5 clk=~clk;
  logic rst=1,in_valid=0,in_ready,out_valid,out_ready=0;
  logic [1:0] in_mode,out_mode;
  logic signed [40:0] in_dot;
  logic signed [8:0] in_e_star;
  logic [31:0] in_strict_dot,in_global,in_k16_index,in_tag,out_bits,out_tag;
  logic [7:0] in_scale;
  logic in_first,in_last,in_invalid,in_discarded;
  logic [4:0] out_fault;
  logic [31:0] in_group,in_epoch,in_scale_group,in_scale_epoch,in_scale_k16_index;
  logic [31:0] out_group,out_epoch,free_group,free_k16_index,free_epoch;
  logic in_scale_valid=1,buffer_free,output_committed;
  localparam integer COUNT=@COUNT@;
  logic [486:0] vec[0:COUNT-1];
  logic [134:0] q[0:COUNT-1];
  logic [134:0] held_payload;
  logic [31:0] rng=32'h95841278;
  logic held=0,accepted=0,done=0;
  integer sent=0,head=0,tail=0,cycles=0,stalls=0,backpressure=0,scale_wait=0;
  otpu_nvfp4_scale_acc dut(.*);
  initial begin
    $readmemh("vectors.hex",vec);
    // Six resets interrupt conversion, scale, accumulation, final scale and
    // stalled output; phase 2 also discards a partial non-last row.
    repeat(2) @(negedge clk);
    for(integer phase=0;phase<7;phase=phase+1) begin
      rst=0; in_valid=1; in_mode=0; in_dot=0; in_e_star=0;
      in_strict_dot=32'h3f800000; in_scale=8'h38; in_global=32'h3f800000;
      in_k16_index=0; in_tag=32'hdead; in_first=1; in_last=(phase!=2 && phase!=6);
      in_group=5;in_epoch=7;in_scale_group=5;in_scale_epoch=7;in_scale_k16_index=0;
      in_scale_valid=1;
      in_invalid=0; in_discarded=0; out_ready=0;
      @(posedge clk);
      if(!in_ready) $fatal(1,"reset preflight not ready");
      @(negedge clk); in_valid=0;
      repeat(phase) @(negedge clk);
      rst=1;
      #1;
      if(in_ready || out_valid || buffer_free || output_committed) $fatal(1,"reset handshake not gated");
      @(negedge clk);
    end
    rst=0;
    while(cycles<120000 && !done) begin
      @(negedge clk);
      rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
      out_ready=(cycles%113 < 39) ? 1'b0 : (rng[0] | rng[7]);
      in_scale_valid=cycles%97 >= 17 && (rng[10] || rng[12]);
      if(!in_valid || accepted) begin
        in_valid=sent<COUNT && (rng[3] || rng[4]);
        if(sent<COUNT)
          {in_mode,in_dot,in_e_star,in_strict_dot,in_scale,in_global,
           in_first,in_last,in_invalid,in_discarded,in_k16_index,in_tag}=vec[sent][262:71];
      end
      if(sent<COUNT)
        {in_group,in_epoch,in_scale_group,in_scale_k16_index,in_scale_epoch}=vec[sent][486:327];
      @(posedge clk);
      accepted=in_valid && in_ready;
      if(buffer_free !== accepted || (buffer_free &&
         {free_group,free_k16_index,free_epoch} !== {in_group,in_k16_index,in_epoch}))
        $fatal(1,"buffer ownership release mismatch");
      if(output_committed !== (out_valid && out_ready && out_fault==0))
        $fatal(1,"early or failed output commit");
      if(held && (!out_valid || held_payload !== {out_group,out_epoch,out_bits,out_fault,out_mode,out_tag}))
        $fatal(1,"stalled output changed");
      held=out_valid && !out_ready;
      held_payload={out_group,out_epoch,out_bits,out_fault,out_mode,out_tag};
      if(held) stalls=stalls+1;
      if(in_valid && !in_ready) backpressure=backpressure+1;
      if(in_valid && !in_scale_valid) scale_wait=scale_wait+1;
      if(out_valid && out_ready) begin
        if(head==tail || {out_group,out_epoch,out_bits,out_fault,out_mode,out_tag} !== q[head])
          $fatal(1,"row mismatch head=%0d got=%h want=%h",head,
                      {out_group,out_epoch,out_bits,out_fault,out_mode,out_tag},q[head]);
        head=head+1;
      end
      if(accepted) begin
        if(in_last) begin q[tail]={vec[sent][326:263],vec[sent][70:0]};tail=tail+1;end
        sent=sent+1;
      end
      #1;
      cycles=cycles+1;
      done=sent==COUNT && head==tail && !out_valid;
    end
    if(!done) $fatal(1,"watchdog sent=%0d head=%0d tail=%0d",sent,head,tail);
    else if(stalls<100 || backpressure<100 || scale_wait<100 || head!=@ROWS@)
      $fatal(1,"coverage gate stalls=%0d backpressure=%0d rows=%0d",stalls,backpressure,head);
    else begin
      $display("TEST_PASS blocks=%0d rows=%0d reset_phases=7 stalls=%0d backpressure=%0d scale_wait=%0d cycles=%0d",
                  COUNT,head,stalls,backpressure,scale_wait,cycles);
      $finish;
    end
  end
endmodule
"""


@pytest.mark.parametrize("backend",["iverilog","verilator"])
def test_ordered_scale_accumulator(tmp_path, backend):
    rows = stream_rows()
    write_stream_vectors(tmp_path,rows)
    tb = STREAM_TB.replace("@COUNT@",str(sum(map(len,rows)))).replace("@ROWS@",str(len(rows)))
    simulate(tmp_path,backend,[FP,ACC],tb)


def write_stream_vectors(path, rows, raw=None):
    lines = []
    ordinal = 0
    for row in rows:
        result,faults,mode,tag = expected_row(row)
        for b in row:
            word = packed([(b["mode"],2),(b["dot"],41),(b["e"],9),(b["strict"],32),
                           (b["scale"],8),(b["global_"],32),(b["first"],1),(b["last"],1),
                           (b["invalid"],1),(b["discarded"],1),(b["index"],32),(b["tag"],32),
                           (result,32),(faults,5),(mode,2),(tag,32)])
            word = packed([(b["group"],32),(b["epoch"],32),(b["scale_group"],32),
                           (b["scale_index"],32),(b["scale_epoch"],32),
                           (row[0]["group"],32),(row[0]["epoch"],32),(word,263)])
            if raw is not None:
                a,c,mask = raw[ordinal]
                word = packed([(sum(x << (16*i) for i,x in enumerate(a)),256),
                               (sum(x << (4*i) for i,x in enumerate(c)),64),(mask,16),(word,487)])
            lines.append(f"{word:0206x}" if raw is not None else f"{word:0122x}")
            ordinal += 1
    (path / "vectors.hex").write_text("\n".join(lines)+"\n")


INTEGRATED_INSTANCES = r"""
  logic [255:0] raw_a;
  logic [63:0] raw_w;
  logic [15:0] raw_mask;
  logic [127:0] raw_identity,dot_identity;
  logic dot_valid,dot_ready,dot_invalid,dot_discarded;
  logic signed [40:0] dot_value;
  logic signed [8:0] dot_e;
  logic [7:0] dot_scale;
  logic [31:0] dot_id;
  otpu_nvfp4_dot16 #(.TAG_W(128)) dot_unit(
    .clk(clk),.rst(rst),.in_valid(in_valid),.in_ready(in_ready),
    .in_a(raw_a),.in_w(raw_w),.in_mask(raw_mask),
    .in_scale(in_scale),.in_tag(raw_identity),
    .out_valid(dot_valid),.out_ready(dot_ready),.out_dot(dot_value),.out_e_star(dot_e),
    .out_discarded(dot_discarded),.out_invalid(dot_invalid),.out_scale(dot_scale),.out_tag(dot_identity));
  assign dot_id=dot_identity[31:0];
  otpu_nvfp4_scale_acc dut(
    .clk(clk),.rst(rst),.in_valid(dot_valid),.in_ready(dot_ready),.in_mode(2'd2),
    .in_dot(dot_value),.in_e_star(dot_e),.in_strict_dot(32'd0),.in_scale(dot_scale),
    .in_global(vec[dot_id][170:139]),.in_first(vec[dot_id][138]),.in_last(vec[dot_id][137]),
    .in_invalid(dot_invalid | vec[dot_id][136]),
    .in_discarded(dot_discarded | vec[dot_id][135]),
    .in_k16_index(vec[dot_id][134:103]),.in_tag(vec[dot_id][102:71]),
    .in_group(dot_identity[127:96]),.in_epoch(dot_identity[95:64]),
    .in_scale_valid(in_scale_valid),.in_scale_group(dot_identity[127:96]),
    .in_scale_k16_index(dot_identity[63:32]),.in_scale_epoch(dot_identity[95:64]),
    .out_valid(out_valid),.out_ready(out_ready),.out_bits(out_bits),.out_tag(out_tag),
    .out_mode(out_mode),.out_fault(out_fault),.out_group(out_group),.out_epoch(out_epoch),
    .buffer_free(buffer_free),.output_committed(output_committed),.free_group(free_group),
    .free_k16_index(free_k16_index),.free_epoch(free_epoch));
"""


@pytest.mark.parametrize("backend",["iverilog","verilator"])
def test_dot16_to_epilogue(tmp_path, backend):
    rng = random.Random(335)
    rows,raw = [],[]
    for tag in range(220):
        length = 1 if tag==0 else rng.randrange(1,7)
        g = rng.randrange(0x7f800000)
        row = []
        for i in range(length):
            top = rng.randrange(30,245)
            a = [((top-rng.randrange(21)) << 7) | rng.randrange(128) |
                 (rng.randrange(2) << 15) for _ in range(16)]
            c = [rng.randrange(16) for _ in range(16)]
            mask = rng.randrange(1 << 16)
            if tag%11==0: a[0]=0x7fc1;mask|=1  # invalid propagates through the real boundary
            if tag%13==0: a[1]=1;mask|=2       # discarded alignment poisons the row
            scale = 127 if tag%17==0 else rng.randrange(127)
            invalid = any(mask >> j & 1 and (x >> 7)&255==255 for j,x in enumerate(a))
            if invalid: z,e,discarded = 0,0,False
            else:
                dot = N.aligned_dot16(a,c,mask=mask)
                z,e,discarded = dot.integer,dot.e_star,dot.discarded
            row.append(block(mode=2,dot=z,e=e,scale=scale,global_=g,first=int(i==0),
                             last=int(i==length-1),invalid=int(invalid or scale>126),
                             discarded=int(discarded),index=i,tag=tag,group=tag%4,epoch=tag))
            raw.append((a,c,mask))
        rows.append(row)
    write_stream_vectors(tmp_path,rows,raw)
    tb = STREAM_TB.replace("otpu_nvfp4_scale_acc dut(.*);",INTEGRATED_INSTANCES)
    tb = tb.replace("logic [486:0] vec", "logic [822:0] vec")
    # Drive every DUT input before the sampling edge. Using `sent` directly as
    # a DUT input would race the scoreboard's posedge acceptance bookkeeping.
    tb = tb.replace("@(posedge clk);", """
      if(sent<COUNT) begin
        raw_a=vec[sent][822:567]; raw_w=vec[sent][566:503];
        raw_mask=vec[sent][502:487];
        raw_identity={vec[sent][486:455],vec[sent][454:423],vec[sent][134:103],32'(sent)};
      end
      @(posedge clk);""")
    tb = tb.replace("phase<7", "phase<13").replace("reset_phases=7", "reset_phases=13")
    tb = tb.replace("buffer_free !== accepted", "buffer_free !== (dot_valid && dot_ready)")
    tb = tb.replace("{in_group,in_k16_index,in_epoch}))", "{dot_identity[127:96],dot_identity[63:32],dot_identity[95:64]}))")
    tb = tb.replace("@COUNT@",str(len(raw))).replace("@ROWS@",str(len(rows)))
    simulate(tmp_path,backend,[FP,ACC,ROOT / "rtl/mxu/otpu_nvfp4_dot16.sv"],tb)
