"""NV41 through production F2 OCL CDC/control/ownership/native AXI; simulated HBM only."""
from dataclasses import replace
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import pytest
from opentpu import nvfp4 as N,target
from test_nvfp4_ocl import fixture
ROOT=Path(__file__).resolve().parents[1]


def vectors():
    rng=random.Random(557);rows=[]
    for mode in (0,1):
        for k in range(1,65):
            a=[rng.randrange(110,141)<<7|rng.randrange(128)|rng.randrange(2)<<15 for _ in range(k)]
            c=[rng.randrange(16) for _ in a];s=[rng.randrange(127) for _ in range((k+15)//16)]
            rows.append((k,mode,a,c,s))
        for a,c,s in [([0x7f80],[0],[0x38]),([0x7fc1],[2],[0]),
                      ([0x7f7f]*16,[7]*16,[0]),([0xff7f]*16,[7]*16,[0x38]),
                      ([1]*16,[1]*16,[0x38]),([0x8000]*16,[2]*16,[0x38]),
                      ([0x4c00,0x3f80,0xcc00,0x3f80],[2]*4,[0x38]),
                      ([0x3f80],[2],[0x7f])]:rows.append((len(a),mode,a,c,s))
    result=[]
    from opentpu.host.nvfp4_ocl import encode
    for i,(k,mode,a,c,s) in enumerate(rows):
        j,w=fixture(k,mode,i)
        try:bits=N.strict_row(a,N.pack_codes(c),s,j.global_bits,profile=j.profile);status=1
        except N.NumericalFault:bits=0;status=2
        d=list(struct.unpack('<64I',encode(j,w)));windows=sum((x.words() for x in w),[])
        data=[struct.pack('<'+'H'*len(a),*a).ljust(128,b'\0'),N.pack_codes(c).ljust(64,b'\xff'),bytes(s).ljust(64,b'\x7f')]
        result.append((d,windows,data,bits,status,0))
    # Malformed raw wire images intentionally bypass the host encoder.
    original=result[-1]
    fresh=original[0].copy();fresh[24]+=1
    mutations=[(0,0,2),(1,0x20000,2),(2,252,2),(3,1,2),(4,0,4),(4,65,4),
               (5,2,3),(6,0x80000000,2),(7,1,2),(8,0,3),(16,0,5),(18,0,5),
               (24,0,6),(26,99,7),(28,2,7),(29,0xffffffff,7),(30,1,2),
               (31,0xffffffff,2),(32,1,2),(58,1,2),(63,1,2)]
    for word,value,error in mutations:
        d=fresh.copy();d[word]=value
        result.append((d,original[1],original[2],0,0,error))
    for n,value,error in [(0,0,7),(2,0x1001,2),(4,0,7),(6,2,8),(7,1,8),
                           (26,0x1040,8),(30,1,8),(3,0xffffffff,7)]:
        w=original[1].copy();w[n]=value
        result.append((fresh,w,original[2],0,0,error))
    return result


def make_tb(count,nxr,enabled):
    src=(ROOT/'rtl/boards/f2/otpu_f2_hbm_sys.sv').read_text().split(') (',1)[1].split(');',1)[0]
    names=[]
    # Bind unused production inputs to zero; actual bus signals declared below.
    for line in src.splitlines():
        m=re.match(r'\s*input\s+logic\s*(?:\[[^\]]+\]\s*)*(\w+)',line)
        if m:names.append(m[1])
    known={'clk':'clk','clk_f':'clk','rst':'rst','aclk':'sclk','arst':'rst','hbm_ready':'ready'}
    known.update({n:n for n in names if n.startswith('ocl_') or n.startswith('ch_')})
    binds=["."+n+"("+known.get(n,"'0")+")" for n in names]
    outputs=['ocl_awready','ocl_wready','ocl_bresp','ocl_bvalid','ocl_arready','ocl_rdata','ocl_rresp','ocl_rvalid',
             'ch_awaddr','ch_awvalid','ch_wdata','ch_wstrb','ch_wvalid','ch_bready','ch_araddr','ch_arlen','ch_arvalid','ch_rready']
    binds += [f'.{n}({n})' for n in outputs]
    return TB.replace('@BIND@',',\n'.join(binds)).replace('@COUNT@',str(count)).replace('@NXR@',str(nxr)).replace('@ENABLE@',str(enabled))


@pytest.mark.parametrize('nxr,enabled',[(0,1),(3,1),(3,0)])
def test_production_ocl_mailbox(tmp_path,nxr,enabled):
    if not shutil.which('verilator'):pytest.skip('Verilator unavailable')
    rows=vectors()
    def words(v):return ''.join(f'{x:08x}' for x in reversed(v))
    (tmp_path/'descriptors.hex').write_text('\n'.join(words(r[0]) for r in rows)+'\n')
    (tmp_path/'windows.hex').write_text('\n'.join(words(r[1]) for r in rows)+'\n')
    (tmp_path/'data.hex').write_text('\n'.join(f'{int.from_bytes(b,"little"):0128x}' for r in rows for block in r[2] for b in [block[:64],*([block[64:]] if len(block)>64 else [])])+'\n')
    (tmp_path/'expected.hex').write_text('\n'.join(f'{(r[5]<<40)|(r[4]<<32)|r[3]:012x}' for r in rows)+'\n')
    (tmp_path/'tb.sv').write_text(make_tb(len(rows),nxr,enabled))
    sources=[ROOT/'rtl'/x for x in target.sources('f2-hbm:nvfp4',family='generic')]
    cmd=['verilator','--binary','--timing','--top-module','tb','-j','2','-Wno-fatal',*map(str,sources),'tb.sv']
    p=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True,timeout=180)
    (tmp_path/'compile.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
    log=tmp_path/'simulation.log'
    with log.open('w') as out:
        p=subprocess.run([str(tmp_path/'obj_dir/Vtb')],cwd=tmp_path,stdout=out,stderr=subprocess.STDOUT,text=True,timeout=180)
    result=log.read_text();print(result[-3000:])
    assert p.returncode==0 and 'TEST_PASS' in result,result[-10000:]


TB=r'''
module tb;
 logic clk=0,sclk=0,rst=1,ready=1;
 always #5 clk=~clk;
 always #2 sclk=~sclk;
 logic [31:0] ocl_awaddr=0,ocl_wdata=0,ocl_araddr=0,ocl_rdata;
 logic [3:0] ocl_wstrb=15;
 logic ocl_awvalid=0,ocl_awready,ocl_wvalid=0,ocl_wready,ocl_bvalid,ocl_bready=0;
 logic ocl_arvalid=0,ocl_arready,ocl_rvalid,ocl_rready=0;
 logic [1:0] ocl_bresp,ocl_rresp;
 logic [1:0][33:0] ch_awaddr,ch_araddr;
 logic [1:0][511:0] ch_wdata,ch_rdata;
 logic [1:0][63:0] ch_wstrb;
 logic [1:0][7:0] ch_arlen;
 logic [1:0] ch_awvalid,ch_wvalid,ch_bready,ch_arvalid,ch_rready;
 logic [1:0] ch_awready,ch_wready,ch_bvalid=0,ch_arready,ch_rvalid=0,ch_rlast=3;
 logic [1:0][1:0] ch_bresp=0,ch_rresp=0;
 otpu_f2_hbm_sys #(.D(128),.MCOLS(4),.ACT_ROWS(4),.ACT_BLOCKS(8),.TMEM_WORDS(256),.IMEM_WORDS(64),
 .FIFO_DEPTH(16),.LANES(8),.WIN(4),.RPB(64),.WPB(1),.TMEM_IMPL(3),.TMEM_PMP(0),.MXU_IMPL(2),
 .ULANES(8),.TRACE_DEPTH(16),.PQ_WIN(16),.NCH(2),.DRAM_LOG2(30),.NXR(@NXR@),.NVFP4_MAILBOX(@ENABLE@)) dut (@BIND@);
 logic [511:0] mem[512];
 logic [2047:0] descriptors[@COUNT@];
 logic [1023:0] windows[@COUNT@];
 logic [511:0] data[4*@COUNT@];
 logic [47:0] expected[@COUNT@];
 logic [33:0] aw_saved;
 logic [511:0] wd_saved;
 logic [63:0] wm_saved;
 logic have_aw=0,have_w=0,have_ar=0,have_ar1=0,allow_legacy=0,hold_b=0,hold_r=0,inject_b=0,inject_r=0;
 integer read_index,read_remaining=0,read_remaining1=0,read_delay=0,read_delay1=0,write_delay=0,cycles=0,reads=0,writes=0;
 logic [31:0] random_state=32'h7364842;
 always @(posedge clk)begin
   if(rst)begin
     have_aw<=0;have_w<=0;have_ar<=0;have_ar1<=0;ch_bvalid<=0;ch_rvalid<=0;reads<=0;writes<=0;
     cycles<=0;random_state<=32'h7364842;
   end else begin
     cycles<=cycles+1;random_state<={random_state[30:0],random_state[31]^random_state[21]^random_state[1]^random_state[0]};
     if((ch_arvalid[1] && !allow_legacy) || ch_awvalid[1])$fatal(1,"sidecar used neighbor HBM channel");
     if(ch_arvalid[1] && ch_arready[1])begin
       if(ch_arlen[1]>15 || ch_araddr[1]<'h20000000 || ch_araddr[1]>='h20008000)$fatal(1,"legacy loader bank1 bounds addr=%h len=%d",ch_araddr[1],ch_arlen[1]);
       have_ar1<=1;read_delay1<=6;read_remaining1<=int'(ch_arlen[1])+1;
     end
     if(have_ar1 && !hold_r)begin
       if(read_delay1>0)read_delay1<=read_delay1-1;
       else if(!ch_rvalid[1])begin ch_rvalid[1]<=1;ch_rdata[1]<=0;ch_rresp[1]<=0;ch_rlast[1]<=read_remaining1==1;end
     end
     if(ch_rvalid[1] && ch_rready[1])begin
       ch_rvalid[1]<=0;read_remaining1<=read_remaining1-1;read_delay1<=3;
       if(read_remaining1==1)have_ar1<=0;
     end
     if(ch_arvalid[0] && ch_arready[0])begin
       if((!allow_legacy && ch_arlen[0]!=0) || ch_arlen[0]>15 || ch_araddr[0]>=32768 || ch_araddr[0][5:0]!=0)$fatal(1,"read bounds/burst");
       have_ar<=1;read_index<=int'(ch_araddr[0]>>6);read_delay<=2+int'(random_state[3:1]);read_remaining<=int'(ch_arlen[0])+1;reads<=reads+1;
     end
     if(have_ar && !hold_r)begin
       if(read_delay>0)read_delay<=read_delay-1;
       else if(!ch_rvalid[0])begin ch_rvalid[0]<=1;ch_rdata[0]<=mem[read_index];ch_rresp[0]<=inject_r ? 2 : 0;ch_rlast[0]<=read_remaining==1;end
     end
     if(ch_rvalid[0] && ch_rready[0])begin
       ch_rvalid[0]<=0;read_remaining<=read_remaining-1;read_index<=read_index+1;read_delay<=3;
       if(read_remaining==1)have_ar<=0;
     end
     if(ch_awvalid[0] && ch_awready[0])begin have_aw<=1;aw_saved<=ch_awaddr[0];end
     if(ch_wvalid[0] && ch_wready[0])begin have_w<=1;wd_saved<=ch_wdata[0];wm_saved<=ch_wstrb[0];end
     if(have_aw && have_w && !ch_bvalid[0] && !hold_b)begin
       if(write_delay<7)write_delay<=write_delay+1;
       else begin
         if(aw_saved!=34'h4000 || wm_saved!=64'hf)$fatal(1,"write protection/strobe");
         for(integer b=0;b<64;b++)if(wm_saved[b])mem[int'(aw_saved>>6)][8*b+:8]<=wd_saved[8*b+:8];
         ch_bvalid[0]<=1;ch_bresp[0]<=inject_b ? 2 : 0;have_aw<=0;have_w<=0;write_delay<=0;writes<=writes+1;
       end
     end
     if(ch_bvalid[0] && ch_bready[0])ch_bvalid[0]<=0;
   end
 end
 assign ch_awready={1'b1,!rst && !have_aw && random_state[1]};
 assign ch_wready={1'b1,!rst && !have_w && random_state[2]};
 assign ch_arready={!rst && !have_ar1 && !ch_rvalid[1] && random_state[5],!rst && !have_ar && !ch_rvalid[0] && random_state[4]};
 // Host AXI-Lite driver: independent AW/W arrival and held response checks.
 task automatic wr(input logic [31:0] addr,value,input logic [1:0] response=0,input logic [3:0] strobes=15);
   bit ad,dd;integer age;
   ad=0;dd=0;age=0;
   @(negedge sclk);ocl_awaddr=addr;ocl_wdata=value;ocl_wstrb=strobes;
   while(!ad || !dd)begin
     ocl_awvalid=!ad && age>=int'(addr[4:2]);
     ocl_wvalid=!dd && age>=int'(addr[6:5]);
     @(posedge sclk);
     if(ocl_awvalid && ocl_awready)ad=1;
     if(ocl_wvalid && ocl_wready)dd=1;
     @(negedge sclk);age++;
   end
   ocl_awvalid=0;ocl_wvalid=0;
   while(!ocl_bvalid)@(negedge sclk);
   repeat(3)begin @(negedge sclk);if(!ocl_bvalid || ocl_bresp!==response)$fatal(1,"write response addr=%h got=%b want=%b",addr,ocl_bresp,response);end
   ocl_bready=1;@(negedge sclk);ocl_bready=0;
 endtask
 task automatic rd(input logic [31:0] addr,output logic [31:0] value,input logic [1:0] response=0);
   logic [31:0] saved;
   @(negedge sclk);ocl_araddr=addr;ocl_arvalid=1;
   do @(posedge sclk);while(!ocl_arready);
   @(negedge sclk);ocl_arvalid=0;
   while(!ocl_rvalid)@(negedge sclk);
   saved=ocl_rdata;
   repeat(3)begin @(negedge sclk);if(!ocl_rvalid || ocl_rdata!==saved || ocl_rresp!==response)$fatal(1,"read stability/response addr=%h",addr);end
   value=saved;ocl_rready=1;@(negedge sclk);ocl_rready=0;
 endtask
 task automatic check(input logic [31:0] addr,want);
   logic [31:0] v;rd(addr,v);if(v!==want)$fatal(1,"register addr=%h got=%h want=%h",addr,v,want);
 endtask
 task automatic configure(input integer j);
   wr('h428,descriptors[0][18*32+:32]);wr('h42c,descriptors[0][19*32+:32]);
   wr('h430,descriptors[0][16*32+:32]);wr('h434,descriptors[0][17*32+:32]);
   for(integer w=0;w<32;w++)wr('h500+4*w,windows[j][32*w+:32]);
   wr('h414,5);
 endtask
 task automatic stage(input integer j);
   for(integer w=0;w<64;w++)wr('h600+4*w,descriptors[j][32*w+:32]);
 endtask
 task automatic wait_completion;
   logic [31:0] v;integer n;
   v=0;n=0;
   while(!v[1] && n<200)begin rd('h410,v);n++;end
   if(!v[1])$fatal(1,"completion did not drain");
 endtask
 task automatic hard_reset;
   @(negedge clk);rst=1;hold_b=0;hold_r=0;inject_b=0;inject_r=0;
   mem['h1000/64]=data[0];mem['h1040/64]=data[1];mem['h2000/64]=data[2];
   mem['h3000/64]=data[3];mem['h4000/64]='hdeadbeef;
   repeat(20)@(negedge clk);rst=0;repeat(30)@(negedge clk);
 endtask
 logic [31:0] v;integer successes=0,failures=0,rejections=0,before_reads,before_writes,accepted=0;
 initial begin
   $readmemh("descriptors.hex",descriptors);$readmemh("windows.hex",windows);
   $readmemh("data.hex",data);$readmemh("expected.hex",expected);
   for(integer i=0;i<512;i++)mem[i]='hdeadbeef;
   hard_reset();$display("PHASE discovery");$fflush();check('h3c,5);rd('h5c,v);
   if(v[6]!=@ENABLE@)$fatal(1,"capability opt-in");
   wr('h38,'h87654321);check('h38,'h87654321);
   if(!@ENABLE@)begin
     rd('h400,v);if(v==32'h3134564e)$fatal(1,"disabled discovery leaked");
     $display("TEST_PASS default_off legacy register behavior NXR=@NXR@");$finish;
   end
   check('h400,'h3134564e);check('h404,'h10000);check('h408,256);
   rd('h1000,v);if(v!=0)$fatal(1,"outside first4KiB reached core");
   wr('h1000,123);check('h400,'h3134564e);
   // Reset before first ARM must still prohibit generation rollback.
   wr('h414,4);wr('h414,4);
   for(integer w=0;w<32;w++)wr('h500+4*w,windows[0][32*w+:32]);
   wr('h428,1);wr('h42c,0);wr('h430,5);wr('h434,0);
   wr('h414,5,2);check('h418,5);check('h4d0,2);
   hard_reset();
   for(integer j=0;j<@COUNT@;j++)begin
     $display("CASE j=%0d cycles=%0d",j,cycles);$fflush();
     mem['h1000/64]=data[4*j];mem['h1040/64]=data[4*j+1];mem['h2000/64]=data[4*j+2];
     mem['h3000/64]=data[4*j+3];mem['h4000/64]='hdeadbeef;
     configure(j);stage(j);before_reads=reads;before_writes=writes;
     if(expected[j][47:40]!=0)begin
       wr('h414,1,2);check('h418,{24'd0,expected[j][47:40]});
       if(reads!=before_reads || writes!=before_writes || dut.nv_owns)$fatal(1,"rejected descriptor accessed memory");
       check('h41c,accepted);rejections++;
     end else begin
       hold_b=1;
       wr('h414,1);accepted++;check('h418,0);check('h41c,accepted);
       if(expected[j][39:32]==1)begin
         wait(have_aw && have_w); // result remains private until actual B response
         check('h440,0);check('h444,0);rd('hc,v);if(v[3])$fatal(1,"legacy WR_IDLE during sidecar store");
         wr('h38,99,2);wr('h600,0,2);wr('h414,1,2);wr('h414,3,2);check('h418,11);
         repeat(12)@(negedge clk);if(!dut.nv_owns)$fatal(1,"ownership released before completion");
         hold_b=0;
       end else hold_b=0;
       wait_completion();
       rd('h440,v);
       if(expected[j][39:32]==1 ? v!='h10001 : (v[7:0]!=2 || v[15:8]==0 || v[31:16]!=0))$fatal(1,"completion fault/status j=%0d got=%h",j,v);
       check('h444,expected[j][31:0]);check('h448,accepted);check('h44c,0);
       for(integer n=0;n<10;n++)check('h450+4*n,descriptors[j][32*(16+n)+:32]);
       if(mem['h4000/64][31:0] !== (expected[j][39:32]==1 ? expected[j][31:0] : 32'hdeadbeef))$fatal(1,"output golden mismatch j=%0d",j);
       if(mem['h4000/64][511:32]!==480'd0)$fatal(1,"masked output padding changed");
       repeat(10)@(negedge clk);check('h444,expected[j][31:0]);wr('h38,99,2);
       wr('h414,2);wr('h38,'h87654321);check('h38,'h87654321);
       repeat(12)@(negedge clk);if(!dut.legacy_idle)$fatal(1,"legacy count/idle corrupted after ACK j=%0d",j);
       if(expected[j][39:32]==1)successes++;else failures++;
     end
   end
   // Completeness tracks each byte; failed control strobes cannot launch a job.
   hard_reset();
   wr('h428,descriptors[0][18*32+:32]);wr('h42c,descriptors[0][19*32+:32]);
   wr('h430,descriptors[0][16*32+:32]);wr('h434,descriptors[0][17*32+:32]);
   for(integer w=0;w<31;w++)wr('h500+4*w,windows[0][32*w+:32]);
   wr('h57c,0,0,7);wr('h414,5,2);check('h418,5);
   wr('h57c,0,0,8);wr('h414,5);
   for(integer w=0;w<63;w++)wr('h600+4*w,descriptors[0][32*w+:32]);
   wr('h6fc,0,0,1);wr('h414,1,2,7);check('h418,2);
   wr('h414,1,2);check('h418,10);check('h41c,0);
   wr('h6fc,0,0,14);wr('h414,1);wait_completion();wr('h414,2);
   wr('h414,1,2);check('h418,10); // accepting cleared the old descriptor byte mask
   rd('h401,v,2);wr('h401,0,2);check('h400,'h3134564e);
   // Existing RUN excludes the sidecar even with a fully staged descriptor.
   hard_reset();configure(0);stage(0);wr('h8,1);wr('h414,1,2);check('h418,9);
   wr('h8,0);wr('h414,1);wait_completion();wr('h414,2);
   // A pending loader read excludes admission; drain rather than stealing its return.
   hard_reset();configure(0);stage(0);hold_r=1;allow_legacy=1;
   wr('h10,'h1000);wr('h14,1);wr('h8,2);wait(have_ar);
   wr('h414,1,2);check('h418,9);hold_r=0;wait(dut.legacy_idle);allow_legacy=0;
   wr('h414,1);wait_completion();wr('h414,2);
   // Generation maximum never wraps, including before another descriptor acceptance.
   hard_reset();configure(0);wr('h428,'hffffffff);wr('h42c,'hffffffff);wr('h414,5);
   wr('h414,4,2);check('h418,13);check('h4d0,'hffffffff);check('h4d4,'hffffffff);
   // Coordinated hard reset discards pending native reads and stale private results.
   hard_reset();configure(0);stage(0);hold_r=1;wr('h414,1);wait(have_ar);
   hard_reset();check('h410,0);configure(0);stage(0);wr('h414,1);wait_completion();wr('h414,2);
   $display("PHASE cancel_reset");$fflush();
   // Cancel/reset accepted reads must drain their response before credit release.
   for(integer reset_case=0;reset_case<2;reset_case++)begin
     hard_reset();configure(0);stage(0);hold_r=1;wr('h414,1);wait(have_ar);
     wr('h414,reset_case==0 ? 3 : 4);check('h440,0);wr('h500,1,2);
     hold_r=0;wait_completion();check('h440,reset_case==0 ? 3 : 4);check('h444,0);
     wr('h414,2);if(reset_case==1)check('h4d0,descriptors[0][18*32+:32]+1);
   end
   // Reset after irreversible output starts defers invalidation until old success ACK.
   hard_reset();configure(0);stage(0);hold_b=1;wr('h414,1);wait(have_aw && have_w);
   wr('h414,4);check('h440,0);hold_b=0;wait_completion();check('h440,'h10001);wr('h414,2);
   check('h4d0,descriptors[0][18*32+:32]+1);stage(0);wr('h414,1,2);check('h418,10);
   // Uncalibrated fabric cannot admit; recovery uses normal readiness synchronization.
   hard_reset();configure(0);stage(0);ready=0;repeat(30)@(negedge clk);wr('h414,1,2);check('h418,9);
   ready=1;repeat(30)@(negedge clk);wr('h414,1);wait_completion();wr('h414,2);
   // Native bus read/B faults poison private results; B failure may have modified Y.
   for(integer which=0;which<2;which++)begin
     hard_reset();configure(0);stage(0);inject_r=which==0;inject_b=which==1;
     wr('h414,1);wait_completion();check('h440,which==0 ? 'h8002 : 'h18002);check('h444,0);wr('h414,2);
     stage(1);wr('h414,1,2);check('h418,9); // native error sticky until coordinated reset
   end
   $display("TEST_PASS production OCL jobs=%0d success=%0d numeric_fail=%0d rejected=%0d NXR=@NXR@ cancellation/reset/readiness/bus_faults",@COUNT@,successes,failures,rejections);
   $finish;
 end
 initial begin #30000000;$fatal(1,"timeout");end
endmodule
'''
