"""Bounded logical descriptor/private-result controller; no model or DMA."""
from copy import deepcopy
from dataclasses import replace
import random

import pytest

from opentpu import nvfp4 as N
from opentpu.mxu_job import Buffer,W4A16Job,PROFILE_DIGESTS,local_job_fields
from test_nvfp4_scale_acc_rtl import simulate,packed,ROOT,FP,ACC
from test_nvfp4_strict_rtl import STRICT

SESSION=0x100000007


def job(k=16,mode=0,request=0):
    profile=N.STRICT_FTZ if mode else N.STRICT_GRADUAL
    return W4A16Job(Buffer(1<<40,2*k,2*k),Buffer(0x2000,(k+1)//2,(k+1)//2),
                    Buffer(0x3000,(k+15)//16,(k+15)//16),Buffer(0x4000,4,4),
                    tokens=1,outputs=1,k=k,global_bits=0x3f812345,profile=profile,
                    profile_digest=PROFILE_DIGESTS[profile],context=(1<<60)+5,
                    generation=SESSION,request=(1<<50)+request,position=(1<<48)+3,
                    epoch=(1<<55)+request)


def test_local_adapter_subset_and_full_identity():
    d=local_job_fields(job(63,1,123),SESSION)
    assert d['mode']==1 and d['context_id']>1<<32 and d['epoch']>1<<32
    assert d['digest']==int(PROFILE_DIGESTS[N.STRICT_FTZ],16)
    for j,g in [(job(65),SESSION),(replace(job(),dependencies=(1,)),SESSION),
                (job(),SESSION+1)]:
        with pytest.raises(ValueError):local_job_fields(j,g)


def desc_pack(d):
    f=[(d[k],32) for k in ('abi_major','abi_minor','flags','reserved')]
    f += [(d['contract'],7),(d['mode'],2),(d['digest'],256)]
    f += [(d[k],32) for k in ('tokens','outputs','k','global_bits')]
    for b in ('a','w','scales','y'):
        f += [(d[b][k],64) for k in ('address','span','stride')]
    f += [(d[k],64) for k in ('context_id','generation','request_id','position','epoch')]
    f += [(d['dependency_count'],32)]
    assert sum(w for _,w in f)==1641
    return packed(f)


def desc_ok(d):
    # Independent typed Python validator, not the RTL validation expression.
    if d['contract']!=127 or d['mode']>1 or d['dependency_count']:
        return False
    profile=N.STRICT_FTZ if d['mode'] else N.STRICT_GRADUAL
    try:
        j=W4A16Job(*(Buffer(**d[b]) for b in ('a','w','scales','y')),
                   tokens=d['tokens'],outputs=d['outputs'],k=d['k'],global_bits=d['global_bits'],
                   profile=profile,profile_digest=f"{d['digest']:064x}",abi_major=d['abi_major'],
                   abi_minor=d['abi_minor'],flags=d['flags'],reserved=d['reserved'],
                   context=d['context_id'],generation=d['generation'],request=d['request_id'],
                   position=d['position'],epoch=d['epoch'])
        local_job_fields(j,SESSION)
        return True
    except ValueError:return False


def packet(d,i,a,c,scale):
    n=min(16,d['k']-16*i)
    return dict(a=a,w=c,mask=(1<<n)-1,scale=scale,first=int(i==0),
                last=int(i==(d['k']+15)//16-1),group=0,k=i,scale_group=0,scale_k=i,
                context=d['context_id'],generation=d['generation'],request=d['request_id'],
                position=d['position'],epoch=d['epoch'],scale_epoch=d['epoch'])


def packet_pack(p):
    f=[(sum(x<<(16*i) for i,x in enumerate(p['a'])),256),
       (sum(x<<(4*i) for i,x in enumerate(p['w'])),64),(p['mask'],16),(p['scale'],8),
       (p['first'],1),(p['last'],1)]
    f += [(p[k],32) for k in ('group','k','scale_group','scale_k')]
    f += [(p[k],64) for k in ('context','generation','request','position','epoch','scale_epoch')]
    assert sum(w for _,w in f)==858
    return packed(f)


def cases():
    rng=random.Random(337)
    rows=[]
    for mode in (0,1):
        for k in range(1,65):
            d=local_job_fields(job(k,mode,len(rows)),SESSION)
            ps=[]
            for i in range((k+15)//16):
                a=[(rng.randrange(110,141)<<7)|rng.randrange(128)|(rng.randrange(2)<<15)
                   for _ in range(16)]
                c=[rng.randrange(16) for _ in range(16)]
                ps.append(packet(d,i,a,c,rng.randrange(127)))
            rows.append((d,ps))
    # Strict cancellation witness, BF16 extremes, signed-zero and masked special values.
    for mode in (0,1):
        for a,c,k in [([0x4c00,0x3f80,0xcc00,0x3f80]+[0]*12,[2]*16,16),
                      ([1]*16,[1]*16,16),([0x8000]*16,[2]*16,16),
                      ([0x7f7f]*16,[7]*16,16),([0x7f80]*16,[0]*16,16),
                      ([0x3f80]+[0x7fc1]*15,[2]*16,1)]:
            d=local_job_fields(job(k,mode,len(rows)),SESSION)
            d['global_bits']=0x3f800000
            rows.append((d,[packet(d,0,a,c,0x38)]))
    baseline=local_job_fields(job(16,0,len(rows)),SESSION)
    for field,value in [('abi_major',2),('abi_minor',1),('flags',1),('reserved',1),
                        ('mode',2),('mode',3),('digest',baseline['digest']^1),
                        ('tokens',2),('outputs',0),('k',0),('k',65),('k',0xffffffff),
                        ('global_bits',0x7f800000),('global_bits',0x80000000),
                        ('dependency_count',1),('generation',SESSION^(1<<40))]:
        d=deepcopy(baseline);d[field]=value;rows.append((d,[]))
    for bit in range(7):
        d=deepcopy(baseline);d['contract']^=1<<bit;rows.append((d,[]))
    for name,field,value in [('a','address',1),('a','span',31),('a','stride',31),
                             ('a','address',(1<<64)-2),('w','span',7),
                             ('scales','span',0),('y','stride',3),('y','address',0x2000),
                             ('y','span',(1<<64)-1)]:
        d=deepcopy(baseline);d[name][field]=value;rows.append((d,[]))
    # Legal one-past end exactly 2^64; large addresses must never truncate.
    d=deepcopy(baseline);d['a']['address']=(1<<64)-32;d['y']['address']=(1<<64)-36
    rows.append((d,[packet(d,0,[0x3f80]*16,[2]*16,0x38)]))
    changes=[{'k':0,'scale_k':0},{'k':2,'scale_k':2}, {'group':1,'scale_group':1},
             {'epoch':(1<<63),'scale_epoch':(1<<63)}, {'scale_group':1},{'scale_k':9},
             {'scale_epoch':(1<<63)}, {'generation':SESSION^(1<<40)},
             {'context':1<<63},{'request':1<<63},{'position':1<<63},
             {'first':1},{'last':0},{'mask':1},{'scale':127}]
    for change in changes:
        d=local_job_fields(job(32,0,len(rows)),SESSION)
        ps=[packet(d,i,[0x3f80]*16,[2]*16,0x38) for i in range(2)]
        ps[1].update(change);rows.append((d,ps))
    d=local_job_fields(job(17,0,len(rows)),SESSION)
    ps=[packet(d,i,[0x3f80]*16,[2]*16,0x38) for i in range(2)]
    ps[0]['first']=0
    rows.append((d,ps))
    d=local_job_fields(job(17,0,len(rows)),SESSION)
    ps=[packet(d,i,[0x3f80]*16,[2]*16,0x38) for i in range(2)]
    ps[1]['mask']=0xffff
    rows.append((d,ps))
    for a,c,g,scale in [(0x7f7f,7,0,0),(0x4000,2,0x7f7fffff,0x38)]:
        d=local_job_fields(job(1,0,len(rows)),SESSION);d['global_bits']=g
        rows.append((d,[packet(d,0,[a]*16,[c]*16,scale)]))
    # Successful reuse after every rejection/numerical/protocol failure.
    d=local_job_fields(job(17,1,len(rows)),SESSION)
    rows.append((d,[packet(d,i,[0x3f80]*16,[2]*16,0x38) for i in range(2)]))
    return rows


def expected(d,ps):
    if not desc_ok(d):return False,0,False
    aa=[];cc=[];scales=[];fault=False
    for i,p in enumerate(ps):
        template=packet(d,i,p['a'],p['w'],p['scale'])
        fault |= any(p[k]!=template[k] for k in template if k not in ('a','w','scale'))
        n=min(16,d['k']-16*i)
        aa.extend(p['a'][:n]);cc.extend(p['w'][:n]);scales.append(p['scale'])
    try:
        bits=N.strict_row(aa,N.pack_codes(cc),scales,d['global_bits'],
                          profile=N.STRICT_FTZ if d['mode'] else N.STRICT_GRADUAL)
    except N.NumericalFault:fault=True;bits=0
    return True,0 if fault else bits,bool(fault)


TB=r"""
module tb;
  import otpu_nvfp4_job::*;
  logic clk=0;always #5 clk=~clk;
  logic rst=1;
  logic [63:0] session_generation=64'h100000007;
  logic desc_valid=0,desc_ready,reply_valid,reply_ready=0,reply_accepted;
  logic [5:0] reply_fault;logic [63:0] reply_request;
  descriptor_t descriptor,original;
  logic packet_valid=0,packet_ready,packet_scale_valid=0,packet_first,packet_last;
  logic [255:0] packet_a;logic [63:0] packet_w;logic [15:0] packet_mask;
  logic [7:0] packet_scale;
  logic [31:0] packet_group,packet_k,scale_group,scale_k,free_group,free_k;
  logic [63:0] packet_context,packet_generation,packet_request,packet_position,packet_epoch,scale_epoch,free_epoch;
  logic buffer_free,result_valid,result_ready=0,result_success,output_committed,session_invalidated;
  logic [31:0] result_bits;logic [7:0] result_fault;
  logic [63:0] result_context,result_generation,result_request,result_position,result_epoch;
  localparam integer COUNT=@COUNT@;
  logic [1674:0] dv[0:COUNT-1];
  logic [857:0] pv[0:4*COUNT-1];
  logic hd,hr,hp,hy,held_reply=0,held_result=0;
  logic [70:0] previous_reply;
  logic [360:0] previous_result;
  logic [31:0] rng=32'h81625793;
  integer cycles=0,rejected=0,completed=0,failed=0,packet_wait=0,result_stalls=0,prefix_id;
  otpu_nvfp4_local_job dut(.*);

  task automatic step;
    @(posedge clk);
    hd=desc_valid && desc_ready;hr=reply_valid && reply_ready;
    hp=packet_valid && packet_ready;hy=result_valid && result_ready;
    if(rst || session_invalidated) begin
      held_reply=0;held_result=0;
      if(output_committed || buffer_free || result_valid) $fatal(1,"invalidation handshake");
    end else begin
      if(held_reply && (!reply_valid || previous_reply !== {reply_accepted,reply_fault,reply_request}))
        $fatal(1,"descriptor reply changed under stall");
      if(held_result && (!result_valid || previous_result !==
         {result_success,result_bits,result_fault,result_context,result_generation,result_request,result_position,result_epoch}))
        $fatal(1,"private completion changed under stall");
      held_reply=reply_valid && !reply_ready;
      held_result=result_valid && !result_ready;
      previous_reply={reply_accepted,reply_fault,reply_request};
      previous_result={result_success,result_bits,result_fault,result_context,result_generation,result_request,result_position,result_epoch};
      if(output_committed !== (result_valid && result_ready && result_success))
        $fatal(1,"early/private output commit");
      if(buffer_free !== hp || (hp && {free_group,free_k,free_epoch} !== {packet_group,packet_k,packet_epoch}))
        $fatal(1,"packet ownership mismatch");
      if((!result_valid || !result_success) && result_bits!=0) $fatal(1,"private result leaked");
      if(!result_valid && {result_context,result_generation,result_request,result_position,result_epoch}!=0)
        $fatal(1,"private identity leaked");
      if(packet_valid && !packet_ready) packet_wait=packet_wait+1;
      if(held_result) result_stalls=result_stalls+1;
    end
    cycles=cycles+1;
    if(cycles>200000) $fatal(1,"watchdog");
    #1;
  endtask

  task automatic load_packet(input integer address);
    {packet_a,packet_w,packet_mask,packet_scale,packet_first,packet_last,
     packet_group,packet_k,scale_group,scale_k,packet_context,packet_generation,
     packet_request,packet_position,packet_epoch,scale_epoch}=pv[address];
  endtask

  initial begin
    if($bits(descriptor)!=1641) $fatal(1,"descriptor shape drift");
    $readmemh("descriptors.hex",dv);$readmemh("packets.hex",pv);
    repeat(2) @(negedge clk);
    // Sweep reset and transient session changes through reply, arithmetic,
    // epilogue and stalled private completion. First case has K=1.
    for(integer kind=0;kind<2;kind=kind+1) begin
      for(integer phase=0;phase<45;phase=phase+1) begin
        rst=0;session_generation=64'h100000007;reply_ready=0;result_ready=0;
        prefix_id=phase==38 ? 16 : 0; // K17: invalidate after a non-last block finishes
        descriptor=dv[prefix_id][1674:34];original=descriptor;desc_valid=1;
        step();if(!hd) $fatal(1,"prefix descriptor not ready");
        @(negedge clk);desc_valid=0;
        if(phase!=0) begin
          reply_ready=1;step();if(!hr) $fatal(1,"prefix reply");
          @(negedge clk);reply_ready=0;load_packet(4*prefix_id);packet_valid=1;packet_scale_valid=1;
          if(phase!=44) begin
            step();if(!hp) $fatal(1,"prefix packet");
            @(negedge clk);packet_valid=0;packet_scale_valid=0;
            repeat(phase-1) begin step();@(negedge clk);end
          end
        end
        if(kind==0) begin
          rst=1;#1;
          if(reply_valid || desc_ready || packet_ready || result_valid || buffer_free || output_committed)
            $fatal(1,"reset did not gate all interfaces");
          step();@(negedge clk);rst=0;
        end else begin
          session_generation=64'h100000008;
          if(phase>=40 && phase<44) result_ready=1; // invalidation must beat ready-high retirement
          #1;
          if(!session_invalidated || result_valid || output_committed || packet_ready)
            $fatal(1,"session transition did not invalidate");
          step();@(negedge clk);session_generation=64'h100000007;result_ready=0;
          packet_valid=0;packet_scale_valid=0;
          if(!result_valid || result_success || result_fault!=8'h80 || result_generation!=original.generation ||
             result_request!=original.request_id || result_context!=original.context_id ||
             result_position!=original.position || result_epoch!=original.epoch || result_bits!=0)
            $fatal(1,"lost generation not retained");
          repeat(3) begin step();@(negedge clk);end
          result_ready=1;reply_ready=1;step();if(!hy) $fatal(1,"lost completion did not drain");
          @(negedge clk);result_ready=0;reply_ready=0;
        end
        packet_valid=0;packet_scale_valid=0;
        repeat(3) begin step();@(negedge clk);if(result_valid) $fatal(1,"stale output after drain");end
      end
    end
    for(integer j=0;j<COUNT;j=j+1) begin
      descriptor=dv[j][1674:34];original=descriptor;desc_valid=1;
      step();if(!hd) $fatal(1,"descriptor unavailable between jobs");
      @(negedge clk);desc_valid=0;descriptor='1; // mutate live port; snapshot must survive
      repeat(3+j%5) begin step();@(negedge clk);end
      if(!reply_valid || reply_accepted !== dv[j][33] || reply_request!==original.request_id ||
         (reply_accepted ? reply_fault!=0 : reply_fault==0)) $fatal(1,"descriptor verdict j=%0d",j);
      reply_ready=1;step();if(!hr) $fatal(1,"reply missing");
      @(negedge clk);reply_ready=0;
      if(!dv[j][33]) begin
        rejected=rejected+1;
        repeat(3) begin step();@(negedge clk);if(result_valid || packet_ready) $fatal(1,"rejected job allocated work");end
      end else begin
        for(integer p=0;p<int'((original.k+15)>>4);p=p+1) begin
          load_packet(4*j+p);packet_valid=1;packet_scale_valid=0;
          hp=0;
          while(!hp) begin
            rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
            packet_scale_valid=rng[0] && rng[8];
            step();@(negedge clk);
          end
          packet_valid=0;packet_scale_valid=0;
        end
        while(!result_valid) begin step();@(negedge clk);end
        if(result_success !== !dv[j][0] || result_bits !== dv[j][32:1] ||
           result_context!==original.context_id || result_generation!==original.generation ||
           result_request!==original.request_id || result_position!==original.position || result_epoch!==original.epoch ||
           (result_success ? result_fault!=0 : result_fault==0))
          $fatal(1,"completion mismatch j=%0d got=%h fault=%h want=%h fault=%b",j,result_bits,result_fault,dv[j][32:1],dv[j][0]);
        if(!result_success) failed=failed+1;
        repeat(4+j%13) begin step();@(negedge clk);end
        result_ready=1;step();if(!hy) $fatal(1,"completion missing");
        @(negedge clk);result_ready=0;completed=completed+1;
      end
    end
    if(rejected<25 || completed<100 || failed<10 || packet_wait<100 || result_stalls<100)
      $fatal(1,"coverage gate");
    $display("TEST_PASS local_jobs=%0d rejected=%0d completed=%0d failed=%0d reset_phases=45 generation_phases=45 packet_wait=%0d result_stalls=%0d cycles=%0d",
                COUNT,rejected,completed,failed,packet_wait,result_stalls,cycles);
    $finish;
  end
endmodule
"""


@pytest.mark.parametrize('backend',['iverilog','verilator'])
def test_local_job_rtl(tmp_path,backend):
    rows=cases();ds=[];ps=[]
    for d,packets in rows:
        accepted,bits,fault=expected(d,packets)
        ds.append(f"{packed([(desc_pack(d),1641),(int(accepted),1),(bits,32),(int(fault),1)]):0419x}")
        ps.extend(f'{packet_pack(p):0215x}' for p in packets)
        ps.extend('0'*215 for _ in range(4-len(packets)))
    (tmp_path/'descriptors.hex').write_text('\n'.join(ds)+'\n')
    (tmp_path/'packets.hex').write_text('\n'.join(ps)+'\n')
    sources=[FP,ROOT/'rtl/mxu/otpu_nvfp4_job.sv',STRICT,ACC,ROOT/'rtl/mxu/otpu_nvfp4_local_job.sv']
    simulate(tmp_path,backend,sources,TB.replace('@COUNT@',str(len(rows))))
