// C2 standalone int8 dual-mode MXU integration experiment.
// ONE product bank serves both modes: legacy = shared W / distinct ACT columns;
// decode = distinct W rows / broadcast ACT[0]. Not an encoded production opcode.
// A command is one row group. All its needed operand/scale chunks must arrive
// before the non-stallable array starts. Two banks overlap the next group's fill
// with the current group's compute. Independent ordered ingress streams may stall
// arbitrarily. Same-mode groups overlap; a mode change drains all commits.
// The external ACC interface supplies old TMEM words with old_valid. Commit is a
// four-wide ready/valid transaction (bank arbitration belongs to its adapter).
module otpu_c2_mxu
  import otpu_fp::*;
#(parameter int D=128, C=4, DEPTH=256, RF=8) (
  input logic clk, rst,
  input logic cmd_valid, cmd_decode, cmd_acc,
  output logic cmd_ready,
  input logic [8:0] cmd_kb,
  input logic [C-1:0] cmd_mask,
  input logic [31:0] cmd_tag,
  input logic [C-1:0][31:0] cmd_addr,
  input logic [C-1:0] w_valid, s_valid, a_valid,
  output logic [C-1:0] w_ready, s_ready, a_ready,
  input logic [C-1:0][D*8-1:0] w_data, a_data,
  input logic [C-1:0][7:0] s_dw,
  input logic [C-1:0][31:0] s_sw, a_da, a_sa,
  output logic old_req, old_ready,
  output logic [31:0] old_tag,
  output logic [C-1:0] old_mask,
  output logic [C-1:0][31:0] old_addr,
  input logic old_valid,
  input logic [C-1:0][31:0] old_data,
  output logic out_valid, out_decode, out_acc,
  input logic out_ready,
  output logic [31:0] out_tag,
  output logic [C-1:0] out_mask,
  output logic [C-1:0][31:0] out_addr, out_data,
  output logic [C-1:0][47:0] out_integer,
  output logic pf_advance, pf_mode_wait, pf_credit_wait, pf_old_wait, pf_out_wait,
  output logic [C-1:0] pf_w_wait, pf_s_wait, pf_a_wait,
  output logic [15:0] pf_live, pf_results,
  output logic [C-1:0][8:0] pf_w_level, pf_s_level, pf_a_level
);
  localparam int CLS=D/8, LROW=CLS+C+5, EP=12;
  localparam int AW=$clog2(DEPTH), FW=$clog2(RF);
  initial begin
    if (D != 128 || C != 4) $fatal(1,"C2 milestone supports D=128 C=4");
    if (DEPTH < 2 || DEPTH > 256 || (DEPTH & (DEPTH-1)) != 0)
      $fatal(1,"C2 operand depth must be a power of two <=256");
    if (RF < 2 || (RF & (RF-1)) != 0) $fatal(1,"C2 RF must be power of two");
  end
  typedef struct packed {
    logic decode, acc;
    logic [31:0] tag;
    logic [C-1:0] mask;
    logic [C-1:0][31:0] addr;
  } meta_t;
  typedef struct packed {
    meta_t meta;
    logic [C-1:0][47:0] x;
    logic [C-1:0][31:0] sw, sa;
  } result_t;
  meta_t command, fill_command, command_bank[2];
  logic filling, mode, mode_valid, fill_bank, compute_bank;
  logic [1:0] occupied, filled;
  logic [8:0] kb, fkb, kb_bank[2], issued;
  logic [C-1:0][8:0] wn, sn, an;
  logic [D*8-1:0] wm[2][C][DEPTH], am[2][C][DEPTH];
  logic [39:0] sm[2][C][DEPTH];
  logic [63:0] asc[2][C][DEPTH];
  logic [C-1:0] need_w, need_a;
  logic fill_complete;
  logic [FW:0] live, rn;
  logic [FW-1:0] rh, rt;
  result_t results[RF], head;
  wire accept=cmd_valid && cmd_ready;
  wire commit=out_valid && out_ready;
  assign command=command_bank[compute_bank];
  assign fill_command=command_bank[fill_bank];
  assign kb=kb_bank[compute_bank];
  assign fkb=kb_bank[fill_bank];
  assign cmd_ready=!rst && !filling && !occupied[fill_bank] && live < RF &&
                   (!mode_valid || cmd_decode == mode || live == 0);
  assign pf_mode_wait=cmd_valid && mode_valid && cmd_decode != mode && live != 0;
  assign pf_credit_wait=cmd_valid && !filling && live == RF;
  assign pf_live=16'(live);
  assign pf_results=16'(rn);
  assign pf_w_level=wn;
  assign pf_s_level=sn;
  assign pf_a_level=an;
  always_comb begin
    need_w=fill_command.decode ? fill_command.mask : C'(1);
    need_a=fill_command.decode ? C'(1) : fill_command.mask;
    fill_complete=filling;
    for (int c=0;c<C;c++) begin
      w_ready[c]=filling && need_w[c] && wn[c]<fkb;
      s_ready[c]=filling && need_w[c] && sn[c]<fkb;
      a_ready[c]=filling && need_a[c] && an[c]<fkb;
      pf_w_wait[c]=filling && need_w[c] && wn[c]<fkb;
      pf_s_wait[c]=filling && need_w[c] && sn[c]<fkb;
      pf_a_wait[c]=filling && need_a[c] && an[c]<fkb;
      if ((need_w[c] && (wn[c]+9'(w_valid[c]&&w_ready[c])!=fkb ||
                          sn[c]+9'(s_valid[c]&&s_ready[c])!=fkb)) ||
          (need_a[c] && an[c]+9'(a_valid[c]&&a_ready[c])!=fkb)) fill_complete=0;
    end
  end
  assign pf_advance=filled[compute_bank];
  // S0: choose operands before the shared physical PEs. The invalid columns are
  // masked in bm2 and at retirement; no input/read is requested for their padding.
  logic [C-1:0][D*8-1:0] a0, w0;
  logic [C-1:0][31:0] da0, sa0, sw0;
  logic [C-1:0][7:0] dw0;
  always_comb for (int c=0;c<C;c++) begin
    a0[c]=am[compute_bank][command.decode ? 0:c][issued[AW-1:0]];
    {da0[c],sa0[c]}=asc[compute_bank][command.decode ? 0:c][issued[AW-1:0]];
    w0[c]=wm[compute_bank][command.decode ? c:0][issued[AW-1:0]];
    {sw0[c],dw0[c]}=sm[compute_bank][command.decode ? c:0][issued[AW-1:0]];
  end
  logic [C-1:0][D-1:0][7:0] wd;
  logic [C-1:0][31:0] da1, sw1, sap1;
  logic [C-1:0][7:0] dw1;
  logic [C-1:0] valid1;
  logic [C-1:0][7:0][16:0] bm2;
  always_ff @(posedge clk) begin
    wd<=w0; da1<=da0; dw1<=dw0; sw1<=sw0;
    valid1<=command.mask & {C{pf_advance}};
    for(int c=0;c<C;c++) begin
      sap1[c]<=fp_mul(sa0[c],32'h37800000); // exact repo Sa * 2^-16
      for(int g=0;g<8;g++) begin
        int sh;
        sh=16-int'(dw1[c])-int'(da1[c][(g/2)*8+:8]);
        bm2[c][g]<=(valid1[c] && sh>=0) ? 17'(17'd1<<sh) : 17'd0;
      end
    end
  end
  logic signed [23:0] preg[C][D];
  logic [47:0] pc[C][D];
  logic [C-1:0][47:0] integers;
  for(genvar c=0;c<C;c++) begin:g_column
    for(genvar i=0;i<D;i++) begin:g_product
      logic [7:0] aa,ww;
      otpu_skew #(.W(8),.N(i%CLS+c)) u_a(.clk,.en(1'b1),.d(a0[c][i*8+:8]),.q(aa));
      otpu_skew #(.W(8),.N(i%CLS+c)) u_w(.clk,.en(1'b1),.d(wd[c][i]),.q(ww));
      otpu_pe #(.FIRST(i%CLS==0),.LAST(i%CLS==CLS-1),.WREG(0)) u_pe(
        .clk,.en(1'b1),.act(aa),.wlo(ww),.whi(8'd0),.sel(1'b0),
        .pcin(i%CLS==0 ? 48'd0 : pc[c][i==0 ? 0:i-1]),.pcout(pc[c][i]),.p(preg[c][i]));
    end
    logic [7:0][23:0] ca;
    logic [7:0][16:0] bm;
    logic first;
    logic [47:0] acc;
    for(genvar g=0;g<8;g++) assign ca[g]=preg[c][g*CLS+CLS-1];
    otpu_skew #(.W(136),.N(CLS-1+c)) u_b(.clk,.en(1'b1),.d(bm2[c]),.q(bm));
    otpu_skew #(.W(1),.N(CLS+4+c)) u_f(.clk,.en(1'b1),
      .d(pf_advance && issued==0),.q(first));
    otpu_mxce #(.CPG(1)) u_ce(.clk,.a(ca),.d(192'd0),.bm,.first,.acc);
    otpu_skew #(.W(48),.N(C-1-c)) u_align(.clk,.en(1'b1),.d(acc),.q(integers[c]));
  end
  meta_t tail_meta;
  logic [C-1:0][31:0] tail_sw,tail_sa;
  logic [LROW-1:0] row_valid;
  otpu_delay #(.W($bits(meta_t)),.N(LROW)) u_meta(.clk,.en(1'b1),.d(command),.q(tail_meta));
  otpu_skew #(.W(C*64),.N(LROW-1)) u_scales(.clk,.en(1'b1),
    .d({sw1,sap1}),.q({tail_sw,tail_sa}));
  wire push=row_valid[LROW-1];
  assign head=results[rh];
  wire epi_en=!out_valid || out_ready;
  wire pop=epi_en && rn!=0 && (!head.meta.acc || old_valid);
  assign old_req=rn!=0 && head.meta.acc;
  assign old_ready=epi_en && old_req;
  assign old_tag=head.meta.tag;
  assign old_mask=head.meta.mask;
  assign old_addr=head.meta.addr;
  assign pf_old_wait=old_ready && !old_valid;
  assign pf_out_wait=out_valid && !out_ready;
  logic [EP-1:0] ev;
  meta_t emeta;
  logic [C-1:0][47:0] ex;
  logic [C-1:0][31:0] plain,sum;
  for(genvar c=0;c<C;c++) begin:g_epilogue
    logic [31:0] y,old;
    otpu_mx_lane #(.LM(2)) u_lane(.clk,.en(epi_en),.x(head.x[c]),.sw(head.sw[c]),.sa(head.sa[c]),.y);
    otpu_delay #(.W(32),.N(8)) u_old(.clk,.en(epi_en),.d(old_data[c]),.q(old));
    otpu_fadd #(.LAT(4)) u_add(.clk,.en(epi_en),.a(old),.b(y),.y(sum[c]));
    otpu_delay #(.W(32),.N(4)) u_plain(.clk,.en(epi_en),.d(y),.q(plain[c]));
    assign out_data[c]=emeta.acc ? sum[c]:plain[c];
  end
  otpu_delay #(.W($bits(meta_t)),.N(EP)) u_emeta(.clk,.en(epi_en),.d(head.meta),.q(emeta));
  otpu_delay #(.W(C*48),.N(EP)) u_ex(.clk,.en(epi_en),.d(head.x),.q(ex));
  assign out_valid=ev[EP-1];
  assign out_tag=emeta.tag;
  assign out_mask=emeta.mask;
  assign out_addr=emeta.addr;
  assign out_decode=emeta.decode;
  assign out_acc=emeta.acc;
  assign out_integer=ex;
  always_ff @(posedge clk) begin
    if(rst) begin
      filling<=0;mode_valid<=0;mode<=0;issued<=0;fill_bank<=0;compute_bank<=0;
      occupied<=0;filled<=0;
      for(int b=0;b<2;b++) begin command_bank[b]<='0;kb_bank[b]<=0;end
      wn<='0;sn<='0;an<='0;live<=0;rn<=0;rh<=0;rt<=0;
      row_valid<='0;ev<='0;
    end else begin
      row_valid<={row_valid[LROW-2:0],pf_advance && issued+1==kb};
      if(epi_en) ev<={ev[EP-2:0],pop};
      case({accept,commit})
        2'b10:live<=live+1'b1;
        2'b01:live<=live-1'b1;
        default:;
      endcase
      if(accept) begin
        filling<=1;mode<=cmd_decode;mode_valid<=1;kb_bank[fill_bank]<=cmd_kb;
        occupied[fill_bank]<=1;
        command_bank[fill_bank]<='{decode:cmd_decode,acc:cmd_acc,tag:cmd_tag,mask:cmd_mask,addr:cmd_addr};
        wn<='0;sn<='0;an<='0;
      end
      for(int c=0;c<C;c++) begin
        if(w_valid[c] && w_ready[c]) begin wm[fill_bank][c][wn[c][AW-1:0]]<=w_data[c];wn[c]<=wn[c]+1'b1;end
        if(s_valid[c] && s_ready[c]) begin sm[fill_bank][c][sn[c][AW-1:0]]<={s_sw[c],s_dw[c]};sn[c]<=sn[c]+1'b1;end
        if(a_valid[c] && a_ready[c]) begin
          am[fill_bank][c][an[c][AW-1:0]]<=a_data[c];asc[fill_bank][c][an[c][AW-1:0]]<={a_da[c],a_sa[c]};an[c]<=an[c]+1'b1;
        end
      end
      if(fill_complete) begin
        filling<=0;filled[fill_bank]<=1;fill_bank<=!fill_bank;
      end
      if(pf_advance) begin
        issued<=issued+1'b1;
        if(issued+1==kb) begin
          issued<=0;filled[compute_bank]<=0;occupied[compute_bank]<=0;compute_bank<=!compute_bank;
        end
      end
      if(push) begin
        results[rt]<='{meta:tail_meta,x:integers,sw:tail_sw,sa:tail_sa};rt<=rt+1'b1;
      end
      if(pop) rh<=rh+1'b1;
      case({push,pop})
        2'b10:rn<=rn+1'b1;
        2'b01:rn<=rn-1'b1;
        default:;
      endcase
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk) if(!rst) begin
    if(accept && (cmd_kb==0 || cmd_kb>DEPTH || cmd_mask==0)) $fatal(1,"C2 invalid command");
    if(accept && mode_valid && cmd_decode!=mode && live!=0) $fatal(1,"C2 mode changed before drain");
    if(push && rn==RF && !pop) $fatal(1,"C2 result overflow");
    if(commit && live==0) $fatal(1,"C2 unreserved commit");
    for(int c=0;c<C;c++) begin
      if(s_valid[c] && s_ready[c] && sn[c]!=0 && s_sw[c]!=sm[fill_bank][c][0][39:8])
        $fatal(1,"C2 Sw changes within a row");
      if(a_valid[c] && a_ready[c] && an[c]!=0 && a_sa[c]!=asc[fill_bank][c][0][31:0])
        $fatal(1,"C2 Sa changes within a row");
    end
    if(filling && (!occupied[fill_bank] || filled[fill_bank])) $fatal(1,"C2 fill bank ownership");
    if(pf_advance && (!occupied[compute_bank] || (filling && fill_bank==compute_bank)))
      $fatal(1,"C2 compute bank ownership");
  end
`endif
endmodule
