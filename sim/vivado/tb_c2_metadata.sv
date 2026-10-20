`timescale 1ns/1ps
module tb;
 reg clk=0,rst=0,accept=0,fill_bank=0,compute_bank=0,read_advance=0;
 reg cmd_decode=0,cmd_acc=0;reg[31:0]cmd_tag=0;reg[3:0]cmd_mask=0;reg[127:0]cmd_addr=0;
 reg[191:0]integers=192'h123456789abcdef0123456789abcdef0123456789abcdef0;
 reg[127:0]tail_sw=128'h123456789abcdeffedcba9876543210,tail_sa=128'hfeedbeef0123456789abcdef12345678;
 reg out_ready=1,old_valid=1;reg[31:0]cfg_rows=0;
 wire out_valid;wire[3:0]result_write;wire[7:0]result_addr;wire[191:0]out_integer;wire[127:0]out_sw,out_sa;
 wire[31:0]result_count;wire done;wire[3:0]fifo_level;
 metadata_faithful dut(.clk,.rst,.accept,.fill_bank,.compute_bank,.read_advance,.cmd_decode,.cmd_tag,.cmd_mask,.integers,.tail_sw,.tail_sa,.out_ready,.cfg_rows,.out_valid,.result_write,.result_addr,.out_integer,.out_sw,.out_sa,.result_count,.done,.fifo_level);
 always #5 clk=~clk;
 reg[165:0]expected[0:63];integer sent=0,received=0,failures=0,cycles=0,ones=0,total=0,batch,k,j,m,guard;
 reg[3:0]masks[0:7];reg[31:0]expected_count=0;reg[3:0]expected_write;
 always @(posedge clk)begin
  if(!rst && $time>200)begin
   cycles=cycles+1;
   if(out_valid&&out_ready)begin
    expected_write=expected[received][165]?expected[received][131:128]:(4'b0001<<expected[received][133:132]);
    if(received>=sent || result_write!==expected_write || result_addr!==expected[received][141:134] || out_integer!==integers || out_sw!==tail_sw || out_sa!==tail_sa)begin
     failures=failures+1;if(failures<=16)$display("META_MISMATCH received=%0d expected_write=%h actual_write=%h expected_addr=%h actual_addr=%h",received,expected_write,result_write,expected[received][141:134],result_addr);
    end
    ones=0;for(j=0;j<4;j=j+1)if(expected[received][128+j])ones=ones+1;
    expected_count=expected_count+ones;received=received+1;
    #1;if(result_count!==expected_count)begin failures=failures+1;$display("COUNT_MISMATCH got=%h expected=%h",result_count,expected_count);end
   end
  end
 end
 initial begin
  masks[0]=1;masks[1]=1;masks[2]=3;masks[3]=7;masks[4]=15;masks[5]=2;masks[6]=4;masks[7]=8;
  for(k=0;k<32;k=k+1)for(j=0;j<4;j=j+1)if((k<16 && j==0) || (k>=16 && masks[k%8][j]))total=total+1;
  cfg_rows=total;
  #200;@(negedge clk);rst=1;repeat(3)@(negedge clk);rst=0;
  for(batch=0;batch<8;batch=batch+1)begin
   for(k=0;k<4;k=k+1)begin
    @(negedge clk);accept=1;fill_bank=sent%2;cmd_mask=(sent<16)?4'b0001:masks[sent%8];cmd_decode=(sent>=16);cmd_acc=0;
    cmd_tag=32'd80+sent;cmd_addr={32'h45670000+32'(sent),32'h89ab0000+32'(sent),32'hcdef0000+32'(sent),32'h13570000+32'(sent)};
    expected[sent]={cmd_decode,cmd_acc,cmd_tag,cmd_mask,cmd_addr};sent=sent+1;
    @(negedge clk);accept=0;compute_bank=fill_bank;read_advance=1;
    @(negedge clk);read_advance=0;
   end
   // Drain all latency stages with deterministic output backpressure.
   guard=0;
   while(received<sent && guard<200)begin
    @(negedge clk);out_ready=(guard%7>=2);old_valid=(guard%5!=0);guard=guard+1;
   end
   @(negedge clk);out_ready=1;old_valid=1;
   if(received!=sent)$fatal(1,"Metadata drain timeout received=%0d sent=%0d",received,sent);
  end
  repeat(45)@(negedge clk);
  if(result_count!==total || !done || fifo_level!==0 || received!=32 || failures!=0)$fatal(1,"Final metadata failure count=%0d wanted=%0d done=%b fifo=%0d received=%0d failures=%0d",result_count,total,done,fifo_level,received,failures);
  $display("METADATA_FAITHFUL_POST_SYNTH_RESULT commands=32 count=%0d received=%0d failures=%0d done=%b",total,received,failures,done);
  $finish;
 end
 initial begin #100000;$fatal(1,"Metadata watchdog");end
endmodule
