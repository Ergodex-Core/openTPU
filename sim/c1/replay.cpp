// E1/S1 command boundary test: a new generation is offered while the old command's
// final fp32 result is held. It must not be accepted before complete retirement.
#include "Votpu_c1_top.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
int main(){try{
  Votpu_c1_top t;
  int loaded=0,commands=0,outputs=0,group=0,hold=0,last_first=-1;
  bool pending=false;uint32_t row=0,generation=0;
  for(int cycle=0;cycle<2000;cycle++){
    t.rst=cycle<2;t.act_we=0;t.cmd_valid=0;t.req_ready=1;t.rsp_valid=pending&&!t.rst;
    if(t.rsp_valid){t.rsp_slot=0;t.rsp_row=row;t.rsp_generation=generation;t.rsp_group=group;
      t.rsp_dw=0;t.rsp_sw=0x3f800000;
      for(int j=0;j<8;j++)t.rsp_weights[j]=generation==100?0x01010101:0xffffffff;}
    if(!t.rst&&loaded<4){t.act_we=1;t.act_addr=loaded;t.act_da=0;t.act_sa=0x3f800000;
      for(int j=0;j<8;j++)t.act_data[j]=0x01010101;}
    else if(!t.rst&&commands<2){t.cmd_valid=1;t.cmd_n0=commands==0?3:9;t.cmd_n=commands==0?3:2;
      t.cmd_generation=commands==0?100:200;t.cmd_kb=1;t.cmd_k0=0;t.cmd_ab=0;}
    // Stall specifically the old command's final result for ten cycles. Presentation
    // of the new command overlaps the entire old command, including this stall.
    t.out_ready=1;t.clk=0;t.eval();
    if(t.out_valid&&t.out_generation==100&&t.out_row==5&&hold<10){t.out_ready=0;hold++;t.eval();}
    if(!t.rst){
      if(t.cmd_valid&&t.cmd_ready){
        if(commands==1&&(last_first<0||cycle<=last_first))throw std::runtime_error("early next command");
        commands++;
      }
      if(t.act_we)loaded++;
      if(t.rsp_valid&&t.rsp_ready){if(t.rsp_drop)throw std::runtime_error("response dropped");
        if(++group==4){pending=false;group=0;}}
      if(t.req_valid&&t.req_ready){if(pending)throw std::runtime_error("S1 overlap");
        pending=true;row=t.req_row;generation=t.req_generation;group=0;}
      if(t.out_valid){
        bool first=outputs<3;uint32_t expected_row=first?3+outputs:9+outputs-3;
        if(t.out_row!=expected_row||t.out_generation!=(first?100U:200U)||
           t.out_bits!=(first?0x43000000U:0xc3000000U))throw std::runtime_error("output/tag instability");
        if(t.out_ready){outputs++;if(outputs==3)last_first=cycle;}
      }
    }
    t.clk=1;t.eval();
    if(outputs==5&&!t.busy){if(commands!=2||hold!=10)throw std::runtime_error("boundary not exercised");
      t.final();std::cout<<"two commands; five exact outputs; ten stalled final-output cycles\n";return 0;}
  }
  throw std::runtime_error("timeout");
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
