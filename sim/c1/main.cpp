// C1 cycle harness. Memory timing is an explicit abstract per-PC service model;
// all request/beat/result events below are sampled from RTL ready/valid handshakes.
#include "Votpu_c1_top.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#ifndef C1_ENGINES
#define C1_ENGINES 16
#endif
#ifndef C1_SLOTS
#define C1_SLOTS 4
#endif
constexpr int E=C1_ENGINES, S=C1_SLOTS;
constexpr int SW=S<=2?1:S<=4?2:3;
template<class T> uint32_t word(const T& p,int i) {
  if constexpr(std::is_integral_v<T>) return i<2?uint32_t(uint64_t(p)>>(32*i)):0;
  else return p[i];
}
template<class T> void word(T& p,int i,uint32_t v) {
  if constexpr(std::is_integral_v<T>) {
    uint64_t m=uint64_t(0xffffffff)<<(32*i);
    p=T((uint64_t(p)&~m)|(uint64_t(v)<<(32*i)));
  } else p[i]=v;
}
template<class T> uint32_t get(const T& p,int b,int n) {
  uint64_t x=word(p,b/32);
  if (b%32+n>32) x|=uint64_t(word(p,b/32+1))<<32;
  return uint32_t((x>>(b%32)) & (n==32?0xffffffffULL:((1ULL<<n)-1)));
}
template<class T> void set(T& p,int b,int n,uint32_t x) {
  uint64_t mask=n==32?0xffffffffULL:((1ULL<<n)-1);
  uint64_t v=word(p,b/32);
  if (b%32+n>32) v|=uint64_t(word(p,b/32+1))<<32;
  v=(v&~(mask<<(b%32)))|((uint64_t(x)&mask)<<(b%32));
  word(p,b/32,uint32_t(v));
  if (b%32+n>32) word(p,b/32+1,uint32_t(v>>32));
}
struct Act {uint32_t addr,da,sa;std::array<uint32_t,8>w;};
struct Beat {uint32_t dw;std::array<uint32_t,8>w;};
struct Row {uint32_t sw;std::vector<Beat> beats;};
struct Request {uint32_t slot,row,gen,group;uint64_t due;};
int main(int argc,char**argv) {try {
  if(argc!=11) throw std::runtime_error("input trace latency interval pc0delay seed req% out% reset_after");
  std::ifstream input(argv[1]);std::ofstream trace(argv[2]);
  if(!input||!trace) throw std::runtime_error("file open failed");
  int latency=std::stoi(argv[3]), interval=std::stoi(argv[4]),delay=std::stoi(argv[5]);
  std::mt19937 rng(std::stoul(argv[6]));
  int reqpct=std::stoi(argv[7]), outpct=std::stoi(argv[8]),reset_after=std::stoi(argv[9]);
  int interleave=std::stoi(argv[10]);
  uint32_t n0,n,kb,k0,ab,na;input>>n0>>n>>kb>>k0>>ab>>na;
  std::vector<Act> acts(na);std::map<uint32_t,Row> rows;
  for(auto&a:acts){input>>a.addr>>a.da>>a.sa;for(auto&w:a.w)input>>w;}
  for(uint32_t i=0;i<n;i++) {uint32_t id;Row r;input>>id>>r.sw;r.beats.resize(kb*4);
    for(auto&b:r.beats){input>>b.dw;for(auto&w:b.w)input>>w;}rows.emplace(id,r);}
  if(!input)throw std::runtime_error("malformed fixture");
  Votpu_c1_top top;std::vector<Request> pending[E];uint64_t nextbeat[E]={};uint32_t previous_occ[E]={};
  uint64_t cycle=0,command_cycle=0,final_cycle=0;uint32_t generation=1,retired=0;
  size_t act_index=0;bool sent=false,reset_done=false,held=false;int reset_left=2;
  uint32_t heldrow=0,heldgen=0,heldbits=0,maxocc=0;uint64_t drops=0,beats=0,requests=0;
  for(;cycle<2000000;cycle++) {
    bool midreset=reset_after>=0&&!reset_done&&sent&&cycle==command_cycle+uint64_t(reset_after);
    if(midreset){reset_left=2;reset_done=true;generation++;sent=false;retired=0;held=false;}
    top.rst=reset_left>0;top.act_we=0;top.cmd_valid=0;top.req_ready=0;top.rsp_valid=0;
    top.out_ready=!top.rst&&(rng()%100<uint32_t(outpct));
    if(!top.rst) {
      if(act_index<acts.size()) {
        const auto&a=acts[act_index];top.act_we=1;top.act_addr=a.addr;top.act_da=a.da;top.act_sa=a.sa;
        for(int j=0;j<8;j++)set(top.act_data,j*32,32,a.w[j]);
      } else if(!sent){top.cmd_valid=1;top.cmd_n0=n0;top.cmd_n=n;top.cmd_kb=kb;top.cmd_k0=k0;
        top.cmd_ab=ab;top.cmd_generation=generation;}
      for(int e=0;e<E;e++) if(rng()%100<uint32_t(reqpct))top.req_ready|=1U<<e;
    }
    int chosen[E];for(int e=0;e<E;e++) {
      chosen[e]=-1;if(top.rst||cycle<nextbeat[e])continue;
      std::vector<int> ready;for(size_t j=0;j<pending[e].size();j++)if(pending[e][j].due<=cycle)ready.push_back(j);
      if(ready.empty())continue;
      int j=ready[interleave?rng()%ready.size():0];chosen[e]=j;const auto&r=pending[e][j];
      const auto&data=rows.at(r.row);const auto&b=data.beats[r.group];top.rsp_valid|=1U<<e;
      set(top.rsp_slot,e*SW,SW,r.slot);set(top.rsp_row,e*32,32,r.row);set(top.rsp_generation,e*32,32,r.gen);
      set(top.rsp_group,e*10,10,r.group);set(top.rsp_dw,e*8,8,b.dw);set(top.rsp_sw,e*32,32,data.sw);
      for(int k=0;k<8;k++)set(top.rsp_weights,e*256+k*32,32,b.w[k]);
    }
    top.clk=0;top.eval();
    if(!top.rst) {
      if(held&&(!top.out_valid||top.out_row!=heldrow||top.out_generation!=heldgen||top.out_bits!=heldbits))
        throw std::runtime_error("unstable output under backpressure");
      held=top.out_valid&&!top.out_ready;heldrow=top.out_row;heldgen=top.out_generation;heldbits=top.out_bits;
      if(top.cmd_valid&&top.cmd_ready){sent=true;command_cycle=cycle;
        trace<<"{\"event\":\"command\",\"cycle\":"<<cycle<<",\"generation\":"<<generation<<"}\n";}
      if(top.act_we)act_index++;
      for(int e=0;e<E;e++) {
        auto occ=get(top.occupancy,e*(SW+1),SW+1);if(occ>S)throw std::runtime_error("slot bound exceeded");
        if(occ>maxocc)maxocc=occ;
        if(occ!=previous_occ[e]){trace<<"{\"event\":\"occupancy\",\"cycle\":"<<cycle<<",\"pc\":"<<e
          <<",\"count\":"<<occ<<",\"generation\":"<<generation<<"}\n";previous_occ[e]=occ;}
        if(chosen[e]>=0&&(top.rsp_ready&(1U<<e))) {
          auto&r=pending[e][chosen[e]];bool drop=(top.rsp_drop>>e)&1;beats++;drops+=drop;
          if(drop!=(r.gen!=generation))throw std::runtime_error("unexpected accept/drop response");
          trace<<"{\"event\":\"beat\",\"cycle\":"<<cycle<<",\"pc\":"<<e<<",\"row\":"<<r.row
            <<",\"generation\":"<<r.gen<<",\"group\":"<<r.group<<",\"drop\":"<<(drop?"true":"false")<<"}\n";
          nextbeat[e]=cycle+interval;
          if(drop||++r.group==kb*4)pending[e].erase(pending[e].begin()+chosen[e]);
        }
        if((top.req_valid&top.req_ready)&(1U<<e)) {
          Request r{get(top.req_slot,e*SW,SW),get(top.req_row,e*32,32),get(top.req_generation,e*32,32),0,
                    cycle+uint64_t(latency+(e==0?delay:0))};
          if(r.row%E!=uint32_t(e)||!rows.count(r.row)||r.gen!=generation||
             get(top.req_kb,e*8,8)!=kb||get(top.req_k0,e*8,8)!=k0)throw std::runtime_error("bad request tag");
          for(const auto&q:pending[e])if(q.gen==r.gen&&(q.slot==r.slot||q.row==r.row))throw std::runtime_error("duplicate live request");
          pending[e].push_back(r);requests++;
          trace<<"{\"event\":\"request\",\"cycle\":"<<cycle<<",\"pc\":"<<e<<",\"slot\":"<<r.slot
            <<",\"row\":"<<r.row<<",\"generation\":"<<r.gen<<"}\n";
        }
      }
      if(top.int_valid&&top.int_ready)trace<<"{\"event\":\"integer\",\"cycle\":"<<cycle<<",\"row\":"<<top.int_row
        <<",\"generation\":"<<top.int_generation<<",\"acc_bits\":"<<top.int_acc<<"}\n";
      if(top.out_valid&&top.out_ready) {
        if(top.out_generation!=generation||top.out_row!=n0+retired)throw std::runtime_error("unordered/stale output");
        trace<<"{\"event\":\"result\",\"cycle\":"<<cycle<<",\"row\":"<<top.out_row<<",\"generation\":"<<top.out_generation
          <<",\"bits\":"<<top.out_bits<<"}\n";retired++;if(retired==n)final_cycle=cycle;
      }
    }
    if(midreset)trace<<"{\"event\":\"reset\",\"cycle\":"<<cycle<<"}\n";
    top.clk=1;top.eval();if(reset_left)reset_left--;
    if(retired==n&&(!reset_after||reset_after<0||reset_done)&&cycle>=final_cycle+2) {
      if(top.busy)throw std::runtime_error("busy after pipeline drained");break;
    }
  }
  if(cycle==2000000)throw std::runtime_error("timeout");
  trace<<"{\"event\":\"summary\",\"cycles\":"<<final_cycle-command_cycle+1<<",\"command_cycle\":"<<command_cycle
    <<",\"end_cycle\":"<<final_cycle+1<<",\"generation\":"<<generation<<",\"max_occupancy\":"<<maxocc
    <<",\"requests\":"<<requests<<",\"served_beats\":"<<beats<<",\"dropped_beats\":"<<drops<<"}\n";
  top.final();return 0;
} catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
