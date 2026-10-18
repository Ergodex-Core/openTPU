// SPDX-License-Identifier: Apache-2.0
// Ramulator MIT-licensed library is an external, pinned build dependency.
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include "base/base.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"
#include "dram/dram.h"

struct Bridge {
  Ramulator::IFrontEnd* frontend;
  Ramulator::IMemorySystem* memory;
  uint64_t cycle = 0;
  int write_tail;
  std::deque<std::pair<uint64_t, uint64_t>> completed;
};
static thread_local std::string error;
extern "C" {
const char* otpu_error() { return error.c_str(); }
void* otpu_create(const char* yaml) {
  try {
    auto cfg = YAML::LoadFile(yaml);
    auto b = std::make_unique<Bridge>();
    b->frontend = Ramulator::Factory::create_frontend(cfg);
    b->memory = Ramulator::Factory::create_memory_system(cfg);
    b->frontend->connect_memory_system(b->memory);
    b->memory->connect_frontend(b->frontend);
    auto d = b->memory->get_ifce<Ramulator::IDRAM>();
    b->write_tail = d->m_timing_vals("nCWL") + d->m_timing_vals("nBL");
    return b.release();
  } catch (const std::exception& e) { error=e.what(); return nullptr; }
}
int otpu_send(void* p, uint64_t id, uint64_t address, int write) {
  try {
    auto b = static_cast<Bridge*>(p);
    return b->frontend->receive_external_requests(write, address, 0,
      [b,id,write](Ramulator::Request&) {
        b->completed.emplace_back(id, b->cycle + (write ? b->write_tail : 0));
      });
  } catch (const std::exception& e) { error=e.what(); return -1; }
}
int otpu_tick(void* p) {
  try { auto b=static_cast<Bridge*>(p); ++b->cycle; b->memory->tick(); return 0; }
  catch (const std::exception& e) { error=e.what(); return -1; }
}
int otpu_poll(void* p, uint64_t* id) {
  auto b=static_cast<Bridge*>(p);
  for (auto i=b->completed.begin(); i!=b->completed.end(); ++i) {
    if (i->second <= b->cycle) { *id=i->first; b->completed.erase(i); return 1; }
  }
  return 0;
}
int otpu_tck_ps(void* p) { return int(static_cast<Bridge*>(p)->memory->get_tCK()*1000+0.5); }
int otpu_tx_bytes(void* p) {
  auto d=static_cast<Bridge*>(p)->memory->get_ifce<Ramulator::IDRAM>();
  return d->m_internal_prefetch_size*d->m_channel_width/8;
}
void otpu_destroy(void* p) {
  auto b=static_cast<Bridge*>(p);
  delete b->memory; delete b->frontend; delete b;
}
}
