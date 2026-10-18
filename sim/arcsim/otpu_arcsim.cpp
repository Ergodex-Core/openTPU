// The ArcSim harness of sim/verilator/tb_top.sv (docs/arcsim.md). ArcSim compiles the whole
// testbench -- tb_top, the design, the memory models -- into a self-driven model whose generated
// driver, tb_top_main(), runs the testbench's own clocks and initial blocks to $finish, so the
// harness is the Verilator executable's command line and nothing of the testbench: it hands the
// +plusargs to the runtime (the same +dir=, +max_cycles=, +trace, +axi_*, +arg<k>= ... rtlsim
// passes Vtb_top), then runs the model. What it adds is the DRAM image store of
// sim/verilator/otpu_native_mem.sv under -DOTPU_ARCSIM: the model's 4 GiB per slice lives here
// (DPI-C otpu_mem_*), loaded from <dir>/dram_<s>.bin (big-endian words, as rtlsim writes them and
// $fread reads them) and dumped to <dir>/dram_out_<s>.bin (native little-endian words, as
// $fwrite "%u" writes them). Zero pages are never touched: calloc's lazily mapped memory keeps
// a 4 GiB slice at the size of what the program used.
//
// Multi-threading: ARC_THREADS=<n> is the runtime's (an eval split into tasks, --partition-eval-
// tasks at build time); the store needs no locks, as a beat belongs to one channel and the
// channels perform one command a cycle.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

extern "C" void circt_sv_set_plusargs(const char *joined);
#ifndef OTPU_ARCSIM_MAIN
#define OTPU_ARCSIM_MAIN tb_top_main             // the generated driver: <top>_main
#endif
extern "C" void OTPU_ARCSIM_MAIN();

namespace {

// Verilator 5.048's $urandom (verilated.cpp): a xoroshiro128** stream seeded through splitmix64
// from (seed << 32) ^ seed, the seed $urandom(seed)'s (whose own draw is the stream's first), or,
// before any, Verilator's default -- two lrand48 words from the C library's unseeded state (the
// same library here). otpu_native_mem's draws under -DOTPU_ARCSIM come from this stream in the
// order Verilator makes them, so both simulators stall and jitter the memory identically.
struct VlRng {
  uint64_t s0 = 0, s1 = 0;
  bool seeded = false;
  static std::pair<uint64_t, uint64_t> splitmix(uint64_t x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return {x, z ^ (z >> 31)};
  }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  void seed(uint64_t s) {
    auto a = splitmix(s);
    auto b = splitmix(a.first);
    s0 = a.second;
    s1 = b.second;
    seeded = true;
  }
  void seed_default() {
    uint64_t hi = static_cast<uint32_t>((lrand48() << 16) ^ lrand48());
    uint64_t lo = static_cast<uint32_t>((lrand48() << 16) ^ lrand48());
    seed((hi << 32) ^ lo);
  }
  uint32_t draw() {
    if (!seeded) seed_default();
    uint64_t r = rotl(s0 * 5, 7) * 9;
    uint64_t t1 = s1 ^ s0;
    uint64_t n0 = rotl(s0, 24) ^ t1 ^ (t1 << 16);
    uint64_t n1 = rotl(t1, 37);
    s0 = n0;
    s1 = n1;
    return static_cast<uint32_t>(r);
  }
};
VlRng g_rng;

struct Poke {
  int64_t at;                             // the cycle (poke_<s>.txt) or the run (pokeb_<s>.txt)
  int64_t word;
  uint32_t value;
};

struct Store {
  uint32_t *words = nullptr;
  int64_t n = 0;
  std::vector<Poke> pokes, pokesb;        // the host's writes during a run / between runs
  size_t next_poke = 0;
};

constexpr int kMaxSlices = 64;
Store g_store[kMaxSlices];
std::string g_dir;

void die(const char *what) {
  std::fprintf(stderr, "otpu_arcsim: %s\n", what);
  std::fflush(nullptr);
  std::_Exit(2);
}

Store &slot(int sid) {
  if (sid < 0 || sid >= kMaxSlices) die("slice id out of range");
  return g_store[sid];
}

}  // namespace

// <dir>/poke_<s>.txt: lines "cycle word value" (hex, in cycle order; the memory model's
// otpu_mem_poke writes each at its cycle); <dir>/pokeb_<s>.txt: lines "run word value" (hex;
// otpu_mem_pokeb writes run r's when run r - 1 ends: tb_top +runs)
static void read_pokes(const std::string &path, std::vector<Poke> &out) {
  FILE *f = std::fopen(path.c_str(), "r");
  if (!f) return;
  unsigned long long a, b, v;
  while (std::fscanf(f, "%llx %llx %llx", &a, &b, &v) == 3)
    out.push_back(Poke{static_cast<int64_t>(a), static_cast<int64_t>(b), static_cast<uint32_t>(v)});
  std::fclose(f);
}

extern "C" unsigned otpu_urandom() { return g_rng.draw(); }

// $urandom(seed): reseed, and the call's own draw (discarded by the model) is the first value
extern "C" void otpu_urandom_seed(int seed) {
  int64_t s = seed;
  g_rng.seed((static_cast<uint64_t>(s) << 32) ^ static_cast<uint64_t>(s));
  (void)g_rng.draw();
}

extern "C" void otpu_mem_init(int sid, int64_t words) {
  Store &s = slot(sid);
  if (s.words) die("otpu_mem_init twice for one slice");
  if (words <= 0) die("otpu_mem_init: no words");
  s.words = static_cast<uint32_t *>(std::calloc(static_cast<size_t>(words), sizeof(uint32_t)));
  if (!s.words) die("otpu_mem_init: out of memory");
  s.n = words;
  if (g_dir.empty()) return;
  read_pokes(g_dir + "/poke_" + std::to_string(sid) + ".txt", s.pokes);
  read_pokes(g_dir + "/pokeb_" + std::to_string(sid) + ".txt", s.pokesb);
  std::string path = g_dir + "/dram_" + std::to_string(sid) + ".bin";
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return;                        // no image: zeros, as the $fread path leaves them
  // big-endian words in the file ($fread's order); read in chunks, swap in place
  constexpr size_t kChunk = 1u << 20;    // words
  int64_t at = 0;
  while (at < words) {
    size_t want = static_cast<size_t>(std::min<int64_t>(kChunk, words - at));
    size_t got = std::fread(s.words + at, sizeof(uint32_t), want, f);
    for (size_t i = 0; i < got; ++i) s.words[at + i] = __builtin_bswap32(s.words[at + i]);
    at += static_cast<int64_t>(got);
    if (got < want) break;
  }
  std::fclose(f);
}

extern "C" void otpu_mem_poke(int sid, int64_t cyc) {
  Store &s = slot(sid);
  while (s.next_poke < s.pokes.size() && s.pokes[s.next_poke].at <= cyc) {
    const Poke &p = s.pokes[s.next_poke++];
    if (p.word < 0 || p.word >= s.n) die("poke beyond the image");
    s.words[p.word] = p.value;
  }
}

extern "C" void otpu_mem_pokeb(int sid, int epoch) {
  Store &s = slot(sid);
  for (const Poke &p : s.pokesb)
    if (p.at == epoch) {
      if (p.word < 0 || p.word >= s.n) die("pokeb beyond the image");
      s.words[p.word] = p.value;
    }
}

extern "C" int otpu_mem_rd(int sid, int64_t i) {
  Store &s = slot(sid);
  if (i < 0 || i >= s.n) die("otpu_mem_rd beyond the image");
  return static_cast<int>(s.words[i]);
}

extern "C" void otpu_mem_wr(int sid, int64_t i, int v) {
  Store &s = slot(sid);
  if (i < 0 || i >= s.n) die("otpu_mem_wr beyond the image");
  s.words[i] = static_cast<uint32_t>(v);
}

// mask: bit k = write byte k (the little-endian byte k of the word, as the channels' byte mask
// names it and as mem_lo[i][8 * k +: 8] <= ... wrote it)
extern "C" void otpu_mem_wrm(int sid, int64_t i, int mask, int v) {
  Store &s = slot(sid);
  if (i < 0 || i >= s.n) die("otpu_mem_wrm beyond the image");
  uint32_t m = 0;
  for (int k = 0; k < 4; ++k)
    if (mask & (1 << k)) m |= 0xFFu << (8 * k);
  s.words[i] = (s.words[i] & ~m) | (static_cast<uint32_t>(v) & m);
}

extern "C" void otpu_mem_dump(int sid) {
  Store &s = slot(sid);
  if (!s.words) die("otpu_mem_dump before otpu_mem_init");
  if (g_dir.empty()) return;
  std::string path = g_dir + "/dram_out_" + std::to_string(sid) + ".bin";
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) die("otpu_mem_dump: cannot open dram_out");
  constexpr size_t kChunk = 1u << 22;    // words
  int64_t at = 0;
  while (at < s.n) {
    size_t want = static_cast<size_t>(std::min<int64_t>(kChunk, s.n - at));
    if (std::fwrite(s.words + at, sizeof(uint32_t), want, f) != want) die("otpu_mem_dump: short write");
    at += static_cast<int64_t>(want);
  }
  if (std::fclose(f) != 0) die("otpu_mem_dump: close failed");
}

int main(int argc, char **argv) {
  std::string joined;
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '+') {
      std::fprintf(stderr, "otpu_arcsim: unknown argument %s (plusargs only)\n", argv[i]);
      return 2;
    }
    if (!joined.empty()) joined += ";";
    joined += argv[i];
    if (std::strncmp(argv[i], "+dir=", 5) == 0) g_dir = argv[i] + 5;
  }
  circt_sv_set_plusargs(joined.c_str());
  OTPU_ARCSIM_MAIN();                    // returns only without a $finish (no events left)
  std::fflush(nullptr);
  return 0;
}
