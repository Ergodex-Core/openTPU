// Explicit-width PCIe accesses. No device management or image loading here.
#include <stdint.h>
#include <stddef.h>
#include <smmintrin.h>
#if !defined(__x86_64__)
#error F2 host MMIO implementation currently requires x86-64
#endif
uint32_t otpu_reg_read(volatile void *base, size_t off) {
    uint32_t v=*(volatile uint32_t *)((volatile uint8_t *)base+off);
    __asm__ __volatile__("lfence" ::: "memory");
    return v;
}
void otpu_reg_write(volatile void *base, size_t off, uint32_t v) {
    *(volatile uint32_t *)((volatile uint8_t *)base+off)=v;
    __asm__ __volatile__("sfence" ::: "memory");
}
void otpu_mem_read(volatile void *base,size_t off,void *dest,size_t n) {
    volatile uint8_t *s=(volatile uint8_t *)base+off;
    uint8_t *d=dest;
    while(n && ((uintptr_t)s&7)) { *d++=*s++; --n; }
    while(n>=8) {
        uint64_t v=*(volatile uint64_t *)s;
        __builtin_memcpy(d,&v,8);s+=8;d+=8;n-=8;
    }
    while(n--) *d++=*s++;
    __asm__ __volatile__("lfence" ::: "memory");
}
void otpu_mem_write(volatile void *base,size_t off,const void *src,size_t n) {
    volatile uint8_t *d=(volatile uint8_t *)base+off;
    const uint8_t *s=src;
    while(n && ((uintptr_t)d&7)) { *d++=*s++; --n; }
    while(n>=8) {
        uint64_t v;__builtin_memcpy(&v,s,8);
        *(volatile uint64_t *)d=v;s+=8;d+=8;n-=8;
    }
    while(n--) *d++=*s++;
    __asm__ __volatile__("sfence" ::: "memory");
}

int otpu_stream_read_supported(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse4.1") != 0;
}

// Only for ordinary HBM through its WC mapping, never control registers.
// Consume complete 64-byte streaming lines, fence each transfer boundary, and
// keep short/unaligned edges scalar. No read extends past the requested range.
__attribute__((target("sse4.1")))
void otpu_mem_read_stream(volatile void *base,size_t off,void *dest,size_t n) {
    volatile uint8_t *s=(volatile uint8_t *)base+off;
    uint8_t *d=dest;
    _mm_mfence();
    while(n && ((uintptr_t)s&63)) { *d++=*s++; --n; }
    while(n>=64) {
        __m128i a=_mm_stream_load_si128((__m128i *)(s+0));
        __m128i b=_mm_stream_load_si128((__m128i *)(s+16));
        __m128i c=_mm_stream_load_si128((__m128i *)(s+32));
        __m128i e=_mm_stream_load_si128((__m128i *)(s+48));
        _mm_storeu_si128((__m128i *)(d+0),a);
        _mm_storeu_si128((__m128i *)(d+16),b);
        _mm_storeu_si128((__m128i *)(d+32),c);
        _mm_storeu_si128((__m128i *)(d+48),e);
        s+=64;d+=64;n-=64;
    }
    while(n--) *d++=*s++;
    _mm_mfence();
}
