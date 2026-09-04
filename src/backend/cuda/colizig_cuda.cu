// colizig CUDA backend — block-FP8 matmul behind a C ABI, compiled by nvcc into
// a standalone colizig_cuda.dll that the engine loads at runtime. CUDA is never
// a build dependency of the engine; `zig build cuda` produces this DLL on demand.
//
//   nvcc -O3 -arch=sm_75 --shared -o colizig_cuda.dll colizig_cuda.cu -lcudart
//
// The FP8 decode mirrors src/ops/fp8.zig: E4M3 [s:1][e:4][m:3] bias 7,
// value = e4m3(byte) * scale[o/128, i/128]  (per-128x128-block f32 scale).
//
// Phase 10b: a bounded VRAM weight cache. A matmul call carries a `key`
// (0 = uncached, upload every time — 10a behaviour); a non-zero key is looked up
// in the resident set, so a hot expert's ~1.6 MB of weights are uploaded once and
// then only the tiny activation vectors cross PCIe.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>  // _mm_getcsr / _mm_setcsr

// The CUDA runtime flips the host thread's SSE control word to flush-to-zero on
// init and around some calls, which would silently change every *CPU* float op
// afterwards (E4M3 subnormal weights are common). Each ABI entry saves/restores.
struct CsrGuard {
    unsigned saved;
    CsrGuard() : saved(_mm_getcsr()) {}
    ~CsrGuard() { _mm_setcsr(saved); }
};

#define FP8_BLOCK 128

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

__constant__ float c_e4m3[256];

// y[S,O] = x[S,I] @ dequant(w)^T ; w row-major [O,I] E4M3 ; scales [nblk(O),nblk(I)]
// One thread block per output row. Accumulation mirrors the CPU reference:
// f32 within a 128-column block, f64 across blocks with the block scale folded
// once per block — keeps greedy decode token-for-token identical to the CPU.
__global__ void mm_fp8_kernel(float* __restrict__ y, const float* __restrict__ x,
                              const uint8_t* __restrict__ w, const float* __restrict__ scales,
                              int S, int I, int O, int nbi) {
    const int o = blockIdx.x;
    if (o >= O) return;
    const uint8_t* wrow = w + (size_t)o * I;
    const float* scl = scales + (size_t)(o / FP8_BLOCK) * nbi;
    const int nblocks = (I + FP8_BLOCK - 1) / FP8_BLOCK;

    extern __shared__ float red[];

    for (int s = 0; s < S; ++s) {
        const float* xs = x + (size_t)s * I;
        double total = 0.0;
        for (int b = 0; b < nblocks; ++b) {
            const int base = b * FP8_BLOCK;
            const int end = min(base + FP8_BLOCK, I);
            float acc = 0.0f;
            for (int i = base + threadIdx.x; i < end; i += blockDim.x)
                acc += xs[i] * c_e4m3[wrow[i]];
            red[threadIdx.x] = acc;
            __syncthreads();
            for (int n = blockDim.x >> 1; n > 0; n >>= 1) {
                if (threadIdx.x < n) red[threadIdx.x] += red[threadIdx.x + n];
                __syncthreads();
            }
            if (threadIdx.x == 0) total += (double)red[0] * (double)scl[b];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[(size_t)s * O + o] = (float)total;
    }
}

// ---- device context + VRAM weight cache -----------------------------------

#define MAX_SLOTS 4096

struct Slot {
    uint64_t key;      // 0 = empty
    void* w_dev;   size_t w_cap;
    void* scl_dev; size_t scl_cap;
    uint64_t clock;    // LRU
};

struct Ctx {
    int inited;
    char name[128];
    // scratch for activations (small) and for uncached / overflow weight uploads
    void* d_x;   size_t d_x_cap;
    void* d_y;   size_t d_y_cap;
    void* d_w;   size_t d_w_cap;
    void* d_scl; size_t d_scl_cap;
    // weight cache
    size_t vram_budget;
    size_t vram_used;
    uint64_t lru_clock;
    int n_slots;
    Slot slot[MAX_SLOTS];
    uint64_t hits, misses;
    uint64_t uploaded;   // bytes of weights+scales pushed H2D
};
static Ctx g;

static int ensure(void** p, size_t* cap, size_t need) {
    if (*cap >= need) return 0;
    if (*p) cudaFree(*p);
    if (cudaMalloc(p, need) != cudaSuccess) { *p = 0; *cap = 0; return -1; }
    *cap = need;
    return 0;
}

static void build_lut(float* t) {
    for (int b = 0; b < 256; ++b) {
        int sgn = (b & 0x80) ? -1 : 1;
        int e = (b >> 3) & 0x0f;
        int m = b & 0x07;
        double v;
        if (e == 0) v = (m == 0) ? 0.0 : ldexp((double)m / 8.0, 1 - 7);
        else if (e == 0x0f && m == 0x07) { t[b] = nanf(""); continue; }
        else v = ldexp(1.0 + (double)m / 8.0, e - 7);
        t[b] = (float)(sgn * v);
    }
}

// Return the slot holding `key`, or a slot to (re)use for it. On eviction the
// old buffers are kept (just re-tagged) if they are big enough. Returns -1 only
// if nothing can be allocated at all.
static int cache_slot_for(uint64_t key, size_t wb, size_t sb) {
    // hit?
    for (int i = 0; i < g.n_slots; ++i)
        if (g.slot[i].key == key) return i;

    // room for a fresh slot within budget?
    if (g.n_slots < MAX_SLOTS && g.vram_used + wb + sb <= g.vram_budget) {
        int i = g.n_slots++;
        Slot& s = g.slot[i];
        s.key = 0; s.w_dev = 0; s.w_cap = 0; s.scl_dev = 0; s.scl_cap = 0; s.clock = 0;
        if (ensure(&s.w_dev, &s.w_cap, wb) || ensure(&s.scl_dev, &s.scl_cap, sb)) {
            g.n_slots--;               // roll back
            return -1;
        }
        g.vram_used += s.w_cap + s.scl_cap;
        return i;
    }

    // evict LRU (only among slots whose buffers already fit)
    int victim = -1;
    uint64_t best = ~0ull;
    for (int i = 0; i < g.n_slots; ++i)
        if (g.slot[i].w_cap >= wb && g.slot[i].scl_cap >= sb && g.slot[i].clock < best) {
            best = g.slot[i].clock; victim = i;
        }
    if (victim < 0) {
        // no fitting slot — grow the globally-LRU one
        for (int i = 0; i < g.n_slots; ++i)
            if (g.slot[i].clock < best) { best = g.slot[i].clock; victim = i; }
        if (victim < 0) return -1;
        Slot& s = g.slot[victim];
        g.vram_used -= s.w_cap + s.scl_cap;
        if (ensure(&s.w_dev, &s.w_cap, wb) || ensure(&s.scl_dev, &s.scl_cap, sb)) return -1;
        g.vram_used += s.w_cap + s.scl_cap;
    }
    g.slot[victim].key = 0;  // caller fills it
    return victim;
}

EXPORT int colizig_cuda_init(void) {
    CsrGuard _csr;
    if (g.inited) return 0;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return -1;
    if (cudaSetDevice(0) != cudaSuccess) return -2;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -3;
    snprintf(g.name, sizeof g.name, "%s (sm_%d%d, %zu MiB)",
             prop.name, prop.major, prop.minor, (size_t)(prop.totalGlobalMem >> 20));
    float lut[256];
    build_lut(lut);
    if (cudaMemcpyToSymbol(c_e4m3, lut, sizeof lut) != cudaSuccess) return -4;
    g.vram_budget = 0;  // cache disabled until set
    g.inited = 1;
    return 0;
}

EXPORT const char* colizig_cuda_device_name(void) {
    return g.inited ? g.name : "";
}

// Bytes of VRAM the weight cache may use. 0 disables it (every keyed call then
// behaves like key 0). Clamp to the free VRAM reported by the driver.
EXPORT void colizig_cuda_set_vram_budget(uint64_t bytes) {
    CsrGuard _csr;
    if (!g.inited) return;
    size_t freeb = 0, totalb = 0;
    if (cudaMemGetInfo(&freeb, &totalb) == cudaSuccess && bytes > freeb - (freeb / 8))
        bytes = freeb - (freeb / 8);
    g.vram_budget = bytes;
}

EXPORT void colizig_cuda_stats(uint64_t* hits, uint64_t* misses, uint64_t* uploaded_mib, uint64_t* resident) {
    if (hits) *hits = g.hits;
    if (misses) *misses = g.misses;
    if (uploaded_mib) *uploaded_mib = g.uploaded >> 20;
    if (resident) *resident = (uint64_t)g.n_slots;
}

EXPORT void colizig_cuda_shutdown(void) {
    CsrGuard _csr;
    if (!g.inited) return;
    if (g.d_x) cudaFree(g.d_x);
    if (g.d_y) cudaFree(g.d_y);
    if (g.d_w) cudaFree(g.d_w);
    if (g.d_scl) cudaFree(g.d_scl);
    for (int i = 0; i < g.n_slots; ++i) {
        if (g.slot[i].w_dev) cudaFree(g.slot[i].w_dev);
        if (g.slot[i].scl_dev) cudaFree(g.slot[i].scl_dev);
    }
    Ctx z; memset(&z, 0, sizeof z);
    g = z;
}

// Synchronous. `key` 0 = uncached (upload w every call); non-zero = VRAM-cached.
// Returns 0 on success, negative on failure (caller falls back to CPU).
EXPORT int colizig_cuda_matmul_fp8(float* y, const float* x, const unsigned char* w,
                                   const float* scales, int S, int I, int O, uint64_t key) {
    CsrGuard _csr;
    if (!g.inited || S < 1 || I < 1 || O < 1) return -1;
    const int nbi = (I + FP8_BLOCK - 1) / FP8_BLOCK;
    const int nbo = (O + FP8_BLOCK - 1) / FP8_BLOCK;
    const size_t xb = (size_t)S * I * sizeof(float);
    const size_t wb = (size_t)O * I;
    const size_t sb = (size_t)nbo * nbi * sizeof(float);
    const size_t yb = (size_t)S * O * sizeof(float);

    if (ensure(&g.d_x, &g.d_x_cap, xb) || ensure(&g.d_y, &g.d_y_cap, yb)) return -2;
    if (cudaMemcpy(g.d_x, x, xb, cudaMemcpyHostToDevice) != cudaSuccess) return -3;

    const void* wdev;
    const void* sdev;

    if (key != 0 && g.vram_budget != 0) {
        int idx = cache_slot_for(key, wb, sb);
        if (idx < 0) {
            key = 0;  // cache full of bigger blocks — fall through to scratch
        } else if (g.slot[idx].key == key) {
            g.hits++;
            g.slot[idx].clock = ++g.lru_clock;
            wdev = g.slot[idx].w_dev;
            sdev = g.slot[idx].scl_dev;
        } else {
            g.misses++;
            if (cudaMemcpy(g.slot[idx].w_dev, w, wb, cudaMemcpyHostToDevice) != cudaSuccess) return -3;
            if (cudaMemcpy(g.slot[idx].scl_dev, scales, sb, cudaMemcpyHostToDevice) != cudaSuccess) return -3;
            g.uploaded += wb + sb;
            g.slot[idx].key = key;
            g.slot[idx].clock = ++g.lru_clock;
            wdev = g.slot[idx].w_dev;
            sdev = g.slot[idx].scl_dev;
        }
    }

    if (key == 0 || g.vram_budget == 0) {
        if (ensure(&g.d_w, &g.d_w_cap, wb) || ensure(&g.d_scl, &g.d_scl_cap, sb)) return -2;
        if (cudaMemcpy(g.d_w, w, wb, cudaMemcpyHostToDevice) != cudaSuccess) return -3;
        if (cudaMemcpy(g.d_scl, scales, sb, cudaMemcpyHostToDevice) != cudaSuccess) return -3;
        g.uploaded += wb + sb;
        wdev = g.d_w;
        sdev = g.d_scl;
    }

    int threads = 256;
    while (threads > I && threads > 32) threads >>= 1;
    mm_fp8_kernel<<<O, threads, threads * sizeof(float)>>>(
        (float*)g.d_y, (const float*)g.d_x, (const uint8_t*)wdev, (const float*)sdev, S, I, O, nbi);
    if (cudaGetLastError() != cudaSuccess) return -4;
    // The blocking D2H copy on the default stream waits for the kernel, and
    // returns an error if the kernel faulted — no separate sync needed.
    if (cudaMemcpy(y, g.d_y, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return -5;
    return 0;
}
