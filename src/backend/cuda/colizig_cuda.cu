// colizig CUDA backend — Phase 10a: one kernel (block-FP8 matmul) behind a
// C ABI, compiled by nvcc into a standalone colizig_cuda.dll that the engine
// loads at runtime (std.DynLib). CUDA is never a build dependency of the engine
// itself; `zig build cuda` produces this DLL only when asked.
//
//   nvcc -O3 -arch=sm_75 --shared -o colizig_cuda.dll colizig_cuda.cu -lcudart
//
// The FP8 decode mirrors src/ops/fp8.zig exactly: E4M3 [s:1][e:4][m:3] bias 7,
// value = e4m3(byte) * scale[o/128, i/128]  (per-128x128-block f32 scale).
// The 256-entry decode table is built host-side with the same f64 math and
// uploaded to __constant__ memory, so a byte decodes identically to the CPU LUT.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>  // _mm_getcsr / _mm_setcsr

// The CUDA runtime/driver flips the host thread's SSE control word to
// flush-to-zero + denormals-are-zero on init and around some calls. That would
// silently change every *CPU* float op in the engine afterwards (E4M3 subnormal
// weights are common), so each ABI entry point below saves the caller's MXCSR
// and restores it before returning. RAII guard.
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
// One thread block per output row. Threads stride over I; the per-thread partials
// are block-reduced in shared memory. Accumulation mirrors the CPU reference
// (src/ops/fp8.zig): f32 within a 128-column block, then f64 across blocks with
// the block scale folded once per block — this keeps the GPU logits close enough
// to the CPU/colibri path that greedy decode does not diverge.
__global__ void mm_fp8_kernel(float* __restrict__ y, const float* __restrict__ x,
                              const uint8_t* __restrict__ w, const float* __restrict__ scales,
                              int S, int I, int O, int nbi) {
    const int o = blockIdx.x;
    if (o >= O) return;
    const uint8_t* wrow = w + (size_t)o * I;
    const float* scl = scales + (size_t)(o / FP8_BLOCK) * nbi;
    const int nblocks = (I + FP8_BLOCK - 1) / FP8_BLOCK;

    extern __shared__ float red[];  // blockDim.x floats

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

// ---- persistent device context (buffers reused across calls; 10a does NOT
//      cache weights — every call re-uploads w) --------------------------------

struct Ctx {
    int inited;
    char name[128];
    void* d_x;   size_t d_x_cap;
    void* d_w;   size_t d_w_cap;
    void* d_scl; size_t d_scl_cap;
    void* d_y;   size_t d_y_cap;
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
        if (e == 0) v = (m == 0) ? 0.0 : ldexp((double)m / 8.0, 1 - 7);      // subnormal
        else if (e == 0x0f && m == 0x07) { t[b] = nanf(""); continue; }       // NaN
        else v = ldexp(1.0 + (double)m / 8.0, e - 7);                          // normal
        t[b] = (float)(sgn * v);
    }
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
    g.inited = 1;
    return 0;
}

EXPORT const char* colizig_cuda_device_name(void) {
    return g.inited ? g.name : "";
}

EXPORT void colizig_cuda_shutdown(void) {
    CsrGuard _csr;
    if (!g.inited) return;
    if (g.d_x) cudaFree(g.d_x);
    if (g.d_w) cudaFree(g.d_w);
    if (g.d_scl) cudaFree(g.d_scl);
    if (g.d_y) cudaFree(g.d_y);
    Ctx z; memset(&z, 0, sizeof z);
    g = z;
}

// Synchronous. Returns 0 on success, negative on failure (caller falls back to CPU).
EXPORT int colizig_cuda_matmul_fp8(float* y, const float* x, const unsigned char* w,
                                   const float* scales, int S, int I, int O) {
    CsrGuard _csr;
    if (!g.inited || S < 1 || I < 1 || O < 1) return -1;
    const int nbi = (I + FP8_BLOCK - 1) / FP8_BLOCK;
    const int nbo = (O + FP8_BLOCK - 1) / FP8_BLOCK;
    const size_t xb = (size_t)S * I * sizeof(float);
    const size_t wb = (size_t)O * I;
    const size_t sb = (size_t)nbo * nbi * sizeof(float);
    const size_t yb = (size_t)S * O * sizeof(float);

    if (ensure(&g.d_x, &g.d_x_cap, xb) || ensure(&g.d_w, &g.d_w_cap, wb) ||
        ensure(&g.d_scl, &g.d_scl_cap, sb) || ensure(&g.d_y, &g.d_y_cap, yb))
        return -2;

    if (cudaMemcpy(g.d_x, x, xb, cudaMemcpyHostToDevice) != cudaSuccess) return -3;
    if (cudaMemcpy(g.d_w, w, wb, cudaMemcpyHostToDevice) != cudaSuccess) return -3;
    if (cudaMemcpy(g.d_scl, scales, sb, cudaMemcpyHostToDevice) != cudaSuccess) return -3;

    int threads = 256;
    while (threads > I && threads > 32) threads >>= 1;
    mm_fp8_kernel<<<O, threads, threads * sizeof(float)>>>(
        (float*)g.d_y, (const float*)g.d_x, (const uint8_t*)g.d_w,
        (const float*)g.d_scl, S, I, O, nbi);
    if (cudaGetLastError() != cudaSuccess) return -4;
    if (cudaDeviceSynchronize() != cudaSuccess) return -4;
    if (cudaMemcpy(y, g.d_y, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return -5;
    return 0;
}
