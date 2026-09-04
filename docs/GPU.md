# CUDA backend (`--cuda`) — Phase 10a

The engine is CPU-first. Phase 10 adds an **optional** GPU path for the one
kernel that dominates decode — the block-FP8 MoE-expert matmul — behind the
`--cuda` flag. CUDA is never a build or run dependency: no DLL, no device, or a
missing symbol just means every op stays on the CPU.

## Layout

```
src/backend/cuda/colizig_cuda.cu   the kernel + a tiny C ABI (nvcc → DLL)
src/backend/gpu.zig                runtime loader (LoadLibraryA) + dispatch
src/ops/fp8.zig                    matmulFp8 tries gpu.matmulFp8 first
build.zig                          `zig build cuda` step (opt-in)
build_cuda.ps1                     imports the MSVC env, then `zig build cuda`
```

`std.DynLib` has no Windows implementation in Zig 0.16, so `gpu.zig` calls
`LoadLibraryA` / `GetProcAddress` / `FreeLibrary` directly; on non-Windows the
module is an always-unavailable stub.

## Building `colizig_cuda.dll`

Needs the CUDA toolkit (`nvcc`) **and** an MSVC host compiler (`cl.exe`). The
normal `zig build` and `zig build test` never touch this.

```powershell
# imports "x64 Native Tools" env, puts CUDA on PATH, runs `zig build cuda`
powershell -ExecutionPolicy Bypass -File build_cuda.ps1
# or, from a VS "x64 Native Tools" prompt with nvcc on PATH:
zig build cuda -Dcuda-arch=native
```

The DLL installs to `zig-out/bin/` next to `colizig.exe`. `cudart64_*.dll` must be
reachable at runtime (`%CUDA_PATH%\bin` is on PATH after a normal CUDA install).

## Running

```
colizig chat  <MODEL_DIR> --cuda
colizig forward <MODEL_DIR> --tokens ... --cuda
colizig forward <MODEL_DIR> --tokens ... --cuda-verify   # recompute on CPU, print divergence
```

`--cuda` prints one line — `--cuda: CUDA backend up — <device>` — or the reason it
fell back to CPU.

## The kernel

`y[S,O] = x[S,I] @ dequant(w)ᵀ`, `w` row-major `[O,I]` E4M3, one f32 scale per
128×128 block. One thread block per output row; threads stride over `I`;
accumulation is **f32 within a 128-column block, f64 across blocks** with the
block scale folded once per block — matching `src/ops/fp8.zig` so greedy decode
does not drift. The 256-entry E4M3 decode table is built host-side and uploaded
to `__constant__` memory.

Verified: `--cuda` greedy decode on the real checkpoint is **token-for-token
identical** to the CPU path (and to colibri); `--cuda-verify` reports
`max|Δ| ≈ 1e-7` per matmul over a full decode.

### Gotcha fixed here

The CUDA runtime flips the host thread's SSE control word to
**flush-to-zero / denormals-are-zero** on init and around some calls. That
silently changes every *CPU* float op afterwards (E4M3 subnormal weights are
common) and made the first `--cuda` runs produce plausible-but-wrong tokens.
Each ABI entry point in the `.cu` now saves the caller's MXCSR and restores it
before returning (`CsrGuard`).

### Serialisation

The DLL keeps **one** device context with shared upload buffers, so
`gpu.matmulFp8` holds a spinlock across the call and `moe.forwardDense` skips its
per-expert CPU thread fan-out when the GPU is up (the GPU is the parallelism).

## Status / what 10a is not

10a **re-uploads each expert's ~1.5 MB of weights over PCIe on every call** — no
VRAM weight cache. On the T1000 Max-Q (4 GB, ~128 GB/s, PCIe-limited) that makes
`--cuda` decode currently **slower than the warm CPU path**; prefill (fewer,
larger matmuls) already benefits. The number this produces is the baseline that
motivates **10b**: a bounded VRAM expert cache filled from the `.colizig_usage`
priors, so hot experts are resident and only cold ones stream. See
`docs/BENCHMARK.md` for the measured figures.
