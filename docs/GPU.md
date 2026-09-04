# CUDA backend (`--cuda`) - Phase 10

The engine is CPU-first. Phase 10 adds an **optional** GPU path for the one
kernel that dominates decode - the block-FP8 MoE-expert matmul - behind the
`--cuda` flag. CUDA is never a build or run dependency: no DLL, no device, or a
missing symbol just means every op stays on the CPU.

- **10a**: plumbing + the kernel; every call re-uploads its weights.
- **10b**: `--vram <size>` - a bounded LRU cache of resident expert weights in
  VRAM (`key`-addressed), so a hot expert's ~1.6 MB is uploaded once.

**Measured verdict on the dev box (Quadro T1000 Max-Q, 4 GB): `--cuda` is ~2×
slower than the warm CPU path and 10b does not change that** - see "Result"
below. The backend is kept because it is correct and will help on a GPU with
more VRAM; on 4 GB it is a demo, not a speedup.

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

`--cuda` prints one line - `--cuda: CUDA backend up - <device>` - or the reason it
fell back to CPU.

## The kernel

`y[S,O] = x[S,I] @ dequant(w)ᵀ`, `w` row-major `[O,I]` E4M3, one f32 scale per
128×128 block. One thread block per output row; threads stride over `I`;
accumulation is **f32 within a 128-column block, f64 across blocks** with the
block scale folded once per block - matching `src/ops/fp8.zig` so greedy decode
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

### `--cuda-verify`

Recompute each GPU matmul on the CPU and print the divergence; a summary line
reports the worst `relΔ` over the run. `max|Δ| ≈ 1e-7` per matmul on the real
checkpoint - the GPU arithmetic is not the problem, the byte movement is.

## 10b - VRAM weight cache

`--vram <size>` (default: most of free VRAM) turns on a bounded LRU cache inside
the DLL. Each `Fp8Matrix` carries a stable `key` (`1 + ((layer*512 + expert)*4 +
role)`); a keyed call checks the resident set - a **hit** skips the ~1.6 MB
weight upload and runs the kernel straight from VRAM, a **miss** uploads and
admits (evicting the LRU slot). `--cuda` prints
`cuda: N matmuls  VRAM cache X% hit (R resident experts)  M MiB pushed H2D`.

## Result (T1000 Max-Q, 4 GB, real checkpoint, warm)

| | decode tok/s | notes |
|---|---|---|
| CPU (12 threads) | **0.75** | memory-bandwidth-bound |
| `--cuda` 10a | 0.35 | re-uploads every call |
| `--cuda` 10b | 0.35 | VRAM cache **26 % hit** - no better |

Why 10b doesn't help *here*: 4 GB holds ~1850 weight slots ≈ **13 experts per
layer**, but a short generation routes to ~50+ distinct experts per layer, so the
cache thrashes (26 % hit) and 74 % of calls still push 1.6 MB over the laptop's
PCIe link. The per-call synchronous kernel launch (~1700 / token) and pageable
(un-pinned, mmap-backed) H2D copies are a fixed tax on top.

The MoE-in-VRAM approach needs a card that can hold most of the working set -
**8 GB+** would put the hit rate where it pays. On 4 GB the alternatives are 10c
(pinned staging + async streams + batched multi-expert kernels) or a different
tiering (resident dense weights + `lm_head` in VRAM, experts on the CPU). Both
are open; neither is obviously worth it on a 35 W Max-Q part.
