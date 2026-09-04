# ops/ - numeric kernels

Small, dependency-free f32 kernels shared by the Qwen4-Exp forward pass. Forms
match colibri so results line up with the reference engine. Phase 2:
single-threaded, scalar + `@Vector` SIMD; thread fan-out is Phase 8.

## Conventions

- **Activations and accumulation are f32.** BF16 weights are widened per element
  (`bf16ToF32`), never rounded back to BF16 mid-dot - same as colibri.
- Sum-of-squares in RMSNorm accumulates in **f64**, then the reciprocal sqrt is
  cast to f32.
- Weight matrices are **row-major `[O, I]`**; `matmul` computes `y[s,o] =
  Σ_i x[s,i]·W[o,i]` (i.e. `x @ Wᵀ`).

## `matmul.zig`

| fn | weights | notes |
|---|---|---|
| `matmul(y, x, w, S, I, O)` | `[]const f32` | |
| `matmulBf16(y, x, w, S, I, O)` | `[]const u16` | raw bf16 bit patterns |
| `dotF32(a, b)` / `dotBf16(a, b)` | | 4 `@mulAdd` (FMA) accumulator chains + two tail loops |
| `bf16ToF32(u16)` / `f32ToBf16(f32)` | | round-to-nearest-even |

Vector width = `std.simd.suggestVectorLength(f32)` - 8 on this build (Zig 0.16
`zig build` targets the native CPU: AVX2 + FMA on the dev box). Verified in tests
against a naive triple loop with a non-vector-multiple inner dim.

`dotBf16` widens a whole lane at once - `@as(@Vector(N,u32), raw) << @splat(16)`
then `@bitCast` - not a scalar `bf16ToF32` per element. GDN/QSA/residual/router/
shared-expert all route through it.

Both dots run **four independent `@mulAdd` accumulator chains**: one serial
`acc += va*vb` chain stalls at ~1 FMA / 4 cycles (the FMA latency), four hide it.
`@mulAdd` also contracts the mul+add into one fused instruction - which is what
colibri's `gcc -O3 -march=native` emits too (`-ffp-contract=fast`), so the result
stays numerically aligned with the C reference. Micro-benchmark (single-thread,
L1-resident, `I=2560 O=640`): **1.46× vs the single-accumulator loop** (17 →
25 GFLOP/s). End-to-end the gain is smaller - the threaded MoE decode is
memory-bandwidth-bound, not FLOP-bound (see `BENCHMARK.md`). The model's inner
widths (2560, 640, 256, 128, vocab) are all multiples of `4·lanes`, so the fast
loop carries them; the two tail loops cover the fixture's odd shapes.

## `rmsnorm.zig`

- `rms0(out, x, w, eps)` - Qwen4-Exp norms are **zero-centered**: effective scale
  is `1 + w`. `out_i = x_i · rsqrt(mean(x²) + eps) · (1 + w_i)`.
- `rmsGated(out, x, gate, w, eps, sigmoid_gate)` - DeltaNet's `RMSNormGated`
  (from Qwen3-Next), **not** zero-centered, times a gate branch
  (`sigmoid` or `silu`). `out_i = x_i · rsqrt(mean(x²) + eps) · w_i · g(gate_i)`.

## `rope.zig`

`rope(x, rotary_dim, pos, theta)` - NeoX split-half: for `i in 0..rotary_dim/2`,
rotate the pair `(x[i], x[i + half])` by `pos / theta^(2i/rotary_dim)`. Only the
first `rotary_dim` lanes are touched (partial rotary factor 0.25 → dim 64 on the
real model). Text-only: `pos` is the token index; mRoPE sectioning is out of
scope.

## `softmax.zig`

`softmax(x)` / `softmaxInto(dst, src)` - max-subtracted, sum in f64. Shift
invariant; large constant offsets do not overflow.

## `activation.zig`

`sigmoid` (stable two-branch form), `silu`, `softplus` (with colibri's `x > 20`
shortcut), `geluTanh` (`gelu_pytorch_tanh`, used by the vision tower - not wired).

## FP8 (`ops/fp8.zig`)

`e4m3ToF32(byte)` decodes an OCP E4M3 byte with **no scale applied** - routed
experts use 128×128 block scales, the PLE table a scalar scale, applied by those
subsystems. Single-value form is a 256-entry comptime table (`e4m3_lut`). The
generic `tensors.decodeInto` still returns `error.UnsupportedDType` for FP8.

`dequantRow(out, w, scale_row)` - decodes a whole E4M3 weight row to f32 (block
scale folded) with **`@Vector` ops only, no gather**. E4M3 `[s:1][e:4][m:3]`
bias 7 maps onto the f32 field layout directly:

    f32_bits = (s << 31) | ((e + 120) << 23) | (m << 20)

bit-exact for normals; subnormals become `float(m)·2^-9` with the sign OR'd back;
NaN is `@select`ed in. The per-lane-chunk decode is factored into `decodeLane`.
A lane-chunk never straddles a 128-scale block (`comptime assert block % lanes ==
0`), so the scale is one broadcast per chunk. Bit-identical to `e4m3_lut` for
every value in a real checkpoint (tested). A 256-entry-LUT + gather decode was
tried and is *slower* here - `vpgatherdps` is throttled on this class of CPU and
LLVM does not emit it well from a scalar `lut[byte]` loop.

`matmulFp8(y, x, w, scales, S, I, O)` - block-FP8 GEMM for the routed experts,
fans over `parallel.chunks`:

- **`S == 1`** (the decode path) → `dotFp8Row`: `decodeLane` feeds two `@mulAdd`
  chains directly, with **no f32 weight-row buffer** and no store/reload. The
  dequant is ~90 % of this kernel's cost (micro-benchmarked), so cutting the
  buffer round-trip is worth ~9 % on the kernel.
- **`S > 1`** (prefill) → `dequantRow` to an f32 stack buffer once, then `dotF32`
  against each of the `S` input rows - the dequant cost amortises over `S`.

Both match a "dequantize then f32 matmul" reference within 1e-4 (rel). The
`S == 1` reduction order differs from `S > 1` only by f32 rounding. Greedy output
on the real checkpoint is unchanged (token-for-token equal to colibri).
