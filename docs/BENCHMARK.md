# `benchmark` / `stress` + instrumentation

## `src/runtime/timers.zig` — `Timers`

Per-subsystem wall-clock accounting via `std.Io.Timestamp` (`.awake`, monotonic).
Phases: `embed`, `gated_residual` (all three reads summed), `deltanet`, `qsa`,
`moe`, `ple`, `lm_head`. Whole-call timing — the phases overlap the dense
matmuls inside them by design.

Attached through `model.Opts.timers`. `forwards` counts forward passes so the
report can show ms/forward.

## `src/runtime/meter.zig` — `Meter`

A pass-through `std.mem.Allocator` that tracks `current` and `peak` live bytes.
This is **our** allocation accounting, not process RSS, but it lines up with the
memory-model tiers and needs no platform call. `benchmark` / `stress` wrap `gpa`
in a `Meter` and report `peak`.

## `compute_stall_due_to_io`

The demand-path expert load time. `ExpertCache` takes an optional `io`
(`State.init(..., io)`); when set, `fill` times `loadExpert` for demand (not
prefetch) misses into `CacheStats.demand_ns`. `benchmark` sums it across layers.
The count form is `CacheStats.demand_loads`.

## `benchmark <MODEL_DIR> [--prompt-len N] [--steps N] [--expert-cap K] [--ram-limit G] [--context N]`

Synthetic prompt `[1,2,…,prompt_len]`, prefill (→ TTFT), then `--steps` greedy
decode steps (→ tok/s). Prints the brief §23 `=== QWEN38 RUNTIME ===` report:
model shape + load time; tokens / TTFT / tok/s; memory (resident weights, plan
target vs limit, tracked peak); experts (requests, hits, demand loads + ms,
prefetch hits); I/O scheduler (serviced by priority, queue peak, predictor
accuracy); PLE table shape; QSA context / GDN layers; then the phase timing
table.

On the tiny fixture the numbers are toy-scale (KB, sub-ms); on a real checkpoint
they are the real thing. Example (`--prompt-len 8 --steps 20 --expert-cap 2`):

```
Tokens:  generated 20   TTFT 9.77 ms   decode tok/s 936
Experts: requests 224   hits 56.7%   demand loads 97 (3.87 ms ← compute_stall_due_to_io)
phase:   moe 0.765 ms/fwd   gated_residual 0.235   deltanet 0.076   ...
```

## `stress <MODEL_DIR> [--context N] [--steps N] [--ram-limit G]`

Sweeps RAM budgets (default ladder 8/12/16/24/32 GiB, or the single `--ram-limit`)
and for each: runs `budget.plan`; if it doesn't fit, prints `DOES NOT FIT`;
otherwise runs a short forward under a `Meter` and prints
`ram limit | plan target | tracked peak | tok/s | hit rate | status`, where
status asserts `tracked peak ≤ limit` (brief §26). The memory plan already
guarantees `plan.total_resident ≤ ram_limit` when it fits; the tracked peak is
the empirical check.

## Real-checkpoint results (2026-09, 32 GB laptop, single NVMe, 6 threads)

Released `Qwen/Qwen3.8-Flash-Next-FP8`, `--expert-cap 512 --ram-limit 24G`.
Cold page cache. Coherent English output; not yet logit-checked vs the reference.

> These figures predate the `zig build` → ReleaseFast default and were likely
> measured on a Debug build — treat them as a lower bound. The "A/B vs colibri"
> section below has current optimized numbers.

| | first real run | after the perf pass |
|---|---|---|
| TTFT (16-token prompt) | ~99 s | **~29 s** |
| decode | ~0.04 tok/s | **~0.25 tok/s** (≈ 4 s/token) |
| tracked peak (private RAM) | 11.6 GiB | **7.0 GiB** |

Per-forward phase split after the pass (prefill, S=16): `moe` ~7 s (genuine FP8
arithmetic + ~1.5 s of cold expert page faults, mostly overlapped by the matmul
worker fan-out), `deltanet` ~2.3 s, `lm_head` ~1.8 s, `qsa`/`gated_residual` < 1 s.

What moved the needle, in order:
1. **`dotBf16` SIMD** — vector widen+shift instead of a scalar `inline for` lane
   loop; bit-identical; hit every BF16 matmul (GDN/QSA/residual/router/shared).
2. **O(1) tensor index** — `Weights.find` was a linear scan of 152k names; the
   MoE demand path does 6 lookups/expert, so this alone was ~15 s/run.
3. **FP8 LUT + row-dequant + SIMD dot** in `matmulFp8`.
4. **prefill token→expert grouping** — evaluate each distinct routed expert once
   against its batch (`moe.forwardGrouped`); the real model routes 16 tokens to
   only ~76 distinct experts/layer.
5. **experts borrow the shard mmap** (no `dupe`) — cut private RAM, made the
   cache cheap enough to run `--expert-cap 512`.

Tried and reverted: async `std.Io.Group` expert-page prewarm — no win, the matmul
fan-out already overlaps the cold reads and the demand path waits < 20 ms/forward.

## SIMD E4M3 dequant (2026-09)

`fp8.dequantRow` decodes the E4M3 weight row with pure `@Vector` ops (the f32 bit
pattern is built arithmetically — no 256-entry gather, which Zig 0.16 can't
vectorise). Bit-identical to the LUT. On the real model: **MoE decode compute
−23%**, decode **0.20 → 0.23 tok/s**. The remaining MoE decode cost is now split
~half genuine FP8 arithmetic, ~half cold expert page-fault I/O (~1.2 s/forward),
so the next gains are on the I/O side (warm the cache with learned hot-expert
priors; split expert reads across both NVMe drives).

## Learned expert priors (2026-09)

`runtime/expert_usage.zig` counts every `(layer, expert)` routing and merges the
totals into `<model_dir>/.colizig_usage` (192 KB) on exit. Next run, `chat` /
`forward` pre-fill each bounded `ExpertCache` with its layer's historically
hottest experts and fault those E4M3 pages in parallel (bounded by
`ram_budget − fixed_resident`). `--no-usage` opts out.

Cache-flushed A/B (cap 64, 20 tokens, "The capital of France is"):

| | `--no-usage` | learned priors |
|---|---|---|
| decode | 0.290 tok/s | **0.322 tok/s** (+11%) |
| demand loads | 3212 | **429** (−87%) |
| expert hit rate | ~71% | **~96%** |

Total wall time is unchanged for a short run (the ~7 s parallel page-warm at
startup offsets the decode savings); it turns net-positive as the generation
grows, and the priors keep accumulating across sessions. Bit-identical output.

## Dual-SSD expert mirror (2026-09) — implemented, no win here

`--mirror <dir>` mmaps a second copy of the checkpoint's shards on another drive;
`Weights.view` alternates routed-expert reads ~55/45 between the two. Correct
(bit-identical output), opt-in, zero cost unused.

A/B on this box (model on the slower BG4, 65/131 shards mirrored to the faster
XG6), cache-flushed: **no measurable change** — 20-token decode 0.286→0.273 tok/s,
64-token prefill TTFT 89.1→91.2 s, MoE prefill 51.7 s both. Same reason the async
prewarm gave nothing: after the LUT/O(1)-index/borrow/SIMD pass the MoE is
**FP8-arithmetic-bound**, not expert-I/O-bound — `demand_ns` is ~0.4 s/run and the
matmul's `parallel.chunks` workers already overlap the cold FP8 page faults. The
mirror would matter on a genuinely slow single drive or a much faster CPU; it
does not on an i7-10750H + two NVMe SSDs.

## Phase 9b — MoE per-expert threading (2026-09)

`moe.forwardDense` (the S==1 decode path, `cap >= topk`) now pulls the top-k
experts into the cache serially, then fans their SwiGLU evaluation across worker
threads — each expert writes a disjoint `eo` row, the reduce runs in top-k order,
so the result is **bit-identical** to the serial version. A `threadlocal
in_worker` flag in `parallel.chunks` keeps each expert's own matmuls serial
(the fan-out is over experts, not over matmul rows — coarser tasks, less
`std.Io.Group` overhead).

Warm-cache decode-only benchmark, `--threads 1` vs `12`:

| | threads 1 | threads 12 |
|---|---|---|
| decode | 0.060 tok/s | **0.339 tok/s** (5.6×) |
| MoE / forward | 9292 ms | **1433 ms** (6.5×) |
| deltanet / forward | 1923 ms | 448 ms (4.3×) |

MoE scales ~6.5× on 6 cores / 12 threads — decode ~0.34 tok/s warm, in colibri's
range. The gain is masked on a cold run (the first forwards fault ~2.3 GB of
expert weights from SSD).

## A/B vs colibri, both optimized (2026-09-03)

> Build note: `zig build` now defaults to **ReleaseFast** (was Debug — a Debug
> build is ~3× slower and the earlier "colibri is 1.4–2× faster" A/B was
> measuring one). `-Doptimize=Debug` for a safety-checked build.

`bench_ab2.ps1` — same checkpoint, prompt, token count, `--expert-cap 64`,
12 threads, greedy. Each engine: flush OS page cache + wipe learned priors →
**cold** run → immediate **warm** run (hot page cache + priors it just wrote).
colibri built `-O3 -march=native -fopenmp` (mingw64).

Prompt "The capital of France is", 24 new tokens:

| | colibri | qwen38-zig |
|---|---|---|
| model load | 11.3 s | 13.2 s |
| TTFT (cold / warm) | 18.1 / 17.7 s | **8.2 / 5.4 s** |
| decode (cold / warm) | 0.39 / 0.38 tok/s | **0.78 / 1.02 tok/s** |
| peak working set | 23.9 GB | 27.3 GB |
| greedy output | — | **token-identical to colibri, cold and warm** |

Second prompt (48 tokens, reasoning-heavy): colibri 0.32 tok/s both, qwen38-zig
0.55 cold / 0.51 warm, TTFT ~31 s vs ~13 s. Core generation matches; the two
diverge by one token near step ~46 (different prompt framing in the harness +
accumulated f32 rounding between two independent implementations).

**qwen38-zig is now ~1.5–2.5× faster on decode and ~2× on TTFT at parity**, with
identical greedy output on matched token input. Why the reversal:

- Phase 9b MoE per-expert threading + the SIMD FP8/BF16 kernels + O(1) tensor
  index compound once compiled optimized.
- qwen38-zig **borrows** the expert mmap — a warm OS page cache directly speeds
  decode (0.78 → 1.02) and TTFT (8.2 → 5.4). colibri **copies** experts into its
  heap every process (RSS ~24 GB), so a warm page cache barely moves its
  throughput (0.39 → 0.38); it needs its persistent `coli chat` server to
  amortise that copy across turns.
- Prefill token→expert grouping → ~2× TTFT.

Caveats: one sample per config (run-to-run variance); colibri at cap 64 is
expert-starved (60–74 % hit rate) and its multi-turn server mode is not measured
here; qwen38-zig's 27 GB working set is ~7 GB private + reclaimable mmap page
cache, colibri's 24 GB is private; neither is logit-checked against HF weights
(colibri is the reference and the output matches it).

### expert-cap / thread sweep (warm, `bench_cap.ps1`)

qwen38-zig only, "capital of France" 24 tok, all warm, back-to-back (the machine
was thermally loaded from the A/B above — read the **columns relative to each
other**, not the absolute tok/s):

| cap | threads | TTFT s | decode tok/s |
|---|---|---|---|
| 64  | 12 | 8.9 | 0.68 |
| 64  | 6  | 13.5 | 0.55 |
| 512 | 12 | **5.9** | **0.74** |
| 512 | 6  | 6.1 | 0.66 |

- **cap 512 > cap 64**: +9 % decode / −34 % TTFT at 12 threads (+21 % decode at
  6). No eviction → routed experts stay hot; startup `warmCaches` also pre-faults
  the historically-hot set. Costs a one-time ~15–20 GB page-warm on a cold start
  and a bigger (reclaimable) working set; private RAM barely moves. **Use
  `--expert-cap 512` on a ≥ 24 GB box.**
- **12 threads > 6**: the MoE expert fan-out and the matmul row fan-out both use
  the extra logical cores. `--threads 0` (auto = 12 here) is right; don't pin to
  physical-core count.
- Sustained benchmarking on this laptop thermally throttles — the first
  interactive use of a session runs faster than a marathon like this suggests.

## AVX2 / FMA kernel pass (2026-09-03)

`ops/matmul.zig` `dotF32` / `dotBf16` rewritten to four independent `@mulAdd`
(FMA) accumulator chains; `ops/fp8.zig` gained a fused `dotFp8Row` for the
`S == 1` decode path (decode straight into FMA chains, no f32 weight-row buffer).

**Micro-benchmark** (`tools/microbench.zig` — single thread, L1-resident,
`I=2560 O=640`, run it with `zig run -O ReleaseFast tools/microbench.zig`):

| kernel | µs/matmul | GFLOP/s |
|---|---|---|
| f32 dot, 1 accumulator | 193 | 17.0 |
| f32 dot, 4× `@mulAdd` | **133** | **24.7** (1.46×) |
| FP8: arith dequant only | 462 | — |
| FP8: split (dequant→buf→dot) | 514 | 6.4 |
| FP8: fused decode+FMA | **469** | 7.0 |
| FP8: 256-LUT scalar (auto-vec) | 1462 | 2.2 |
| FP8: 256-LUT manual gather | 504 | 6.5 |

Takeaways: the FMA/accumulator change is a real 1.46× on the **pure f32/bf16**
dot (GDN, QSA, gated-residual, LM head, router, shared expert). The **FP8 expert**
kernel is ~90 % dequant and there is no easy win there on AVX2 — the arithmetic
decode already beats every LUT/gather variant (`vpgatherdps` is throttled on
Comet Lake); fusing the decode into the dot saves ~9 % by dropping the buffer
round-trip.

**End-to-end** (controlled A/B, `forward`, cap 512, 12 threads, warm, 4 reps):
decode 0.689 → 0.71 tok/s (**+3 %**), TTFT −2 %, **greedy output token-for-token
identical** to the pre-change build and to colibri. The end-to-end gain is far
below the 1.46× kernel gain because the **12-thread MoE decode is
memory-bandwidth-bound**, not FLOP-bound: ~1 GB of expert weights stream from the
mmap page cache per token, and the i7-10750H's ~40 GB/s shared bus caps that
around 1 tok/s regardless of how fast the dot runs. Same pattern as the I/O-side
optimisations, mirrored: after the SIMD pass, compute is no longer the wall.

The remaining real lever is the **GPU backend** (Phase 10 — deferred) or moving
fewer bytes (int4, out of scope).

## Not done

QSA per-head + prefill row-chunking (both small next to MoE); MTP speculative
decode; GPU backend (Phase 10, deferred).
