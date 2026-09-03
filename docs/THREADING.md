# Threaded kernels (`src/runtime/parallel.zig`)

Opt-in loop parallelism for the hot paths, via `std.Io.Group` (Zig 0.16's
concurrency primitive — `std.Thread.Pool` was removed).

## The switch

`parallel.enable(io, max_tasks)` sets a process-wide `?Io` + fan-out cap
(`max_tasks == 0` → CPU count). `disable()` clears it. Unit tests leave it off,
so their results stay deterministic; the inference CLIs
(`forward` / `chat` / `benchmark` / `stress`) turn it on, controlled by
`--threads` (0 = auto, 1 = single-threaded, N = cap at N).

## `chunks(n, work, ctx, body)`

Splits `[0, n)` into up to `min(max_tasks, n)` contiguous ranges and runs
`body(ctx, lo, hi)` on each through `Group.async` + `await`. `body` must only
write **disjoint or append-only** state across chunks.

`work` is the loop's approximate total scalar-op count. Below
`parallel.min_work` (96 Ki) it runs **serially even when enabled** — a small
loop cannot amortize the `Io.Group` coordination cost. (The first, un-thresholded
version of this was ~24× *slower* on the toy fixture; the threshold is the fix,
matching the `if` clause on an OpenMP pragma.)

## What is threaded

| kernel | fanned dimension | `work` |
|---|---|---|
| `matmul` / `matmulBf16` (`ops/matmul.zig`) | output rows `O` | `O·I·S` |
| `matmulFp8` (`ops/fp8.zig`) | output rows `O` | `O·I·S` |
| GDN gated-delta recurrence (`qwen38/gdn.zig`) | value heads `VH` | `VH·kd·vd·3` |
| MoE decode expert eval (`qwen38/moe.zig` `forwardDense`) | top-k experts | `K·3·I·H` |

Every fanned task writes `y[… + o]` (matmul), its own `rec[h]` / `core[h]` /
`delta[h]` slice (GDN), or its own `eo[z]` row (MoE) — disjoint. **The output is
bit-identical to the serial path**, verified by:

- `ops/matmul.zig` — a `[512×256]` matmul, threaded vs serial, `expectEqual`.
- `qwen38/model.zig` — the whole `forward`, threaded vs serial, `expectEqual`.
- `forward` on the real checkpoint — `--threads 1` vs `12` token-for-token equal.

### Nested fan-out

`parallel.zig` keeps a `threadlocal in_worker` flag: a `chunks` call made from
*inside* a worker body runs serially. So the MoE expert fan-out (`forwardDense`)
does not spawn a second fan-out per expert matmul — the split is over experts,
which are coarser tasks with less coordination overhead for the same work. The
top-k experts are pulled into the `ExpertCache` serially *before* the fan-out
(the cache handoff stays single-threaded); only the pure-compute SwiGLU eval is
threaded.

matmul threading alone covers most of the real model's FLOPs: every projection
(`q/k/v/o`, gated residual `down`/`up`, router, shared expert) and every routed
expert GEMM and the LM head.

## What is not threaded

- QSA per-query-head attention — ~94 ms/forward warm, noise next to MoE's 1.4 s,
  and it needs per-head score/softmax scratch to avoid races. Not worth it.
- `Model.load` (48 layers loaded sequentially — a one-time cost).
- Prefill batching (colibri's bounded 32-row chunks; `Scratch` is sized to the
  whole prompt).

## Measuring it

The wall-clock speedup only shows on the real checkpoint (the fixture's matmuls
are ~2 Ki ops and stay serial). On the fixture, `benchmark --threads 0` and
`--threads 1` are within noise, as intended.
