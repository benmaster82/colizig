# Mixture-of-Experts (`src/qwen38/moe.zig`)

Every layer has an MoE block: a router over 512 experts, top-10 routed + 1
always-on shared expert. Ported from colibri's `q38_moe_decode`.

## Dimensions (`Dims`, real model)

| | value |
|---|---|
| `experts` / `topk` | 512 / 10 |
| `inter` (routed) / `shared_inter` | 640 / 640 |
| `norm_topk` | true (renormalize gates over the selected experts) |

## Resident per-layer weights (`Layer`) - `layers.i.mlp.*`

| field | tensor | shape | dtype |
|---|---|---|---|
| `router` | `gate.weight` | `[experts, hidden]` | BF16 |
| `sh_gate_proj` / `sh_up_proj` | `shared_expert.{gate,up}_proj.weight` | `[shared_inter, hidden]` | BF16 |
| `sh_down_proj` | `shared_expert.down_proj.weight` | `[hidden, shared_inter]` | BF16 |
| `sh_gate` | `shared_expert_gate.weight` | `[hidden]` | F32 |

Real cost ≈ (router 2.5 MiB + shared 9.4 MiB) per layer × 48 ≈ 570 MiB resident.

## Streamed experts - `Fp8Matrix` + `ExpertCache`

Each routed expert is 3 block-FP8 matrices (`experts.<e>.{gate,up,down}_proj`):

| matrix | shape | + scales `weight_scale_inv` |
|---|---|---|
| `gate` / `up` | `[inter, hidden]` E4M3 | `[⌈inter/128⌉, ⌈hidden/128⌉]` = `[5, 20]` f32 |
| `down` | `[hidden, inter]` E4M3 | `[20, 5]` f32 |

≈ 4.7 MiB per expert (colibri: same). `Fp8Matrix.matmul` folds each 128-column
block's scale into the accumulation (`ops/fp8.zig : matmulFp8`), verified against
a dequantize-then-matmul reference.

`ExpertCache` is a **bounded LRU** keyed by expert id:

- `get(w, dims, id)` → hit bumps recency; miss loads the 3 matrices into a slot,
  evicting the least-recently-used slot when full.
- `CacheStats`: `hits`, `misses`, `loads`, `evictions`, `bytes_resident`.
- `get`'s result is valid only until the next `get` on the same cache. `forward`
  finishes all matmuls for one expert before requesting the next, so a tiny cap
  (even `< topk`) is correct, just thrashy.

Capacity comes from the memory-budget plan (`runtime/budget.zig`); on the real
model cap 16/32/64 ≈ 3.5/7.0/14 GiB across 48 layers.

## Per-token forward

1. `logits = router @ xᵀ`; softmax (max-subtracted; keep the unnormalized `exp`
   values and their sum `all`).
2. Top-k by softmax weight (monotonic). `top = Σ selected exp-values`.
   `den = norm_topk ? top : all`; `gate_z = exp_value(idx_z) / den`.
3. Shared expert: `sh = silu(x·gate_projᵀ) ⊙ (x·up_projᵀ)`, `shared = sh·down_projᵀ`;
   `sgate = σ(x · sh_gate)`.
4. Each routed expert `z`: `eh = silu(x·gateᵀ) ⊙ (x·upᵀ)`, `eo = eh·downᵀ`;
   `y += gate_z · eo`.
5. `y += sgate · shared`.

The router, shared expert and routed experts all consume the **same** MoE input
`x` (already normalized by the gated-residual read, upstream).

## Decode vs prefill path

`forward` splits on `S`:

- **`S == 1` (`forwardDense`)** - per-token: route, pull the top-k experts into
  the cache serially, then fan their SwiGLU evaluation over `parallel.chunks`
  (one task per expert; each writes a disjoint `eo` row). The accumulate into
  `out` still runs in top-k order, so the result is **bit-identical** to the
  serial path. When threading is off, or `cap < topk`, it stays a plain serial
  loop. Warm-cache decode: MoE scales ~6.5× on 6c/12t (see `BENCHMARK.md`).
- **`S > 1` (`forwardGrouped`)** - route every token, sort the `S·topk`
  `(token, slot)` pairs by expert id (`std.sort.pdq`), then evaluate each
  **distinct** expert once against the batch of tokens that picked it - one pass
  over its ~4.7 MiB E4M3 weights instead of one per token (a 16-token prompt
  routes to only ~76 distinct experts/layer on the real model). The paths
  accumulate `out` in a different order → f32 rounding difference only (< 1e-4).

`ExpertCache` **borrows** the shard mmap: `Fp8Matrix.data` slices a `Weights`
shard, never copied. A demand load only decodes the scale table; the E4M3 pages
fault in on first `matmul` and are OS-reclaimable, so `--expert-cap 512` (no
eviction) costs only ~1 KB/expert of decoded scales.

## Guarantees / tests

- `matmulFp8` == reference (dequantize then f32 matmul).
- MoE output finite; deterministic across independent cache instances.
- Grouped prefill ≈ per-token dense path (< 1e-3 abs).
- Warm cache (cap ≥ experts): replaying the same tokens is all hits, 0 evictions.
- Tight cache (cap 1): each distinct expert evicts the previous.
- Full numerical validation vs upstream awaits the logit cross-check.

## Not yet

- Fused BF16 expert layout (`experts.gate_up_proj` / `down_proj`).
- Parallel *expert loads* (the cache handoff is still serial; only the compute
  fans out).
- Router replay/trace hooks (colibri's `rt_*`) - not needed here.
