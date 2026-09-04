# Qwen Sparse Attention (`src/qwen38/qsa.zig`)

The `full_attention` path for 12 of the 48 layers (every 4th). Ported from
colibri's `q38_attention`. Weights are BF16-resident; activations f32.

## Dimensions (`Dims`, real model)

| | value |
|---|---|
| `q_heads` / `kv_heads` / `head_dim` | 24 / 2 / 256 (`group` = 12) |
| `rotary_dim` / `theta` | 64 / 1e7 (partial RoPE) |
| indexer: `idx_qheads` / `idx_kheads` / `idx_dim` | 4 / 1 / 128 |
| `idx_budget` / `idx_ratio` | 2048 / 4 → keep ≤ 512 blocks, `maxSelected` = 2051 |

## Weights (`Layer`) - `layers.i.self_attn.*`

| field | tensor | shape |
|---|---|---|
| `q` | `q_proj.weight` | `[q_heads·head_dim·2, hidden]` - **query ‖ output-gate** |
| `k` / `v` | `{k,v}_proj.weight` | `[kv_heads·head_dim, hidden]` |
| `o` | `o_proj.weight` | `[hidden, q_heads·head_dim]` |
| `q_norm` / `k_norm` | `{q,k}_norm.weight` | `[head_dim]` |
| `idx_qk` | `indexer.index_qk_proj.weight` | `[(idx_qheads+1)·idx_dim, hidden]` |
| `idx_qn` / `idx_kn` | `indexer.{q,k}_layernorm.weight` | `[idx_dim]` |

## Context cache (`Cache`) - the big context-dependent consumer

- `k` - `[kv_heads][cap][head_dim]`, stored **normalized + RoPE'd**
- `v` - `[kv_heads][cap][head_dim]`, raw
- `ik` - `[cap][idx_dim]`, raw indexer key (normalized/RoPE'd later, per block)

Per-token cost `= (2·kv_heads·head_dim + idx_dim)·4` bytes = **54 KiB** on the
real model (matches `MEMORY_BUDGET.md`). Sized from `--context`. `forward`
appends in order and asserts `cache.len == pos_base`.

## Per-token forward

1. Project `qp` (`q_heads·2·head_dim`), `kp`, `vp`, `ip` (`(idx_qheads+1)·idx_dim`).
2. Append to cache: `k = rope(rms0(k, k_norm))`, `v` raw, indexer key raw.
3. Indexer query per head: `rope(rms0(ip_h, idx_qn), pos)`.
4. For each complete `idx_ratio`-token block `b` (`blocks = visible/idx_ratio`):
   pool its `idx_ratio` cached indexer keys (mean), `rms0(idx_kn)`,
   `rope(pos = b·idx_ratio)`, `score = Σ_h max(qidx_h · pool, 0) / √idx_dim`.
5. Sort blocks by score (desc); take the top `idx_budget/idx_ratio`, expand each
   to its token indices; append the causal tail `[blocks·idx_ratio, visible)`.
6. Full attention per query head `h` over the selected tokens
   (`kv head = h / group`): softmax(`q·k/√head_dim`), weighted `v`, then
   `× σ(gate)` where gate is the 2nd `head_dim` slice of `qp_h`.
7. `o_proj`.

Early tokens (`visible ≤ idx_ratio`) have no complete block → the causal tail is
the whole prefix (dense attention).

## Guarantees / tests

- Output finite; fully deterministic on a fresh cache.
- **Prefill == chunked decode**: running `[0..T)` in one call equals running it in
  any split, because each token's attention is fully determined by the in-order
  KV cache. Verified in unit tests and `selftest`.
- `nsel ≤ maxSelected`, every selected index `< visible`.
- Full numerical validation vs upstream awaits the Python reference harness.

## Not yet

- Threading over query heads / blocks (Phase 8).
- Reusing a shared prefix's cached K/V/index rows across prompts (colibri's
  serve-path prefix slot) - an inference-loop concern, Phase 7+.
