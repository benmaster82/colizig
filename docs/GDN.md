# Gated DeltaNet (`src/qwen38/gdn.zig`)

Linear-attention path for 36 of the 48 layers (`layer_types` == `linear_attention`).
Ported from colibri's `q38_deltanet`. Weights are BF16-resident; activations f32.

## Dimensions (`Dims`, real model)

| | value | |
|---|---|---|
| `hidden` | 2560 | |
| `kheads` / `vheads` | 16 / 48 | `rep = vheads/kheads = 3` (Q,K heads shared) |
| `kdim` / `vdim` | 128 / 128 | `vdim ≤ 512` enforced (stack-sized delta in the reference) |
| `convk` | 4 | causal depthwise conv kernel |
| `conv_dim` | 10240 | `= 2·kheads·kdim + vheads·vdim` (= 4096 + 6144) |
| `k()` / `v()` | 2048 / 6144 | conv-output block widths: `[q | k | v]` |

## Weights (`Layer`) - `layers.i.linear_attn.*`

| field | tensor | shape | dtype |
|---|---|---|---|
| `qkv` | `in_proj_qkv.weight` | `[conv_dim, hidden]` | BF16 |
| `z` | `in_proj_z.weight` | `[v, hidden]` | BF16 |
| `b`, `a` | `in_proj_b/a.weight` | `[vheads, hidden]` | BF16 |
| `out` | `out_proj.weight` | `[hidden, v]` | BF16 |
| `conv` | `conv1d.weight` | `[conv_dim, convk]` | F32 |
| `dt_bias`, `a_log` | `dt_bias`, `A_log` | `[vheads]` | F32 |
| `norm` | `norm.weight` | `[vdim]` | F32 |

BF16 matrices are copied out of the shard mmap into owned, aligned buffers
(`NativeMatrix`) and multiplied with `matmulBf16` (per-element widen, f32 accum).

## Persistent state (`GdnState`) - allocated once per GDN layer

- `rec` - `[vheads][kdim·vdim]` f32 recurrent state (real: 48·128·128·4 ≈ 3.1 MiB/layer)
- `ring` - `[conv_dim][convk-1]` f32 causal-conv history

`reset()` zeroes both (new sequence). Never reallocated per token.

## Per-token pipeline (`forward`)

1. `qkv = x·qkvᵀ`, `z = x·zᵀ`, `bb = x·bᵀ`, `aa = x·aᵀ` (one matmul each over the whole chunk).
2. **Causal depthwise conv1d + SiLU** per channel `c` of `conv_dim`:
   `conv[c] = silu( w[c,K-1]·qkv[c] + Σ_{tap<K-1} w[c,tap]·ring[c,tap] )`, then shift
   `ring[c]` left and store `qkv[c]` at the end.
3. Split `conv` into `qi[0:k]`, `ki[k:2k]`, `vi[2k:2k+v]`.
4. Per value head `h`: gather the shared key head `h/rep`, L2-normalize
   (`Σx² + 1e-6`), scale - `qscale = 1/√Σ · 1/√kdim`, `kscale = 1/√Σ`.
5. **Gated-delta recurrence** per head, state `[kdim, vdim]`:
   - `α = exp(−exp(A_log[h]) · softplus(aa[h] + dt_bias[h]))`, `β = σ(bb[h])`
   - `state *= α`
   - `δ[j] = (v[j] − Σ_d k[d]·state[d,j]) · β`
   - `state[d,j] += k[d]·δ[j]`
   - `core[h,j] = Σ_d q[d]·state[d,j]`
6. `norm[h] = RMSNormGated(core[h], gate = z[h], weight = norm)` (sigmoid gate).
7. `out = norm·outᵀ`.

## Guarantees / tests

- **Chunk-boundary invariant**: feeding `[A,B]` in one call == `[A]` then `[B]`
  (conv ring + recurrence are strictly token-causal). Checked in unit tests and
  `selftest`.
- **Zero input from a fresh state → zero output.**
- Determinism; state advance changes the output for a repeated token.
- Unit tests also compare `forward` to a naive in-test reimplementation on random
  weights. Full validation vs upstream `Qwen4ExpForCausalLM` awaits the Python
  reference harness (brief §25).

## Threading / batching

The gated-delta recurrence fans over value heads via `parallel.chunks` (Phase 9,
bit-identical - each head owns its `rec`/`core`/`delta` slice). Not yet: prefill
row-batching (colibri's bounded 32-row chunks) - the recurrence and conv already
produce chunk-invariant results, so it is purely a memory/robustness change.
