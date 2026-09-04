# Qwen4-Exp tensor shapes

Shapes for the real checkpoint, using the config constants from
`ARCHITECTURE.md`:

```
H  hidden            2560      W   hc_width           10240   (= hc_count·H)
V  vocab           248320      R   hc_lowrank           320
QH q_heads             24      D   head_dim             256
KVH kv_heads            2      CD  dn_conv_dim        10240   (= 2·kh·kd + vh·vd)
E  experts            512      I   moe_inter            640     SI shared_inter 640
kh/vh dn key/val heads 16/48   kd/vd dn key/val dim  128/128   ck dn_conv_k 4
iq/ik idx q/kv heads   4/1     id  idx_dim              128     ib idx_budget 2048  ir idx_ratio 4
PE ple_embed_dim     2560      nh  ngram_heads           16     nd ngram_head_dim 160
pk ple_conv_k           4      np  split_ngram_parts    128
```

Tensor names are shown without the `model.language_model.` / `model.` prefix.
Layer index `i` runs 0..47; PLE lives on layer **1**.

## Persistent (whole model) - resident, BF16 unless noted

| name | shape | dtype |
|---|---|---|
| `embed_tokens.weight` | `[V, H]` | BF16 |
| `lm_head.weight` | `[V, H]` | BF16 |
| `hyper_connection_mixer.hc_norm.weight` | `[W]` | F32 |
| `hyper_connection_mixer.input_mix_weight_down.weight` | `[R, W]` | BF16 |
| `hyper_connection_mixer.input_mix_weight_up.weight` | `[W, R]` | BF16 |

## Per layer (×48) - resident

**Gated residual** - `X ∈ {attn, mlp}`, `layers.i.X_hyper_connection.*`:

| suffix | shape | dtype | category |
|---|---|---|---|
| `hc_norm.weight` | `[W]` | F32 | gated_residual |
| `input_mix_weight_down.weight` | `[R, W]` | BF16 | gated_residual |
| `input_mix_weight_up.weight` | `[W, R]` | BF16 | gated_residual |
| `block_inject_weight.weight` | `[hc_count, W]` = `[4, W]` | BF16 | gated_residual |

**MoE common** - `layers.i.mlp.*`:

| suffix | shape | dtype | category |
|---|---|---|---|
| `gate.weight` | `[E, H]` | BF16 | router |
| `shared_expert.gate_proj.weight` | `[SI, H]` | BF16 | shared_expert |
| `shared_expert.up_proj.weight` | `[SI, H]` | BF16 | shared_expert |
| `shared_expert.down_proj.weight` | `[H, SI]` | BF16 | shared_expert |
| `shared_expert_gate.weight` | `[H]` | F32 | shared_expert |

**QSA layers** (12) - `layers.i.self_attn.*`:

| suffix | shape | dtype | category |
|---|---|---|---|
| `q_proj.weight` | `[QH·D·2, H]` = `[12288, H]` | BF16 | attn_dense |
| `k_proj.weight` | `[KVH·D, H]` = `[512, H]` | BF16 | attn_dense |
| `v_proj.weight` | `[512, H]` | BF16 | attn_dense |
| `o_proj.weight` | `[H, QH·D]` = `[H, 6144]` | BF16 | attn_dense |
| `q_norm.weight` / `k_norm.weight` | `[D]` = `[256]` | F32 | attn_dense |
| `indexer.index_qk_proj.weight` | `[(iq+ik)·id, H]` = `[640, H]` | BF16 | qsa_indexer |
| `indexer.q_layernorm.weight` / `indexer.k_layernorm.weight` | `[id]` = `[128]` | F32 | qsa_indexer |

**Gated DeltaNet layers** (36) - `layers.i.linear_attn.*`:

| suffix | shape | dtype | category |
|---|---|---|---|
| `in_proj_qkv.weight` | `[CD, H]` = `[10240, H]` | BF16 | deltanet |
| `in_proj_z.weight` | `[vh·vd, H]` = `[6144, H]` | BF16 | deltanet |
| `in_proj_b.weight` / `in_proj_a.weight` | `[vh, H]` = `[48, H]` | BF16 | deltanet |
| `conv1d.weight` | `[CD, ck]` = `[10240, 4]` | F32 | deltanet |
| `dt_bias` / `A_log` | `[vh]` = `[48]` | F32 | deltanet |
| `norm.weight` | `[vd]` = `[128]` | F32 | deltanet |
| `out_proj.weight` | `[H, vh·vd]` = `[H, 6144]` | BF16 | deltanet |

## Per layer × per expert (×48 × 512) - **streamable**, native E4M3

`layers.i.mlp.experts.e.*` (per-expert layout):

| suffix | shape | dtype |
|---|---|---|
| `gate_proj.weight` | `[I, H]` | F8_E4M3 |
| `gate_proj.weight_scale_inv` | `[⌈I/128⌉, ⌈H/128⌉]` = `[5, 20]` | F32 |
| `up_proj.weight` | `[I, H]` | F8_E4M3 |
| `up_proj.weight_scale_inv` | `[5, 20]` | F32 |
| `down_proj.weight` | `[H, I]` | F8_E4M3 |
| `down_proj.weight_scale_inv` | `[20, 5]` | F32 |

Fused alternative (upstream text class): `layers.i.mlp.experts.gate_up_proj`
`[E, 2I, H]`, `layers.i.mlp.experts.down_proj` `[E, H, I]` - sliced per expert.

The block-scale sidecars are small enough to keep **resident** for every expert
(colibri's "FP8 scale bank", ~28 MiB) so a cache miss is a single FP8 read.

## PLE - layer 1 only

**Dense + meta - resident:**

| name | shape | dtype | category |
|---|---|---|---|
| `layers.1.ple.key_proj.weight` | `[W, PE]` | BF16 | ple_dense |
| `layers.1.ple.value_proj.weight` | `[H, PE]` | BF16 | ple_dense |
| `layers.1.ple.norm_key.weight` / `norm_query.weight` / `norm_conv.weight` | `[W]` | F32 | ple_dense |
| `layers.1.ple.conv1d.weight` | `[W, pk]` = `[W, 4]` | F32 | ple_dense |
| `layers.1.ple.ple_embedding.layer_multipliers` | `[ngram_size]` = `[3]` | I64 | ple_meta |
| `layers.1.ple.ple_embedding.ngram_heads_vocab_sizes` | `[nh]` = `[16]` | I64 | ple_meta |
| `layers.1.ple.ple_embedding.ngram_heads_offsets` | `[16]` | I64 | ple_meta |
| `layers.1.ple.ple_embedding.ngram_embedding.weight_scale` | `[1]` | F32 | ple_meta |

**N-gram table - cold, never resident:**

| name | shape | dtype | category |
|---|---|---|---|
| `layers.1.ple.ple_embedding.ngram_embedding.shard_p.weight`, `p = 0..127` | `[rows_p, nd]` = `[rows_p, 160]` | F8_E4M3 | ple_table |

`Σ rows_p ≈ 51e9 / 160 ≈ 320M` rows. Per token: `nh = 16` positioned reads of
`nd = 160` bytes each.

## Runtime state (allocated, not checkpoint tensors)

| state | shape | when |
|---|---|---|
| GDN recurrent | `[vh, kd, vd]` f32 per GDN layer | persistent across tokens |
| GDN causal-conv ring | `[CD, ck-1]` f32 per GDN layer | persistent |
| PLE causal-conv ring | `[W, pk-1, ngram_size]` f32 | persistent |
| QSA K / V / indexer-key cache | `(2·KVH·D + id)` f32 per QSA layer **per token** | grows with context |

## Classification legend (brief §4.5)

- **persistent**: whole-model tensors above.
- **per-layer**: everything under `layers.i.` that is not per-expert.
- **per-token**: the QSA K/V/IK cache; PLE row reads.
- **cacheable / streamable**: `moe_expert` (bounded per-layer LRU).
- **deterministic-address**: all resident + streamable tensors (offset fixed at
  manifest time).
- **dynamically-addressed**: `ple_table` rows (address = hash of token history).
