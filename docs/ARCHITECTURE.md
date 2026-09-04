# Qwen3.8-Flash-Next / Qwen4-Exp - architecture

Values below are from the HF `config.json` of `Qwen/Qwen3.8-Flash-Next-FP8`,
cross-checked against colibri's `c/qwen38_core.h` and `docs/qwen38.md`. The
engine reads them from `config.json` at runtime - nothing here is hard-coded
(`src/model/config.zig`).

**Verified against the real checkpoint** (2026-09, metadata only): the strict
`config.json` parser accepts the released file with **no discrepancy** (brief
rule 13); the tensor-name classifier categorizes **all 152,089** real tensors
with **zero `unknown`** (147,456 routed-expert tensors = 512×48×6, 128 PLE table
shards, 3,101 MTP + 333 vision correctly ignored); the config-derived resident
estimate is **9.22 GiB** and the memory plan reproduces colibri's published
figures (28 MiB scale bank, 54 KiB/token context, cap-16 = 3.5 GiB, cap-64
wants 32 GB).  The full-weight forward is still unvalidated - see
`docs/REFERENCE.md`.

## Config

| field | value | derived / notes |
|---|---|---|
| `model_type` (root / text) | `qwen4_exp` / `qwen4_exp_text` | classes `Qwen4ExpForConditionalGeneration` / `Qwen4ExpForCausalLM` |
| `hidden_size` | 2560 | |
| `num_hidden_layers` | 48 | |
| `layer_types` | `12 × [linear_attention×3, full_attention×1]` | 36 Gated DeltaNet + 12 QSA; `full_attention_interval: 4` |
| `vocab_size` | 248320 | |
| `max_position_embeddings` | 262144 | native context ceiling |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 2 / 256 | GQA, group size 12 |
| `output_gate_type` | `sigmoid` | q_proj output width = `q_heads·head_dim·2` (gated attention) |
| `rope_parameters` | `rope_type: default`, `rope_theta: 1e7`, `partial_rotary_factor: 0.25`, `mrope_interleaved: true`, `mrope_section: [11,11,10]` | `rotary_dim = int(256·0.25) = 64` |
| `num_experts` / `num_experts_per_tok` | 512 / 10 | + 1 shared expert |
| `moe_intermediate_size` / `shared_expert_intermediate_size` | 640 / 640 | routed expert ≈ 3·640·2560 B ≈ 4.7 MiB (E4M3) |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 48 | Gated DeltaNet |
| `linear_key_head_dim` / `linear_value_head_dim` / `linear_conv_kernel_dim` | 128 / 128 / 4 | `dn_conv_dim = 2·(16·128) + 48·128 = 10240` |
| `indexer_n_heads` / `indexer_kv_heads` / `indexer_head_dim` | 4 / 1 / 128 | QSA lightweight indexer |
| `indexer_budget` / `indexer_compress_ratio` | 2048 / 4 | ≈2048-token attention budget, 4-token micro-blocks |
| `hc_count` / `hc_lowrank` | 4 / 320 | 4-branch gated residual ("hyper connections"); `hc_width = 4·2560 = 10240` |
| `ple_embed_dim` | 2560 | |
| `ngram_size` / `heads_per_ngram` | 3 / 8 | `ngram_heads = (3-1)·8 = 16`; `ngram_head_dim = 2560/16 = 160` |
| `ple_conv_kernel_size` | 4 | causal conv over the PLE stream |
| `split_ngram_parts` | 128 | row-partitioned n-gram table shards |
| `ple_layer_ids` | `[2]` (1-based) | PLE injected at layer index **1** (0-based) |
| `ngram_vocab_size_base` / `make_ngram_vocab_size_divisible_by` | 20_000_000 / 128 | |
| quantization | `fp8`, `weight_block_size: [128,128]`, `modules_to_convert: ["ple.ple_embedding.ngram_embedding"]` | routed experts + PLE table are E4M3 block-FP8; everything else BF16 |
| `mtp_num_hidden_layers` | 1 | **MTP layer skipped** (like colibri) |
| `vision_config` | depth 27, hidden 1152, 16 heads, patch 16, merge 2, out_hidden 2560 | **text-only engine**: vision tensors indexed, never loaded |

## Layer stack

```
                    token embedding
                          │
        ┌─────────────────┴─────────────────┐  ×12 blocks
        │  L: linear_attention  (Gated DeltaNet)   ┐
        │  L: linear_attention  (Gated DeltaNet)   │ each layer:
        │  L: linear_attention  (Gated DeltaNet)   │  4-branch gated residual
        │  L: full_attention    (Qwen Sparse Attn) │  + 512-expert top-10 MoE
        └──────────────────────────────────────────┘  + 1 shared expert
                          │        (PLE injected once, at layer 1)
                    final gated-residual mixer
                          │
                       LM head → logits
```

- **Gated DeltaNet** (36 layers): `in_proj_qkv` → causal conv1d (k=4) → per-head
  Q/K normalize+scale → gated linear-attention recurrent state update
  (`state *= α; state += kᵀ·((v − k·state)·β); out = q·state`) → `RMSNormGated`
  → `out_proj`. Recurrent state is **persistent across tokens** (`GdnState`,
  allocated once).
- **Qwen Sparse Attention** (12 layers): projects Q/K/V (+ gate) and a
  lightweight indexer key; pools every complete 4-token block, scores blocks
  against the indexer query, keeps the best ~512 blocks + a ≤3-token causal
  tail, then runs full 24-head attention over only those tokens. Partial RoPE
  (dim 64, θ=1e7). K/V + indexer key cached per token.
- **Gated residual** ("hyper connections", `hc_count = 4`): each layer reads a
  4-branch hidden state through a low-rank (320) mixer with per-branch
  sigmoid gates and a block-inject weight; keeps cross-layer read/write dynamic.
- **MoE**: `mlp.gate` router → top-10 of 512 experts (+ 1 always-on shared
  expert), each expert a SwiGLU in native E4M3 with 128×128 block scales.
- **PLE / hashed n-gram** (layer 1 only): see `QWEN38_TENSORS.md` and
  `MEMORY_MODEL.md`. Per token, hash the bigram/trigram history to 16 row
  addresses, read 16×160 B E4M3 rows, project and fold into the residual via a
  causal conv. The ~51B-row table **never becomes resident**.

## Forward pipeline (target - Phase 2+)

```
token ─▶ embed ─▶ [gated-residual read ─▶ (GDN | QSA) ─▶ gated-residual write
                    ─▶ gated-residual read ─▶ MoE(router→experts)+shared ─▶ write
                    ─▶ (layer 1 only) PLE inject] ×48 ─▶ final mixer ─▶ LM head ─▶ logits
```

`INFERENCE_PIPELINE.md` expands this with the I/O-scheduling view.

## Discrepancies with the project brief (brief rule 13)

1. **Name.** Brief "Qwen3.8-Flash-Next" ↔ `model_type: qwen4_exp_text`. The
   loader validates on `qwen4_exp_text`.
2. **"QSA" layers.** `config.layer_types` labels the 12 sparse-attention layers
   `full_attention`; they carry the `self_attn.indexer.*` submodule. The loader
   accepts `full_attention` and `qwen_sparse_attention` as attention layers.
3. **RoPE.** Brief implies plain RoPE. Real: partial (dim 64), θ=1e7, interleaved
   mRoPE. Text-only path collapses mRoPE to 1-D position = token index.
   Multimodal mRoPE sectioning is out of scope.
4. **Attention output gate** (`·2` on q_proj) and the **MTP layer** are not in
   the brief. The gate is handled; MTP is skipped.
5. **Multimodal wrapper.** Checkpoint is vision+text; this engine is text-only,
   like colibri. Vision tensors are indexed and reported, never loaded.
6. Brief §2's hidden 2560 / 24 heads / 2 KV / context 262144 / 512 experts /
   top-10 all match the real config.
