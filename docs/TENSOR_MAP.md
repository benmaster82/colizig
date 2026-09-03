# Tensor map — name → (category, layer, expert)

`src/model/manifest.zig : classify()` turns a safetensors tensor name into a
`Class { category, layer?, expert? }`. Rules, in order:

1. name contains `.visual.` / starts `visual.` → **vision** (ignored)
2. name contains `.mtp.` / starts `mtp.` → **mtp** (ignored)
3. name == `lm_head.weight` → **lm_head**
4. strip a leading `model.language_model.` or `model.` prefix → `rest`
5. `rest` == `embed_tokens.weight` → **embedding**
6. `rest` == `norm.weight` → **final_norm**
7. `rest` starts `hyper_connection_mixer` → **gated_residual** (no layer)
8. `rest` starts `layers.<N>.` → parse `N`, `sub` = remainder:
   | `sub` matches | category |
   |---|---|
   | contains `hyper_connection` / `hc_norm`, or starts `attn_hyper` / `mlp_hyper` | gated_residual |
   | starts `self_attn.indexer.` | qsa_indexer |
   | starts `self_attn.` | attn_dense |
   | starts `linear_attn.` | deltanet |
   | == `mlp.gate.weight` | router |
   | starts `mlp.shared_expert` | shared_expert |
   | starts `mlp.experts.` + `gate_up_proj` / `down_proj` | moe_expert (fused; `expert = null`) |
   | starts `mlp.experts.<E>.…` | moe_expert (`expert = E`) |
   | contains `ngram_embedding.weight_scale` | ple_meta |
   | contains `ngram_embedding.shard_` / ends `ngram_embedding.weight` | ple_table |
   | starts `ple.ple_embedding.` | ple_meta |
   | starts `ple.` | ple_dense |
   | contains `norm` | norm |
   | otherwise | **unknown** |
9. anything else → **unknown**

Both checkpoint layouts are accepted: the multimodal export
(`model.language_model.*` for text, `model.visual.*` for vision) and the
standalone text export (`model.*`). The detected prefix is reported by
`inspect` as `tensor prefix`.

Routed experts may be per-expert block-FP8 matrices
(`…experts.<E>.{gate,up,down}_proj.weight` + `…weight_scale_inv`) or the fused
BF16 3-D tensors `…experts.gate_up_proj` / `…experts.down_proj` emitted by the
upstream text class — both classify as `moe_expert`.

## Category → residency

See `MEMORY_MODEL.md`. Summary: `moe_expert` → streamable, `ple_table` → cold,
`vision` / `mtp` / `unknown` → ignored, everything else → resident.

## Deterministic vs dynamic addressing

- **deterministic-address**: every resident and streamable tensor — its byte
  offset is fixed in the shard, recorded in `TensorLocation.offset` at manifest
  time.
- **dynamically-addressed**: PLE table rows — the row index is
  `head_offset[h] + (hash(cur, p1[, p2]) mod head_vocab[h])`, computed per token
  (`q38_hash_row`); the containing shard and local offset follow from the
  cumulative shard row counts (`split_ngram_parts = 128`), still without scanning.
