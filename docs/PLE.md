# PLE — hashed n-gram embedding (`src/qwen38/ple.zig`)

Injected once, at layer `ple_layer` (index **1**). Adds a large (~51B-parameter)
associative memory with almost no per-token compute. Ported from colibri's
`q38_hash_row` / `q38_ple_row` / `q38_ple`.

## Dimensions (`Dims`, real model)

| | value |
|---|---|
| `hc_width` (= hc_count·hidden) | 10240 |
| `ple_dim` | 2560 (= ngram_heads·ngram_head_dim) |
| `ngram_size` / `heads_per_ngram` | 3 / 8 → `ngram_heads` 16 |
| `ngram_head_dim` | 160 |
| `ple_conv_kernel_size` | 4 → `stateLen` = `(4-1)·3` = 9 |
| `split_ngram_parts` | 128 |

Heads `0..7` are **bigram** (hash of `cur, prev`), `8..15` **trigram**
(`cur, prev, prev2`).

## The table (`Table`) — streamed, never resident

- Shards `layers.1.ple.ple_embedding.ngram_embedding.shard_<p>.weight`,
  `p = 0..127`, each `[rows_p, 160]` E4M3 (also accepts a single
  `…ngram_embedding.weight` and F32 shards). Held as `View`s into the shard
  mmap; `part_start[]` is the cumulative row count.
- Hash parameters (i64 tensors under `ple.ple_embedding.`):
  `layer_multipliers[3]`, `ngram_heads_vocab_sizes[16]`,
  `ngram_heads_offsets[16]`; scalar `ngram_embedding.weight_scale`.

### Address (`hashRow`) — deterministic, no scan

```
x  = u64(cur)·mult[0]  ^  u64(p1)·mult[1]         (all u64, wrapping)
                       ^  u64(p2)·mult[2]          (trigram heads only)
r  = i64(x) % vocab[head]      (C truncation; += vocab if negative)
row = offset[head] + r
```

`row → (shard p, local)` follows from `part_start[]`; the byte offset is
`local · 160` (E4M3) — no table scan, ever. This is why PLE prefetch is
*deterministic* while expert prefetch is only predictive.

### `readRow(row, out[160])`

Locate the shard, decode `160` E4M3 bytes × `weight_scale` (or read F32).

## Per-token forward

1. Gather `ngram_heads · ngram_head_dim` (= `ple_dim`) values: 16 rows.
2. `keys = emb · key_projᵀ` (`[hc_width]`), `value = emb · value_projᵀ` (`[hidden]`).
3. Per gated-residual branch `b` (`hc_count` = 4):
   - `kn = rms0(keys_b, norm_key_b)`, `qn = rms0(hyper_b, norm_query_b)`
   - `dot = (kn · qn) / √hidden`; `shaped = copysign(√max(|dot|, 1e-6), dot)`;
     `g = σ(shaped)`
   - `gated_b = g · value`; `norm_b = rms0(gated_b, norm_conv_b)`
4. **Dilated causal conv** over `hc_width` channels: taps at `k·ngram_size`
   into a length-`stateLen` ring (`{t, t−3, t−6, t−9}` with kernel 4 / dilation 3);
   `out[d] = gated[d] + silu(conv)`, then shift the ring and append `norm[d]`.
5. Advance the bigram/trigram history (`eos` resets it).

`out` (`[S, hc_width]`) is **added into** the 4-branch residual by the caller.

## State (`State`)

- `ring` — `hc_width · stateLen` f32 causal-conv history
- `history[2]` + `history_len` — the bigram/trigram window

`reset()` for a new sequence.

## Prefetch (`prefetchRows`)

Computes every row address for a chunk of tokens (history simulated, not
mutated) and reads all `S · 16` rows up front. `forward(..., prefetch)` then
consumes that buffer. Result is **bit-identical** to inline reads — verified in
tests and `selftest`. The bounded async I/O queue that would overlap these reads
with the two preceding layers' compute is Phase 7.

## Guarantees / tests

- Hash addresses land in `[offset[h], offset[h]+vocab[h])` and inside the table.
- Prefetched forward == inline forward, bit for bit.
- Chunk-boundary invariance (ring + history are token-causal).
- Output finite.
- Full numerical validation vs upstream awaits the Python reference harness.
