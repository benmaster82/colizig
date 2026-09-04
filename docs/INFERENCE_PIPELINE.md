# Inference pipeline (target design - Phases 2–7)

Phase 1 implements none of this; it is recorded here so the metadata core (config,
manifest, budget) is built against the right end state.

## Per-token flow

```
             token id
                │
          embed_tokens  (resident)
                │
   ┌────────────┼─────────────────────────────────────────┐  × 48 layers
   │  gated-residual READ  (4-branch, low-rank mixer)      │
   │        │                                              │
   │   layer is linear_attention ?                         │
   │     yes → Gated DeltaNet:                             │
   │           in_proj_qkv → causal conv1d(k=4)            │
   │           → per-head Q/K norm+scale                   │
   │           → recurrent state update (persistent)      │
   │           → RMSNormGated(z) → out_proj                │
   │     no  → Qwen Sparse Attention:                      │
   │           q/k/v/gate proj + indexer qk proj          │
   │           → pool 4-token blocks, score vs indexer q  │
   │           → top ~512 blocks + ≤3 causal tail          │
   │           → full 24-head attention over those tokens │
   │           → cache K/V/indexer-key for this token     │
   │  gated-residual WRITE                                 │
   │        │                                              │
   │  gated-residual READ                                  │
   │   MoE:  router → top-10 of 512 experts               │
   │         + shared expert (always on)                  │
   │         → load missing experts (LRU / IoScheduler)   │
   │         → SwiGLU GEMM per expert (E4M3, block scales)│
   │         → weighted accumulate                        │
   │  gated-residual WRITE                                 │
   │        │                                              │
   │  layer == 1 ?  → PLE inject:                          │
   │        hash bigram/trigram history → 16 row addrs    │
   │        → 16×160 B reads (prefetched at forward start)│
   │        → key/value proj → gated dot → causal conv    │
   │        → fold into residual                          │
   └──────────────────────────────────────────────────────┘
                │
      final gated-residual mixer
                │
           lm_head  → logits
```

## I/O-scheduling view

```
        token t
          │
   ┌──────┴───────────────────────────┐
   │ PLE addresses (deterministic)    │──▶ enqueue 16 reads, HIGH-1 layer of lead
   │ prev-layer router (predictive)   │──▶ enqueue next-layer experts, MEDIUM
   └──────────────────────────────────┘
          │
     IoScheduler  (bounded queue, priority by
     distance-to-consumer · type · cache state · confidence)
          │
   ┌──────┼───────┐
   ▼      ▼       ▼
  SSD    RAM     VRAM
          │
          ▼
       COMPUTE ──▶ logits ──▶ token t+1
```

Priorities: **HIGH** = data the next layer needs · **MEDIUM** = predicted
next-layer working set · **LOW** = speculative. Speculative work must never
starve mandatory work. Prefill and decode use different execution strategies
(prefill groups tokens by expert; decode optimizes the single-token path).

The primary metric for the whole project is **`compute_stall_due_to_io`** - time
compute spends blocked on a read the scheduler failed to hide.
