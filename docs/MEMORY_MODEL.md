# Memory model

The engine treats the checkpoint as a **working set**, not a file to load.

## Tiers and states

```
        MODEL
          │
   ┌──────┼──────┐
   ▼      ▼      ▼
resident cached cold
  RAM    RAM    SSD
   │
   ▼
 VRAM (optional)
```

`src/runtime/budget.zig` declares:

- **Tier** - `ssd`, `ram`, `vram`
- **State** - `cold` · `prefetched` · `resident` · `in_use` · `evictable`

Phase 1 uses only the budget-accounting side. The state machine, eviction, and
VRAM staging arrive with the inference kernels (Phase 4+ / Phase 7).

## Residency classes

Every tensor is classified by name (`Manifest.classify`, see `TENSOR_MAP.md`)
into a `Category`, and each category maps to a `Residency`:

| Residency | meaning | categories |
|---|---|---|
| **resident** | always in RAM, native dtype (BF16) | embedding, lm_head, final_norm, gated_residual, attn_dense, qsa_indexer, deltanet, router, shared_expert, norm, ple_meta, ple_dense |
| **streamable** | loaded on demand into a bounded per-layer LRU, kept in native E4M3 | moe_expert |
| **cold** | never materialized; single positioned reads per token | ple_table |
| **ignored** | present in the checkpoint, never read | vision, mtp, unknown |

`unknown` is *reported loudly* (`inspect` fails with the offending names) rather
than silently skipped - a checkpoint the classifier does not fully understand is
a bug to fix, not a thing to guess at.

## What must be resident

The dense set - DeltaNet + QSA projections, norms, gated-residual mixers,
embedding, shared experts, LM head, PLE dense + meta - in native BF16. colibri
measures this at **9.2 GiB** on the real checkpoint. Everything else is sized by
a knob (`--ram-limit`, `--context`) or by the prompt.

## What stays on disk

- **Routed experts** (~120.8 GB): streamed; only a bounded LRU of decoded-in-
  place E4M3 experts is resident. Cache capacity is chosen by the planner.
- **PLE n-gram table** (~51B params): 0 bytes resident. Per token the engine
  hashes the bigram/trigram history to 16 addresses and issues 16 × 160-byte
  positioned reads. Addresses are a pure function of token ids - knowable before
  any compute - which is why PLE prefetch is *deterministic* and expert prefetch
  is only *predictive* (one layer ahead, from the previous router).

## Deterministic vs predictive prefetch

| kind | source | lead time |
|---|---|---|
| **deterministic** - PLE rows | token history → exact address (`q38_hash_row`) | whole forward pass (PLE is on layer 1 of 48) |
| **predictive** - next-layer experts | layer N router → likely layer N+1 working set | one layer |

The unified `IoScheduler` (Phase 7) prioritizes by distance-to-consumer,
request type, cache state and confidence, and must never let speculative work
starve mandatory work.
