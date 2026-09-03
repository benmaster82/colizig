# I/O scheduler & prefetch (`src/runtime/io.zig`, `src/runtime/predict.zig`)

The project's defining feature is **intelligent movement of model state between
storage tiers** (brief §13). Phase 7b builds the scheduling architecture and the
measurement; the payoff is only visible under real memory pressure with a real
slow disk.

## `Scheduler` — bounded typed priority queue

```
ResourceKey = ple(row) | expert(layer, id) | qsa(layer)
Priority    = high    (needed by the next layer)
            | medium  (predicted next working set)
            | low     (speculative)
```

- **`submit(key, prio)`** — dedups; a re-submit at a stronger priority *upgrades*
  the queued entry. Returns `false` when the bounded queue (`capacity`) is full —
  the caller must then do a synchronous read.
- **`next()`** — pops the highest-priority pending key, FIFO within a priority.
  Strict priority means **mandatory work is never starved by speculative work**
  (verified: a HIGH request submitted behind 50 LOW requests comes out first).
- **`cancel(key)`** — drop a queued key (e.g. it was just demand-served).
- **`Stats`** — `submitted`, `deduped`, `upgraded`, `rejected_full`,
  `serviced[high|med|low]`, `cancelled`, `queue_peak`.

## `LastTokenPredictor` (brief §14)

`PrefetchPrediction { layer, expert, confidence, expected_use_distance }` is the
abstraction. The concrete predictor exploits MoE routing's token-to-token
locality: the experts a layer routed for the previous token are the guess for
the next. `record(layer, ids)` scores the previous prediction and stores the new
ids; `Stats.accuracy()` is the running hit rate. `predictScored` emits
`PrefetchPrediction` rows with `confidence = accuracy`.

## Wiring into `forward`

`forward(model, state, sc, ids, logits, opts)` with
`Opts { io, scheduler, predictor }`:

1. **PLE** — before layer 0, `ple.prefetchRowsAsync` fans the `S · ngram_heads`
   row reads out through `std.Io.Group` (genuinely concurrent when `io` is set).
   Deterministic: every address is a pure function of the token ids. Result is
   bit-identical to serial reads. Injected at layer 1, so on the real model
   there are ~2 layers of compute to hide behind.
2. **Experts** — entering layer `i`, submit `predictor.predict(i)` at MEDIUM,
   then drain the scheduler: each `.expert` key → `ExpertCache.prefetch` (a
   no-op if already resident). After the MoE sub-block, `predictor.record(i,
   routed)` with the last token's routed experts.
3. `ExpertCache` tracks whether a resident expert got there by prefetch:
   `Slot.prefetched`. A demand `get` that hits a prefetched slot counts
   `prefetch_hits`; a prefetched slot evicted unused counts `prefetch_wasted`.
   `demand_loads` (expert loads on the critical path) is the count form of
   **`compute_stall_due_to_io`**.

The forward output is **bit-identical** with prefetch on or off — prefetch only
changes *when* bytes are read, never *which*.

## `forward` CLI telemetry

```
qwen38-zig forward <dir> --tokens <csv> [--steps N] [--expert-cap K]
```

`--expert-cap` overrides the memory-plan capacity — set it small to see the
scheduler work (a model that fits in RAM has nothing to prefetch). Prints:

```
-- I/O scheduler --
  scheduler queue peak / serviced HIGH·MED·LOW
  predictor accuracy
  expert demand loads        (compute_stall_due_to_io proxy)
  expert prefetch loads / hits / wasted
  expert prefetch cover      (% of routed accesses served warm)
```

## Concurrency status & follow-ups

- **Concurrent now**: PLE row reads (`std.Io.Group`).
- **Cooperative now**: expert prefetch runs on the compute thread. The queue
  policy, dedup, no-starvation guarantee and the hit-rate / stall telemetry are
  real; true compute/IO overlap for experts needs the evented `Io` backend
  (Uring / Kqueue / IOCP) and a real checkpoint on a real disk to measure a
  wall-clock benefit.
- **Follow-ups**: `qsa` requests aren't scheduled yet (QSA K/V is recomputed, not
  streamed); confidence-weighted / adaptive predictors; a shared cross-layer
  expert budget; the wall-clock (not count) form of `compute_stall_due_to_io`.
