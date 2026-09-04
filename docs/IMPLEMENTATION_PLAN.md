# Implementation plan

## Principle

> The model is not a file that must be loaded. It is a computation graph whose
> working set changes over time.

Build a **small verified core** and grow it into the full engine. Every phase
ends with: build ✅ · test ✅ · benchmark (once kernels exist) · document.

## Toolchain

- **Zig 0.16.0** (2026-04-13), pinned in `build.zig.zon` (`minimum_zig_version`).
  Installed via `winget install zig.zig`.
- No third-party dependencies. `std` only (brief rule 9). Zig 0.16's `std.Io`
  interface (the `init: std.process.Init` main signature, `std.Io.Dir` /
  `std.Io.File`, `*std.Io.Writer`) is used throughout.

## Phases

| # | scope | status |
|---|---|---|
| **1** | config parser · manifest parser · tensor-metadata loader · memory-budget manager · `inspect` CLI · docs | **done** |
| **2** | mmap/`View` materialization · `ops/` (matmul, rmsnorm, rope, softmax, activation) · embeddings · LM head · `selftest` CLI | **done** |
| **3** | Gated DeltaNet (projection → causal conv → Q/K/V → beta/gate → recurrent update → out proj); persistent `GdnState` | **done** |
| **4** | MoE forward: router → top-10 → expert load → GEMM → weighted accumulate; block-FP8 dequant; bounded per-layer expert LRU with stats | **done** |
| **5** | PLE / hashed n-gram: token history → hash → address → table lookup; `split_ngram_parts` partitioning; deterministic prefetch | **done** |
| **6** | Qwen Sparse Attention: indexer → micro-block pooling → block ranking → top-k → full attention over selected tokens; context KV cache | **done** |
| **7a** | Gated residual (`hyper_connection_mixer`); **end-to-end forward** (embed → 48 layers → final mixer → LM head), PLE inject at layer 1; `Model`/`State`/`Scratch`; `forward` CLI on raw token ids + greedy decode | **done — awaiting approval** |
| **7b** | Bounded typed priority `Scheduler` (no speculative starvation); `LastTokenPredictor`; concurrent PLE row reads (`std.Io.Group`); predictive expert prefetch wired into `forward`; prefetch-hit / demand-load / accuracy telemetry | **done — awaiting approval** |
| **8a** | Byte-level BPE tokenizer (`tokenizer.json`), ChatML template, `chat` CLI (one-shot: template → encode → forward → greedy decode → detokenize) | **done — awaiting approval** |
| **8b** | Independent NumPy reference forward + oracle test — cross-validates the whole Zig forward against a second port of the spec | **done — awaiting approval** |
| **8c** | `benchmark` (§23 runtime report: tok/s, TTFT, per-phase timers, `compute_stall_due_to_io`) and `stress` (§26 RAM-budget sweep); `Timers` + tracked-bytes `Meter` | **done — awaiting approval** |
| **9** | Threaded kernels — `runtime/parallel.zig` fans the matmul output-row loop and the GDN recurrent head loop over `std.Io.Group` workers, gated by a work threshold (toy models stay serial). `--threads` flag | **done — awaiting approval** |
| 9b | MoE per-expert threading (decode path fans the top-k SwiGLU evals) | **done** |
| — | AVX2 / FMA kernel pass (`dotF32`/`dotBf16` 4× `@mulAdd`; fused `dotFp8Row` for S==1) | **done** |
| MTP | Speculative decode via the checkpoint's MTP head | **investigated, declined** — see below |
| **10a** | CUDA backend plumbing + the block-FP8 matmul on the GPU (`--cuda`), bit-identical | **done** |
| **10b** | bounded VRAM LRU cache of resident expert weights (`--vram`) | **done — no speedup on 4 GB** |
| 10c | pinned staging + async streams + batched multi-expert kernels | *open, uncertain on a Max-Q* |

## Phase 1 deliverables

```
build.zig, build.zig.zon          Zig 0.16 build: `run`, `test`, `gen-fixture`
src/main.zig                      subcommand dispatch, clean exit codes
src/cli/args.zig                  --ram-limit / --context / --profile / --gpu
src/cli/inspect.zig               `inspect` (metadata only)
src/model/config.zig              Cfg + strict parse + validate  (std.json)
src/model/safetensors.zig         8-byte len + JSON header reader; DType; no bodies
src/model/manifest.zig            ModelManifest: config + index → []TensorLocation + classification
src/runtime/budget.zig            MemoryManager planning half: Plan, plan(), Profile, cap ladder
src/util/units.zig                size parse + humanize
tools/gen_tiny_fixture.zig        writes test/fixtures/tiny/
docs/*.md                         this set
```

## Phase 2 deliverables

```
src/ops/matmul.zig                y = x @ Wᵀ (f32 + bf16 weights), @Vector SIMD, bf16<->f32
src/ops/rmsnorm.zig               rms0 (zero-centered, 1+w) + rmsGated (Qwen3-Next DeltaNet)
src/ops/rope.zig                  NeoX split-half partial RoPE
src/ops/softmax.zig               stable softmax
src/ops/activation.zig            sigmoid / silu / softplus / geluTanh
src/model/tensors.zig             View + decodeInto (F32/F16/BF16); e4m3ToF32 helper; FP8 errors clearly
src/model/weights.zig             Weights: mmap each shard whole (buffer fallback); view/materialize; embed(); lmHead()
src/cli/selftest.zig              `selftest [MODEL_DIR]` — ops self-consistency + weight-materialization checks
tools/gen_tiny_fixture.zig        now writes deterministic non-zero F32/BF16 bodies
```

`selftest` is a bring-up diagnostic, **not** an inference run — the transformer
layers do not exist yet. FP8 decoding (routed experts, PLE table) is deliberately
an error in `tensors.zig` until Phase 4 / 5 supply the block/scalar scales.

## Phase 3 deliverables

```
src/model/tensors.zig    + NativeMatrix (resident BF16/F32 weight matrix, mmap→owned, matmul)
src/model/weights.zig    + matrixBySuffix / vectorBySuffix / materializeView helpers
src/qwen38/gdn.zig        Dims, GdnState (persistent rec + conv ring), Layer (weights),
                          Scratch, forward()  — port of colibri q38_deltanet
src/cli/selftest.zig     + GDN checks (finite, chunk-boundary invariance, zero-input→zero)
```

Numerical validation so far is by invariant + a naive in-test reimplementation;
full comparison against the upstream `Qwen4ExpForCausalLM` is the deferred Python
reference harness (brief §25). See `docs/GDN.md`.

## Phase 4 deliverables

```
src/ops/fp8.zig          e4m3ToF32, matmulFp8 (128×128 block-scaled), nblk; demo bytes
src/model/tensors.zig    e4m3ToF32 re-exported from ops/fp8
src/qwen38/moe.zig        Dims, Fp8Matrix, Expert, ExpertCache (bounded LRU + CacheStats),
                          Layer (router + shared expert, resident), Scratch, forward()
                          — port of colibri q38_moe_decode
src/cli/selftest.zig     + MoE checks (finite, routes topk, warm-replay all hits,
                          tight cache evicts) and FP8 numeric check via ops tests
tools/gen_tiny_fixture.zig  FP8 expert bytes = exact small values; positive block scales
```

Targets the **per-expert block-FP8** layout of `Qwen3.8-Flash-Next-FP8`
(`experts.<e>.{gate,up,down}_proj.weight` + `_scale_inv`).  The fused BF16
alternative from the upstream text class is not wired.  Prefill still runs the
per-token decode path — expert-major grouping is a Phase 8 throughput change that
does not affect results.  See `docs/MOE.md`.

## Phase 5 deliverables

```
src/qwen38/ple.zig        Dims, Table (streamed n-gram table: hashRow + readRow,
                          shard-partitioned addressing, no scan), State (conv ring
                          + bigram/trigram history), Layer (key/value proj + norms
                          + conv, resident), Scratch, prefetchRows(), forward()
                          — port of colibri q38_hash_row / q38_ple_row / q38_ple
src/cli/selftest.zig     + PLE checks (addresses in range, prefetch bit-identical,
                          chunk-invariant, finite)
tools/gen_tiny_fixture.zig  + Builder.addRaw for the i64 hash-parameter tensors
                          (multipliers, per-head vocab/offset that exactly fill the table)
```

The n-gram table is memory-mapped; a row "read" is a page fault serviced by the
OS page cache.  `prefetchRows` demonstrates the deterministic-address property
(all addresses known before any compute); the bounded async I/O queue is Phase 7.
See `docs/PLE.md`.

## Phase 6 deliverables

```
src/qwen38/qsa.zig        Dims, Cache (K normalized+RoPE'd / V raw / indexer key,
                          context-sized), Layer (q/k/v/o + q/k norm + indexer
                          qk-proj + indexer norms, resident), Scratch, forward()
                          — port of colibri q38_attention
src/cli/selftest.zig     + QSA checks (finite; prefill == chunked-decode via the
                          KV cache; deterministic)
```

The sigmoid output gate (2nd half of `q_proj`) and the GQA head mapping are
handled.  `forward` appends tokens to the cache at absolute positions and asserts
in-order growth (`cache.len == pos_base`).  Block scoring uses `std.sort.pdq`
(descending score, block-index tie-break).  See `docs/QSA.md`.

## Phase 7a deliverables

```
src/qwen38/residual.zig  Dims, Gated (norm + down/up + optional inject), Scratch,
                         read() / apply()  — port of colibri q38_gr_read/q38_gr_apply
src/qwen38/model.zig      Model (all per-layer resident weights + PLE table),
                         State (per-layer GDN/QSA/expert-cache + PLE + pos),
                         Scratch, forward(), generateGreedy()  — port of `step`
src/cli/forward.zig      `forward <dir> --tokens <csv> [--steps N]` — runs the full
                         forward on raw token ids, prints top-8 logits + greedy decode
src/cli/args.zig         + --tokens / --steps
src/main.zig             + `forward` command; `chat`/`benchmark`/`stress` message
                         now points at `forward` and cites the missing tokenizer
```

Still **synchronous** — a cache miss loads the expert inline. No tokenizer:
ids in, ids out. `forward` validates every token id against the vocabulary and
sizes the context / expert-cache from the Phase 1 memory plan.

Verified: `forward` is finite; one-shot prefill == token-by-token decode
(`≤ 2e-3` rel, over all 4 layers of the tiny fixture); out-of-vocab ids rejected;
greedy decode stays in vocabulary and stops at EOS. See `docs/FORWARD.md`.

## Phase 7b deliverables

```
src/runtime/io.zig       ResourceKey/Priority, Scheduler (bounded, dedup+upgrade,
                         strict-priority `next`, cancel), Stats
src/runtime/predict.zig  PrefetchPrediction, LastTokenPredictor (per-layer, tracks accuracy)
src/qwen38/ple.zig       + prefetchRowsAsync — S·ngram_heads row reads fan out via std.Io.Group
src/qwen38/moe.zig       + ExpertCache.prefetch + Slot.prefetched flag;
                         CacheStats gains prefetch_hits/prefetch_wasted/demand_loads/prefetch_loads;
                         moe.forward reports the last token's routed experts
src/qwen38/model.zig     forward(...) gains Opts { io, scheduler, predictor };
                         PLE prefetch hoisted before layer 0; per-layer expert prefetch
                         driven by predict→submit→drain; predictor.record after each MoE
src/cli/forward.zig      wires the scheduler + predictor; prints the I/O telemetry block;
src/cli/args.zig         + --expert-cap (force a small cache to exercise the scheduler)
```

**Concurrency reality**: PLE row reads are genuinely concurrent (`std.Io.Group`
fan-out).  Expert prefetch executes *cooperatively* on the compute thread — the
bounded priority queue, dedup, no-starvation policy and the prefetch-hit /
`compute_stall_due_to_io` (= demand-load count) telemetry are all real, but true
compute/IO overlap for experts needs the evented `Io` backend and a real
slow-disk checkpoint to matter.  Documented in `docs/IO_SCHEDULER.md`.

Verified: `forward` output is **bit-identical** with prefetch on vs off; the
scheduler's `next` returns HIGH before MEDIUM before LOW even behind a full LOW
backlog; the bounded queue rejects when full; `--expert-cap 1` drives visible
prefetch loads / hits / waste.

## Phase 8a deliverables

```
src/qwen38/tokenizer.zig     byte-level BPE: GPT-2 byte↔unicode map, an approximation
                             of Qwen's split regex, BPE merges from tokenizer.json,
                             special-token splitting; encode / decode
src/qwen38/chat_template.zig ChatML framing (<|im_start|>role\ncontent<|im_end|>\n)
src/cli/chat.zig             `chat <dir> --prompt "..." [--system ...] [--steps N]`
                             one-shot: template → encode → forward → greedy → detokenize
src/cli/args.zig             + --prompt / --system
tools/gen_tiny_fixture.zig   + a minuscule tokenizer.json (a-z / 0-9 / merges / ChatML specials)
src/main.zig                 `chat` is a real command; only `benchmark`/`stress` remain NOT IMPLEMENTED
```

The pre-tokenizer is exact for ASCII and treats most non-ASCII codepoints as
letters — word boundaries in exotic scripts may differ from HF `tokenizers`.
`chat` is single-turn (`--prompt`); the harness runs non-interactively so a REPL
comes later. See `docs/TOKENIZER.md`.

## Phase 8b deliverables

```
tools/reference/qwen38_ref.py    NumPy forward — a second, dense/loopy port of the
                                 same colibri-derived spec (all 5 subsystems)
tools/reference/build_oracle.py  reads the Zig-generated fixture, writes
                                 test/fixtures/tiny/oracle.json (3 token sets)
tools/reference/requirements.txt numpy only (no torch / transformers)
build.zig                        + `zig build oracle` step (needs python)
src/qwen38/model.zig             + test "matches the NumPy reference oracle" —
                                 skipped when oracle.json is absent
```

Both implementations read the **same** fixture weights; agreement (< 2e-2 max
abs on the final logits, argmax exact) is strong evidence that both are faithful
to the spec.  This is *not* validation against the released `Qwen4ExpForCausalLM`
weights/outputs — that needs the checkpoint + `transformers` and remains a
further step.  See `docs/REFERENCE.md`.

## Phase 8c deliverables

```
src/runtime/timers.zig   Timers — per-phase (embed / gated_residual / deltanet /
                         qsa / moe / ple / lm_head) wall-clock via std.Io.Timestamp
src/runtime/meter.zig    Meter — pass-through allocator tracking current + peak bytes
src/qwen38/model.zig     Opts + timers; State.init(+ io) times demand expert loads
                         (ExpertCache.stats.demand_ns = compute_stall_due_to_io)
src/cli/benchmark.zig    `benchmark <dir> [--prompt-len N] [--steps N] [--expert-cap K]`
                         — the brief §23 "=== QWEN38 RUNTIME ===" report
src/cli/stress.zig       `stress <dir> [--context N] [--steps N] [--ram-limit G]`
                         — sweep RAM budgets, check tracked peak ≤ limit (brief §26)
src/cli/args.zig         + --prompt-len
```

All of the brief's CLI is now real (`inspect`, `chat`, `benchmark`, `stress`,
plus `forward` / `selftest`).  `Meter` tracks *our* allocations, not process RSS
— an honest proxy that lines up with the memory-model accounting.  See
`docs/BENCHMARK.md`.

## Real-checkpoint validation (metadata, 2026-09)

The ~5 MB of metadata (`config.json`, `model.safetensors.index.json`,
`tokenizer.json`) of `Qwen/Qwen3.8-Flash-Next-FP8` was downloaded (the full
185 GB does not fit the machine's free disk).  Added:

```
src/model/manifest.zig   metadata-only fallback: when the shard files are absent
                         but the index is present, build the manifest from the
                         weight_map alone (names + categories, no shapes)
src/runtime/budget.zig   estimateResidentBytes(cfg) — config-derived BF16 dense size
src/cli/inspect.zig      handles the metadata-only manifest
src/cli/tokenize.zig     `tokenize <dir> --prompt "..."` — encode/decode via
                         the checkpoint's real tokenizer.json (no weights)
```

Results: config parser accepts the real file unchanged (**no rule-13
discrepancy**); classifier categorizes all 152,089 tensors with zero unknowns;
resident estimate 9.22 GiB and the full memory plan match colibri's numbers;
the real 248 k-vocab tokenizer round-trips ASCII text.  The full-weight forward
is still unvalidated (needs the download or a colibri-diff).

## Phase 9 deliverables

```
src/runtime/parallel.zig  process-wide switch + `chunks(n, work, ctx, body)`:
                          splits [0,n) over std.Io.Group workers when enabled AND
                          `work >= min_work` (96 Ki ops), else serial
src/ops/matmul.zig        matmul / matmulBf16 fan the output-row loop
src/ops/fp8.zig           matmulFp8 fans the output-row loop
src/qwen38/gdn.zig        the gated-delta recurrent head loop fans over value heads
                          (per-head state / delta / core slices — disjoint)
src/cli/{forward,chat,benchmark,stress}.zig  `parallel.enable(io, opts.threads)`
src/cli/args.zig          + --threads (0 = auto / CPU count, 1 = single-threaded)
```

Each fanned task writes a **disjoint** slice, so the result is **bit-identical**
to the serial path — verified (`matmul` large-shape threaded == serial; the whole
`forward` threaded == serial).  The work threshold means the toy fixture never
actually threads (its matmuls are ~2 Ki ops), so `--threads 0` does no harm
there; the real model's projections (≥ 1.6 M MACs each) and expert GEMMs do fan
out.  The wall-clock benefit can only be shown on the real checkpoint — on this
machine the very first (un-thresholded) attempt was ~24× *slower* on the fixture
purely from `Io.Group` coordination overhead, which is exactly what the threshold
now avoids.  See `docs/THREADING.md`.

## Phase 9b deliverables

```
src/runtime/parallel.zig  threadlocal `in_worker` — a `chunks` call from inside a
                          worker runs serial (so the MoE expert fan-out does not
                          spawn a nested per-matmul fan-out over the same pool)
src/qwen38/moe.zig        forwardDense (S==1 decode, cap >= topk): pull the top-k
                          experts into the cache serially, then fan their SwiGLU
                          eval over `parallel.chunks` (module-level EvalCtx /
                          evalExpertBody / evalExpertsParallel). Each expert
                          writes a disjoint `eo` row; the accumulate runs in
                          top-k order → bit-identical to serial. Scratch eg/eu/eh
                          grown to topk*inter, eo to topk*hidden.
```

Coarser tasks than Phase 9 (one per expert, not one per matmul-row-block), so
less `std.Io.Group` coordination for the same work. Warm-cache decode benchmark
`--threads 1` vs `12`: MoE 9292 → 1433 ms/forward (6.5×), deltanet 4.3×, decode
0.060 → 0.339 tok/s. `forward` token output verified **bit-identical** to serial
and to colibri. The gain only shows warm — a cold run's first forwards fault
~2.3 GB of expert weights off SSD, masking the compute speedup. See
`docs/BENCHMARK.md`.

Not pursued: QSA per-head threading (~94 ms/forward warm — noise next to MoE's
1.4 s, and needs per-head score/softmax scratch to avoid races) and prefill
row-chunking (results-invariant robustness only).

### AVX2 / FMA kernel pass

`ops/matmul.zig` dots → 4 `@mulAdd` accumulator chains (1.46× on the pure
f32/bf16 dot, micro-benchmarked); `ops/fp8.zig` → fused `dotFp8Row` for the
`S == 1` decode path (no f32 weight-row buffer). `zig build` already targets the
native CPU (AVX2+FMA). Greedy output token-for-token unchanged. **End-to-end only
+3 %** — the 12-thread MoE decode is memory-bandwidth-bound (~1 GB experts/token
over a ~40 GB/s bus), not FLOP-bound. CPU compute levers are now exhausted; the
GPU backend (Phase 10, deferred) is the remaining one. See `docs/BENCHMARK.md`.

### MTP speculative decode — investigated, declined (2026-09-04)

The checkpoint's MTP head (`mtp.*`, `mtp_num_hidden_layers: 1`,
`mtp_use_hidden_state_from_layer: null` → last layer, `layer_types: ["full_attention"]`)
is **not a lightweight head**: it is a full QSA decoder layer carrying its own
**512-expert block-FP8 MoE** + shared expert + two hyper-connections +
`fc_embedding`/`fc_hidden` fusion of `embed(x_{t+1})` and `h_last`. It reuses the
main `embed_tokens`, `norm`, `lm_head` (`mtp_use_dedicated_embeddings: false`,
`tie_word_embeddings: false`).

A single MTP forward ≈ 0.15× a main forward (≈10 expert reads vs ≈480). With
draft depth 1 and a realistic 70–85 % greedy acceptance the speculative loop
(1 main verify pass at S=2 + 1 MTP forward per round) yields ≈ **1.3–1.5×**
decode — a whole phase (`mtp.zig`, `speculative.zig`, a second expert cache,
manifest/budget/fixture changes) for a moderate, memory-bound-limited gain.
Not worth it ahead of the GPU backend; revisit if/when a GPU path exists (MTP
verification batches well on a GPU).

## Phase 10a deliverables

```
src/backend/cuda/colizig_cuda.cu   block-FP8 matmul kernel + C ABI (nvcc → DLL)
src/backend/gpu.zig                LoadLibraryA loader + serialised dispatch + --cuda-verify
src/ops/fp8.zig                    matmulFp8 → gpu.matmulFp8 first, else CPU (matmulFp8Cpu)
src/qwen38/moe.zig                 forwardDense stays serial when the GPU is up
src/cli/{args,forward,chat,benchmark}.zig   --cuda / --cuda-verify
build.zig                          `zig build cuda` (opt-in; nvcc + MSVC)
build_cuda.ps1                     wrapper that imports the MSVC env first
```

`--cuda` greedy decode on the real checkpoint is **token-for-token identical** to
CPU and colibri. Hardware: NVIDIA Quadro T1000 Max-Q (4 GB, sm_75), CUDA 13.0.
See `docs/GPU.md`. The one non-obvious fix: the CUDA runtime sets the host
thread's SSE flush-to-zero mode, corrupting every subsequent CPU float op —
each ABI call now saves/restores MXCSR.

**10b** adds `--vram <size>`: a bounded LRU cache of resident expert weights in
VRAM, keyed by `Fp8Matrix.key`. A hit skips the ~1.6 MB weight upload.

**Measured**: on the 4 GB T1000 the cache holds ~13 experts/layer, a short
generation routes to ~50+, so it thrashes at **26 % hit** and `--cuda` decode
stays **~2× slower than the warm CPU** (0.35 vs 0.75 tok/s). The MoE-in-VRAM
approach needs an 8 GB+ card to reach a hit rate that pays. 10c (pinned staging,
async streams, batched kernels) or a dense-weights-in-VRAM pivot are the
remaining options; on a 35 W Max-Q neither is clearly worth it. The backend is
kept — correct, opt-in, and useful on better hardware. Figures in
`docs/BENCHMARK.md` / `docs/GPU.md`.

## Real-checkpoint bring-up + perf pass (2026-09)

The released `Qwen/Qwen3.8-Flash-Next-FP8` (172.8 GiB, 131 shards) was downloaded
and the engine runs it **end-to-end**: `inspect` classifies all 152,089 tensors
(0 unknown), `chat` streams coherent English, memory stays bounded (~7 GiB
private on a 32 GB box). Not yet cross-checked logit-for-logit against
`Qwen4ExpForCausalLM`.

The first real run was ~32 s to first token and ~0.04 tok/s. A profiling pass
took it to ~29 s TTFT (16-token prompt) / ~0.25 tok/s, all changes tests + oracle
clean:

- `src/ops/matmul.zig` — `dotBf16` widening vectorised (bit-identical).
- `src/model/weights.zig` — `Weights.find` is an O(1) name index, not a linear
  scan of the manifest; `lmHead` fans over `parallel.chunks`.
- `src/ops/fp8.zig` — 256-entry E4M3 LUT; `matmulFp8` dequants each weight row
  once then SIMD-dots.
- `src/qwen38/moe.zig` — `Fp8Matrix` borrows the shard mmap (no copy);
  `forwardGrouped` evaluates each distinct routed expert once per prefill.
- `src/runtime/budget.zig` — plan splits **private resident** (hard requirement)
  from **expert stream** (reclaimable page cache); see `MEMORY_BUDGET.md`.
- `src/cli/chat.zig` — no `--prompt` → interactive multi-turn REPL, live token
  streaming, per-reply `tok/s` (state carries across turns; `/reset`, `/exit`).

Tried and reverted: async `std.Io.Group` expert-page prewarm (no win — the matmul
fan-out already overlaps cold reads).

Still open: logit validation vs the reference weights; QSA per-head threading;
GPU backend. (Per-expert MoE decode threading: done — Phase 9b above.)

## Exit codes

`0` ok · `1` usage / not-implemented / context-does-not-fit / selftest-failed ·
`2` bad CLI flags · `3` bad checkpoint (config, index, shard, unknown tensors).

## Verification

```sh
zig build oracle          # (needs python+numpy) writes test/fixtures/tiny/oracle.json
zig build test            # 59 tests; auto-runs gen-fixture; oracle test runs if oracle.json exists
zig build run -- inspect test/fixtures/tiny --profile tiny --context 4096      # exit 0
zig build run -- inspect test/fixtures/tiny --ram-limit 512M --context 262144  # exit 1, explains
zig build run -- selftest test/fixtures/tiny                                   # exit 0, subsystem + tokenizer checks
zig build run -- forward test/fixtures/tiny --tokens 3,1,4,1,5 --steps 6 --expert-cap 1   # I/O scheduler visibly working
zig build run -- forward test/fixtures/tiny --tokens 9999                      # exit 3, token out of vocab
zig build run -- chat test/fixtures/tiny --prompt "hello world" --steps 12     # exit 0, ChatML + greedy reply
printf 'hi\n/exit\n' | zig build run -- chat test/fixtures/tiny --steps 2      # exit 0, interactive REPL (EOF-terminated)
zig build run -- benchmark test/fixtures/tiny --prompt-len 8 --steps 20        # exit 0, runtime report
zig build run -- stress test/fixtures/tiny --context 8192 --steps 10           # exit 0, RAM-budget sweep table
```

When a real `config.json` + `model.safetensors.index.json` are available,
`inspect <dir>` runs on metadata alone (no weight download) and its figures
should land near colibri's published numbers — see `MEMORY_BUDGET.md`.
