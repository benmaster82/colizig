# Project: qwen38-zig — Minimal Zig Inference Engine for Qwen3.8-Flash-Next on Non-Optimized Hardware

## Role

You are a senior systems engineer specializing in:

- Zig
- C/C++ low-level systems
- LLM inference runtimes
- Mixture-of-Experts architectures
- memory-mapped files
- asynchronous I/O
- CPU SIMD
- GPU acceleration
- model quantization
- memory hierarchies
- cache/prefetch algorithms

You are going to build a **small experimental inference engine in Zig** capable of running **Qwen3.8-Flash-Next** on hardware that is NOT optimized for large-model inference.

The project is explicitly experimental.

Do NOT attempt to reproduce a full production framework such as llama.cpp, vLLM or TensorRT-LLM.

The objective is to explore how far a highly memory-efficient Zig runtime can push a ~125B MoE model plus ~51B hashed N-gram/PLE parameters using:

```text
SSD
 ↓
RAM
 ↓
optional VRAM
 ↓
CPU/GPU compute
```

The engine must prioritize **low resident memory and intelligent data movement** over maximum throughput.

---

# 1. Primary objective

Build:

```text
qwen38-zig
```

a minimal native Zig inference engine for:

**Qwen3.8-Flash-Next**

with these architectural goals:

```text
                 Qwen3.8
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
       GDN         QSA          MoE
        │           │            │
        │           │       512 experts
        │           │       top-10/token
        │           │            │
        └───────────┼────────────┘
                    │
              Memory Scheduler
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        VRAM       RAM       SSD
```

The engine must be able to operate even when the entire model cannot fit into RAM or VRAM.

---

# 2. Important architectural target

The target architecture contains approximately:

```text
125B main parameters
~51B hashed N-gram / PLE parameters

48 transformer blocks

36 Gated DeltaNet blocks
12 Qwen Sparse Attention blocks

512 MoE experts
10 selected experts/token

hidden size: 2560

24 attention heads
2 KV heads

context:
262144 native target
```

Do NOT assume that the entire model can be resident.

The runtime must treat the model as a **working set**.

---

# 3. Core design principle

The most important architectural principle is:

> The model is not a file that must be loaded. It is a computation graph whose working set changes over time.

Therefore implement a memory hierarchy:

```text
                    MODEL
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     resident       cached        cold
        │             │             │
       RAM           RAM           SSD
        │
        ▼
       VRAM
```

The runtime must know:

- what is currently needed
- what will probably be needed next
- what can be evicted
- what can be prefetched
- what should never be loaded

---

# 4. Phase 1 — Repository and architecture analysis

Before writing significant code:

1. Study the official Qwen3.8-Flash-Next architecture.
2. Study the official model configuration.
3. Study the current Colibrì Qwen3.8 implementation as a reference.
4. Document all tensor shapes required by the forward pass.
5. Identify which tensors are:
   - persistent
   - per-layer
   - per-token
   - cacheable
   - streamable
   - deterministic-address
   - dynamically-addressed

Create:

```text
docs/ARCHITECTURE.md
docs/MEMORY_MODEL.md
docs/QWEN38_TENSORS.md
docs/INFERENCE_PIPELINE.md
```

Do NOT copy Colibrì code.

Use it only as an architectural reference.

---

# 5. Phase 2 — Minimal runtime

Create a small Zig runtime with:

```text
src/
    main.zig

    model/
        config.zig
        loader.zig
        tensors.zig

    runtime/
        scheduler.zig
        memory.zig
        cache.zig
        prefetch.zig
        io.zig

    ops/
        matmul.zig
        rmsnorm.zig
        rope.zig
        softmax.zig
        activation.zig

    qwen38/
        model.zig
        gdn.zig
        qsa.zig
        moe.zig
        ple.zig
        residual.zig
        vision.zig
        tokenizer.zig

    cli/
        chat.zig
        benchmark.zig
```

Keep the code modular.

---

# 6. Model loading

Do NOT load the complete checkpoint.

Implement:

```text
ModelManifest
```

which reads:

```text
config.json
model.safetensors.index.json
```

or equivalent model metadata.

The manifest must know:

```text
tensor name
dtype
shape
file
offset
size
layer
expert
category
```

Example:

```zig
const TensorLocation = struct {
    file: []const u8,
    offset: u64,
    size: u64,
    dtype: DType,
    shape: []usize,
    layer: ?u32,
    expert: ?u32,
};
```

---

# 7. Memory-mapped storage

Prefer:

```text
mmap
```

or platform-equivalent mechanisms.

Do not automatically read the entire tensor into RAM.

Implement:

```zig
TensorView
```

which represents:

```text
file + offset + length
```

without necessarily materializing the tensor.

The runtime should be able to create:

```text
TensorView
     │
     ▼
mmap / pread
     │
     ▼
working buffer
```

only when computation requires it.

---

# 8. Memory manager

Implement a central memory manager:

```zig
MemoryManager
```

with tiers:

```text
SSD
RAM
VRAM (optional)
```

and states:

```text
COLD
PREFETCHED
RESIDENT
IN_USE
EVICTABLE
```

The memory manager must enforce a configurable resident memory limit:

```bash
qwen38-zig --ram-limit 8G
qwen38-zig --ram-limit 16G
qwen38-zig --ram-limit 32G
```

The engine must NEVER silently exceed this limit.

---

# 9. Expert cache

Implement a per-layer MoE expert cache.

Target:

```text
512 experts
top-10/token
```

Cache key:

```text
(layer, expert_id)
```

Use LRU initially.

Later make the policy pluggable:

```text
LRU
LFU
frequency
adaptive
```

Example:

```zig
const ExpertKey = struct {
    layer: u32,
    expert: u32,
};
```

Track:

```text
hits
misses
loads
evictions
bytes
load latency
last use
```

---

# 10. Expert loading

Experts should remain in their compact/native representation whenever possible.

Prefer:

```text
SSD
 ↓
FP8
 ↓
RAM cache
 ↓
compute
```

instead of:

```text
SSD FP8
 ↓
huge FP32 expansion
 ↓
RAM
```

Avoid unnecessary copies.

The implementation should make quantization/dequantization boundaries explicit.

---

# 11. PLE / N-gram subsystem

Implement Qwen3.8's hashed N-gram / PLE mechanism as a separate subsystem:

```text
src/qwen38/ple.zig
```

The PLE table is approximately:

```text
~51B parameters
```

and must NOT be loaded into RAM.

Implement:

```text
token history
     ↓
hash
     ↓
PLE address
     ↓
SSD/RAM lookup
```

Support the model's partitioning:

```text
split_ngram_parts = 128
```

The runtime must be able to calculate the exact storage location of a PLE row without scanning the table.

---

# 12. PLE asynchronous prefetch

This is one of the key research components.

PLE addresses are known from token history.

Therefore:

```text
token t
   │
   ▼
calculate PLE addresses
   │
   ▼
enqueue asynchronous reads
   │
   ├───────────────┐
   │               │
   ▼               ▼
GDN layer 0      GDN layer 1
   │               │
   └───────┬───────┘
           ▼
       PLE consumer
```

Implement a bounded I/O queue.

Do NOT create an unlimited thread pool.

---

# 13. Unified I/O scheduler

This is the central experimental feature.

Create:

```zig
IoScheduler
```

It must handle:

```text
PLE requests
Expert requests
QSA requests
```

with priorities.

Example:

```zig
const RequestKind = enum {
    ple,
    expert,
    qsa,
};
```

Priority should depend on:

```text
distance_to_consumer
request_type
cache state
estimated latency
confidence
```

For example:

```text
HIGH:
data needed by next layer

MEDIUM:
next-layer expert prediction

LOW:
speculative future data
```

The scheduler must prevent speculative work from starving mandatory work.

---

# 14. Prefetch prediction

Implement two types of prefetch:

## Deterministic

PLE:

```text
token history
→ exact address
```

## Predictive

Experts:

```text
layer N router
→ likely layer N+1 working set
```

Initially use:

```text
next-layer expert prefetch
```

Then create an abstraction allowing future predictors.

Example:

```zig
const PrefetchPrediction = struct {
    key: ResourceKey,
    confidence: f32,
    expected_use_distance: u32,
};
```

---

# 15. Gated DeltaNet

Implement the actual Qwen3.8 GDN path.

Do not replace it with standard attention.

Implement:

```text
projection
↓
causal convolution
↓
Q/K/V
↓
beta/gate
↓
recurrent state update
↓
output projection
```

The recurrent state must be persistent across tokens.

Expose:

```zig
GdnState
```

and avoid allocating it on every token.

---

# 16. Qwen Sparse Attention

Implement QSA separately.

Pipeline:

```text
Q/K/V
  │
  ▼
lightweight indexer
  │
  ▼
micro-block representation
  │
  ▼
block ranking
  │
  ▼
top relevant blocks
  │
  ▼
full attention over selected tokens
```

Support:

```text
indexer compression ratio = 4
attention budget ≈ 2048 tokens
```

Do NOT initially optimize QSA aggressively.

Correctness first.

---

# 17. QSA memory strategy

QSA is one of the largest context-dependent memory consumers.

Therefore implement configurable context:

```bash
--context 4096
--context 8192
--context 32768
--context 131072
--context 262144
```

Print memory estimates before allocation.

Example:

```text
Qwen3.8 Memory Plan
-------------------
Context:       32768
GDN state:     ...
QSA cache:     ...
Expert cache:  ...
PLE working:   ...
Scratch:       ...
Total resident target: ...
```

If the requested context cannot fit within the configured memory budget, fail gracefully with an explanation.

---

# 18. MoE forward

Implement:

```text
hidden
 ↓
router
 ↓
top-10 experts
 ↓
load missing experts
 ↓
expert GEMM
 ↓
weighted accumulation
```

For prefill, group tokens by expert.

For decode, optimize for single-token inference.

Do not assume prefill and decode have identical optimal execution strategies.

---

# 19. CPU-first implementation

The first complete implementation must work WITHOUT a GPU.

Target:

```text
x86-64
ARM64
```

where practical.

Use portable Zig.

Use SIMD where safe.

Provide:

```text
scalar
SIMD
```

paths.

The engine must remain functional on:

```text
8 GB RAM
16 GB RAM
32 GB RAM
```

hardware.

Performance is secondary to proving the memory architecture.

---

# 20. Optional GPU backend

Only after CPU inference works.

Create:

```text
src/backend/
    cpu.zig
    gpu.zig
```

Do NOT make CUDA a hard dependency.

The architecture should allow:

```text
CPU-only
CPU + GPU
GPU-heavy
```

execution modes.

---

# 21. Hardware profiles

Create predefined profiles:

```text
--profile tiny
--profile laptop
--profile desktop
--profile gpu
```

Example:

```text
tiny:
RAM  = 8 GB
VRAM = 0

laptop:
RAM  = 16 GB
VRAM = 4-8 GB

desktop:
RAM  = 32 GB
VRAM = 8-16 GB

gpu:
RAM  = 64+ GB
VRAM = 24+ GB
```

These are scheduler budgets, not assumptions about actual hardware.

---

# 22. Zero-copy philosophy

Avoid unnecessary:

```text
SSD → RAM → temporary → RAM → GPU
```

copies.

Prefer:

```text
SSD → mapped/cache buffer → compute
```

where possible.

Every major memory copy should be measurable.

Add counters:

```text
bytes_read
bytes_copied
bytes_moved
allocations
deallocations
```

---

# 23. Instrumentation

The runtime must expose detailed statistics.

Example:

```text
=== QWEN38 RUNTIME ===

Tokens:
  generated:       128
  tok/s:            0.73

Memory:
  resident:         9.4 GiB
  peak:             10.1 GiB
  limit:            16.0 GiB

Experts:
  requests:         1280
  hits:              923
  misses:            357
  hit rate:          72.1%

PLE:
  requests:         2048
  bytes:             5.2 MiB
  cache hits:        31%
  late prefetch:     4.2%

I/O:
  reads:             ...
  throughput:        ... GB/s
  queue depth:       ...
  wait/token:        ... ms

QSA:
  context:           8192
  selected tokens:   2048

GDN:
  recurrent layers:  36
```

---

# 24. Benchmark suite

Create:

```text
bench/
    memory.zig
    moe.zig
    ple.zig
    gdn.zig
    qsa.zig
    end_to_end.zig
```

Measure:

```text
tokens/sec
first-token latency
I/O latency
expert cache hit rate
PLE hit rate
prefetch usefulness
RAM peak
VRAM peak
SSD bandwidth
CPU utilization
```

Most importantly measure:

```text
compute_stall_due_to_io
```

because the project's primary objective is hiding I/O latency.

---

# 25. Correctness testing

Implement small deterministic tests for every subsystem.

For example:

```text
GDN:
Zig result vs reference

PLE:
hash/address vs reference

QSA:
selected blocks vs reference

MoE:
router top-k vs reference

RMSNorm:
numerical tolerance

RoPE:
numerical tolerance
```

Create a Python reference harness if necessary.

The Python code is only for validation.

The final runtime remains Zig.

---

# 26. Memory stress testing

The engine must include:

```bash
qwen38-zig stress \
    --ram-limit 8G \
    --context 8192 \
    --tokens 100
```

and verify:

```text
resident <= limit
```

Repeat for:

```text
8G
12G
16G
24G
32G
```

Generate a report:

```text
memory_limit
peak_resident
tok/s
I/O wait
cache hit rate
```

---

# 27. CLI

Minimum CLI:

```bash
qwen38-zig inspect MODEL

qwen38-zig benchmark MODEL

qwen38-zig chat MODEL

qwen38-zig chat MODEL \
    --ram-limit 16G \
    --context 8192

qwen38-zig benchmark MODEL \
    --ram-limit 8G \
    --context 8192 \
    --tokens 128
```

`inspect` must NOT load the model.

It should display:

```text
Architecture
Tensor count
Parameter count
Experts
Active experts
PLE size
Estimated resident memory
Estimated context memory
```

---

# 28. Important engineering constraint

Do NOT begin by implementing every feature.

Work incrementally:

```text
Phase 1
  model metadata
  tensor loader
  memory manager

Phase 2
  basic tensor operations
  embeddings
  RMSNorm
  LM head

Phase 3
  GDN

Phase 4
  MoE

Phase 5
  PLE

Phase 6
  QSA

Phase 7
  unified scheduler

Phase 8
  optimization

Phase 9
  optional GPU
```

At the end of every phase:

```text
build
test
benchmark
document
```

---

# 29. Do not fake support

This is extremely important.

Never implement:

```text
TODO
return zero
placeholder tensor
fake expert
fake attention
```

and claim Qwen3.8 support.

If a subsystem is incomplete, explicitly report:

```text
NOT IMPLEMENTED
```

The runtime should fail clearly rather than silently producing incorrect results.

---

# 30. Reference implementation

Use the following projects as architectural references:

- Official Qwen3.8-Flash-Next implementation
- Hugging Face Qwen3.8 configuration/model definition
- Colibrì v1.10.0 Qwen3.8 implementation

Do not copy proprietary/non-compatible code.

The goal is an independent Zig implementation.

---

# 31. Research objective

The project should answer this question experimentally:

> How small can the resident working set of Qwen3.8-Flash-Next become while still achieving useful inference throughput on ordinary consumer hardware?

The key variables are:

```text
RAM limit
SSD speed
expert cache capacity
PLE cache capacity
I/O queue depth
prefetch distance
prefetch confidence
context length
CPU SIMD
GPU availability
```

Eventually produce a graph:

```text
RAM
│
│            ●
│        ●
│    ●
│ ●
└───────────────────
       tok/s
```

and another:

```text
Prefetch hit rate
│
│          ●
│       ●
│    ●
│ ●
└───────────────────
    I/O wait/token
```

---

# 32. Final architectural goal

The final runtime should conceptually behave like:

```text
                    TOKEN
                      │
                      ▼
              ┌──────────────┐
              │   Scheduler  │
              └──────┬───────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
      PLE            GDN           MoE
       │              │             │
       │              │        router
       │              │             │
       │              │        expert IDs
       │              │             │
       └──────────────┼─────────────┘
                      │
                I/O Scheduler
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
         SSD         RAM         VRAM
          │           │           │
          └───────────┼───────────┘
                      ▼
                   COMPUTE
                      │
                      ▼
                    TOKEN
```

The defining feature of this project is therefore NOT raw FLOPS.

It is:

> **intelligent movement of model state between storage tiers.**

---

# 33. Development rules for the coding agent

1. Inspect before modifying.
2. Keep commits small and logical.
3. Never rewrite large portions unnecessarily.
4. Preserve working functionality.
5. Add tests with every new subsystem.
6. Benchmark before and after performance changes.
7. Document every architectural decision.
8. Do not introduce dependencies unless absolutely necessary.
9. Prefer Zig standard library functionality.
10. Keep platform-specific code isolated.
11. Never silently fall back to incorrect computation.
12. Never claim benchmark results that were not actually measured.
13. If the official Qwen3.8 architecture differs from assumptions in this prompt, stop and document the discrepancy before implementing it.
14. If an optimization changes numerical results, measure and document the error.

---

# 34. First task

Do NOT immediately start writing the complete engine.

First:

1. Clone/create the project.
2. Inspect the official Qwen3.8 architecture.
3. Inspect Colibrì v1.10.0's `qwen38` implementation.
4. Produce:

```text
docs/IMPLEMENTATION_PLAN.md
docs/TENSOR_MAP.md
docs/MEMORY_BUDGET.md
```

5. Create the initial Zig project.
6. Implement only:
   - configuration parser
   - checkpoint manifest parser
   - tensor metadata loader
   - memory budget manager
   - `inspect` CLI command

7. Run tests.
8. Show the resulting architecture and memory budget.
9. STOP and wait for approval before implementing the inference kernels.

Do not skip this planning stage.

The project should evolve from a **small verified core** into the full inference engine rather than attempting a monolithic implementation.