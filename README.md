<p align="center">
  <img src="assets/logo.svg" alt="colizig" width="280">
</p>

<h1 align="center">colizig</h1>
<p align="center"><b>An experimental Zig inference engine for Qwen mixture-of-experts models —
built to run checkpoints bigger than your RAM, on hardware you already own.</b></p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#cli-reference">CLI reference</a> ·
  <a href="#architecture--how-it-works">How it works</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#contributing--we-want-your-hardware">Contributing</a>
</p>

---

## What this project is trying to prove

Most engines answer "how many tokens per second". colizig asks a different
question first: **how small can the *resident* working set be while inference
stays useful, on an ordinary laptop?** A large MoE checkpoint is treated as a
computation graph whose working set changes *every token* — most of a 512-expert
model is cold at any given instant — and the engine streams that state across
SSD → RAM → (optionally) VRAM instead of insisting the whole thing fits in
memory up front.

This is **not** a llama.cpp / vLLM competitor and it doesn't try to be the
fastest engine around. It's a testbed for intelligent movement of model state
between storage tiers, written from scratch in Zig, with nothing faked: a
subsystem that isn't finished returns an error instead of a plausible-looking
wrong number.

Two model families are supported today, both natively **block-FP8**:

| | Qwen3.8-Flash-Next <sub>(`Qwen4-Exp`)</sub> | Qwen3-MoE <sub>(e.g. `Qwen3-30B-A3B`)</sub> |
|---|---|---|
| Size | ~176B, 512 experts, top-10 + 1 shared | ~30B, 128 experts, top-8, no shared |
| Attention | Gated DeltaNet (36 layers) + Qwen Sparse Attention (12 layers) | plain GQA (32Q/4KV) + per-head QK-RMSNorm |
| Extras | hashed n-gram PLE, hyper-connection residual, MTP head | — (a much plainer transformer) |
| On this laptop | ~1 tok/s warm decode, ~7 GB private RAM | ~3 tok/s warm decode |
| HF checkpoint | `Qwen/Qwen3.8-Flash-Next-FP8` | `Qwen/Qwen3-30B-A3B-FP8` |

Both run through the **same** CLI, the **same** chat interface, and the
**same** OpenAI-compatible HTTP server — the engine dispatches on
`config.json`'s `model_type` at load time, so nothing about how you use it
changes with the model.

The C engine [JustVugg/colibri](https://github.com/JustVugg/colibri) already
proved this checkpoint can be streamed from disk; it's used here purely as an
**architectural reference** (no code copied — see [`docs/REFERENCE.md`](docs/REFERENCE.md))
and as an independent correctness check: colizig's greedy decode is
token-for-token identical to it on real weights.

## Features

- **Two model families, one engine** — Qwen4-Exp (Qwen3.8-Flash-Next) and
  Qwen3-MoE (Qwen3-30B-A3B and siblings), auto-detected from `config.json`.
  All model-generic code (`chat`, `serve`, the MoE/FP8 kernels) is written once
  and made `comptime`-generic over whichever model module matches.
- **Memory-tiered streaming, not a fixed footprint.** A planner
  (`runtime/budget.zig`) works out what must stay resident (attention, norms,
  router) vs. what can be evicted (MoE experts) and picks the largest bounded
  LRU expert cache that fits your `--ram-limit`; it refuses to run rather than
  silently blow the budget.
- **Block-FP8 (E4M3) kernels** for weights and, optionally, GPU matmul — SIMD
  dequant (`@Vector`, no LUT/gather), fused decode+FMA dot products, thread-fanned
  across `std.Io.Group` workers.
- **Learned expert priors.** Routing history is persisted to
  `<model_dir>/.colizig_usage` and used to pre-warm the expert cache with each
  layer's historically hot experts on the next run (`--no-usage` to disable).
- **Optional CUDA backend** (`--cuda`, `src/backend/`) — routes the block-FP8
  expert matmul to a runtime-loaded `colizig_cuda.dll` with an LRU VRAM weight
  cache (`--vram`), CPU fallback if the DLL isn't present. Bit-identical output,
  never a build dependency of the base engine. See [Benchmarks](#benchmarks) —
  this is exactly the kind of result we'd love more data points on.
- **Dual-SSD expert mirror** (`--mirror <dir>`) — split routed-expert reads
  across two copies of the checkpoint on different drives.
- **Sampling** — greedy by default; `--temperature` / `--top-k` / `--top-p` /
  `--seed` for real sampling.
- **`chat`** — a styled interactive REPL (ColiZig look: a pixel-art banner, a
  greyed "thinking" box for `<think>` reasoning, streamed tokens, a per-reply
  `tok/s` line) or one-shot with `--prompt`. `/think`, `/reset`, `/exit`.
- **`serve`** — an OpenAI-compatible HTTP API (`POST /v1/chat/completions`,
  streaming SSE or not, `GET /v1/models`) built directly on Zig 0.16's
  `std.http.Server` / `std.Io.net`, model kept warm across requests.
- **`inspect`** — architecture + full memory-budget plan from `config.json` and
  safetensors headers alone, **without loading a single tensor body**.
- **`benchmark` / `stress`** — runtime telemetry (TTFT, tok/s, expert hit rate,
  per-subsystem phase timing) and a RAM-budget sweep.
- **Two independent correctness nets**: a from-scratch NumPy reference
  implementation cross-validated against the Zig forward pass (logits agree to
  <1e-2/<2e-2), and token-for-token identity against colibri on real weights.
- **Threaded kernels** (`runtime/parallel.zig`) — matmul, MoE per-expert
  evaluation and the GDN recurrence fan out over worker threads, gated by a
  work-size threshold so small models stay serial; verified bit-identical to
  the single-threaded path.

## Status

| area | state |
|---|---|
| config / manifest / tensor metadata / memory-budget planner | ✅ |
| `inspect`, `selftest`, `forward`, `tokenize` CLIs | ✅ |
| block-FP8 (E4M3) kernels, SIMD dequant, AVX2/FMA matmul | ✅ |
| Qwen4-Exp: GDN, MoE (+shared expert), hashed n-gram PLE, QSA, gated residual | ✅ |
| Qwen3-MoE: GQA + per-head QK-norm + MoE (no shared/PLE/GDN) | ✅ |
| threaded kernels, I/O scheduler, learned expert priors, dual-SSD mirror | ✅ |
| tokenizer + ChatML + unified `chat` REPL (both model families) | ✅ |
| `serve` (OpenAI-compatible HTTP, both model families) | ✅ |
| sampling (temperature / top-k / top-p / seed) | ✅ |
| CUDA backend (`--cuda`, `--vram`) — correct, not yet a win on ≤4 GB cards | ✅ |
| NumPy reference oracle for both model families | ✅ |
| real-weight validation vs. an independent C engine (colibri) | ✅ |
| async/batched CUDA (10c), QSA per-head threading, MTP speculative decode, int4 | ❌ not started |

`selftest` runs kernel + plumbing checks against a bundled tiny synthetic
fixture — nothing is faked, an incomplete subsystem returns an error rather
than a wrong number.

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the phase-by-phase
history, [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the verified model facts, and
[`docs/DESIGN_BRIEF.md`](docs/DESIGN_BRIEF.md) for the original project brief.

## Quick start

### Prerequisites

- **Zig 0.16.0** — pinned in `build.zig.zon`. Get it from
  [ziglang.org/download](https://ziglang.org/download/) (or `winget install zig.zig`
  on Windows). Any other version is not guaranteed to build.
- Optional, only for the GPU backend: an **NVIDIA GPU + CUDA toolkit** (`nvcc`)
  and, on Windows, MSVC build tools. Everything else works with the CPU only.
- Optional, only for regenerating the NumPy oracle: **Python 3 + NumPy**.

### Build

```sh
git clone https://github.com/benmaster82/colizig.git
cd colizig

zig build                 # zig-out/bin/colizig, ReleaseFast by default
zig build test             # unit + regression tests (ReleaseFast)
zig build test -Doptimize=Debug   # same tests, with UB/bounds checks
```

```sh
zig build gen-fixture      # (re)writes the tiny synthetic fixtures under test/fixtures/
zig build oracle           # regenerates the NumPy reference oracle.json (needs python+numpy)
zig build cuda              # optional: builds colizig_cuda.dll (needs nvcc; see docs/GPU.md)
```

`zig build test` never needs Python or a GPU — the oracle test is skipped
gracefully if `oracle.json` isn't present, and the CUDA backend is loaded at
runtime only, never a build dependency.

### Try it without downloading anything

```sh
zig build run -- inspect  test/fixtures/tiny --profile tiny
zig build run -- selftest test/fixtures/tiny
zig build run -- chat     test/fixtures/tiny --prompt "hello world" --steps 12
```

### Get a real checkpoint

Either model family works. The smaller one (Qwen3-30B-A3B, ~30 GB) is the
faster way to try the engine for real:

```sh
hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir C:\Models\Qwen3-30B-A3B-FP8
```

> If the download hangs at 0 B/s behind a corporate proxy, it's almost always
> the `xet` transport failing TLS — retry with `HF_HUB_DISABLE_XET=1`.

Then:

```sh
# metadata + a memory-budget plan, no weights touched:
zig build run -- inspect C:\Models\Qwen3-30B-A3B-FP8

# interactive chat:
zig build run -- chat C:\Models\Qwen3-30B-A3B-FP8 --expert-cap 128

# or serve it over HTTP:
zig build run -- serve C:\Models\Qwen3-30B-A3B-FP8 --port 8080 --expert-cap 128
curl http://127.0.0.1:8080/v1/chat/completions -H "content-type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi!\"}]}"
```

The 176B `Qwen/Qwen3.8-Flash-Next-FP8` (~173 GiB on disk) works exactly the
same way — swap the model directory and, on a memory-constrained box, raise
`--ram-limit` / lower `--expert-cap` to taste; `inspect` will tell you up front
whether your budget fits.

## CLI reference

```
colizig inspect   <MODEL_DIR> [options]                                    architecture + memory plan, no weights loaded
colizig selftest  [MODEL_DIR]                                              kernel + subsystem bring-up checks
colizig forward   <MODEL_DIR> --tokens <csv> [--steps N] [--expert-cap K]  end-to-end forward on raw token ids
colizig chat      <MODEL_DIR> [--prompt "..."] [--system "..."] [--steps N] interactive REPL, or one-shot with --prompt
colizig serve     <MODEL_DIR> [--port 8080] [--host 127.0.0.1]             OpenAI-compatible HTTP API
colizig tokenize  <MODEL_DIR> --prompt "..."                               encode/decode with the checkpoint's tokenizer.json
colizig benchmark <MODEL_DIR> [--prompt-len N] [--steps N] [--expert-cap K] runtime telemetry (TTFT, tok/s, hit rate, phases)
colizig stress    <MODEL_DIR> [--context N] [--steps N] [--ram-limit G]    sweep RAM budgets, report fit/no-fit
```

| flag | applies to | meaning |
|---|---|---|
| `--ram-limit <size>` | inspect/chat/forward/benchmark/stress | resident memory budget (`8G`, `16GiB`, …); default from `--profile` |
| `--context <n>` | inspect/stress | context length in tokens (default 8192) |
| `--profile <name>` | inspect | `tiny \| laptop \| desktop \| gpu` — a starting RAM/VRAM budget |
| `--expert-cap <n>` | forward/chat/benchmark | override the planned per-layer expert-cache size |
| `--threads <n>` | all compute commands | worker fan-out; `0` = auto (CPU count), `1` = single-threaded |
| `--no-usage` | chat/forward | don't read/write `.colizig_usage` learned expert priors |
| `--mirror <dir>` | chat/forward/benchmark | a 2nd checkpoint copy on another drive; routed-expert reads split across both |
| `--temperature <f>` / `--top-k <n>` / `--top-p <f>` / `--seed <n>` | chat/forward | sampling; `--temperature 0` (default) = deterministic greedy |
| `--cuda` [`--vram <size>`] [`--cuda-verify`] | chat/forward/benchmark | run the block-FP8 expert matmul on the GPU; `--vram` caps the resident weight cache; `--cuda-verify` cross-checks every GPU matmul against the CPU |
| `--port <n>` / `--host <addr>` | serve | listen address (default `127.0.0.1:8080`) |
| `-h`, `--help` | — | usage |

## Architecture & how it works

- **Layered by tier, not by layer.** Weights are classified on load
  (`model/manifest.zig`) into what's always resident (attention/router/norms,
  a few GiB) vs. what's demand-streamed (MoE experts, the vast majority of
  parameters) — see [`docs/MEMORY_MODEL.md`](docs/MEMORY_MODEL.md) /
  [`docs/MEMORY_BUDGET.md`](docs/MEMORY_BUDGET.md).
- **Experts are mmap-borrowed, not copied.** `moe.zig`'s `Fp8Matrix` reads
  E4M3 bytes straight out of the shard mmap through a bounded LRU cache; the
  OS page cache does the actual eviction, so a second run over the same
  checkpoint gets faster for free.
- **Nothing is hard-coded to one model's dimensions** — every kernel is driven
  by the parsed `config.json`; the two model families share the FP8 matmul,
  the expert cache, the tokenizer, the sampler, the CLI, `chat` and `serve`,
  and differ only in `src/qwen38/` vs `src/qwen3moe/` (attention style, whether
  there's a shared expert, PLE, GDN, hyper-connections).
- **Everything is cross-checked twice**: an independent NumPy port
  (`tools/reference/`) agrees with the Zig forward pass logit-for-logit, and
  greedy decode matches colibri (an independent C engine) token-for-token on
  real weights.

Full docs: [`ARCHITECTURE`](docs/ARCHITECTURE.md) ·
[`MEMORY_MODEL`](docs/MEMORY_MODEL.md) · [`MEMORY_BUDGET`](docs/MEMORY_BUDGET.md) ·
[`MOE`](docs/MOE.md) · [`GDN`](docs/GDN.md) · [`QSA`](docs/QSA.md) ·
[`PLE`](docs/PLE.md) · [`OPS`](docs/OPS.md) · [`THREADING`](docs/THREADING.md) ·
[`IO_SCHEDULER`](docs/IO_SCHEDULER.md) · [`TOKENIZER`](docs/TOKENIZER.md) ·
[`FORWARD`](docs/FORWARD.md) · [`GPU`](docs/GPU.md) · [`REFERENCE`](docs/REFERENCE.md) ·
[`TENSOR_MAP`](docs/TENSOR_MAP.md) / [`QWEN38_TENSORS`](docs/QWEN38_TENSORS.md).

## Benchmarks

All current numbers come from **one machine**: an i7-10750H laptop (6c/12t),
32 GB RAM, dual NVMe, an NVIDIA Quadro T1000 Max-Q (4 GB VRAM). The full,
warts-and-all history — including two things that turned out *not* to help
(async expert prewarm, dual-SSD mirroring on this hardware) — is in
[`docs/BENCHMARK.md`](docs/BENCHMARK.md). Headline numbers, warm cache, greedy:

| model | expert cap | decode | TTFT (short prompt) |
|---|---|---|---|
| Qwen3.8-Flash-Next (176B, 512 experts) | 512 | ~0.7–1.0 tok/s | ~5–8 s |
| Qwen3-30B-A3B (30B, 128 experts) | 128 | ~3.2 tok/s | a few seconds |

CPU decode on this box is **memory-bandwidth-bound**, not FLOP-bound — the
4 GB GPU is currently ~2× *slower* than the CPU because it re-uploads each
routed expert over PCIe every call and the VRAM cache is too small to hold
enough of the working set (see [`docs/GPU.md`](docs/GPU.md) and the Phase 10
sections of `BENCHMARK.md`). That's very likely different on an 8 GB+ card,
more RAM, more/fewer cores, or a faster SSD — **we don't know yet**, and this
is exactly the kind of gap the project needs filled in.

### Reproduce it on your machine

```sh
zig build run -- benchmark <MODEL_DIR> --prompt-len 16 --steps 20 --expert-cap <K>
zig build run -- stress    <MODEL_DIR> --context 8192 --ram-limit 16G
```

`benchmark` prints TTFT, decode tok/s, expert cache hit rate, and a
per-subsystem phase-timing table; `stress` sweeps RAM budgets and reports
fit/no-fit + tracked peak per budget.

## Contributing — we want your hardware

This project genuinely wants outside data and eyes, not just code:

- **Benchmarks on hardware we don't have.** More RAM, more cores, a bigger
  GPU (8 GB+), Apple Silicon, a slower/faster SSD, Linux — anything. Run
  `benchmark`/`stress` as above and open an issue or PR with your numbers,
  CPU/GPU model, OS, and which checkpoint/`--expert-cap`. Negative results
  ("the GPU backend is still slower here") are just as useful as wins — that's
  how the current GPU numbers got documented honestly instead of cherry-picked.
- **Feedback on the design itself** — the memory-tiering approach, the CLI, the
  chat/serve UX — is welcome even without a code change attached.
- **Bug reports and PRs** for anything in [Status](#status) marked ❌, or any
  behavior that doesn't match what `docs/` claims.

Before sending a PR: `zig build test` (and `-Doptimize=Debug` too) should pass;
keep everything driven by `config.json` (no hard-coded model dimensions); if a
subsystem isn't finished, make it return an error rather than a plausible-looking
wrong number, consistent with the rest of the codebase.

## License

MIT — see [`LICENSE`](LICENSE). This is an independent research project, not
affiliated with or endorsed by Alibaba / the Qwen team. "Qwen" and model names
are used only to identify the target architecture. colibri is referenced under
its own license as an architectural reference; no colibri code is included.
