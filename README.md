# colizig

An **experimental** Zig inference engine for Qwen mixture-of-experts models,
streaming experts from disk so a model that does not fit RAM still runs on a
laptop. Two families supported:

- **Qwen3.8-Flash-Next** ([`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8),
  internal codename *Qwen4-Exp*) — GDN + QSA + hashed-n-gram PLE + 512-expert MoE.
- **Qwen3-MoE** ([`Qwen/Qwen3-30B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8)
  and siblings) — plain GQA + QK-norm + 128-expert MoE. `~3 tok/s` CPU decode.

The research question is not "how fast" but **"how small can the resident working
set be while inference stays useful on ordinary consumer hardware?"** The model
is treated as a *computation graph whose working set changes over time*, streamed
across SSD → RAM → (optional) VRAM.

This is not a llama.cpp / vLLM competitor. It is a testbed for **intelligent
movement of model state between storage tiers**.

The C engine [JustVugg/colibri](https://github.com/JustVugg/colibri)
(`c/qwen38.c`) already runs this checkpoint by streaming experts from disk and is
used here as an **architectural reference only** — no code is copied.

## Status — Phases 1–9 + real-checkpoint bring-up

Runs the **released 176B checkpoint** end-to-end on a 32 GB / single-SSD laptop:
config + all 152,089 tensors validated, ~1 tok/s warm decode at `--expert-cap 512`
with ~7 GB private RAM (see [`docs/BENCHMARK.md`](docs/BENCHMARK.md)). Greedy decode
is **token-for-token identical to the independent C engine
[colibri](https://github.com/JustVugg/colibri)** on the real weights
(`docs/REFERENCE.md`).

| area | state |
|---|---|
| config parser + strict validation | ✅ |
| safetensors header reader (no bodies) | ✅ |
| checkpoint manifest + tensor classification | ✅ |
| memory-budget planner (`MemoryManager`, planning half) | ✅ |
| `inspect` CLI | ✅ |
| tiny synthetic fixture generator (deterministic bodies) | ✅ |
| `ops/`: matmul (SIMD), rmsnorm, rope, softmax, activations | ✅ |
| weight materialization: mmap shards, `View` decode (F32/F16/BF16) | ✅ |
| embedding lookup + LM-head projection + `selftest` CLI | ✅ |
| Gated DeltaNet (`src/qwen38/gdn.zig`): conv1d + gated-delta recurrence, persistent `GdnState` | ✅ |
| MoE (`src/qwen38/moe.zig`): router → top-k experts + shared expert, block-FP8 SwiGLU, bounded LRU expert cache | ✅ |
| block-FP8 (E4M3) decode + `matmulFp8` (`src/ops/fp8.zig`) | ✅ |
| PLE hashed n-gram (`src/qwen38/ple.zig`): deterministic address → streamed table lookup, conv ring, prefetch | ✅ |
| Qwen Sparse Attention (`src/qwen38/qsa.zig`): indexer block scoring, top-k + causal tail, GQA full attention, output gate, context KV cache | ✅ |
| gated residual (`src/qwen38/residual.zig`) + **end-to-end forward** (`src/qwen38/model.zig`): embed → 48 layers → final mixer → LM head; `forward` CLI on raw token ids + greedy decode | ✅ |
| I/O scheduler (`src/runtime/io.zig`, bounded priority queue) + `LastTokenPredictor` + concurrent PLE reads + expert-prefetch telemetry | ✅ |
| byte-level BPE tokenizer (`src/qwen38/tokenizer.zig`) + ChatML template + `chat` CLI: one-shot **or** interactive multi-turn REPL with live token streaming + per-reply tok/s | ✅ |
| NumPy reference forward (`tools/reference/`) + oracle test — full forward cross-validated against an independent port | ✅ |
| `benchmark` (runtime telemetry) + `stress` (RAM-budget sweep) + per-phase `Timers` + tracked-bytes `Meter` | ✅ |
| threaded kernels (`runtime/parallel.zig`): matmul + GDN recurrence + MoE per-expert decode fan out over `std.Io.Group`, work-thresholded, bit-identical (MoE decode scales ~6.5× warm) | ✅ |
| perf: FP8/BF16 matmul SIMD-vectorized, O(1) tensor index, experts borrow the shard mmap (no copy), prefill token→expert grouping — ~10× vs first real run | ✅ |
| **real-weight validation**: greedy decode token-for-token identical to colibri (independent C engine) on `Qwen/Qwen3.8-Flash-Next-FP8` | ✅ |
| sampling (`--temperature` / `--top-k` / `--top-p` / `--seed`); learned expert priors (`.colizig_usage`); dual-SSD `--mirror` | ✅ |
| **CUDA backend** (`--cuda` / `--vram`, `src/backend/`): block-FP8 matmul on the GPU via a runtime-loaded `colizig_cuda.dll` + a VRAM weight-cache; bit-identical, optional, CPU fallback (Phase 10a/10b). *Measured ~2× slower than CPU on a 4 GB T1000 — needs 8 GB+ VRAM to win.* | ✅ |
| **Qwen3-MoE** (`model_type: qwen3_moe`, `src/qwen3moe/`): plain GQA + per-head QK-norm + full RoPE + 128-expert MoE (no PLE/GDN/shared expert); `forward` + `chat`. Runs `Qwen3-30B-A3B-FP8` end-to-end at ~3 tok/s CPU. | ✅ |
| 10c async/batched CUDA · QSA per-head threading · MTP speculative decode · qwen3_moe fixture+oracle | ❌ remaining |

`selftest` runs kernel + plumbing checks. Nothing is faked: an incomplete
subsystem returns an error rather than a wrong number (brief §29).

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the roadmap,
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the verified model facts, and
[`docs/DESIGN_BRIEF.md`](docs/DESIGN_BRIEF.md) for the original project brief the
"§N" references throughout the docs point at (it uses the project's earlier
working name, *qwen38-zig*).

## Build & run

Requires **Zig 0.16.0** (pinned in `build.zig.zon`).

```sh
zig build                 # build zig-out/bin/colizig  (ReleaseFast by default)
zig build test            # unit tests (ReleaseFast; add -Doptimize=Debug for UB checks)
zig build gen-fixture     # (re)write test/fixtures/tiny/
zig build cuda            # optional CUDA backend → zig-out/bin/colizig_cuda.dll
                          # (needs nvcc + MSVC; see build_cuda.ps1 and docs/GPU.md)

# inspect a checkpoint directory — reads config.json + safetensors headers only,
# never a tensor body:
zig build run -- inspect <MODEL_DIR> [--ram-limit 16G] [--context 8192] [--profile laptop]

# against the bundled tiny fixture:
zig build run -- inspect test/fixtures/tiny --profile tiny --context 4096

# bring-up checks (ops kernels; + subsystems if a model dir is given):
zig build run -- selftest test/fixtures/tiny

# end-to-end forward on raw token ids:
zig build run -- forward test/fixtures/tiny --tokens 1,2,3 --steps 5

# chat — one-shot with --prompt, or the "ColiZig" interactive REPL without it:
# colibri-styled header, a greyed "thinking" box + the answer in Qwen's colour,
# each token streamed; per-reply "N tok · prefill … · X tok/s".
# /think toggles reasoning · /reset clears the conversation · /exit quits.
# greedy by default; --temperature <f> [--top-k <n>] [--top-p <f>] [--seed <n>] to sample.
zig build run -- chat <MODEL_DIR> --ram-limit 24G --expert-cap 512
zig build run -- chat test/fixtures/tiny --prompt "hello world" --steps 12
# add --cuda to run the block-FP8 matmul on the GPU (needs colizig_cuda.dll):
zig build run -- chat <MODEL_DIR> --expert-cap 512 --cuda

# Qwen3-MoE (Qwen3-30B-A3B-FP8) — same CLI, auto-detected from config.json:
zig build run -- inspect C:\Models\Qwen3-30B-A3B-FP8
zig build run -- chat    C:\Models\Qwen3-30B-A3B-FP8 --expert-cap 128

# runtime telemetry / RAM-budget sweep:
zig build run -- benchmark test/fixtures/tiny --prompt-len 8 --steps 20
zig build run -- stress test/fixtures/tiny --context 8192 --steps 10
```

`inspect` prints the architecture, a tensor inventory, parameter counts, and a
memory-budget plan; it exits non-zero (with an explanation) if the requested
context cannot fit the RAM budget.

The `bench_*.ps1` scripts at the repo root are optional Windows A/B harnesses
(point `-Model` at a checkpoint, or set `$env:QWEN38_MODEL`); the colibri half of
`bench_ab*.ps1` is skipped if no colibri build is present.
`tools/microbench.zig` (`zig run -O ReleaseFast tools/microbench.zig`) times the
FP8 / BF16 inner kernels in isolation.

## License

MIT — see [`LICENSE`](LICENSE). This is an independent research project, not
affiliated with or endorsed by Alibaba / the Qwen team. "Qwen" and model names
are used only to identify the target architecture. colibri is referenced under
its own license as an architectural reference; no colibri code is included.
