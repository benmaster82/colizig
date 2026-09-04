# Reference harness (`tools/reference/`)

Cross-validates the Zig engine's full forward against an **independent NumPy
port of the same specification** (brief §25).  Requires `numpy` only - no
`torch`, no `transformers`.

## What it is / isn't

- **Is**: a second implementation of the Qwen4-Exp forward (embedding, gated
  residual, Gated DeltaNet, Qwen Sparse Attention, MoE, PLE, final mixer, LM
  head), written densely and loopily in `qwen38_ref.py` from the same
  colibri-derived formulas the Zig code ports.  Both read the *same* fixture
  weights; if they agree, both are faithful to the spec, and a structural bug in
  either shows up as a logit mismatch (the way colibri's vision-tower bugs
  showed up as ~1e-2 gaps against an oracle).
- **Isn't**: validation against the released `Qwen/Qwen3.8-Flash-Next-FP8`
  weights - that is the **colibri cross-check** below.

## Files

| file | role |
|---|---|
| `qwen38_ref.py` | `Model(dir).forward(ids) -> logits`; CLI: `qwen38_ref.py <dir> "1,2,3"` |
| `build_oracle.py` | reads `test/fixtures/tiny/`, writes `oracle.json` (3 token sets) |
| `requirements.txt` | `numpy` |

## Use

```sh
zig build gen-fixture                          # deterministic tiny checkpoint
pip install -r tools/reference/requirements.txt
zig build oracle                               # or: python tools/reference/build_oracle.py
zig build test                                 # the oracle test now runs
```

`test/fixtures/tiny/oracle.json` is git-ignored (regenerated). The Zig test
`model.zig : "matches the NumPy reference oracle"` **skips** when it is absent,
There are now two reference forwards - `qwen38_ref.py` (Qwen4-Exp) and
`qwen3moe_ref.py` (Qwen3-MoE, sharing the E4M3 / block-FP8 / RoPE helpers) - and
`build_oracle.py` writes one `oracle.json` per fixture (`test/fixtures/tiny/` and
`test/fixtures/tiny-qwen3/`).

So `zig build test` never requires Python. It compares `forward` logits to the
oracle (max abs < 2e-2, both f32) and requires the argmax to match exactly.

## Keeping it in sync

If `tools/gen_tiny_fixture.zig` changes the fixture weights, re-run
`zig build oracle`. A stale oracle makes the test fail, which is the intended
signal.

## Latest result (NumPy oracle, tiny fixture)

The two implementations agree to ~1e-3 on the final logits across all three token
sets; argmax identical. Example (`--tokens 1,2,3,4,5`): Zig `55:4.103 46:2.881
38:2.105 8:1.792 …`, NumPy `[4.1034, …, 2.881@46, 2.105@38, 1.792@8, …]`.

## colibri cross-check - real checkpoint (2026-09)

An independent engine ([JustVugg/colibri](https://github.com/JustVugg/colibri),
C, no shared code - brief §30) on the **released** `Qwen/Qwen3.8-Flash-Next-FP8`.

Build (single translation unit, `zig cc` as the compiler, OpenMP dropped so it
is single-threaded but bit-exact):

```
zig cc -O2 -D_FILE_OFFSET_BITS=64 -w c/qwen38.c -o qwen38.exe -lm
```

Run both on the raw prompt `"The capital of France is"` (both tokenizers encode
it to `760 6511 314 9338 369`), greedy, 12 new tokens:

```
colibri:   SNAP=<dir> N_NEW=12 ./qwen38.exe 16 8 prompt.txt      # prompt.txt = raw text
colizig: forward <dir> --tokens 760,6511,314,9338,369 --steps 12
```

Result - **token-for-token identical**:

```
11751 13 561 6511 314 9564 369 19241 13 561 6511 314
= " Paris. The capital of Germany is Berlin. The capital of"
```

Side cross-checks: colibri reports "resident matrices 10.88 GiB" vs our
`private resident` 10.90 GiB; expert-routing counts (distinct experts, selections
per token) line up. colibri is itself validated against `transformers`, so this
closes the "same numbers as the real model" gap.

(colibri's repo-root `ref.json` targets a different model's tokenizer and lacks
`schema_version` - use text-mode with a non-`.json` prompt file instead.)
