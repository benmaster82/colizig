# End-to-end forward (`src/qwen38/model.zig`, `src/qwen38/residual.zig`)

Ports colibri's `step` + `q38_gr_read` / `q38_gr_apply`.

## Gated residual — `residual.zig`

The residual state `hyper` is `[S, hc_width]` = `[S, hc_count·hidden]` (4 branches).

**`Gated`** (`hyper_connection_mixer` / `layers.i.{attn,mlp}_hyper_connection`):
`norm[hc_width]` F32, `down[hc_rank, hc_width]` + `up[hc_width, hc_rank]` BF16,
and (per-layer only) `inject[hc_count, hc_width]` BF16.

**`read(g, hyper, S) → mixed[S, hidden] (+ inject[S, hc_count])`**
1. per branch `b`: `norm_b = rms0(hyper_b, g.norm_b)`
2. `low = silu((norm · downᵀ) / hc_count)` → `mix = low · upᵀ`
3. `mixed[d] = (Σ_b σ(mix_b[d]) · norm_b[d]) / hc_count`
4. if injecting: `inject = 2·σ((norm · injectᵀ) / hc_count)`

**`apply(hyper, block[S, hidden], inject[S, hc_count])`**:
`hyper_b[d] += inject_b · block[d]` for every branch.

The final mixer is a `read` with no inject; its `mixed` feeds the LM head.

## `Model`

Loads, for all `num_hidden_layers`: `attn_gr`, `mlp_gr`, `moe_layers[i]`, and
either `gdn_layers[i]` or `qsa_layers[i]`; plus `final_gr`, `ple_layer`,
`ple_table`. This is the resident dense set (~9.2 GiB on the real checkpoint).
Embedding and LM head stay streamed via `Weights` (`embed` row lookup, `lmHead`
row-by-row projection).

## `State` (per sequence)

- `gdn[i]` — `GdnState` for each DeltaNet layer (persistent recurrent + conv ring)
- `qsa[i]` — `Cache` for each attention layer, sized to `max_context`
- `experts[i]` — bounded `ExpertCache` per layer (content, not sequence state:
  `reset()` leaves it warm)
- `ple_state` — conv ring + n-gram history
- `pos` — tokens processed so far

## `forward(model, state, sc, ids, logits)`

```
build hyper: embed(ids[s]) → hyper[s, branch 0]; replicate to branches 1..hc_count
for layer i in 0..num_hidden_layers:
    if i == ple_layer:  hyper += ple.forward(ids, hyper)
    mixed,inject = residual.read(attn_gr[i], hyper)
    block = QSA.forward(...) if is_attn[i] else GDN.forward(...)
    residual.apply(hyper, block, inject)
    mixed,inject = residual.read(mlp_gr[i], hyper)
    block = MoE.forward(...)          # loads experts on miss (inline, Phase 7a)
    residual.apply(hyper, block, inject)
mixed = residual.read(final_gr, hyper)          # no inject
logits = lmHead(mixed[last token])
state.pos += S
```

`forward` validates every `ids[s] ∈ [0, vocab)` and every QSA cache has room
(`pos_base + S ≤ cap`) before touching state.

`generateGreedy(prompt, out[])` runs the prompt then feeds back `argmax` up to
`out.len` times, stopping at `eos_token_id`.

## `forward` CLI

```
qwen38-zig forward <MODEL_DIR> --tokens <id,id,...> [--steps N]
                   [--context K] [--ram-limit G] [--profile P]
```

No tokenizer yet — raw integer token ids in, ids out. Sizes the context and
expert-cache capacity from the Phase 1 memory plan; refuses if the plan doesn't
fit. Prints the top-8 logits of the last prompt token and, with `--steps`, the
greedy continuation.

## Guarantees / tests

- Output logits finite.
- **One-shot prefill == token-by-token decode** (`≤ 2e-3` rel across all layers).
- Out-of-vocabulary token ids rejected before any state mutation.
- Greedy decode stays in vocabulary and stops at EOS.
- End-to-end numerical validation vs upstream `Qwen4ExpForCausalLM` still needs
  the Python reference harness (brief §25) — the subsystems are individually
  faithful ports but the composed result is only invariant-checked so far.

## Later phases

Done since 7a: I/O scheduler + expert prefetch (7b), tokenizer + `chat` (8a),
NumPy oracle (8b), `benchmark`/`stress` (8c), threaded kernels incl. MoE
per-expert decode (9/9b), sampling. Still open: logit validation vs upstream
`Qwen4ExpForCausalLM`; QSA per-head threading; GPU backend.
