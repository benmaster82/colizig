# Speculative decoding (`src/runtime/lookup_draft.zig`, `src/qwen3moe/speculative.zig`)

Verify several drafted tokens in one batched forward instead of one token per
forward call - the direct local lever for tok/s (no second machine, no
network protocol; see the project discussion that led here). Greedy-only,
**Qwen3-MoE only** for now (`--speculative N` in `chat`, `N` = max draft
tokens per round, 0 = off). Wired into `chat`'s `Session(Mdl).generate()`,
comptime-gated so it never touches the Qwen4-Exp path
(`src/cli/chat.zig`: `if (comptime Mdl == q3) { ... }`).

## Why this is free: `moe.forwardGrouped` already proved the shape

`forward()` already runs a batch of `S` tokens through the whole model in one
call (that's what prompt prefill is). The only thing missing for speculative
verification was reading the **per-position** prediction instead of only the
last one - `Scratch.h` (`src/qwen3moe/model.zig`) already holds the final
hidden state for every position after `forward()` returns; only `lmHead` was
being called on the last row. `speculativeStep` reuses `forward()` completely
unmodified and just does the extra `lmHead` calls itself.

## Draft: n-gram / prompt lookup (`lookup_draft.zig`)

No second model, no training. Search the token history (prompt + everything
confirmed so far) for the most recent earlier occurrence of the trailing
3-token window and propose whatever followed it, up to `k` tokens - clipped
so the draft never wraps back into the trailing window itself (that would
just be "predicting" tokens already known). Backward linear scan - a chat
turn's history is hundreds to a few thousand tokens, microseconds next to a
forward pass; a hashmap index would be the natural upgrade if that stops
being true. Best on repetitive content: pasted code/text the model echoes or
lightly edits.

## Verify / accept / rollback (`speculativeStep`)

Confirmed context ends at `state.pos = P` (last confirmed token `t`). Draft
proposes `d[1..k]`. Build one batch `ids = [t, d[1], ..., d[k]]` (`S = k+1`)
and run **one** `forward()` call.

For each batch position `i` in `0..k-1` (predicting the token after `ids[i]`):
recompute its prediction - `forward()` only applies the final RMSNorm to the
*last* row of `sc.h` (`src/qwen3moe/model.zig:238`, in place), so every
earlier row is still pre-final-norm. Copy it, norm the copy, `lmHead` it.
Compare the argmax to `d[i+1]`.

**The gotcha that cost a debugging round while implementing this**: position
`k` (the *last* batch position) is different - `forward()` already normed
that row **in place** and already ran `lmHead` on it into the caller-supplied
`logits` buffer. Re-deriving it the same way as the earlier rows
double-applies RMSNorm to an already-normed row and silently desyncs every
following prediction from a token-by-token decode (caught by the "matches
greedy" unit test, not by inspection - the first version of this function
got token 0 wrong on a fresh state, `14 47 14 24 14 61` instead of
`27 32 31 25 0 25`). Position `k`'s correct prediction is just
`argmax(logits)` - no recomputation.

Walk `i = 0..k-1`: first mismatch at `i = m` stops the walk - that position's
argmax is the **correction**. If every draft token matches, the walk reaches
`i = k` and that position's argmax (via the case above) is a free **bonus**
token. Either way exactly one extra token comes from the same verification
pass, so a round always yields `valid = accepted + 1 >= 1` new tokens, even
in the worst case (first draft token already wrong).

Rollback is free: `state.pos = P - 1 + valid`; for every layer,
`state.kv[i].len = state.pos`. Qwen3-MoE's only sequence state is
`attn.Cache` (`src/qwen3moe/attn.zig`), absolute-position indexed - the
rejected suffix's bytes are simply overwritten on the next real forward call,
no extra compute. `k = 0` degenerates to exactly one plain greedy step
(`ids = [t]`, the loop starts at `i = k = 0` and immediately hits the
bonus-token case) - the round loop in `chat.zig` always goes through
`speculativeStep`, using `k = 0` as the natural fallback for "no draft
match" or "context nearly full", no separate code path needed.

## The prefill contract - the second bug this phase found

`speculativeStep` always forwards `last_confirmed` itself - so whatever code
calls the first round must **not** have already forwarded the prompt's last
token. `chat.zig`'s existing `prefill()` forwards the *entire* prompt (that's
what the plain per-token loop expects: prefill leaves `state.pos` at the
prompt length and `logits` already holds the next-token prediction, which
`sampler.pick` consumes directly). Speculative mode needs the opposite: hold
back the last prompt token so the first `speculativeStep` call can forward it.
`Session.wantsSpeculative()` is the single source of truth both call sites
(`oneShot`, `repl`) check to decide how much of the prompt to hand to
`prefill` - see its doc comment on `generate()`. Getting this wrong doubles
the forward of the prompt's last token and desyncs `state.pos` from every
subsequent prediction, the same failure signature as the RMSNorm bug above,
just triggered from the CLI path instead of the unit test - caught by
A/B'ing `chat --speculative 0` against `--speculative N` on the same prompt
and diffing the output (now part of the verification recipe below).

## Verification

```sh
zig build test --summary all      # lookup_draft + speculative unit tests, "matches greedy" invariant
zig build run -- chat <Qwen3-30B dir> --prompt "..." --speculative 4
zig build run -- chat <Qwen3-30B dir> --prompt "..." --speculative 0
# same prompt, both greedy (temperature 0 is the default) -> output must be identical
```

The unit test in `qwen3moe/speculative.zig` runs `generateGreedy` (ground
truth) against the speculative round loop at draft depths 0/2/4 on the tiny
fixture and asserts the exact same token sequence - this is the regression
that would have caught both bugs above immediately, and is what protects this
code going forward.

**`benchmark` doesn't cover this yet** - `src/cli/benchmark.zig` is
Qwen4-Exp-only today (`@import("../qwen38/model.zig")` directly, no arch
dispatch), a pre-existing gap unrelated to this phase. Measuring the real
accept-rate / tok/s gain on a real Qwen3-30B-A3B-FP8 checkpoint goes through
`chat`'s own footer (`{n} tok · ... · {tok/s}`) for now; wiring `benchmark`
would need Qwen3-MoE support there first.

## Qwen4-Exp: not yet - needs GDN/PLE snapshot + replay

Qwen3-MoE's only sequence state is the KV cache (`len`-truncatable, free
rollback). Qwen4-Exp additionally carries **irreversible recurrent state**:

- `gdn.GdnState` (`rec` + `ring`, fixed-size - confirmed via code inspection,
  allocated once from `Dims`, independent of position) mixes every token it
  sees into a small fixed cell. It cannot be truncated - the delta-rule
  recurrence has already blended any rejected draft tokens into `rec` by the
  time you'd want to reject them.
- `ple.State` (`ring` + a 2-token `history`) - same shape, same problem.

The design (not implemented): snapshot both (`@memcpy` - they're small,
fixed-size buffers) **before** the speculative batch. On a full accept
(`valid == k+1`), nothing to do - the state ends up exactly where a
token-by-token decode would leave it. On a partial reject, restore the
snapshot and **replay** - run one more forward over just the accepted prefix
(`ids[0..valid]`, `S = valid <= k`) so GDN/PLE advance through exactly the
right tokens. QSA's `Cache` truncates for free either way (same as Qwen3-MoE's
`attn.Cache`).

This costs an extra forward over the accepted prefix whenever a draft is only
partially right (up to ~2x the batch's work on those rounds - `(k+1) + valid`
vs `k+1`) - real engineering, not a blocker, but enough to be its own phase
rather than folded into this one. A full accept round (the common case with a
good draft heuristic) costs nothing extra.
