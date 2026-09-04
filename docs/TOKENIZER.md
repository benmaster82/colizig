# Tokenizer & chat (`src/qwen38/tokenizer.zig`, `chat_template.zig`, `cli/chat.zig`)

Byte-level BPE (GPT-2 / Qwen family), loaded from the checkpoint's
`tokenizer.json`.

## Pipeline

```
text
 └─ split on special-token literals  (<|im_start|>, <|im_end|>, …)
     └─ per chunk: pre-tokenize   (approximation of Qwen's split regex)
         └─ byte → unicode remap  (GPT-2 bytes_to_unicode, a bijection over 256)
             └─ BPE merges        (lowest-rank adjacent pair first, from model.merges)
                 └─ vocab lookup   → token ids  (fallback: per-byte ids so it never fails)
```

`decode` reverses it: id → token bytes (byte-unicode) → codepoints → bytes → UTF-8.
`skip_special` drops special ids from the output.

## `tokenizer.json` fields used

- `model.vocab` — `{token: id}` (token in byte-unicode form)
- `model.merges` — `["a b", …]` or `[["a","b"], …]`; rank = array index
- `added_tokens` — `[{id, content, special}]`; `special` entries split the stream

## Pre-tokenizer — approximation

Qwen's pretokenizer is a Unicode-property split regex. This implements a
greedy scanner for the same alternation (contractions `'s 't 're …`; letter
runs with an optional leading space; **single digits**; punctuation runs +
trailing newlines; whitespace with the trailing-space-attaches rule). Unicode
classes are approximated: ASCII is exact; a non-ASCII codepoint is a *letter*
unless it falls in a punctuation block (General Punctuation, CJK Symbols,
fullwidth forms). Word boundaries in scripts outside that model may differ from
the HF `tokenizers` library — **validate against the real `tokenizer.json` with
known text↔id pairs once the checkpoint is available** (Phase 8b harness).

## ChatML template (`chat_template.zig`)

```
<|im_start|>{role}\n{content}<|im_end|>\n     for each message
<|im_start|>assistant\n                        if add_generation_prompt
```

The core of Qwen's `chat_template.jinja`. The upstream template also does a
default system message, tool declarations and reasoning-effort / thinking
blocks — not replicated.

## `chat` CLI

```
colizig chat <MODEL_DIR> [--prompt "..."] [--system "..."] [--steps N]
                [--expert-cap K] [--ram-limit G] [--context K]
```

**One-shot** (`--prompt` given): builds `[system?, user]` → renders ChatML →
encodes → prefill → **greedy** decode (stop at `<|im_end|>` / `eos_token_id` /
`--steps`, default 64) → prints the assistant turn (specials stripped).

**Interactive REPL** — "ColiZig" (no `--prompt`): a colibri-styled read-eval
loop. Header with the hummingbird logo in Qwen violet + Zig amber; each turn the
new user message is wrapped as `<|im_end|>\n<|im_start|>user\n…<|im_end|>\n` +
`assistantOpen(think)` (`<|im_start|>assistant\n<think>\n` when reasoning is on —
the Qwen3.8 template default — or the empty `<think>\n\n</think>\n\n` block when
off), encoded, and prefilled in ≤512-token chunks; then greedy decode **streams
each token's detokenized delta** with the reasoning greyed inside a
`┌ thinking / │ … / └─` box and the answer under a `◆ ColiZig` marker in Qwen's
violet.  `▸ you` prompt in Zig amber.  Per-reply footer `N tok · prefill …
· X tok/s`.  `/think` toggles reasoning, `/reset` clears the conversation,
`/exit` (or EOF) quits.  The `State` (KV / DeltaNet / PLE) and the warm expert
cache carry across turns.  Default `--steps` is 512 in REPL mode.

`<think>` / `</think>` (ids 248068 / 248069) are `special:false` **added
tokens** — the loader still splits on them atomically (as HF does for every added
token, special or not), so they encode as one id and the REPL can detect the
`</think>` that ends the reasoning; `decode(skip_special=true)` keeps them (only
`special:true` tokens are dropped).

Both modes use the Phase 7b I/O scheduler + predictor and refuse a tokenizer
whose vocab exceeds the model's, a token id outside the model vocabulary, or a
context that doesn't fit the memory plan.

**Sampling** (`src/runtime/sampler.zig`): `--temperature 0` (default) is exact
greedy/argmax — deterministic, bit-identical to the old loop. `--temperature <f>`
(+ optional `--top-k <n>`, `--top-p <f>`, `--seed <n>`) switches on stochastic
decode: temperature-scaled softmax over the top-k candidates → nucleus (top-p)
prefix → draw. `--seed 0` (default) pulls a fresh seed from `io.random` each run;
a fixed seed reproduces a sample. `--top-k 1` is greedy at any temperature.

## Guarantees / tests

- GPT-2 byte↔unicode map is a bijection.
- Encode→decode round-trips ASCII text.
- Special tokens split the stream and survive the round-trip; `skip_special`
  removes them.
- Pre-tokenizer splits words / single digits / punctuation as expected.
- `selftest <dir>` loads the fixture tokenizer, round-trips "hello world", and
  checks the ChatML render tokenizes with `<|im_start|>`.
- End-to-end vs the upstream tokenizer: **pending the Phase 8b reference harness**
  (the tiny fixture's tokenizer is synthetic).
