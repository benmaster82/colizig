# Memory budget

`src/runtime/budget.zig : plan()` turns `(Cfg, resident_weight_bytes, Options)`
into a `Plan`: where every resident byte goes, and the largest routed-expert
cache that still fits the RAM budget. The engine **never silently exceeds the
budget** - if nothing fits, `Plan.fits == false` with a reason, and `inspect`
exits non-zero.

## Inputs

| input | source |
|---|---|
| `resident_weights` | `Manifest.residentBytes()` - sum of `size` over `resident` tensors, native dtype |
| `--ram-limit` | explicit override, else the profile |
| `--profile` | `tiny` 8 GiB · `laptop` 16 GiB · `desktop` 32 GiB · `gpu` 64 GiB (RAM); VRAM 0 / 6 / 12 / 24 GiB (reported only in Phase 1) |
| `--context` | tokens; default 8192 |
| `--gpu` | `none` (Phase 1) / `auto` (recorded, planned as CPU-only) |

## Formulas

```
nblk(x)            = ⌈x / 128⌉
fp8_scale_bank     = experts · 3 · nblk(inter) · nblk(hidden) · 4 · layers
per_expert_bytes   = 2·inter·hidden + hidden·inter          (E4M3, 1 B/elem)
expert_stream(cap) = cap · per_expert_bytes · layers

gdn_state          = numGdnLayers · ( vheads·kdim·vdim·4
                                    + conv_dim·(conv_k − 1)·4 )
ple_state          = hc_width·(ple_conv_k − 1)·ngram_size·4

per_token_bytes    = numAttnLayers · (2·kv_heads·head_dim + idx_dim)·4
context_state      = per_token_bytes · context

scratch            = 1152 MiB      (constant, calibrated - see below)

private_resident   = resident_weights + fp8_scale_bank + gdn_state
                   + ple_state + context_state + scratch      (the `fixed_resident` field)
recommended_RAM    = private_resident + expert_stream(chosen_cap)   (the `total_resident` field)
```

**Private vs reclaimable.** Since the perf pass, the `ExpertCache` **borrows** each
routed expert's E4M3 bytes straight out of the shard mmap - nothing is copied, so
those pages are OS page cache that faults in on demand and is evicted under
pressure, *not* a private allocation. So `expert_stream` is a soft upper bound on
the hot-expert working set, and only `private_resident` is a hard requirement.
(`resident_weights` is itself slightly conservative: `embed_tokens` and `lm_head`
- ~2.5 GiB together on the real model - are also streamed row-by-row from the
mmap, never materialised.)

**`fits`**: `private_resident ≤ ram_budget`. If that holds the model runs;
otherwise `fits = false` with a reason and `inspect` exits non-zero.

**Cap selection**: walk `cap_ladder = [128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4]`
and take the largest `cap` with `private_resident + expert_stream(cap) ≤ ram_budget`;
if even the smallest doesn't fit, keep the smallest (the page cache just recycles
harder - slower, still correct).

## Real-checkpoint expectation (calibration targets from colibri)

colibri, measured on `Qwen/Qwen3.8-Flash-Next-FP8`:

| quantity | colibri | our formula |
|---|---|---|
| resident weights (native BF16) | **9.2 GiB** | `Manifest.residentBytes()` on the real index - expect ≈ this |
| routed expert (E4M3, one) | **4.7 MiB** | `per_expert_bytes` = 2·640·2560 + 2560·640 = 4,915,200 B ≈ **4.69 MiB** ✅ |
| expert cache, cap 16 / 32 / 64 | 3.5 / 7.0 / 14.1 GiB | `cap · 4.69 MiB · 48` = 3.52 / 7.03 / 14.06 GiB ✅ |
| FP8 scale bank | **28 MiB** | `512·3·nblk(640)·nblk(2560)·4·48` = 512·3·5·20·4·48 = 29.5 MiB ≈ ✅ |
| context state per token | **54 KiB** | `12·(2·2·256 + 128)·4` = 12·1152·4 = **55,296 B = 54 KiB** ✅ |
| context bank @ 8192 | **432 MiB** | 54 KiB · 8192 = **432 MiB** ✅ |
| workspace peak | **≤ ~1.1 GiB** | `scratch` constant = 1152 MiB (this is the calibration source) |
| recurrent + PLE + snapshot state | ~226 MiB (colibri lumps several) | `gdn_state` = 36·(48·128·128·4 + 10240·3·4) ≈ 36·(3.15 MiB + 120 KiB) ≈ **117 MiB**; `ple_state` ≈ 480 KiB. colibri's 226 MiB also includes a prefix snapshot + cached logits we don't model in Phase 1. |

Where our number and colibri's diverge (workspace, the lumped recurrent/snapshot
figure) the difference is **documented, not fudged** (brief rule 14).

**Verified on the real checkpoint** (`inspect D:\…\Qwen38-FP8 --ram-limit 16G`):
resident weights 9.22 GiB, FP8 scale bank 28.13 MiB, GDN state 112 MiB, KV bank
432 MiB @ 8192, private resident 10.90 GiB, expert stream 3.52 GiB @ cap 16,
recommended RAM 14.42 GiB - every figure matches the table above. Measured
`tracked peak` during a run is ~7 GiB (below `private_resident` because the
1.13 GiB scratch estimate is worst-case and embed/lm_head stream rather than
allocate).

## Worked example - tiny fixture

`inspect test/fixtures/tiny --profile tiny --context 2048`:

```
resident weights   102.57 KiB     (H=32, 4 layers, tiny shapes)
FP8 scale bank      192 B          (512→4 experts, nblk = 1×1)
GDN state           2.63 KiB
PLE conv state      4.50 KiB       (128·3·3·4)
context / KV bank   160.00 KiB     (80 B/token · 2048;  1·(2·1·8 + 4)·4 = 80)
scratch (peak)      1.13 GiB
private resident    1.13 GiB       (dominated by the scratch constant at toy scale)
expert stream       768.00 KiB     (cap 128 · 1.5 KiB · 4 - reclaimable page cache)
recommended RAM     ≈ 1.13 GiB
```

At `--ram-limit 512M --context 262144` the context bank plus the 1.13 GiB scratch
push `private resident` past the budget, so `inspect` reports
`DOES NOT FIT: the private resident set (weights + KV + scratch) exceeds the RAM
budget` and exits 1.
