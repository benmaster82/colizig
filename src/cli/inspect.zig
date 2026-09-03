//! `inspect` — report the architecture and a memory-budget plan for a
//! checkpoint, reading ONLY config.json and the safetensors headers.  No tensor
//! body is ever read here.

const std = @import("std");
const args = @import("args.zig");
const manifest_mod = @import("../model/manifest.zig");
const budget = @import("../runtime/budget.zig");
const units = @import("../util/units.zig");

const h = units.human;

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
) !void {
    var m = try manifest_mod.open(gpa, io, opts.model_dir, err);
    defer m.deinit();
    const c = m.cfg;

    try out.print(
        \\== Qwen3.8-Flash-Next (Qwen4-Exp) ==
        \\  source dir:      {s}
        \\  tensor prefix:   {s}.*
        \\  hidden size:     {d}
        \\  layers:          {d}  ({d} Gated DeltaNet + {d} Qwen Sparse Attention, PLE on layer {d})
        \\  attention:       {d} query / {d} KV heads, head_dim {d}, rotary_dim {d}, theta {d:.0}
        \\  MoE:             {d} experts, top-{d} routed + 1 shared, expert width {d}
        \\  Gated DeltaNet:  {d} key / {d} value heads, key_dim {d}, value_dim {d}, conv k={d}
        \\  QSA indexer:     {d} heads, dim {d}, budget {d}, compress ratio {d}
        \\  gated residual:  {d} branches, low-rank {d}
        \\  PLE / n-gram:    ngram_size {d}, {d} heads x {d} dim, {d} table shards
        \\  vocab:           {d}
        \\  native context:  {d} tokens
        \\
    , .{
        opts.model_dir,
        m.text_prefix,
        c.hidden,
        c.layers,
        c.numGdnLayers(),
        c.numAttnLayers(),
        c.ple_layer,
        c.q_heads,
        c.kv_heads,
        c.head_dim,
        c.rotary_dim,
        c.theta,
        c.experts,
        c.topk,
        c.inter,
        c.dn_kheads,
        c.dn_vheads,
        c.dn_kdim,
        c.dn_vdim,
        c.dn_convk,
        c.idx_qheads,
        c.idx_dim,
        c.idx_budget,
        c.idx_ratio,
        c.hc_count,
        c.hc_rank,
        c.ngram_size,
        c.ngram_heads,
        c.ngram_head_dim,
        c.ngram_parts,
        c.vocab,
        c.max_positions,
    });

    if (m.metadata_only) {
        try out.print(
            \\
            \\  (metadata-only: shard files absent — tensor names & categories from
            \\   model.safetensors.index.json; per-tensor shapes/bytes are estimated)
            \\
        , .{});
    }

    // Tensor inventory.
    const n_total = m.tensors.len;
    const n_resident = m.countByResidency(.resident);
    const n_stream = m.countByResidency(.streamable);
    const n_cold = m.countByResidency(.cold);
    const n_vision = m.countByCategory(.vision);
    const n_mtp = m.countByCategory(.mtp);

    try out.print(
        \\-- tensors --
        \\  total indexed:   {d}
        \\  resident:        {d}
        \\  streamable MoE:  {d}
        \\  cold PLE table:  {d}
        \\  ignored (vision {d} / MTP {d})
        \\
    , .{ n_total, n_resident, n_stream, n_cold, n_vision, n_mtp });

    // Parameters (from shapes — only meaningful with the shard headers).
    const p_total = m.paramCount();
    const p_resident = m.paramCountByResidency(.resident);
    const p_stream = m.paramCountByResidency(.streamable);
    const p_cold = m.paramCountByResidency(.cold);
    if (!m.metadata_only) {
        // Dense (incl. shared expert) + `topk` routed experts per layer. `p_stream`
        // already sums all layers, so dividing by the per-layer expert count and
        // multiplying by topk gives the per-token routed contribution.
        const active_per_tok = if (c.experts != 0)
            p_resident + p_stream * c.topk / c.experts
        else
            p_resident;

        try out.print(
            \\-- parameters --
            \\  total:           {d}
            \\  resident dense:  {d}
            \\  routed experts:  {d}
            \\  PLE n-gram:      {d}
            \\  ~active / token: {d}
            \\
        , .{ p_total, p_resident, p_stream, p_cold, active_per_tok });
    }

    try out.print(
        \\-- PLE table (never resident) --
        \\  per-token reads: {d} rows x {d} bytes  (bigram: {d} heads, trigram: {d})
        \\
    , .{
        c.ngram_heads,
        c.ngram_head_dim,
        c.heads_per_ngram,
        c.ngram_heads - c.heads_per_ngram,
    });

    const resident = if (m.metadata_only) budget.estimateResidentBytes(c) else m.residentBytes();
    try out.print(
        \\-- storage --
        \\  resident weights (BF16):   {f}{s}
        \\  checkpoint total on disk:  {f}
        \\
    , .{
        h(resident),
        if (m.metadata_only) "   (estimated from config)" else "",
        if (m.total_size) |ts| h(ts) else h(m.residentBytes() +| m.routedExpertBytes() +| m.pleTableBytes()),
    });

    const p = budget.plan(c, resident, opts.budget);
    try out.writeAll("\n");
    try p.print(out);

    if (!p.fits) {
        try err.writeAll(
            \\
            \\inspect: the requested context does not fit the RAM budget.
            \\  Try a larger --ram-limit, a smaller --context, or --profile desktop/gpu.
            \\
        );
        return error.ContextDoesNotFit;
    }
}
