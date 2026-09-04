//! `forward` for `model_type: qwen3_moe` (Qwen3-30B-A3B and friends). Dispatched
//! from `forward.zig` when `m.cfg.arch == .qwen3_moe`. Raw token ids in, ids out.

const std = @import("std");
const args = @import("args.zig");
const units = @import("../util/units.zig");
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");
const budget = @import("../runtime/budget.zig");
const q3 = @import("../qwen3moe/model.zig");
const parallel = @import("../runtime/parallel.zig");
const usage_mod = @import("../runtime/expert_usage.zig");
const sampler_mod = @import("../runtime/sampler.zig");
const gpu = @import("../backend/gpu.zig");
const timers_mod = @import("../runtime/timers.zig");

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
    prompt: []const i64,
) !void {
    parallel.enable(io, opts.threads);
    defer parallel.disable();
    if (opts.cuda) gpu.init(err);
    gpu.verify = opts.cuda_verify;
    if (opts.cuda) gpu.setVramBudget(if (opts.vram != 0) opts.vram else 16 << 30);
    defer gpu.deinit();
    defer gpu.verifySummary();

    var m = try manifest_mod.open(gpa, io, opts.model_dir, err);
    defer m.deinit();
    var w = try weights_mod.Weights.open(gpa, io, opts.model_dir, &m, err);
    defer w.deinit();

    const steps: usize = opts.steps;
    // Qwen3-30B-A3B KV: 2 · kv_heads · head_dim · 4 B per token per layer.
    const kv_per_tok: u64 = 2 * @as(u64, m.cfg.kv_heads) * m.cfg.head_dim * 4 * m.cfg.layers;
    var want_ctx: usize = prompt.len + steps + 4;
    if (opts.budget.context != 0 and opts.budget.context > want_ctx) want_ctx = opts.budget.context;

    const cap: usize = if (opts.expert_cap != 0) opts.expert_cap else @min(64, m.cfg.experts);

    try out.print(
        "loading Qwen3-MoE ({d} layers, {d} experts top-{d}, KV {f}/tok, expert cap {d}/layer) ...\n",
        .{ m.cfg.layers, m.cfg.experts, m.cfg.topk, units.human(kv_per_tok), cap },
    );
    const t_load0 = std.Io.Timestamp.now(io, .awake);
    var model = try q3.Model.load(gpa, &w);
    defer model.deinit();
    const load_ns = ns(io, t_load0);

    var state = try q3.State.init(gpa, &model, want_ctx, cap, io);
    defer state.deinit();
    var sc = try q3.Scratch.init(gpa, &model, @max(prompt.len, 1), want_ctx);
    defer sc.deinit();

    // learned expert priors (shared file format with the qwen4_exp path)
    var udir: ?std.Io.Dir = if (opts.no_usage) null else openModelDir(io, opts.model_dir) catch null;
    defer if (udir) |*d| d.close(io);
    const ul: u32 = @intCast(m.cfg.layers);
    const ue: u32 = @intCast(m.cfg.experts);
    var usage: ?q3.ExpertUsage = if (udir) |d| (usage_mod.ExpertUsage.load(gpa, io, d, ul, ue) orelse (q3.ExpertUsage.init(gpa, ul, ue) catch null)) else null;
    defer if (usage) |*u| u.deinit();
    defer if (usage) |*u| {
        if (udir) |d| u.save(io, d);
    };
    if (usage) |*u| {
        const per_expert = 3 * @as(u64, m.cfg.inter) * m.cfg.hidden + m.cfg.hidden * m.cfg.inter;
        usage_mod.warmCaches(u, state.experts, &w, model.moe_dims, @as(u64, cap) * per_expert * m.cfg.layers);
    }

    var timers = timers_mod.Timers.init(io);
    const fopts: q3.Opts = .{ .io = io, .timers = &timers, .usage = if (usage) |*u| u else null };

    var sampler = try sampler_mod.Sampler.init(gpa, io, m.cfg.vocab, .{
        .temperature = opts.temperature,
        .top_k = opts.top_k,
        .top_p = opts.top_p,
        .seed = opts.seed,
    });
    defer sampler.deinit();

    const logits = try gpa.alloc(f32, m.cfg.vocab);
    defer gpa.free(logits);

    const t_pf0 = std.Io.Timestamp.now(io, .awake);
    try q3.forward(&model, &state, &sc, prompt, logits, fopts);
    const ttft_ns = ns(io, t_pf0);

    var gen: std.ArrayList(i64) = .empty;
    defer gen.deinit(gpa);
    var stopped_eos = false;
    const t_dec0 = std.Io.Timestamp.now(io, .awake);
    var one: [1]i64 = undefined;
    while (gen.items.len < steps) {
        const next = sampler.pick(logits);
        try gen.append(gpa, @intCast(next));
        if (@as(i64, @intCast(next)) == m.cfg.eos_id) {
            stopped_eos = true;
            break;
        }
        one[0] = @intCast(next);
        try q3.forward(&model, &state, &sc, &one, logits, fopts);
    }
    const dec_ns = ns(io, t_dec0);

    try out.writeAll(if (sampler.greedy()) "greedy: " else "sampled: ");
    for (gen.items, 0..) |t, i| {
        if (i != 0) try out.writeByte(',');
        try out.print("{d}", .{t});
    }
    if (stopped_eos) try out.writeAll("  (stopped at EOS)");
    try out.writeByte('\n');

    const tok_s: f64 = if (dec_ns == 0 or gen.items.len == 0) 0 else @as(f64, @floatFromInt(gen.items.len)) * 1e9 / @as(f64, @floatFromInt(dec_ns));
    try out.print(
        \\-- timing --
        \\  model load:   {d:.1} s
        \\  TTFT:         {d:.2} s   ({d} prompt tokens)
        \\  decode:       {d:.3} tok/s   ({d} tokens in {d:.1} s)
        \\
    , .{
        @as(f64, @floatFromInt(load_ns)) / 1e9,
        @as(f64, @floatFromInt(ttft_ns)) / 1e9,
        prompt.len,
        tok_s,
        gen.items.len,
        @as(f64, @floatFromInt(dec_ns)) / 1e9,
    });
    try timers.print(out);
    gpu.statsLine(out);
}

fn ns(io: std.Io, from: std.Io.Timestamp) u64 {
    const d = from.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds;
    return if (d > 0) @intCast(d) else 0;
}

fn openModelDir(io: std.Io, path: []const u8) !std.Io.Dir {
    return if (std.fs.path.isAbsolute(path))
        std.Io.Dir.openDirAbsolute(io, path, .{})
    else
        std.Io.Dir.cwd().openDir(io, path, .{});
}
