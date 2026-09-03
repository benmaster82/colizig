//! `forward` — run the end-to-end Qwen4-Exp forward on raw token ids.
//!
//! There is no tokenizer yet (Phase 8+), so this takes `--tokens` as a
//! comma-separated list of integer ids and, with `--steps N`, greedily decodes
//! N more.  Output is ids in, ids out.

const std = @import("std");
const args = @import("args.zig");
const units = @import("../util/units.zig");
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");
const budget = @import("../runtime/budget.zig");
const model_mod = @import("../qwen38/model.zig");
const parallel = @import("../runtime/parallel.zig");
const usage_mod = @import("../runtime/expert_usage.zig");
const sampler_mod = @import("../runtime/sampler.zig");

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
) !void {
    if (opts.tokens.len == 0) {
        try err.writeAll("forward: --tokens <id,id,...> is required (no tokenizer yet)\n");
        return error.MissingModelDir;
    }
    parallel.enable(io, opts.threads);
    defer parallel.disable();

    var prompt: std.ArrayList(i64) = .empty;
    defer prompt.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, opts.tokens, ", ");
    while (it.next()) |tok| {
        const v = std.fmt.parseInt(i64, tok, 10) catch {
            try err.print("forward: bad token id \"{s}\"\n", .{tok});
            return error.BadFlag;
        };
        try prompt.append(gpa, v);
    }
    if (prompt.items.len == 0) {
        try err.writeAll("forward: no token ids parsed\n");
        return error.BadFlag;
    }

    var m = try manifest_mod.open(gpa, io, opts.model_dir, err);
    defer m.deinit();
    var w = try weights_mod.Weights.open(gpa, io, opts.model_dir, &m, err);
    defer w.deinit();
    if (opts.mirror.len != 0) {
        const nm = w.attachMirror(gpa, io, opts.mirror);
        try out.print("mirror: {d}/{d} shards from {s} (routed-expert reads split across both drives)\n", .{ nm, w.shards.len, opts.mirror });
    }

    // memory plan → context + expert-cache capacity
    var bopts = opts.budget;
    if (bopts.context < prompt.items.len + opts.steps)
        bopts.context = @intCast(prompt.items.len + opts.steps);
    const plan = budget.plan(m.cfg, m.residentBytes(), bopts);
    if (!plan.fits) {
        try err.print("forward: {s}\n", .{plan.reason});
        return error.ContextDoesNotFit;
    }
    const cap: usize = if (opts.expert_cap != 0) opts.expert_cap else plan.expert_cap;

    try out.print("loading model ({d} layers, resident {f}, expert cache cap {d}/layer) ...\n", .{ m.cfg.layers, units.human(m.residentBytes()), cap });
    const t_load0 = std.Io.Timestamp.now(io, .awake);
    var model = try model_mod.Model.load(gpa, &w);
    defer model.deinit();
    const load_ns = elapsedNs(io, t_load0);

    const ctx: usize = plan.context;
    var state = try model_mod.State.init(gpa, &model, ctx, cap, io);
    defer state.deinit();
    var sc = try model_mod.Scratch.init(gpa, &model, @max(prompt.items.len, 1), ctx);
    defer sc.deinit();

    // learned expert priors: load (or start fresh), warm the caches, save on exit
    var udir: ?std.Io.Dir = if (opts.no_usage) null else (if (std.fs.path.isAbsolute(opts.model_dir))
        std.Io.Dir.openDirAbsolute(io, opts.model_dir, .{})
    else
        std.Io.Dir.cwd().openDir(io, opts.model_dir, .{})) catch null;
    defer if (udir) |*d| d.close(io);
    const ul: u32 = @intCast(m.cfg.layers);
    const ue: u32 = @intCast(m.cfg.experts);
    var usage: ?model_mod.ExpertUsage = if (udir) |d| (usage_mod.ExpertUsage.load(gpa, io, d, ul, ue) orelse (model_mod.ExpertUsage.init(gpa, ul, ue) catch null)) else null;
    defer if (usage) |*u| u.deinit();
    defer if (usage) |*u| {
        if (udir) |d| u.save(io, d);
    };
    if (usage) |*u| {
        const stream = @as(u64, cap) * plan.per_expert_bytes * m.cfg.layers;
        const headroom = plan.ram_budget -| plan.fixed_resident;
        usage_mod.warmCaches(u, state.experts, &w, model.moe_dims, @min(stream, headroom));
    }

    const logits = try gpa.alloc(f32, m.cfg.vocab);
    defer gpa.free(logits);

    var sched = model_mod.Scheduler.init(gpa, 4096);
    defer sched.deinit();
    var predictor = try model_mod.Predictor.init(gpa, m.cfg.layers, m.cfg.topk);
    defer predictor.deinit();
    const fopts: model_mod.Opts = .{ .io = io, .scheduler = &sched, .predictor = &predictor, .usage = if (usage) |*u| u else null };

    var sampler = try sampler_mod.Sampler.init(gpa, io, m.cfg.vocab, .{
        .temperature = opts.temperature,
        .top_k = opts.top_k,
        .top_p = opts.top_p,
        .seed = opts.seed,
    });
    defer sampler.deinit();

    // prefill → time to first token
    const t_pf0 = std.Io.Timestamp.now(io, .awake);
    try model_mod.forward(&model, &state, &sc, prompt.items, logits, fopts);
    const ttft_ns = elapsedNs(io, t_pf0);

    try out.writeAll("prompt: ");
    try printIds(out, prompt.items);
    try out.writeAll("\ntop logits: ");
    try printTopK(out, logits, 8);
    try out.writeByte('\n');

    var decode_ns: u64 = 0;
    var generated: usize = 0;
    if (opts.steps > 0) {
        const gen = try gpa.alloc(i64, opts.steps);
        defer gpa.free(gen);
        var one: [1]i64 = undefined;
        var stopped_eos = false;
        const t_dec0 = std.Io.Timestamp.now(io, .awake);
        while (generated < opts.steps) : (generated += 1) {
            const best = sampler.pick(logits);
            if (@as(i64, @intCast(best)) == m.cfg.eos_id) {
                stopped_eos = true;
                break;
            }
            gen[generated] = @intCast(best);
            one[0] = @intCast(best);
            try model_mod.forward(&model, &state, &sc, one[0..1], logits, fopts);
        }
        decode_ns = elapsedNs(io, t_dec0);
        try out.writeAll(if (sampler.greedy()) "greedy: " else "sampled: ");
        try printIds(out, gen[0..generated]);
        if (stopped_eos) try out.writeAll("  (stopped at EOS)");
        try out.writeByte('\n');
    }

    const tok_s: f64 = if (decode_ns == 0 or generated == 0) 0 else @as(f64, @floatFromInt(generated)) * 1e9 / @as(f64, @floatFromInt(decode_ns));
    try out.print(
        \\-- timing --
        \\  model load:   {d:.1} s
        \\  TTFT:         {d:.2} s   ({d} prompt tokens)
        \\  decode:       {d:.3} tok/s   ({d} tokens in {d:.1} s)
        \\
    , .{
        @as(f64, @floatFromInt(load_ns)) / 1e9,
        @as(f64, @floatFromInt(ttft_ns)) / 1e9,       prompt.items.len,
        tok_s,                                        generated,
        @as(f64, @floatFromInt(decode_ns)) / 1e9,
    });

    // I/O scheduler + prefetch telemetry
    var demand: u64 = 0;
    var pf_hits: u64 = 0;
    var pf_loads: u64 = 0;
    var pf_wasted: u64 = 0;
    var expert_reqs: u64 = 0;
    for (state.experts) |ec| {
        demand += ec.stats.demand_loads;
        pf_hits += ec.stats.prefetch_hits;
        pf_loads += ec.stats.prefetch_loads;
        pf_wasted += ec.stats.prefetch_wasted;
        expert_reqs += ec.stats.hits + ec.stats.misses;
    }
    try out.print(
        \\-- I/O scheduler --
        \\  scheduler queue peak:   {d}   (bounded, HIGH before MEDIUM before LOW)
        \\  serviced HIGH/MED/LOW:  {d} / {d} / {d}
        \\  predictor accuracy:     {d:.1}%   ({d}/{d})
        \\  expert demand loads:    {d}   (compute_stall_due_to_io proxy)
        \\  expert prefetch loads:  {d}   (hits {d}, wasted {d})
        \\  expert prefetch cover:  {d:.1}%   of {d} routed accesses
        \\
    , .{
        sched.stats.queue_peak,
        sched.stats.serviced[0],
        sched.stats.serviced[1],
        sched.stats.serviced[2],
        100.0 * predictor.stats.accuracy(),
        predictor.stats.correct,
        predictor.stats.predicted,
        demand,
        pf_loads,
        pf_hits,
        pf_wasted,
        if (expert_reqs == 0) @as(f64, 0) else 100.0 * @as(f64, @floatFromInt(pf_hits)) / @as(f64, @floatFromInt(expert_reqs)),
        expert_reqs,
    });
}

fn elapsedNs(io: std.Io, from: std.Io.Timestamp) u64 {
    const d = from.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds;
    return if (d > 0) @intCast(d) else 0;
}

fn printIds(out: *std.Io.Writer, ids: []const i64) !void {
    for (ids, 0..) |t, i| {
        if (i != 0) try out.writeByte(',');
        try out.print("{d}", .{t});
    }
}

fn printTopK(out: *std.Io.Writer, logits: []const f32, k: usize) !void {
    var idx: [16]usize = undefined;
    const kk = @min(k, @min(logits.len, idx.len));
    for (0..kk) |r| {
        var best: usize = 0;
        var bv: f32 = -std.math.inf(f32);
        outer: for (logits, 0..) |v, i| {
            for (idx[0..r]) |used| if (used == i) continue :outer;
            if (v > bv) {
                bv = v;
                best = i;
            }
        }
        idx[r] = best;
        if (r != 0) try out.writeByte(' ');
        try out.print("{d}:{d:.3}", .{ best, bv });
    }
}
