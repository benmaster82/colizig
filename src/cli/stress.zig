//! `stress` — sweep RAM budgets, verify the memory plan holds and our tracked
//! peak stays under the limit, report tok/s and the expert-cache hit rate at
//! each (brief §26).

const std = @import("std");
const args = @import("args.zig");
const units = @import("../util/units.zig");
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");
const budget = @import("../runtime/budget.zig");
const model_mod = @import("../qwen38/model.zig");
const meter_mod = @import("../runtime/meter.zig");
const parallel = @import("../runtime/parallel.zig");

const h = units.human;
const G: u64 = 1024 * 1024 * 1024;
const default_ladder = [_]u64{ 8 * G, 12 * G, 16 * G, 24 * G, 32 * G };

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
) !void {
    parallel.enable(io, opts.threads);
    defer parallel.disable();

    // metadata is cheap and shared across the sweep
    var m = try manifest_mod.open(gpa, io, opts.model_dir, err);
    defer m.deinit();
    const resident = m.residentBytes();
    const ctx: u32 = if (opts.budget.context == 0) 8192 else opts.budget.context;
    const prompt_len: usize = if (opts.prompt_len == 0) 8 else opts.prompt_len;
    const steps: usize = if (opts.steps == 0) 16 else opts.steps;

    var single: [1]u64 = undefined;
    const ladder: []const u64 = if (opts.budget.ram_limit) |r| blk: {
        single[0] = r;
        break :blk single[0..1];
    } else &default_ladder;

    try out.print(
        \\stress: context {d}, prompt {d} + {d} decode steps
        \\
        \\ {s:>10} | {s:>12} | {s:>12} | {s:>10} | {s:>9} | {s}
        \\ {s:->10}-+-{s:->12}-+-{s:->12}-+-{s:->10}-+-{s:->9}-+-{s:->8}
        \\
    , .{
        ctx,         prompt_len,    steps,
        "ram limit", "plan target", "tracked peak",
        "tok/s",     "hit rate",    "status",
        "",          "",            "",
        "",          "",            "",
    });

    for (ladder) |limit| {
        const plan = budget.plan(m.cfg, resident, .{ .ram_limit = limit, .context = ctx });
        if (!plan.fits) {
            try out.print(" {f:>10} | {f:>12} | {s:>12} | {s:>10} | {s:>9} | DOES NOT FIT\n", .{
                h(limit), h(plan.fixed_resident),
                "—",
                "—",
                "—",
            });
            continue;
        }

        var meter = meter_mod.Meter.init(gpa);
        const row = runOne(meter.allocator(), io, opts.model_dir, m.cfg, plan, ctx, prompt_len, steps) catch |e| {
            try out.print(" {f:>10} | {f:>12} | {s:>12} | {s:>10} | {s:>9} | error: {s}\n", .{
                h(limit),      h(plan.total_resident),
                "—",
                "—",
                "—",
                @errorName(e),
            });
            continue;
        };
        const ok = meter.peak <= limit;
        try out.print(" {f:>10} | {f:>12} | {f:>12} | {d:>10.3} | {d:>8.1}% | {s}\n", .{
            h(limit),                                            h(plan.total_resident), h(meter.peak), row.tok_s, row.hit_rate,
            if (ok) "resident <= limit" else "!! EXCEEDS LIMIT",
        });
    }
}

const Row = struct { tok_s: f64, hit_rate: f64 };

fn runOne(
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    cfg: manifest_mod.Cfg,
    plan: budget.Plan,
    ctx: u32,
    prompt_len: usize,
    steps: usize,
) !Row {
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);

    var m = try manifest_mod.open(gpa, io, dir, &sink.writer);
    defer m.deinit();
    var w = try weights_mod.Weights.open(gpa, io, dir, &m, &sink.writer);
    defer w.deinit();
    var model = try model_mod.Model.load(gpa, &w);
    defer model.deinit();

    var state = try model_mod.State.init(gpa, &model, ctx, plan.expert_cap, io);
    defer state.deinit();
    var sc = try model_mod.Scratch.init(gpa, &model, @max(prompt_len, 1), ctx);
    defer sc.deinit();
    var sched = model_mod.Scheduler.init(gpa, 4096);
    defer sched.deinit();
    var predictor = try model_mod.Predictor.init(gpa, cfg.layers, cfg.topk);
    defer predictor.deinit();
    const fopts: model_mod.Opts = .{ .io = io, .scheduler = &sched, .predictor = &predictor };

    const ids = try gpa.alloc(i64, prompt_len);
    defer gpa.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast((i + 1) % cfg.vocab);
    const logits = try gpa.alloc(f32, cfg.vocab);
    defer gpa.free(logits);

    try model_mod.forward(&model, &state, &sc, ids, logits, fopts);
    var one: [1]i64 = undefined;
    const t0 = std.Io.Timestamp.now(io, .awake);
    var n: usize = 0;
    while (n < steps) : (n += 1) {
        var best: usize = 0;
        for (logits, 0..) |v, i| {
            if (v > logits[best]) best = i;
        }
        one[0] = @intCast(best);
        try model_mod.forward(&model, &state, &sc, one[0..1], logits, fopts);
    }
    const dt: u64 = @intCast(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds);

    var er: u64 = 0;
    var eh: u64 = 0;
    for (state.experts) |ec| {
        er += ec.stats.hits + ec.stats.misses;
        eh += ec.stats.hits;
    }
    return .{
        .tok_s = if (dt == 0 or n == 0) 0 else @as(f64, @floatFromInt(n)) * 1e9 / @as(f64, @floatFromInt(dt)),
        .hit_rate = if (er == 0) 0 else 100.0 * @as(f64, @floatFromInt(eh)) / @as(f64, @floatFromInt(er)),
    };
}
