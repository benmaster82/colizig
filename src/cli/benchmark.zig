//! `benchmark` - run a prefill + N decode steps on a synthetic prompt and
//! report the runtime telemetry (brief §23).  On the tiny fixture the numbers
//! are toy-scale; on a real checkpoint they are the real thing.

const std = @import("std");
const args = @import("args.zig");
const units = @import("../util/units.zig");
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");
const budget = @import("../runtime/budget.zig");
const model_mod = @import("../qwen38/model.zig");
const meter_mod = @import("../runtime/meter.zig");
const parallel = @import("../runtime/parallel.zig");
const gpu = @import("../backend/gpu.zig");

const h = units.human;

pub fn run(
    gpa_in: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
) !void {
    parallel.enable(io, opts.threads);
    defer parallel.disable();
    if (opts.cuda) gpu.init(err);
    if (opts.cuda) gpu.setVramBudget(if (opts.vram != 0) opts.vram else 16 << 30);
    defer gpu.deinit();
    var meter = meter_mod.Meter.init(gpa_in);
    const gpa = meter.allocator();

    var m = try manifest_mod.open(gpa, io, opts.model_dir, err);
    defer m.deinit();
    var w = try weights_mod.Weights.open(gpa, io, opts.model_dir, &m, err);
    defer w.deinit();
    if (opts.mirror.len != 0) {
        const nm = w.attachMirror(gpa, io, opts.mirror);
        try out.print("mirror: {d}/{d} shards from {s}\n", .{ nm, w.shards.len, opts.mirror });
    }

    const prompt_len: usize = if (opts.prompt_len == 0) 16 else opts.prompt_len;
    const steps: usize = if (opts.steps == 0) 32 else opts.steps;

    var bopts = opts.budget;
    if (bopts.context < prompt_len + steps) bopts.context = @intCast(prompt_len + steps);
    const plan = budget.plan(m.cfg, m.residentBytes(), bopts);
    if (!plan.fits) {
        try err.print("benchmark: {s}\n", .{plan.reason});
        return error.ContextDoesNotFit;
    }
    const cap: usize = if (opts.expert_cap != 0) opts.expert_cap else plan.expert_cap;

    const ids = try gpa.alloc(i64, prompt_len);
    defer gpa.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast((i + 1) % m.cfg.vocab);

    var timers = model_mod.Timers.init(io);
    var sched = model_mod.Scheduler.init(gpa, 4096);
    defer sched.deinit();
    var predictor = try model_mod.Predictor.init(gpa, m.cfg.layers, m.cfg.topk);
    defer predictor.deinit();
    const fopts: model_mod.Opts = .{ .io = io, .scheduler = &sched, .predictor = &predictor, .timers = &timers };

    const load_start = std.Io.Timestamp.now(io, .awake);
    var model = try model_mod.Model.load(gpa, &w);
    defer model.deinit();
    const load_ns: u64 = @intCast(load_start.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds);

    var state = try model_mod.State.init(gpa, &model, plan.context, cap, io);
    defer state.deinit();
    var sc = try model_mod.Scratch.init(gpa, &model, @max(prompt_len, 1), plan.context);
    defer sc.deinit();
    const logits = try gpa.alloc(f32, m.cfg.vocab);
    defer gpa.free(logits);

    // prefill → time to first token
    const pf_start = std.Io.Timestamp.now(io, .awake);
    try model_mod.forward(&model, &state, &sc, ids, logits, fopts);
    const ttft_ns: u64 = @intCast(pf_start.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds);

    // decode
    var one: [1]i64 = undefined;
    var generated: usize = 0;
    const dec_start = std.Io.Timestamp.now(io, .awake);
    while (generated < steps) : (generated += 1) {
        var best: usize = 0;
        for (logits, 0..) |v, i| {
            if (v > logits[best]) best = i;
        }
        if (@as(i64, @intCast(best)) == m.cfg.eos_id) break;
        one[0] = @intCast(best);
        try model_mod.forward(&model, &state, &sc, one[0..1], logits, fopts);
    }
    const dec_ns: u64 = @intCast(dec_start.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds);
    const tok_s: f64 = if (dec_ns == 0 or generated == 0) 0 else @as(f64, @floatFromInt(generated)) * 1e9 / @as(f64, @floatFromInt(dec_ns));

    // aggregate cache / scheduler stats
    var er: u64 = 0;
    var eh: u64 = 0;
    var edemand: u64 = 0;
    var edemand_ns: u64 = 0;
    var epf_hits: u64 = 0;
    for (state.experts) |ec| {
        er += ec.stats.hits + ec.stats.misses;
        eh += ec.stats.hits;
        edemand += ec.stats.demand_loads;
        edemand_ns += ec.stats.demand_ns;
        epf_hits += ec.stats.prefetch_hits;
    }
    const hit_rate: f64 = if (er == 0) 0 else 100.0 * @as(f64, @floatFromInt(eh)) / @as(f64, @floatFromInt(er));

    try out.print(
        \\=== QWEN38 RUNTIME ===
        \\
        \\Model:
        \\  layers:            {d}  ({d} GDN / {d} QSA)
        \\  experts:           {d} routed, top-{d} + shared
        \\  load time:         {d:.1} ms
        \\
        \\Tokens:
        \\  prompt:            {d}
        \\  generated:         {d}
        \\  TTFT:              {d:.2} ms
        \\  decode tok/s:      {d:.3}
        \\
        \\Memory:
        \\  resident weights:  {f}
        \\  plan target:       {f}   (limit {f}, expert cache cap {d}/layer)
        \\  tracked peak:      {f}
        \\
        \\Experts:
        \\  requests:          {d}
        \\  hits:              {d}   ({d:.1}%)
        \\  demand loads:      {d}   ({d:.2} ms  ← compute_stall_due_to_io)
        \\  prefetch hits:     {d}
        \\
        \\I/O scheduler:
        \\  serviced HIGH/MED/LOW:  {d} / {d} / {d}   (queue peak {d})
        \\  predictor accuracy:     {d:.1}%
        \\
        \\PLE:
        \\  table shards:      {d}   ({d} rows, {d} reads x {d} B/token)
        \\
        \\QSA:  context {d}   GDN:  recurrent layers {d}
        \\
    , .{
        m.cfg.layers,                m.cfg.numGdnLayers(),                      m.cfg.numAttnLayers(),
        m.cfg.experts,               m.cfg.topk,                                @as(f64, @floatFromInt(load_ns)) / 1e6,
        prompt_len,                  generated,                                 @as(f64, @floatFromInt(ttft_ns)) / 1e6,
        tok_s,                       h(m.residentBytes()),                      h(plan.total_resident),
        h(plan.ram_budget),          plan.expert_cap,                           h(meter.peak),
        er,                          eh,                                        hit_rate,
        edemand,                     @as(f64, @floatFromInt(edemand_ns)) / 1e6, epf_hits,
        sched.stats.serviced[0],     sched.stats.serviced[1],                   sched.stats.serviced[2],
        sched.stats.queue_peak,      100.0 * predictor.stats.accuracy(),        model.ple_table.shards.len,
        model.ple_table.totalRows(), m.cfg.ngram_heads,                         m.cfg.ngram_head_dim,
        plan.context,                m.cfg.numGdnLayers(),
    });

    try timers.print(out);
    gpu.statsLine(out);
}
