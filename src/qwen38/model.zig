//! End-to-end Qwen4-Exp forward - wires the embedding, the 48 layers (each a
//! gated-residual-wrapped attention/DeltaNet block and MoE block), the PLE
//! injection at layer 1, the final mixer and the LM head.  Ported from
//! colibri's `step`.
//!
//! Phase 7b: PLE row reads fan out concurrently (`std.Io.Group`) when an `Io`
//! is supplied; a `Scheduler` + `LastTokenPredictor` drive per-layer expert
//! prefetch (cooperative execution - the loads still run on the compute thread,
//! but the queue policy is real and the prefetch hit rate is measured).  No
//! tokenizer - `forward` takes raw token ids.

const std = @import("std");
const Cfg = @import("../model/config.zig").Cfg;
const Weights = @import("../model/weights.zig").Weights;
const gdn = @import("gdn.zig");
const qsa = @import("qsa.zig");
const moe = @import("moe.zig");
const ple = @import("ple.zig");
const residual = @import("residual.zig");
const io_sched = @import("../runtime/io.zig");
const predict = @import("../runtime/predict.zig");
const timers_mod = @import("../runtime/timers.zig");
const usage_mod = @import("../runtime/expert_usage.zig");

pub const Scheduler = io_sched.Scheduler;
pub const Predictor = predict.LastTokenPredictor;
pub const Timers = timers_mod.Timers;
pub const ExpertUsage = usage_mod.ExpertUsage;

/// Optional I/O-scheduling / instrumentation knobs for `forward`.
pub const Opts = struct {
    /// When set, PLE row reads fan out concurrently.
    io: ?std.Io = null,
    /// When both are set, predicted experts are prefetched per layer.
    scheduler: ?*Scheduler = null,
    predictor: ?*Predictor = null,
    /// When set, per-phase wall-clock time is accumulated.
    timers: ?*Timers = null,
    /// When set, per-layer routed-expert selections are counted into it (for
    /// the learned-prior cache warming next run).
    usage: ?*ExpertUsage = null,
};

pub const ForwardError = error{ TokenOutOfVocab, ContextExhausted, BadShape, TooManyTokens };

pub const Model = struct {
    weights: *Weights,
    cfg: Cfg,

    gr_dims: residual.Dims,
    gdn_dims: gdn.Dims,
    qsa_dims: qsa.Dims,
    moe_dims: moe.Dims,
    ple_dims: ple.Dims,

    // per-layer resident weights (length == cfg.layers)
    attn_gr: []residual.Gated,
    mlp_gr: []residual.Gated,
    moe_layers: []moe.Layer,
    gdn_layers: []?gdn.Layer, // non-null iff !is_attn[i]
    qsa_layers: []?qsa.Layer, // non-null iff  is_attn[i]
    final_gr: residual.Gated,

    ple_layer: ple.Layer,
    ple_table: ple.Table,

    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, weights: *Weights) !Model {
        const cfg = weights.manifest.cfg;
        const L = cfg.layers;

        var self: Model = undefined;
        self.allocator = gpa;
        self.weights = weights;
        self.cfg = cfg;
        self.gr_dims = residual.Dims.of(cfg);
        self.gdn_dims = gdn.Dims.of(cfg);
        self.qsa_dims = qsa.Dims.of(cfg);
        self.moe_dims = moe.Dims.of(cfg);
        self.ple_dims = ple.Dims.of(cfg);

        self.attn_gr = try gpa.alloc(residual.Gated, L);
        self.mlp_gr = try gpa.alloc(residual.Gated, L);
        self.moe_layers = try gpa.alloc(moe.Layer, L);
        self.gdn_layers = try gpa.alloc(?gdn.Layer, L);
        self.qsa_layers = try gpa.alloc(?qsa.Layer, L);
        @memset(self.gdn_layers, null);
        @memset(self.qsa_layers, null);

        var loaded: usize = 0;
        errdefer freeLayers(&self, loaded);

        for (0..L) |i| {
            const li: u32 = @intCast(i);
            self.attn_gr[i] = try residual.Gated.load(gpa, weights, li, .attn);
            self.mlp_gr[i] = try residual.Gated.load(gpa, weights, li, .mlp);
            self.moe_layers[i] = try moe.Layer.load(gpa, weights, li, moe.Dims.of(cfg));
            if (cfg.is_attn[i]) {
                self.qsa_layers[i] = try qsa.Layer.load(gpa, weights, li);
            } else {
                self.gdn_layers[i] = try gdn.Layer.load(gpa, weights, li);
            }
            loaded = i + 1;
        }

        self.final_gr = try residual.Gated.load(gpa, weights, null, .final);
        errdefer self.final_gr.deinit();
        self.ple_layer = try ple.Layer.load(gpa, weights, cfg.ple_layer);
        errdefer self.ple_layer.deinit();
        self.ple_table = try ple.Table.load(gpa, weights, cfg);

        return self;
    }

    fn freeLayers(self: *Model, upto: usize) void {
        const gpa = self.allocator;
        for (0..upto) |i| {
            self.attn_gr[i].deinit();
            self.mlp_gr[i].deinit();
            self.moe_layers[i].deinit();
            if (self.gdn_layers[i]) |*g| g.deinit();
            if (self.qsa_layers[i]) |*q| q.deinit();
        }
        gpa.free(self.attn_gr);
        gpa.free(self.mlp_gr);
        gpa.free(self.moe_layers);
        gpa.free(self.gdn_layers);
        gpa.free(self.qsa_layers);
    }

    pub fn deinit(self: *Model) void {
        self.ple_table.deinit();
        self.ple_layer.deinit();
        self.final_gr.deinit();
        freeLayers(self, self.cfg.layers);
        self.* = undefined;
    }
};

/// Per-sequence recurrent + cache state.
pub const State = struct {
    gdn: []?gdn.GdnState,
    qsa: []?qsa.Cache,
    ple_state: ple.State,
    experts: []moe.ExpertCache,
    pos: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, model: *const Model, max_context: usize, expert_cap: usize, io: ?std.Io) !State {
        const L = model.cfg.layers;
        var self: State = undefined;
        self.allocator = gpa;
        self.pos = 0;
        self.gdn = try gpa.alloc(?gdn.GdnState, L);
        self.qsa = try gpa.alloc(?qsa.Cache, L);
        self.experts = try gpa.alloc(moe.ExpertCache, L);
        @memset(self.gdn, null);
        @memset(self.qsa, null);

        var made: usize = 0;
        errdefer {
            for (0..made) |i| {
                if (self.gdn[i]) |*g| g.deinit();
                if (self.qsa[i]) |*q| q.deinit();
                self.experts[i].deinit();
            }
            gpa.free(self.gdn);
            gpa.free(self.qsa);
            gpa.free(self.experts);
        }
        for (0..L) |i| {
            self.experts[i] = try moe.ExpertCache.init(gpa, @intCast(i), expert_cap);
            self.experts[i].io = io;
            if (model.cfg.is_attn[i]) {
                self.qsa[i] = try qsa.Cache.init(gpa, model.qsa_dims, max_context);
            } else {
                self.gdn[i] = try gdn.GdnState.init(gpa, model.gdn_dims);
            }
            made = i + 1;
        }
        self.ple_state = try ple.State.init(gpa, model.ple_dims);
        return self;
    }

    pub fn reset(self: *State) void {
        self.pos = 0;
        for (self.gdn) |*g| if (g.*) |*gs| gs.reset();
        for (self.qsa) |*q| if (q.*) |*qc| qc.reset();
        self.ple_state.reset();
        // expert caches are content, not sequence state - leave them warm
    }

    pub fn deinit(self: *State) void {
        const gpa = self.allocator;
        for (self.gdn) |*g| if (g.*) |*gs| gs.deinit();
        for (self.qsa) |*q| if (q.*) |*qc| qc.deinit();
        for (self.experts) |*e| e.deinit();
        self.ple_state.deinit();
        gpa.free(self.gdn);
        gpa.free(self.qsa);
        gpa.free(self.experts);
        self.* = undefined;
    }
};

pub const Scratch = struct {
    max_tokens: usize,
    hyper: []f32, // [max_tokens * hc_width]
    mixed: []f32, // [max_tokens * hidden]
    block: []f32, // [max_tokens * hidden]
    inject: []f32, // [max_tokens * hc_count]
    ple_out: []f32, // [max_tokens * hc_width]
    routed: []u32, // [topk] - last token's routed experts, for the predictor
    gr: residual.Scratch,
    gd: gdn.Scratch,
    qs: qsa.Scratch,
    mo: moe.Scratch,
    pl: ple.Scratch,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, model: *const Model, max_tokens: usize, max_context: usize) !Scratch {
        const c = model.cfg;
        return .{
            .max_tokens = max_tokens,
            .hyper = try gpa.alloc(f32, max_tokens * c.hc_width),
            .mixed = try gpa.alloc(f32, max_tokens * c.hidden),
            .block = try gpa.alloc(f32, max_tokens * c.hidden),
            .inject = try gpa.alloc(f32, max_tokens * c.hc_count),
            .ple_out = try gpa.alloc(f32, max_tokens * c.hc_width),
            .routed = try gpa.alloc(u32, c.topk),
            .gr = try residual.Scratch.init(gpa, model.gr_dims, max_tokens),
            .gd = try gdn.Scratch.init(gpa, model.gdn_dims, max_tokens),
            .qs = try qsa.Scratch.init(gpa, model.qsa_dims, max_tokens, max_context),
            .mo = try moe.Scratch.init(gpa, model.moe_dims, max_tokens),
            .pl = try ple.Scratch.init(gpa, model.ple_dims),
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *Scratch) void {
        const gpa = self.allocator;
        gpa.free(self.hyper);
        gpa.free(self.mixed);
        gpa.free(self.block);
        gpa.free(self.inject);
        gpa.free(self.ple_out);
        gpa.free(self.routed);
        self.gr.deinit();
        self.gd.deinit();
        self.qs.deinit();
        self.mo.deinit();
        self.pl.deinit();
        self.* = undefined;
    }
};

/// Run the forward for `ids` (raw token ids), advancing `state`.  Writes the
/// logits for the LAST token into `logits` (`logits.len == vocab`).  `opts`
/// enables PLE / expert prefetch; the produced logits are identical regardless.
pub fn forward(model: *Model, state: *State, sc: *Scratch, ids: []const i64, logits: []f32, opts: Opts) !void {
    const c = model.cfg;
    const H = c.hidden;
    const W = c.hc_width;
    const C = c.hc_count;
    const S = ids.len;
    if (S == 0 or S > sc.max_tokens) return error.TooManyTokens;
    if (logits.len != c.vocab) return error.BadShape;

    const pos_base = state.pos;

    for (ids) |t| {
        if (t < 0 or t >= @as(i64, @intCast(c.vocab))) return error.TokenOutOfVocab;
    }
    for (state.qsa) |*q| {
        if (q.*) |qc| {
            if (pos_base + S > qc.cap) return error.ContextExhausted;
        }
    }

    // PLE row reads are deterministic (pure function of the token ids) - start
    // them now, before layer 0, so they overlap the layers that precede the
    // injection point.
    var ple_pf: ?[]f32 = null;
    defer if (ple_pf) |p| sc.allocator.free(p);
    if (c.ple_layer < c.layers) {
        ple_pf = try ple.prefetchRowsAsync(sc.allocator, model.ple_table, model.ple_dims, ids, S, state.ple_state, opts.io);
    }

    // NOTE: an async "prewarm the predicted experts' E4M3 pages N layers ahead
    // on std.Io workers" experiment was tried here and removed - it did not beat
    // the baseline on this hardware.  The MoE matmul already fans over
    // `parallel.chunks` workers, so when one worker stalls on a cold expert page
    // the others keep computing; the demand path itself waits < 20 ms/forward on
    // I/O.  Explicit prewarm only added nested-Group scheduling overhead.

    // embedding, replicated across the hc_count branches
    {
        const t0 = tstart(opts);
        for (ids, 0..) |t, s| {
            const row = sc.hyper[s * W ..][0..H];
            try model.weights.embed(@intCast(t), row);
            for (1..C) |b| @memcpy(sc.hyper[s * W + b * H ..][0..H], row);
        }
        tend(opts, .embed, t0);
    }

    for (0..c.layers) |i| {
        if (i == c.ple_layer) {
            const tp = tstart(opts);
            ple.forward(
                &model.ple_layer,
                model.ple_table,
                &state.ple_state,
                model.ple_dims,
                ids,
                S,
                sc.hyper[0 .. S * W],
                sc.ple_out[0 .. S * W],
                if (ple_pf) |p| p else null,
                &sc.pl,
            );
            for (sc.hyper[0 .. S * W], sc.ple_out[0 .. S * W]) |*h, p| h.* += p;
            tend(opts, .ple, tp);
        }

        // predictive expert prefetch for this layer (from the previous token's
        // routing).  Submitted MEDIUM; serviced here so the experts' cache slots
        // and scale tables are warm by the time the MoE sub-block routes.
        if (opts.predictor != null and opts.scheduler != null) {
            const sch = opts.scheduler.?;
            for (opts.predictor.?.predict(i)) |eid| {
                _ = sch.submit(.{ .expert = .{ .layer = @intCast(i), .id = eid } }, .medium);
            }
            while (sch.next()) |key| switch (key) {
                .expert => |e| moe.ExpertCache.prefetch(&state.experts[e.layer], model.weights, model.moe_dims, e.id) catch {},
                else => {},
            };
        }

        // attention / DeltaNet sub-block
        var tg = tstart(opts);
        residual.read(&model.attn_gr[i], model.gr_dims, sc.hyper[0 .. S * W], S, sc.mixed[0 .. S * H], sc.inject[0 .. S * C], &sc.gr);
        tend(opts, .gated_residual, tg);
        const ta = tstart(opts);
        if (c.is_attn[i]) {
            qsa.forward(&model.qsa_layers[i].?, &state.qsa[i].?, model.qsa_dims, sc.mixed[0 .. S * H], S, pos_base, sc.block[0 .. S * H], &sc.qs);
            tend(opts, .qsa, ta);
        } else {
            gdn.forward(&model.gdn_layers[i].?, &state.gdn[i].?, model.gdn_dims, sc.mixed[0 .. S * H], S, sc.block[0 .. S * H], &sc.gd);
            tend(opts, .deltanet, ta);
        }
        residual.apply(model.gr_dims, sc.hyper[0 .. S * W], sc.block[0 .. S * H], sc.inject[0 .. S * C], S);

        // MoE sub-block
        tg = tstart(opts);
        residual.read(&model.mlp_gr[i], model.gr_dims, sc.hyper[0 .. S * W], S, sc.mixed[0 .. S * H], sc.inject[0 .. S * C], &sc.gr);
        tend(opts, .gated_residual, tg);
        const routed_out: ?[]u32 = if (opts.predictor != null) sc.routed else null;
        const tm0 = tstart(opts);
        const usage_row = if (opts.usage) |u| u.row(@intCast(i)) else null;
        try moe.forward(&model.moe_layers[i], &state.experts[i], model.weights, model.moe_dims, sc.mixed[0 .. S * H], S, sc.block[0 .. S * H], &sc.mo, routed_out, usage_row);
        tend(opts, .moe, tm0);
        if (opts.predictor) |pred| pred.record(i, sc.routed);
        residual.apply(model.gr_dims, sc.hyper[0 .. S * W], sc.block[0 .. S * H], sc.inject[0 .. S * C], S);
    }

    const tf = tstart(opts);
    residual.read(&model.final_gr, model.gr_dims, sc.hyper[0 .. S * W], S, sc.mixed[0 .. S * H], null, &sc.gr);
    tend(opts, .gated_residual, tf);
    const tl = tstart(opts);
    try model.weights.lmHead(sc.mixed[(S - 1) * H ..][0..H], logits);
    tend(opts, .lm_head, tl);

    state.pos = pos_base + S;
    if (opts.timers) |tm| tm.forwards += 1;
}

fn tstart(opts: Opts) ?std.Io.Timestamp {
    if (opts.timers) |tm| return tm.now();
    return null;
}
fn tend(opts: Opts, phase: timers_mod.Phase, t0: ?std.Io.Timestamp) void {
    if (opts.timers) |tm| {
        if (t0) |s| tm.add(phase, s);
    }
}

/// Greedy decode: run `prompt`, then feed back argmax up to `out.len` steps
/// (stopping early on the EOS token).  Returns the number of tokens written.
pub fn generateGreedy(
    model: *Model,
    state: *State,
    sc: *Scratch,
    prompt: []const i64,
    out: []i64,
    opts: Opts,
) !usize {
    const gpa = sc.allocator;
    const logits = try gpa.alloc(f32, model.cfg.vocab);
    defer gpa.free(logits);

    try forward(model, state, sc, prompt, logits, opts);
    var n: usize = 0;
    while (n < out.len) {
        var best: usize = 0;
        for (logits, 0..) |v, i| {
            if (v > logits[best]) best = i;
        }
        out[n] = @intCast(best);
        n += 1;
        if (@as(i64, @intCast(best)) == model.cfg.eos_id) break;
        try forward(model, state, sc, out[n - 1 .. n], logits, opts);
    }
    return n;
}

// ---- tests -----------------------------------------------------------

const testing = std.testing;
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");

test "end-to-end forward on the tiny fixture: prefill == incremental decode" {
    const gpa = testing.allocator;
    const io = std.testing.io;
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);

    var m = manifest_mod.open(gpa, io, "test/fixtures/tiny", &sink.writer) catch |e| switch (e) {
        error.OpenFailed, error.NoCheckpoint => return error.SkipZigTest,
        else => return e,
    };
    defer m.deinit();
    var w = try weights_mod.Weights.open(gpa, io, "test/fixtures/tiny", &m, &sink.writer);
    defer w.deinit();

    var model = try Model.load(gpa, &w);
    defer model.deinit();

    const V = m.cfg.vocab;
    const T = 6;
    const ctx = 16;
    var prng = std.Random.DefaultPrng.init(0xE2E);
    const rnd = prng.random();
    const ids = try gpa.alloc(i64, T);
    defer gpa.free(ids);
    for (ids) |*v| v.* = rnd.intRangeAtMost(i64, 0, @as(i64, @intCast(V - 1)));

    var sc = try Scratch.init(gpa, &model, T, ctx);
    defer sc.deinit();

    // one-shot prefill
    var st1 = try State.init(gpa, &model, ctx, m.cfg.experts, io);
    defer st1.deinit();
    const l1 = try gpa.alloc(f32, V);
    defer gpa.free(l1);
    try forward(&model, &st1, &sc, ids, l1, .{});
    for (l1) |v| try testing.expect(std.math.isFinite(v));

    // token-by-token
    var st2 = try State.init(gpa, &model, ctx, m.cfg.experts, io);
    defer st2.deinit();
    const l2 = try gpa.alloc(f32, V);
    defer gpa.free(l2);
    for (0..T) |s| try forward(&model, &st2, &sc, ids[s .. s + 1], l2, .{});

    for (l1, l2) |a, b| try testing.expectApproxEqRel(a, b, 2e-3);

    // greedy decode runs and stays in-vocab
    var st3 = try State.init(gpa, &model, ctx, m.cfg.experts, io);
    defer st3.deinit();
    const out = try gpa.alloc(i64, 4);
    defer gpa.free(out);
    const got = try generateGreedy(&model, &st3, &sc, ids[0..3], out, .{});
    try testing.expect(got >= 1 and got <= 4);
    for (out[0..got]) |t| try testing.expect(t >= 0 and t < @as(i64, @intCast(V)));

    // out-of-vocab token is rejected
    try testing.expectError(error.TokenOutOfVocab, forward(&model, &st1, &sc, &[_]i64{@intCast(V)}, l1, .{}));

    // --- Phase 7b: prefetch changes nothing but the timing ---
    // A cache too small for one token's working set forces churn, so the
    // predicted experts really do get evicted and re-loaded by prefetch.
    const tight_cap: usize = 1;
    var sched = Scheduler.init(gpa, 64);
    defer sched.deinit();
    var pred = try Predictor.init(gpa, m.cfg.layers, m.cfg.topk);
    defer pred.deinit();
    var stp = try State.init(gpa, &model, ctx, tight_cap, io);
    defer stp.deinit();
    const lp = try gpa.alloc(f32, V);
    defer gpa.free(lp);

    const opts: Opts = .{ .io = io, .scheduler = &sched, .predictor = &pred };
    const tok = [_]i64{ids[0]};
    for (0..6) |_| try forward(&model, &stp, &sc, &tok, lp, opts);

    // prefetch must not change the result - bit-identical on the same trajectory
    var stn = try State.init(gpa, &model, ctx, tight_cap, io);
    defer stn.deinit();
    const ln = try gpa.alloc(f32, V);
    defer gpa.free(ln);
    for (0..6) |_| try forward(&model, &stn, &sc, &tok, ln, .{});
    for (lp, ln) |a, b| try testing.expectEqual(a, b);

    // the scheduler serviced predicted prefetches and the predictor is scoring
    try testing.expect(sched.stats.servicedTotal() > 0);
    try testing.expect(pred.stats.predicted > 0);
    var pf_loads: u64 = 0;
    var demand_no_pf: u64 = 0;
    for (stp.experts) |ec| pf_loads += ec.stats.prefetch_loads;
    for (stn.experts) |ec| demand_no_pf += ec.stats.demand_loads;
    try testing.expect(pf_loads > 0); // experts were loaded ahead of demand
    try testing.expect(demand_no_pf > 0); // without prefetch everything is demand

    // --- Phase 9: threaded kernels produce bit-identical logits ---
    const parallel = @import("../runtime/parallel.zig");
    var sthr = try State.init(gpa, &model, ctx, m.cfg.experts, io);
    defer sthr.deinit();
    const lthr = try gpa.alloc(f32, V);
    defer gpa.free(lthr);
    parallel.enable(io, 4);
    for (0..T) |s| try forward(&model, &sthr, &sc, ids[s .. s + 1], lthr, .{});
    parallel.disable();
    // same token-by-token trajectory as l2, just with the kernels fanned out
    for (l2, lthr) |a, b| try testing.expectEqual(a, b);
}

test "matches the NumPy reference oracle (run tools/reference/build_oracle.py)" {
    const gpa = testing.allocator;
    const io = std.testing.io;
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);

    const oracle_bytes = std.Io.Dir.cwd().readFileAlloc(io, "test/fixtures/tiny/oracle.json", gpa, .limited(4 << 20)) catch
        return error.SkipZigTest;
    defer gpa.free(oracle_bytes);

    var m = manifest_mod.open(gpa, io, "test/fixtures/tiny", &sink.writer) catch |e| switch (e) {
        error.OpenFailed, error.NoCheckpoint => return error.SkipZigTest,
        else => return e,
    };
    defer m.deinit();
    var w = try weights_mod.Weights.open(gpa, io, "test/fixtures/tiny", &m, &sink.writer);
    defer w.deinit();
    var model = try Model.load(gpa, &w);
    defer model.deinit();

    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, oracle_bytes, .{});
    defer parsed.deinit();
    const cases = parsed.value.object.get("cases").?.array;

    const V = m.cfg.vocab;
    const logits = try gpa.alloc(f32, V);
    defer gpa.free(logits);
    const ids_buf = try gpa.alloc(i64, 64);
    defer gpa.free(ids_buf);

    for (cases.items) |case| {
        const co = case.object;
        const jids = co.get("token_ids").?.array;
        const jlog = co.get("final_logits").?.array;
        for (jids.items, 0..) |v, i| ids_buf[i] = v.integer;
        const ids = ids_buf[0..jids.items.len];

        const ctx = ids.len + 4;
        var st = try State.init(gpa, &model, ctx, m.cfg.experts, io);
        defer st.deinit();
        var sc = try Scratch.init(gpa, &model, ids.len, ctx);
        defer sc.deinit();
        try forward(&model, &st, &sc, ids, logits, .{});

        var max_abs: f32 = 0;
        for (jlog.items, logits) |ref_v, got| {
            const ref: f32 = switch (ref_v) {
                .float => |x| @floatCast(x),
                .integer => |x| @floatFromInt(x),
                else => unreachable,
            };
            max_abs = @max(max_abs, @abs(ref - got));
        }
        // two independent ports of the same spec, f32 throughout
        try testing.expect(max_abs < 2e-2);

        var argmax: usize = 0;
        for (logits, 0..) |v, i| {
            if (v > logits[argmax]) argmax = i;
        }
        try testing.expectEqual(@as(i64, @intCast(argmax)), co.get("argmax").?.integer);
    }
}
