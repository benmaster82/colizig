//! End-to-end Qwen3-MoE forward (e.g. Qwen3-30B-A3B).
//!
//! A plain pre-norm transformer: for each of the `L` layers,
//!   h += attn(rmsnorm(h, input_layernorm))
//!   h += moe (rmsnorm(h, post_attention_layernorm))
//! then `rmsnorm(h, model.norm)` and the LM head. No PLE / GDN / hyper-
//! connections / shared expert — the MoE and FP8 kernels are shared with the
//! Qwen4-Exp path (`src/qwen38/moe.zig`); the attention is `qwen3moe/attn.zig`.

const std = @import("std");
const Cfg = @import("../model/config.zig").Cfg;
const Weights = @import("../model/weights.zig").Weights;
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");
const attn = @import("attn.zig");
const moe = @import("../qwen38/moe.zig");
const rms = @import("../ops/rmsnorm.zig").rms;
const usage_mod = @import("../runtime/expert_usage.zig");
const timers_mod = @import("../runtime/timers.zig");

pub const ExpertUsage = usage_mod.ExpertUsage;
pub const Timers = timers_mod.Timers;

pub const Opts = struct {
    io: ?std.Io = null,
    timers: ?*Timers = null,
    usage: ?*ExpertUsage = null,
};

pub const ForwardError = error{ TokenOutOfVocab, ContextExhausted, BadShape, TooManyTokens };

pub const Model = struct {
    weights: *Weights,
    cfg: Cfg,
    attn_dims: attn.Dims,
    moe_dims: moe.Dims,

    attn_layers: []attn.Weights,
    moe_layers: []moe.Layer,
    in_norm: [][]f32, // input_layernorm[layer]  [hidden]
    post_norm: [][]f32, // post_attention_layernorm[layer]  [hidden]
    final_norm: []f32, // model.norm  [hidden]
    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, weights: *Weights) !Model {
        const cfg = weights.manifest.cfg;
        std.debug.assert(cfg.arch == .qwen3_moe);
        const L = cfg.layers;
        const H = cfg.hidden;

        var self: Model = .{
            .weights = weights,
            .cfg = cfg,
            .attn_dims = attn.Dims.of(cfg),
            .moe_dims = moe.Dims.of(cfg),
            .attn_layers = try gpa.alloc(attn.Weights, L),
            .moe_layers = try gpa.alloc(moe.Layer, L),
            .in_norm = try gpa.alloc([]f32, L),
            .post_norm = try gpa.alloc([]f32, L),
            .final_norm = &.{},
            .allocator = gpa,
        };
        var made: usize = 0;
        errdefer {
            for (0..made) |i| {
                self.attn_layers[i].deinit();
                self.moe_layers[i].deinit();
                gpa.free(self.in_norm[i]);
                gpa.free(self.post_norm[i]);
            }
            if (self.final_norm.len != 0) gpa.free(self.final_norm);
            gpa.free(self.attn_layers);
            gpa.free(self.moe_layers);
            gpa.free(self.in_norm);
            gpa.free(self.post_norm);
        }

        var nb: [128]u8 = undefined;
        var sb: [96]u8 = undefined;
        for (0..L) |i| {
            const li: u32 = @intCast(i);
            self.attn_layers[i] = try attn.Weights.load(gpa, weights, li, self.attn_dims);
            self.moe_layers[i] = try moe.Layer.load(gpa, weights, li, self.moe_dims);
            self.in_norm[i] = try weights.vectorBySuffix(nm(&sb, li, "input_layernorm.weight"), &nb);
            self.post_norm[i] = try weights.vectorBySuffix(nm(&sb, li, "post_attention_layernorm.weight"), &nb);
            if (self.in_norm[i].len != H or self.post_norm[i].len != H) return error.BadShape;
            made = i + 1;
        }
        self.final_norm = try weights.vectorBySuffix("norm.weight", &nb);
        if (self.final_norm.len != H) return error.BadShape;
        return self;
    }

    pub fn deinit(self: *Model) void {
        const gpa = self.allocator;
        for (self.attn_layers) |*a| a.deinit();
        for (self.moe_layers) |*m| m.deinit();
        for (self.in_norm) |v| gpa.free(v);
        for (self.post_norm) |v| gpa.free(v);
        gpa.free(self.final_norm);
        gpa.free(self.attn_layers);
        gpa.free(self.moe_layers);
        gpa.free(self.in_norm);
        gpa.free(self.post_norm);
        self.* = undefined;
    }
};

fn nm(buf: []u8, layer: u32, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ layer, suffix }) catch unreachable;
}

pub const State = struct {
    kv: []attn.Cache,
    experts: []moe.ExpertCache,
    pos: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, model: *const Model, max_context: usize, expert_cap: usize, io: ?std.Io) !State {
        const L = model.cfg.layers;
        var self: State = .{
            .kv = try gpa.alloc(attn.Cache, L),
            .experts = try gpa.alloc(moe.ExpertCache, L),
            .allocator = gpa,
        };
        var made: usize = 0;
        errdefer {
            for (0..made) |i| {
                self.kv[i].deinit();
                self.experts[i].deinit();
            }
            gpa.free(self.kv);
            gpa.free(self.experts);
        }
        for (0..L) |i| {
            self.kv[i] = try attn.Cache.init(gpa, model.attn_dims, max_context);
            self.experts[i] = try moe.ExpertCache.init(gpa, @intCast(i), expert_cap);
            self.experts[i].io = io;
            made = i + 1;
        }
        return self;
    }

    pub fn reset(self: *State) void {
        self.pos = 0;
        for (self.kv) |*c| c.reset();
        // expert caches are content, not sequence state — leave them warm
    }

    pub fn deinit(self: *State) void {
        for (self.kv) |*c| c.deinit();
        for (self.experts) |*e| e.deinit();
        self.allocator.free(self.kv);
        self.allocator.free(self.experts);
        self.* = undefined;
    }
};

pub const Scratch = struct {
    max_tokens: usize,
    h: []f32, // [max_tokens * hidden]  running residual
    n: []f32, // [max_tokens * hidden]  normalized input to a sublayer
    b: []f32, // [max_tokens * hidden]  sublayer output
    an: attn.Scratch,
    mo: moe.Scratch,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, model: *const Model, max_tokens: usize, max_context: usize) !Scratch {
        const H = model.cfg.hidden;
        return .{
            .max_tokens = max_tokens,
            .h = try gpa.alloc(f32, max_tokens * H),
            .n = try gpa.alloc(f32, max_tokens * H),
            .b = try gpa.alloc(f32, max_tokens * H),
            .an = try attn.Scratch.init(gpa, model.attn_dims, max_tokens, max_context),
            .mo = try moe.Scratch.init(gpa, model.moe_dims, max_tokens),
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *Scratch) void {
        self.allocator.free(self.h);
        self.allocator.free(self.n);
        self.allocator.free(self.b);
        self.an.deinit();
        self.mo.deinit();
        self.* = undefined;
    }
};

/// `ids` in, `logits` (`[vocab]`) for the LAST token out. Appends `ids.len`
/// tokens to `state` at positions `[state.pos, state.pos + ids.len)`.
pub fn forward(
    model: *const Model,
    state: *State,
    sc: *Scratch,
    ids: []const i64,
    logits: []f32,
    opts: Opts,
) !void {
    const c = model.cfg;
    const H = c.hidden;
    const S = ids.len;
    if (S == 0) return;
    if (S > sc.max_tokens) return error.TooManyTokens;
    if (logits.len != c.vocab) return error.BadShape;
    if (state.pos + S > state.kv[0].cap) return error.ContextExhausted;

    for (ids) |id| if (id < 0 or id >= c.vocab) return error.TokenOutOfVocab;

    const T = opts.timers;
    const h = sc.h[0 .. S * H];
    const n = sc.n[0 .. S * H];
    const b = sc.b[0 .. S * H];

    for (0..S) |s| try model.weights.embed(@intCast(ids[s]), h[s * H ..][0..H]);

    for (0..c.layers) |i| {
        // attention sublayer
        const ta = if (T) |t| t.now() else undefined;
        for (0..S) |s| rms(n[s * H ..][0..H], h[s * H ..][0..H], model.in_norm[i], c.eps);
        try attn.forward(&model.attn_layers[i], &state.kv[i], model.attn_dims, n, S, b, &sc.an);
        for (h, b) |*hv, bv| hv.* += bv;
        if (T) |t| t.add(.qsa, ta);

        // MoE sublayer
        const tm = if (T) |t| t.now() else undefined;
        for (0..S) |s| rms(n[s * H ..][0..H], h[s * H ..][0..H], model.post_norm[i], c.eps);
        const usage_row: ?[]u64 = if (opts.usage) |u| u.row(@intCast(i)) else null;
        try moe.forward(&model.moe_layers[i], &state.experts[i], model.weights, model.moe_dims, n, S, b, &sc.mo, null, usage_row);
        for (h, b) |*hv, bv| hv.* += bv;
        if (T) |t| t.add(.moe, tm);
    }

    state.pos += S;
    const tl = if (T) |t| t.now() else undefined;
    const last = h[(S - 1) * H ..][0..H];
    rms(last, last, model.final_norm, c.eps);
    try model.weights.lmHead(last, logits);
    if (T) |t| {
        t.add(.lm_head, tl);
        t.forwards += 1;
    }
}

/// Greedy decode: run `prompt`, then feed back argmax up to `out.len` times.
/// Returns the number of tokens written; stops early on EOS.
pub fn generateGreedy(
    model: *const Model,
    state: *State,
    sc: *Scratch,
    prompt: []const i64,
    out: []i64,
    opts: Opts,
) !usize {
    const V = model.cfg.vocab;
    const logits = try model.allocator.alloc(f32, V);
    defer model.allocator.free(logits);

    try forward(model, state, sc, prompt, logits, opts);
    var written: usize = 0;
    while (written < out.len) {
        var best: usize = 0;
        for (logits, 0..) |v, k| if (v > logits[best]) {
            best = k;
        };
        out[written] = @intCast(best);
        written += 1;
        if (@as(i64, @intCast(best)) == model.cfg.eos_id) break;
        var one = [_]i64{@intCast(best)};
        try forward(model, state, sc, &one, logits, opts);
    }
    return written;
}

// ---- tests ----------------------------------------------------------------

const testing = std.testing;

test "qwen3_moe forward on the tiny fixture: finite, in-vocab, prefill == incremental decode" {
    const gpa = testing.allocator;
    const io = std.testing.io;
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);

    var m = manifest_mod.open(gpa, io, "test/fixtures/tiny-qwen3", &sink.writer) catch |e| switch (e) {
        error.OpenFailed, error.NoCheckpoint => return error.SkipZigTest,
        else => return e,
    };
    defer m.deinit();
    var w = try weights_mod.Weights.open(gpa, io, "test/fixtures/tiny-qwen3", &m, &sink.writer);
    defer w.deinit();

    var model = try Model.load(gpa, &w);
    defer model.deinit();

    const V = model.cfg.vocab;
    const T = 6;
    const ctx = 16;
    var prng = std.Random.DefaultPrng.init(0x3E2E);
    const rnd = prng.random();
    const ids = try gpa.alloc(i64, T);
    defer gpa.free(ids);
    for (ids) |*v| v.* = rnd.intRangeAtMost(i64, 0, @as(i64, @intCast(V - 1)));

    var sc = try Scratch.init(gpa, &model, T, ctx);
    defer sc.deinit();
    const l1 = try gpa.alloc(f32, V);
    defer gpa.free(l1);
    const l2 = try gpa.alloc(f32, V);
    defer gpa.free(l2);

    var st1 = try State.init(gpa, &model, ctx, model.cfg.experts, io);
    defer st1.deinit();
    try forward(&model, &st1, &sc, ids, l1, .{});
    for (l1) |v| try testing.expect(std.math.isFinite(v));

    var st2 = try State.init(gpa, &model, ctx, model.cfg.experts, io);
    defer st2.deinit();
    for (ids) |id| {
        var one = [_]i64{id};
        try forward(&model, &st2, &sc, &one, l2, .{});
    }
    for (l1, l2) |a, b| try testing.expectApproxEqAbs(a, b, 2e-3);

    var st3 = try State.init(gpa, &model, ctx, model.cfg.experts, io);
    defer st3.deinit();
    const out = try gpa.alloc(i64, 4);
    defer gpa.free(out);
    const n = try generateGreedy(&model, &st3, &sc, ids[0..3], out, .{});
    try testing.expect(n >= 1 and n <= 4);
    for (out[0..n]) |o| try testing.expect(o >= 0 and o < V);

    var one_bad = [_]i64{@intCast(V)};
    try testing.expectError(error.TokenOutOfVocab, forward(&model, &st1, &sc, &one_bad, l1, .{}));
}
