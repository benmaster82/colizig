//! Gated residual ("hyper connections") - the 4-branch cross-layer read/write
//! mixer wrapping every attention and MoE sub-block.  Ported from colibri's
//! `q38_load_gr` / `q38_gr_read` / `q38_gr_apply`.
//!
//! The residual state `hyper` is `[S, hc_width]` = `[S, hc_count · hidden]`.
//! `read` collapses the 4 branches to a single `[S, hidden]` block input;
//! `apply` scatters the block output back across the branches through per-branch
//! injection gates.  The final mixer is a `read` with no injection.

const std = @import("std");
const Cfg = @import("../model/config.zig").Cfg;
const Weights = @import("../model/weights.zig").Weights;
const NativeMatrix = @import("../model/tensors.zig").NativeMatrix;
const act = @import("../ops/activation.zig");
const rms = @import("../ops/rmsnorm.zig");

pub const Dims = struct {
    hidden: usize,
    hc_count: usize,
    hc_rank: usize,
    hc_width: usize, // == hc_count * hidden
    eps: f32,

    pub fn of(cfg: Cfg) Dims {
        return .{
            .hidden = cfg.hidden,
            .hc_count = cfg.hc_count,
            .hc_rank = cfg.hc_rank,
            .hc_width = cfg.hc_width,
            .eps = cfg.eps,
        };
    }
};

pub const Kind = enum { attn, mlp, final };

pub const Gated = struct {
    norm: []f32, // [hc_width]
    down: NativeMatrix, // [hc_rank, hc_width]
    up: NativeMatrix, // [hc_width, hc_rank]
    inject: ?NativeMatrix, // [hc_count, hc_width]  (null for the final mixer)
    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, w: *const Weights, layer: ?u32, kind: Kind) !Gated {
        var nb: [160]u8 = undefined;
        var sb: [128]u8 = undefined;
        const base = switch (kind) {
            .attn => std.fmt.bufPrint(&sb, "layers.{d}.attn_hyper_connection", .{layer.?}) catch unreachable,
            .mlp => std.fmt.bufPrint(&sb, "layers.{d}.mlp_hyper_connection", .{layer.?}) catch unreachable,
            .final => "hyper_connection_mixer",
        };
        var suf: [200]u8 = undefined;

        var self: Gated = undefined;
        self.allocator = gpa;
        self.norm = try w.vectorBySuffix(cat(&suf, base, "hc_norm.weight"), &nb);
        errdefer gpa.free(self.norm);
        self.down = try w.matrixBySuffix(cat(&suf, base, "input_mix_weight_down.weight"), &nb);
        errdefer self.down.deinit();
        self.up = try w.matrixBySuffix(cat(&suf, base, "input_mix_weight_up.weight"), &nb);
        errdefer self.up.deinit();
        self.inject = if (kind == .final)
            null
        else
            try w.matrixBySuffix(cat(&suf, base, "block_inject_weight.weight"), &nb);
        return self;
    }

    pub fn deinit(self: *Gated) void {
        self.allocator.free(self.norm);
        self.down.deinit();
        self.up.deinit();
        if (self.inject) |*i| i.deinit();
        self.* = undefined;
    }
};

fn cat(buf: []u8, base: []const u8, leaf: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ base, leaf }) catch unreachable;
}

pub const Scratch = struct {
    norm: []f32, // [max_tokens * hc_width]
    low: []f32, // [max_tokens * hc_rank]
    mix: []f32, // [max_tokens * hc_width]
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: Dims, max_tokens: usize) !Scratch {
        return .{
            .norm = try gpa.alloc(f32, max_tokens * d.hc_width),
            .low = try gpa.alloc(f32, max_tokens * d.hc_rank),
            .mix = try gpa.alloc(f32, max_tokens * d.hc_width),
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *Scratch) void {
        self.allocator.free(self.norm);
        self.allocator.free(self.low);
        self.allocator.free(self.mix);
        self.* = undefined;
    }
};

/// Read the 4-branch residual into `mixed` (`[S, hidden]`); when `inject` is
/// given and this mixer has an inject weight, also fill `inject` (`[S, hc_count]`).
pub fn read(
    g: *const Gated,
    d: Dims,
    hyper: []const f32,
    S: usize,
    mixed: []f32,
    inject: ?[]f32,
    sc: *Scratch,
) void {
    const H = d.hidden;
    const C = d.hc_count;
    const R = d.hc_rank;
    const W = d.hc_width;
    const Cf: f32 = @floatFromInt(C);
    std.debug.assert(hyper.len == S * W and mixed.len == S * H);

    for (0..S) |s| {
        for (0..C) |b| {
            const off = s * W + b * H;
            rms.rms0(sc.norm[off..][0..H], hyper[off..][0..H], g.norm[b * H ..][0..H], d.eps);
        }
    }

    g.down.matmul(sc.low[0 .. S * R], sc.norm[0 .. S * W], S);
    for (sc.low[0 .. S * R]) |*z| z.* = act.silu(z.* / Cf);
    g.up.matmul(sc.mix[0 .. S * W], sc.low[0 .. S * R], S);

    for (0..S) |s| {
        for (0..H) |dd| {
            var v: f32 = 0;
            for (0..C) |b| {
                const o = s * W + b * H + dd;
                v += act.sigmoid(sc.mix[o]) * sc.norm[o];
            }
            mixed[s * H + dd] = v / Cf;
        }
    }

    if (inject) |inj| {
        if (g.inject) |gi| {
            std.debug.assert(inj.len == S * C);
            gi.matmul(inj, sc.norm[0 .. S * W], S);
            for (inj) |*z| z.* = 2.0 * act.sigmoid(z.* / Cf);
        }
    }
}

/// Scatter a block output `[S, hidden]` back into `hyper` through the per-branch
/// injection gates `[S, hc_count]`.
pub fn apply(d: Dims, hyper: []f32, block: []const f32, inject: []const f32, S: usize) void {
    const H = d.hidden;
    const C = d.hc_count;
    const W = d.hc_width;
    std.debug.assert(hyper.len == S * W and block.len == S * H and inject.len == S * C);
    for (0..S) |s| {
        for (0..C) |b| {
            const a = inject[s * C + b];
            const dst = hyper[s * W + b * H ..][0..H];
            const src = block[s * H ..][0..H];
            for (0..H) |dd| dst[dd] += a * src[dd];
        }
    }
}

// ---- tests -----------------------------------------------------------

const testing = std.testing;
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");

test "gated residual read/apply on the tiny fixture: finite, chunk-invariant" {
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

    const d = Dims.of(m.cfg);
    var attn = try Gated.load(gpa, &w, 0, .attn);
    defer attn.deinit();
    var fin = try Gated.load(gpa, &w, null, .final);
    defer fin.deinit();
    try testing.expect(attn.inject != null and fin.inject == null);

    const T = 4;
    const W = d.hc_width;
    const H = d.hidden;
    const C = d.hc_count;

    var prng = std.Random.DefaultPrng.init(0x6142);
    const r = prng.random();
    const hyper0 = try gpa.alloc(f32, T * W);
    defer gpa.free(hyper0);
    for (hyper0) |*v| v.* = r.float(f32) * 2 - 1;
    const block = try gpa.alloc(f32, T * H);
    defer gpa.free(block);
    for (block) |*v| v.* = r.float(f32) * 2 - 1;

    var sc = try Scratch.init(gpa, d, T);
    defer sc.deinit();

    const mixed = try gpa.alloc(f32, T * H);
    defer gpa.free(mixed);
    const inject = try gpa.alloc(f32, T * C);
    defer gpa.free(inject);

    const hyper = try gpa.dupe(f32, hyper0);
    defer gpa.free(hyper);
    read(&attn, d, hyper, T, mixed, inject, &sc);
    for (mixed) |v| try testing.expect(std.math.isFinite(v));
    for (inject) |v| try testing.expect(v >= 0 and v <= 2); // 2·sigmoid ∈ [0,2]
    apply(d, hyper, block, inject, T);
    for (hyper) |v| try testing.expect(std.math.isFinite(v));

    // process token by token → identical
    const hyper_inc = try gpa.dupe(f32, hyper0);
    defer gpa.free(hyper_inc);
    for (0..T) |s| {
        read(&attn, d, hyper_inc[s * W ..][0..W], 1, mixed[s * H ..][0..H], inject[s * C ..][0..C], &sc);
        apply(d, hyper_inc[s * W ..][0..W], block[s * H ..][0..H], inject[s * C ..][0..C], 1);
    }
    for (hyper, hyper_inc) |a, b| try testing.expectApproxEqRel(a, b, 1e-5);
}
