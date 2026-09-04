//! Gated DeltaNet - the linear-attention path used by 36 of the 48 layers.
//!
//! Ported from colibri's `q38_deltanet`.  Pipeline per token:
//!   in_proj_{qkv,z,b,a}  →  causal depthwise conv1d (kernel CK) + SiLU
//!   →  per-head Q/K L2-normalize + scale  →  gated-delta recurrent state update
//!   →  RMSNormGated(gate = z)  →  out_proj
//!
//! The recurrent state (`GdnState`) is persistent across tokens and allocated
//! once.  Chunk boundaries do not change results: the conv ring and the
//! recurrence are strictly token-causal.
//!
//! Phase 3: single-threaded, one matmul over the whole prompt for the input
//! projections, then a token loop.  Prefill batching / threading is Phase 8.

const std = @import("std");
const Cfg = @import("../model/config.zig").Cfg;
const Weights = @import("../model/weights.zig").Weights;
const NativeMatrix = @import("../model/tensors.zig").NativeMatrix;
const act = @import("../ops/activation.zig");
const rmsnorm = @import("../ops/rmsnorm.zig");
const parallel = @import("../runtime/parallel.zig");

pub const Dims = struct {
    hidden: usize,
    kheads: usize,
    vheads: usize,
    kdim: usize,
    vdim: usize,
    convk: usize,
    conv_dim: usize,
    eps: f32,

    pub fn k(self: Dims) usize { // width of the q (and k) block in the conv output
        return self.kheads * self.kdim;
    }
    pub fn v(self: Dims) usize {
        return self.vheads * self.vdim;
    }
    pub fn rep(self: Dims) usize {
        return self.vheads / self.kheads;
    }

    pub fn of(cfg: Cfg) Dims {
        return .{
            .hidden = cfg.hidden,
            .kheads = cfg.dn_kheads,
            .vheads = cfg.dn_vheads,
            .kdim = cfg.dn_kdim,
            .vdim = cfg.dn_vdim,
            .convk = cfg.dn_convk,
            .conv_dim = cfg.dn_conv_dim,
            .eps = cfg.eps,
        };
    }
};

/// Persistent recurrent + conv state for one GDN layer.
pub const GdnState = struct {
    /// `[vheads][kdim*vdim]` row-major recurrent state.
    rec: []f32,
    /// `[conv_dim][convk-1]` causal-conv history ring.
    ring: []f32,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: Dims) !GdnState {
        const rec = try gpa.alloc(f32, d.vheads * d.kdim * d.vdim);
        errdefer gpa.free(rec);
        const ring = try gpa.alloc(f32, d.conv_dim * (d.convk - 1));
        @memset(rec, 0);
        @memset(ring, 0);
        return .{ .rec = rec, .ring = ring, .allocator = gpa };
    }

    pub fn reset(self: *GdnState) void {
        @memset(self.rec, 0);
        @memset(self.ring, 0);
    }

    pub fn deinit(self: *GdnState) void {
        self.allocator.free(self.rec);
        self.allocator.free(self.ring);
        self.* = undefined;
    }
};

/// Resident weights for one GDN layer.
pub const Layer = struct {
    qkv: NativeMatrix, // [conv_dim, hidden]
    z: NativeMatrix, // [v, hidden]
    b: NativeMatrix, // [vheads, hidden]
    a: NativeMatrix, // [vheads, hidden]
    out: NativeMatrix, // [hidden, v]
    conv: []f32, // [conv_dim, convk]
    dt_bias: []f32, // [vheads]
    a_log: []f32, // [vheads]
    norm: []f32, // [vdim]
    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, w: *const Weights, layer: u32) !Layer {
        var nb: [128]u8 = undefined;
        var sb: [96]u8 = undefined;
        const S = struct {
            fn s(buf: []u8, l: u32, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "layers.{d}.linear_attn.{s}", .{ l, suffix }) catch unreachable;
            }
        }.s;

        var self: Layer = undefined;
        self.allocator = gpa;
        self.qkv = try w.matrixBySuffix(S(&sb, layer, "in_proj_qkv.weight"), &nb);
        errdefer self.qkv.deinit();
        self.z = try w.matrixBySuffix(S(&sb, layer, "in_proj_z.weight"), &nb);
        errdefer self.z.deinit();
        self.b = try w.matrixBySuffix(S(&sb, layer, "in_proj_b.weight"), &nb);
        errdefer self.b.deinit();
        self.a = try w.matrixBySuffix(S(&sb, layer, "in_proj_a.weight"), &nb);
        errdefer self.a.deinit();
        self.out = try w.matrixBySuffix(S(&sb, layer, "out_proj.weight"), &nb);
        errdefer self.out.deinit();
        self.conv = try w.vectorBySuffix(S(&sb, layer, "conv1d.weight"), &nb);
        errdefer gpa.free(self.conv);
        self.dt_bias = try w.vectorBySuffix(S(&sb, layer, "dt_bias"), &nb);
        errdefer gpa.free(self.dt_bias);
        self.a_log = try w.vectorBySuffix(S(&sb, layer, "A_log"), &nb);
        errdefer gpa.free(self.a_log);
        self.norm = try w.vectorBySuffix(S(&sb, layer, "norm.weight"), &nb);
        return self;
    }

    pub fn deinit(self: *Layer) void {
        self.qkv.deinit();
        self.z.deinit();
        self.b.deinit();
        self.a.deinit();
        self.out.deinit();
        self.allocator.free(self.conv);
        self.allocator.free(self.dt_bias);
        self.allocator.free(self.a_log);
        self.allocator.free(self.norm);
        self.* = undefined;
    }
};

/// Scratch buffers reused across `forward` calls for one layer shape.
pub const Scratch = struct {
    qkv: []f32,
    z: []f32,
    bb: []f32,
    aa: []f32,
    norm: []f32,
    conv: []f32,
    q: []f32,
    k: []f32,
    core: []f32,
    delta: []f32,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: Dims, max_tokens: usize) !Scratch {
        return .{
            .qkv = try gpa.alloc(f32, max_tokens * d.conv_dim),
            .z = try gpa.alloc(f32, max_tokens * d.v()),
            .bb = try gpa.alloc(f32, max_tokens * d.vheads),
            .aa = try gpa.alloc(f32, max_tokens * d.vheads),
            .norm = try gpa.alloc(f32, max_tokens * d.v()),
            .conv = try gpa.alloc(f32, d.conv_dim),
            .q = try gpa.alloc(f32, d.vheads * d.kdim),
            .k = try gpa.alloc(f32, d.vheads * d.kdim),
            .core = try gpa.alloc(f32, d.v()),
            .delta = try gpa.alloc(f32, d.vheads * d.vdim), // per-head slice (threaded)
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *Scratch) void {
        const g = self.allocator;
        g.free(self.qkv);
        g.free(self.z);
        g.free(self.bb);
        g.free(self.aa);
        g.free(self.norm);
        g.free(self.conv);
        g.free(self.q);
        g.free(self.k);
        g.free(self.core);
        g.free(self.delta);
        self.* = undefined;
    }
};

/// Run the GDN layer over `x` (`[S, hidden]`), advancing `state`, writing
/// `out` (`[S, hidden]`).  `sc` must be sized for at least `S` tokens.
pub fn forward(
    layer: *const Layer,
    state: *GdnState,
    d: Dims,
    x: []const f32,
    S: usize,
    out: []f32,
    sc: *Scratch,
) void {
    const H = d.hidden;
    const VH = d.vheads;
    const KD = d.kdim;
    const VD = d.vdim;
    const CD = d.conv_dim;
    const CK = d.convk;
    const K = d.k();
    const V = d.v();
    const rep = d.rep();

    std.debug.assert(x.len == S * H and out.len == S * H);

    layer.qkv.matmul(sc.qkv[0 .. S * CD], x, S);
    layer.z.matmul(sc.z[0 .. S * V], x, S);
    layer.b.matmul(sc.bb[0 .. S * VH], x, S);
    layer.a.matmul(sc.aa[0 .. S * VH], x, S);

    var s: usize = 0;
    while (s < S) : (s += 1) {
        const qkv_row = sc.qkv[s * CD ..][0..CD];
        const z_row = sc.z[s * V ..][0..V];
        const b_row = sc.bb[s * VH ..][0..VH];
        const a_row = sc.aa[s * VH ..][0..VH];

        // causal depthwise conv1d + SiLU, per channel; update the ring after read.
        var ch: usize = 0;
        while (ch < CD) : (ch += 1) {
            const wrow = layer.conv[ch * CK ..][0..CK];
            const hist = state.ring[ch * (CK - 1) ..][0 .. CK - 1];
            var val: f32 = wrow[CK - 1] * qkv_row[ch];
            for (0..CK - 1) |tap| val += wrow[tap] * hist[tap];
            sc.conv[ch] = act.silu(val);
            var tap: usize = 0;
            while (tap + 1 < CK - 1) : (tap += 1) hist[tap] = hist[tap + 1];
            hist[CK - 2] = qkv_row[ch];
        }

        const qi = sc.conv[0..K];
        const ki = sc.conv[K .. 2 * K];
        const vi = sc.conv[2 * K .. 2 * K + V];

        // per-head Q/K: gather the shared key head, L2-normalize, scale.
        for (0..VH) |h| {
            const qh = sc.q[h * KD ..][0..KD];
            const kh = sc.k[h * KD ..][0..KD];
            const src = (h / rep) * KD;
            @memcpy(qh, qi[src..][0..KD]);
            @memcpy(kh, ki[src..][0..KD]);
            var qsum: f64 = 1e-6;
            var ksum: f64 = 1e-6;
            for (0..KD) |dd| {
                qsum += @as(f64, qh[dd]) * qh[dd];
                ksum += @as(f64, kh[dd]) * kh[dd];
            }
            const qscale: f32 = @floatCast((1.0 / @sqrt(qsum)) / @sqrt(@as(f64, @floatFromInt(KD))));
            const kscale: f32 = @floatCast(1.0 / @sqrt(ksum));
            for (0..KD) |dd| {
                qh[dd] *= qscale;
                kh[dd] *= kscale;
            }
        }

        // gated-delta recurrent update, per value head - heads are independent
        // (own state slice, own delta scratch, own core slice).
        const RCtx = struct {
            rec: []f32,
            q: []const f32,
            k: []const f32,
            vi: []const f32,
            delta: []f32,
            core: []f32,
            a_log: []const f32,
            dt_bias: []const f32,
            a_row: []const f32,
            b_row: []const f32,
            KD: usize,
            VD: usize,
        };
        parallel.chunks(VH, VH * KD * VD * 3, RCtx{
            .rec = state.rec,
            .q = sc.q,
            .k = sc.k,
            .vi = vi,
            .delta = sc.delta,
            .core = sc.core,
            .a_log = layer.a_log,
            .dt_bias = layer.dt_bias,
            .a_row = a_row,
            .b_row = b_row,
            .KD = KD,
            .VD = VD,
        }, struct {
            fn body(c: RCtx, h0: usize, h1: usize) void {
                var h = h0;
                while (h < h1) : (h += 1) {
                    const st = c.rec[h * c.KD * c.VD ..][0 .. c.KD * c.VD];
                    const qh = c.q[h * c.KD ..][0..c.KD];
                    const kh = c.k[h * c.KD ..][0..c.KD];
                    const vh = c.vi[h * c.VD ..][0..c.VD];
                    const delta = c.delta[h * c.VD ..][0..c.VD];
                    const alpha = @exp(-@exp(c.a_log[h]) * act.softplus(c.a_row[h] + c.dt_bias[h]));
                    const beta = act.sigmoid(c.b_row[h]);

                    for (st) |*cell| cell.* *= alpha;
                    for (0..c.VD) |val| {
                        var prev: f32 = 0;
                        for (0..c.KD) |dd| prev += kh[dd] * st[dd * c.VD + val];
                        delta[val] = (vh[val] - prev) * beta;
                    }
                    for (0..c.KD) |dd| {
                        for (0..c.VD) |val| st[dd * c.VD + val] += kh[dd] * delta[val];
                    }
                    const core = c.core[h * c.VD ..][0..c.VD];
                    for (0..c.VD) |val| {
                        var cur: f32 = 0;
                        for (0..c.KD) |dd| cur += qh[dd] * st[dd * c.VD + val];
                        core[val] = cur;
                    }
                }
            }
        }.body);

        // RMSNormGated with the z projection as the gate.
        const norm_row = sc.norm[s * V ..][0..V];
        for (0..VH) |h| {
            rmsnorm.rmsGated(
                norm_row[h * VD ..][0..VD],
                sc.core[h * VD ..][0..VD],
                z_row[h * VD ..][0..VD],
                layer.norm,
                d.eps,
                true,
            );
        }
    }

    layer.out.matmul(out, sc.norm[0 .. S * V], S);
}

// ---- tests -----------------------------------------------------------

const testing = std.testing;

const TinyGdn = struct {
    d: Dims,
    layer: Layer,
    arena: std.heap.ArenaAllocator,

    fn deinit(self: *TinyGdn) void {
        self.arena.deinit();
    }
};

/// Build a tiny GDN layer with deterministic pseudo-random f32 weights.
fn buildTiny(gpa: std.mem.Allocator, seed: u64) !TinyGdn {
    var arena = std.heap.ArenaAllocator.init(gpa);
    const a = arena.allocator();
    const d: Dims = .{
        .hidden = 6,
        .kheads = 1,
        .vheads = 2,
        .kdim = 3,
        .vdim = 3,
        .convk = 3,
        .conv_dim = 2 * (1 * 3) + 2 * 3, // 12
        .eps = 1e-6,
    };
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const mk = struct {
        fn m(al: std.mem.Allocator, rr: std.Random, rows: usize, cols: usize) !NativeMatrix {
            const buf = try al.alloc(f32, rows * cols);
            for (buf) |*v| v.* = rr.float(f32) * 0.6 - 0.3;
            return .{ .rows = rows, .cols = cols, .storage = .{ .f32 = buf }, .allocator = al };
        }
        fn vec(al: std.mem.Allocator, rr: std.Random, n: usize, lo: f32, hi: f32) ![]f32 {
            const buf = try al.alloc(f32, n);
            for (buf) |*v| v.* = lo + rr.float(f32) * (hi - lo);
            return buf;
        }
    };
    const layer: Layer = .{
        .qkv = try mk.m(a, r, d.conv_dim, d.hidden),
        .z = try mk.m(a, r, d.v(), d.hidden),
        .b = try mk.m(a, r, d.vheads, d.hidden),
        .a = try mk.m(a, r, d.vheads, d.hidden),
        .out = try mk.m(a, r, d.hidden, d.v()),
        .conv = try mk.vec(a, r, d.conv_dim * d.convk, -0.4, 0.4),
        .dt_bias = try mk.vec(a, r, d.vheads, -0.2, 0.2),
        .a_log = try mk.vec(a, r, d.vheads, -1.0, 0.5),
        .norm = try mk.vec(a, r, d.vdim, 0.5, 1.5),
        .allocator = a,
    };
    return .{ .d = d, .layer = layer, .arena = arena };
}

test "GDN: zero input from a fresh state produces zero output" {
    const gpa = testing.allocator;
    var tg = try buildTiny(gpa, 1);
    defer tg.deinit();
    var state = try GdnState.init(gpa, tg.d);
    defer state.deinit();
    var sc = try Scratch.init(gpa, tg.d, 4);
    defer sc.deinit();

    const x = try gpa.alloc(f32, 3 * tg.d.hidden);
    defer gpa.free(x);
    @memset(x, 0);
    const out = try gpa.alloc(f32, 3 * tg.d.hidden);
    defer gpa.free(out);

    forward(&tg.layer, &state, tg.d, x, 3, out, &sc);
    for (out) |v| try testing.expectApproxEqAbs(@as(f32, 0), v, 1e-6);
}

test "GDN: chunk boundaries do not change the result" {
    const gpa = testing.allocator;
    var tg = try buildTiny(gpa, 7);
    defer tg.deinit();
    const H = tg.d.hidden;
    const T = 5;

    var prng = std.Random.DefaultPrng.init(99);
    const r = prng.random();
    const x = try gpa.alloc(f32, T * H);
    defer gpa.free(x);
    for (x) |*v| v.* = r.float(f32) * 2 - 1;

    // one shot
    var s1 = try GdnState.init(gpa, tg.d);
    defer s1.deinit();
    var sc1 = try Scratch.init(gpa, tg.d, T);
    defer sc1.deinit();
    const whole = try gpa.alloc(f32, T * H);
    defer gpa.free(whole);
    forward(&tg.layer, &s1, tg.d, x, T, whole, &sc1);

    // split 2 + 3
    var s2 = try GdnState.init(gpa, tg.d);
    defer s2.deinit();
    var sc2 = try Scratch.init(gpa, tg.d, T);
    defer sc2.deinit();
    const part = try gpa.alloc(f32, T * H);
    defer gpa.free(part);
    forward(&tg.layer, &s2, tg.d, x[0 .. 2 * H], 2, part[0 .. 2 * H], &sc2);
    forward(&tg.layer, &s2, tg.d, x[2 * H ..], 3, part[2 * H ..], &sc2);

    for (whole, part) |wv, pv| try testing.expectApproxEqRel(wv, pv, 1e-4);
}

test "GDN: deterministic and state actually advances" {
    const gpa = testing.allocator;
    var tg = try buildTiny(gpa, 3);
    defer tg.deinit();
    const H = tg.d.hidden;

    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    const tok = try gpa.alloc(f32, H);
    defer gpa.free(tok);
    for (tok) |*v| v.* = r.float(f32) * 2 - 1;

    var st = try GdnState.init(gpa, tg.d);
    defer st.deinit();
    var sc = try Scratch.init(gpa, tg.d, 1);
    defer sc.deinit();

    const o1 = try gpa.alloc(f32, H);
    defer gpa.free(o1);
    const o2 = try gpa.alloc(f32, H);
    defer gpa.free(o2);
    const o3 = try gpa.alloc(f32, H);
    defer gpa.free(o3);

    forward(&tg.layer, &st, tg.d, tok, 1, o1, &sc);
    forward(&tg.layer, &st, tg.d, tok, 1, o2, &sc); // same token, advanced state

    var s_fresh = try GdnState.init(gpa, tg.d);
    defer s_fresh.deinit();
    forward(&tg.layer, &s_fresh, tg.d, tok, 1, o3, &sc);

    var moved = false;
    for (o1, o2) |a, b| moved = moved or @abs(a - b) > 1e-5;
    try testing.expect(moved); // state changed the output
    for (o1, o3) |a, b| try testing.expectApproxEqAbs(a, b, 1e-6); // fresh == first
    for (o1) |v| try testing.expect(std.math.isFinite(v));
}
