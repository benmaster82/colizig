//! Qwen3-MoE attention: plain causal GQA with per-head QK RMSNorm and full
//! RoPE. Much simpler than Qwen4-Exp's QSA — no indexer, no block selection,
//! no output gate. q/k/v/o projections are block-FP8 (E4M3) in the FP8
//! checkpoint; q_norm / k_norm are BF16 vectors of length `head_dim`.
//!
//! Per token: project q/k/v, RMSNorm+RoPE each q and k head, append k/v to the
//! context cache, attend causally over the whole cache, o-project.

const std = @import("std");
const Cfg = @import("../model/config.zig").Cfg;
const W = @import("../model/weights.zig").Weights;
const Fp8Matrix = @import("../qwen38/moe.zig").Fp8Matrix;
const rms = @import("../ops/rmsnorm.zig").rms;
const rope = @import("../ops/rope.zig").rope;

pub const Dims = struct {
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    eps: f32,

    pub fn group(self: Dims) usize {
        return self.q_heads / self.kv_heads;
    }
    pub fn qDim(self: Dims) usize {
        return self.q_heads * self.head_dim;
    }
    pub fn kvDim(self: Dims) usize {
        return self.kv_heads * self.head_dim;
    }

    pub fn of(cfg: Cfg) Dims {
        return .{
            .hidden = cfg.hidden,
            .q_heads = cfg.q_heads,
            .kv_heads = cfg.kv_heads,
            .head_dim = cfg.head_dim,
            .theta = cfg.theta,
            .eps = cfg.eps,
        };
    }
};

/// GPU VRAM-cache key for a per-layer attention projection (disjoint from the
/// MoE expert key space: expert slot `1020 + role`, role 0..3 = q/k/v/o).
pub fn attnKey(layer: u32, role: u2) u64 {
    return Fp8Matrix.expertKey(layer, 1020 + @as(u32, role), 0);
}

// ---- per-layer resident weights -----------------------------------------

pub const Weights = struct {
    q_proj: Fp8Matrix, // [q_heads*head_dim, hidden]
    k_proj: Fp8Matrix, // [kv_heads*head_dim, hidden]
    v_proj: Fp8Matrix, // [kv_heads*head_dim, hidden]
    o_proj: Fp8Matrix, // [hidden, q_heads*head_dim]
    q_norm: []f32, // [head_dim]
    k_norm: []f32, // [head_dim]
    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, w: *const W, layer: u32, d: Dims) !Weights {
        var self: Weights = undefined;
        self.allocator = gpa;

        self.q_proj = try loadFp8(gpa, w, layer, "q_proj");
        errdefer self.q_proj.deinit();
        self.k_proj = try loadFp8(gpa, w, layer, "k_proj");
        errdefer self.k_proj.deinit();
        self.v_proj = try loadFp8(gpa, w, layer, "v_proj");
        errdefer self.v_proj.deinit();
        self.o_proj = try loadFp8(gpa, w, layer, "o_proj");
        errdefer self.o_proj.deinit();
        self.q_proj.key = attnKey(layer, 0);
        self.k_proj.key = attnKey(layer, 1);
        self.v_proj.key = attnKey(layer, 2);
        self.o_proj.key = attnKey(layer, 3);

        var nb: [128]u8 = undefined;
        var sb: [96]u8 = undefined;
        self.q_norm = try w.vectorBySuffix(name(&sb, layer, "q_norm.weight"), &nb);
        errdefer gpa.free(self.q_norm);
        self.k_norm = try w.vectorBySuffix(name(&sb, layer, "k_norm.weight"), &nb);

        if (self.q_proj.rows != d.qDim() or self.q_proj.cols != d.hidden or
            self.k_proj.rows != d.kvDim() or self.v_proj.rows != d.kvDim() or
            self.o_proj.rows != d.hidden or self.o_proj.cols != d.qDim() or
            self.q_norm.len != d.head_dim or self.k_norm.len != d.head_dim)
            return error.AttnShapeMismatch;

        return self;
    }

    pub fn deinit(self: *Weights) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.o_proj.deinit();
        self.allocator.free(self.q_norm);
        self.allocator.free(self.k_norm);
        self.* = undefined;
    }
};

fn name(buf: []u8, layer: u32, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.self_attn.{s}", .{ layer, suffix }) catch unreachable;
}

fn loadFp8(gpa: std.mem.Allocator, w: *const W, layer: u32, proj: []const u8) !Fp8Matrix {
    var nb: [128]u8 = undefined;
    var sb: [96]u8 = undefined;
    const wname = std.fmt.bufPrint(&sb, "layers.{d}.self_attn.{s}.weight", .{ layer, proj }) catch unreachable;
    const wv = w.viewBySuffix(wname, &nb) orelse return error.AttnTensorMissing;
    var sb2: [110]u8 = undefined;
    var nb2: [140]u8 = undefined;
    const sname = std.fmt.bufPrint(&sb2, "layers.{d}.self_attn.{s}.weight_scale_inv", .{ layer, proj }) catch unreachable;
    const sv = w.viewBySuffix(sname, &nb2) orelse return error.AttnScaleMissing;
    return Fp8Matrix.load(gpa, wv, sv);
}

// ---- context KV cache ---------------------------------------------------

pub const Cache = struct {
    k: []f32, // [cap][kv_heads*head_dim]  (post QK-norm + RoPE)
    v: []f32, // [cap][kv_heads*head_dim]  (raw)
    cap: usize,
    len: usize = 0,
    kv_dim: usize,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: Dims, capacity: usize) !Cache {
        return .{
            .k = try gpa.alloc(f32, capacity * d.kvDim()),
            .v = try gpa.alloc(f32, capacity * d.kvDim()),
            .cap = capacity,
            .kv_dim = d.kvDim(),
            .allocator = gpa,
        };
    }
    pub fn reset(self: *Cache) void {
        self.len = 0;
    }
    pub fn deinit(self: *Cache) void {
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.* = undefined;
    }
};

// ---- scratch ----------------------------------------------------------

pub const Scratch = struct {
    q: []f32, // [max_tokens][q_heads*head_dim]
    k: []f32, // [max_tokens][kv_heads*head_dim]
    v: []f32, // [max_tokens][kv_heads*head_dim]
    ao: []f32, // [max_tokens][q_heads*head_dim]  attention output pre o_proj
    scores: []f32, // [max_context]
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: Dims, max_tokens: usize, max_context: usize) !Scratch {
        return .{
            .q = try gpa.alloc(f32, max_tokens * d.qDim()),
            .k = try gpa.alloc(f32, max_tokens * d.kvDim()),
            .v = try gpa.alloc(f32, max_tokens * d.kvDim()),
            .ao = try gpa.alloc(f32, max_tokens * d.qDim()),
            .scores = try gpa.alloc(f32, max_context),
            .allocator = gpa,
        };
    }
    pub fn deinit(self: *Scratch) void {
        inline for (.{ self.q, self.k, self.v, self.ao, self.scores }) |s| self.allocator.free(s);
        self.* = undefined;
    }
};

/// `x` / `out` are `[S][hidden]`. Appends `S` tokens to `cache` at positions
/// `[cache.len, cache.len + S)`; `out` is the attention block output (the caller
/// adds the residual).
pub fn forward(
    wt: *const Weights,
    cache: *Cache,
    d: Dims,
    x: []const f32,
    S: usize,
    out: []f32,
    sc: *Scratch,
) !void {
    const H = d.hidden;
    const HD = d.head_dim;
    const QH = d.q_heads;
    const KVH = d.kv_heads;
    const G = d.group();
    const QD = d.qDim();
    const KVD = d.kvDim();
    const pos_base = cache.len;
    if (pos_base + S > cache.cap) return error.ContextExhausted;
    std.debug.assert(x.len == S * H and out.len == S * H);

    // project + per-head QK-norm + RoPE for every token, then append K/V.
    for (0..S) |s| {
        const xs = x[s * H ..][0..H];
        const q = sc.q[s * QD ..][0..QD];
        const k = sc.k[s * KVD ..][0..KVD];
        const v = sc.v[s * KVD ..][0..KVD];
        wt.q_proj.matmul(q, xs, 1);
        wt.k_proj.matmul(k, xs, 1);
        wt.v_proj.matmul(v, xs, 1);
        const pos = pos_base + s;
        for (0..QH) |h| {
            const qh = q[h * HD ..][0..HD];
            rms(qh, qh, wt.q_norm, d.eps);
            rope(qh, HD, pos, d.theta);
        }
        for (0..KVH) |h| {
            const kh = k[h * HD ..][0..HD];
            rms(kh, kh, wt.k_norm, d.eps);
            rope(kh, HD, pos, d.theta);
        }
        @memcpy(cache.k[(pos_base + s) * KVD ..][0..KVD], k);
        @memcpy(cache.v[(pos_base + s) * KVD ..][0..KVD], v);
    }
    cache.len = pos_base + S;

    // causal attention: query token s (absolute pos_base+s) attends to cache
    // positions [0, pos_base+s].
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HD)));
    for (0..S) |s| {
        const q = sc.q[s * QD ..][0..QD];
        const ao = sc.ao[s * QD ..][0..QD];
        const last = pos_base + s; // inclusive
        for (0..QH) |h| {
            const qh = q[h * HD ..][0..HD];
            const kvh = h / G;
            const oh = ao[h * HD ..][0..HD];
            @memset(oh, 0);
            var mx: f32 = -std.math.inf(f32);
            for (0..last + 1) |t| {
                const kt = cache.k[t * KVD + kvh * HD ..][0..HD];
                var dot: f32 = 0;
                for (qh, kt) |a, b| dot += a * b;
                dot *= scale;
                sc.scores[t] = dot;
                mx = @max(mx, dot);
            }
            var denom: f64 = 0;
            for (0..last + 1) |t| {
                const e = @exp(sc.scores[t] - mx);
                sc.scores[t] = e;
                denom += e;
            }
            const inv: f32 = @floatCast(1.0 / denom);
            for (0..last + 1) |t| {
                const wgt = sc.scores[t] * inv;
                const vt = cache.v[t * KVD + kvh * HD ..][0..HD];
                for (oh, vt) |*o, vv| o.* += wgt * vv;
            }
        }
        // o_proj: [hidden] = ao[q_heads*head_dim] @ o_projᵀ
        wt.o_proj.matmul(out[s * H ..][0..H], ao, 1);
    }
}
