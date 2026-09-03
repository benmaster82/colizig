//! Qwen Sparse Attention — the "full_attention" path for 12 of the 48 layers.
//!
//! Ported from colibri's `q38_attention`.  Per token:
//!   project Q (+ sigmoid output gate), K, V, and a lightweight indexer q/k;
//!   cache normalized+RoPE'd K, raw V, and the raw indexer key;
//!   pool every complete `idx_ratio`-token block of indexer keys, score it
//!   against the indexer query, keep the top `idx_budget/idx_ratio` blocks plus
//!   a causal tail; run full `q_heads` attention over only those tokens; apply
//!   the output gate; `o_proj`.
//!
//! The K/V/indexer-key cache is the largest context-dependent memory consumer
//! (~54 KiB/token on the real model) — size it from `--context` via the Phase 1
//! memory plan.

const std = @import("std");
const Cfg = @import("../model/config.zig").Cfg;
const Weights = @import("../model/weights.zig").Weights;
const NativeMatrix = @import("../model/tensors.zig").NativeMatrix;
const act = @import("../ops/activation.zig");
const rms = @import("../ops/rmsnorm.zig");
const rope = @import("../ops/rope.zig").rope;

pub const Dims = struct {
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    eps: f32,
    idx_qheads: usize,
    idx_kheads: usize, // == 1
    idx_dim: usize,
    idx_budget: usize,
    idx_ratio: usize,

    pub fn group(self: Dims) usize {
        return self.q_heads / self.kv_heads;
    }
    pub fn maxSelected(self: Dims) usize {
        return self.idx_budget + self.idx_ratio - 1;
    }

    pub fn of(cfg: Cfg) Dims {
        return .{
            .hidden = cfg.hidden,
            .q_heads = cfg.q_heads,
            .kv_heads = cfg.kv_heads,
            .head_dim = cfg.head_dim,
            .rotary_dim = cfg.rotary_dim,
            .theta = cfg.theta,
            .eps = cfg.eps,
            .idx_qheads = cfg.idx_qheads,
            .idx_kheads = cfg.idx_kheads,
            .idx_dim = cfg.idx_dim,
            .idx_budget = cfg.idx_budget,
            .idx_ratio = cfg.idx_ratio,
        };
    }
};

// ---- context cache ---------------------------------------------------

pub const Cache = struct {
    k: []f32, // [kv_heads][cap][head_dim]  (normalized + RoPE'd)
    v: []f32, // [kv_heads][cap][head_dim]  (raw)
    ik: []f32, // [cap][idx_dim]             (raw indexer key)
    cap: usize,
    len: usize = 0,
    kv_heads: usize,
    head_dim: usize,
    idx_dim: usize,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: Dims, capacity: usize) !Cache {
        return .{
            .k = try gpa.alloc(f32, d.kv_heads * capacity * d.head_dim),
            .v = try gpa.alloc(f32, d.kv_heads * capacity * d.head_dim),
            .ik = try gpa.alloc(f32, capacity * d.idx_dim),
            .cap = capacity,
            .kv_heads = d.kv_heads,
            .head_dim = d.head_dim,
            .idx_dim = d.idx_dim,
            .allocator = gpa,
        };
    }

    pub fn reset(self: *Cache) void {
        self.len = 0;
    }

    pub fn deinit(self: *Cache) void {
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.allocator.free(self.ik);
        self.* = undefined;
    }

    fn kRow(self: Cache, h: usize, pos: usize) []f32 {
        return self.k[(h * self.cap + pos) * self.head_dim ..][0..self.head_dim];
    }
    fn vRow(self: Cache, h: usize, pos: usize) []f32 {
        return self.v[(h * self.cap + pos) * self.head_dim ..][0..self.head_dim];
    }
    fn ikRow(self: Cache, pos: usize) []f32 {
        return self.ik[pos * self.idx_dim ..][0..self.idx_dim];
    }
};

// ---- resident weights ----------------------------------------------

pub const Layer = struct {
    q: NativeMatrix, // [q_heads*head_dim*2, hidden]  (query || output-gate)
    k: NativeMatrix, // [kv_heads*head_dim, hidden]
    v: NativeMatrix, // [kv_heads*head_dim, hidden]
    o: NativeMatrix, // [hidden, q_heads*head_dim]
    q_norm: []f32, // [head_dim]
    k_norm: []f32, // [head_dim]
    idx_qk: NativeMatrix, // [(idx_qheads+idx_kheads)*idx_dim, hidden]
    idx_qn: []f32, // [idx_dim]
    idx_kn: []f32, // [idx_dim]
    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, w: *const Weights, layer: u32) !Layer {
        var nb: [128]u8 = undefined;
        var sb: [96]u8 = undefined;
        const S = struct {
            fn s(buf: []u8, l: u32, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "layers.{d}.self_attn.{s}", .{ l, suffix }) catch unreachable;
            }
        }.s;
        var self: Layer = undefined;
        self.allocator = gpa;
        self.q = try w.matrixBySuffix(S(&sb, layer, "q_proj.weight"), &nb);
        errdefer self.q.deinit();
        self.k = try w.matrixBySuffix(S(&sb, layer, "k_proj.weight"), &nb);
        errdefer self.k.deinit();
        self.v = try w.matrixBySuffix(S(&sb, layer, "v_proj.weight"), &nb);
        errdefer self.v.deinit();
        self.o = try w.matrixBySuffix(S(&sb, layer, "o_proj.weight"), &nb);
        errdefer self.o.deinit();
        self.q_norm = try w.vectorBySuffix(S(&sb, layer, "q_norm.weight"), &nb);
        errdefer gpa.free(self.q_norm);
        self.k_norm = try w.vectorBySuffix(S(&sb, layer, "k_norm.weight"), &nb);
        errdefer gpa.free(self.k_norm);
        self.idx_qk = try w.matrixBySuffix(S(&sb, layer, "indexer.index_qk_proj.weight"), &nb);
        errdefer self.idx_qk.deinit();
        self.idx_qn = try w.vectorBySuffix(S(&sb, layer, "indexer.q_layernorm.weight"), &nb);
        errdefer gpa.free(self.idx_qn);
        self.idx_kn = try w.vectorBySuffix(S(&sb, layer, "indexer.k_layernorm.weight"), &nb);
        return self;
    }

    pub fn deinit(self: *Layer) void {
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        self.idx_qk.deinit();
        self.allocator.free(self.q_norm);
        self.allocator.free(self.k_norm);
        self.allocator.free(self.idx_qn);
        self.allocator.free(self.idx_kn);
        self.* = undefined;
    }
};

pub const Scratch = struct {
    qp: []f32,
    kp: []f32,
    vp: []f32,
    ip: []f32,
    heads: []f32,
    qidx: []f32,
    pool: []f32,
    qh: []f32,
    rank: []Block,
    selected: []u32,
    attn: []f32,
    allocator: std.mem.Allocator,

    const Block = struct { score: f32, block: u32 };

    pub fn init(gpa: std.mem.Allocator, d: Dims, max_tokens: usize, cache_cap: usize) !Scratch {
        const D = d.head_dim;
        const ID = d.idx_dim;
        const iqk = (d.idx_qheads + d.idx_kheads) * ID;
        return .{
            .qp = try gpa.alloc(f32, max_tokens * d.q_heads * 2 * D),
            .kp = try gpa.alloc(f32, max_tokens * d.kv_heads * D),
            .vp = try gpa.alloc(f32, max_tokens * d.kv_heads * D),
            .ip = try gpa.alloc(f32, max_tokens * iqk),
            .heads = try gpa.alloc(f32, max_tokens * d.q_heads * D),
            .qidx = try gpa.alloc(f32, d.idx_qheads * ID),
            .pool = try gpa.alloc(f32, ID),
            .qh = try gpa.alloc(f32, D),
            .rank = try gpa.alloc(Block, cache_cap / d.idx_ratio + 1),
            .selected = try gpa.alloc(u32, d.maxSelected()),
            .attn = try gpa.alloc(f32, d.maxSelected()),
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *Scratch) void {
        const g = self.allocator;
        inline for (.{ self.qp, self.kp, self.vp, self.ip, self.heads, self.qidx, self.pool, self.qh, self.attn }) |b| g.free(b);
        g.free(self.rank);
        g.free(self.selected);
        self.* = undefined;
    }
};

fn blockLess(_: void, a: Scratch.Block, b: Scratch.Block) bool {
    if (a.score != b.score) return a.score > b.score; // descending
    return a.block < b.block;
}

fn dot(a: []const f32, b: []const f32) f32 {
    var s: f32 = 0;
    for (a, b) |x, y| s += x * y;
    return s;
}

/// Run QSA over `x` (`[S, hidden]`) at absolute positions `pos_base .. pos_base+S`,
/// appending to `cache` and writing `out` (`[S, hidden]`).
pub fn forward(
    layer: *const Layer,
    cache: *Cache,
    d: Dims,
    x: []const f32,
    S: usize,
    pos_base: usize,
    out: []f32,
    sc: *Scratch,
) void {
    const H = d.hidden;
    const QH = d.q_heads;
    const KVH = d.kv_heads;
    const D = d.head_dim;
    const IQ = d.idx_qheads;
    const ID = d.idx_dim;
    const R = d.idx_ratio;
    const iqk = (IQ + d.idx_kheads) * ID;
    std.debug.assert(x.len == S * H and out.len == S * H);
    std.debug.assert(cache.len == pos_base);
    std.debug.assert(pos_base + S <= cache.cap);

    layer.q.matmul(sc.qp[0 .. S * QH * 2 * D], x, S);
    layer.k.matmul(sc.kp[0 .. S * KVH * D], x, S);
    layer.v.matmul(sc.vp[0 .. S * KVH * D], x, S);
    layer.idx_qk.matmul(sc.ip[0 .. S * iqk], x, S);

    // append K (norm+rope), V (raw), indexer key (raw) to the cache
    for (0..S) |s| {
        const pos = pos_base + s;
        for (0..KVH) |h| {
            const kh = sc.kp[s * KVH * D + h * D ..][0..D];
            rms.rms0(kh, kh, layer.k_norm, d.eps);
            rope(kh, d.rotary_dim, pos, d.theta);
            @memcpy(cache.kRow(h, pos), kh);
            @memcpy(cache.vRow(h, pos), sc.vp[s * KVH * D + h * D ..][0..D]);
        }
        @memcpy(cache.ikRow(pos), sc.ip[s * iqk + IQ * ID ..][0..ID]);
    }
    cache.len = pos_base + S;

    for (0..S) |s| {
        const pos = pos_base + s;
        const visible = pos + 1;
        const blocks = visible / R;
        const tail = blocks * R;

        // indexer query per head: norm + rope at pos
        for (0..IQ) |h| {
            const qh = sc.qidx[h * ID ..][0..ID];
            @memcpy(qh, sc.ip[s * iqk + h * ID ..][0..ID]);
            rms.rms0(qh, qh, layer.idx_qn, d.eps);
            rope(qh, d.rotary_dim, pos, d.theta);
        }

        const take = @min(blocks, d.idx_budget / R);
        var nsel: usize = 0;

        for (0..blocks) |b| {
            @memset(sc.pool, 0);
            for (0..R) |rr| {
                const raw = cache.ikRow(b * R + rr);
                for (0..ID) |dd| sc.pool[dd] += raw[dd] / @as(f32, @floatFromInt(R));
            }
            rms.rms0(sc.pool, sc.pool, layer.idx_kn, d.eps);
            rope(sc.pool, d.rotary_dim, b * R, d.theta);
            var score: f32 = 0;
            for (0..IQ) |h| {
                const a = dot(sc.qidx[h * ID ..][0..ID], sc.pool);
                if (a > 0) score += a;
            }
            sc.rank[b] = .{ .score = score / @sqrt(@as(f32, @floatFromInt(ID))), .block = @intCast(b) };
        }
        if (blocks != 0) std.sort.pdq(Scratch.Block, sc.rank[0..blocks], {}, blockLess);

        for (0..take) |z| {
            for (0..R) |rr| {
                sc.selected[nsel] = sc.rank[z].block * @as(u32, @intCast(R)) + @as(u32, @intCast(rr));
                nsel += 1;
            }
        }
        var t = tail;
        while (t < visible) : (t += 1) {
            sc.selected[nsel] = @intCast(t);
            nsel += 1;
        }

        // full attention over the selected tokens, per query head
        for (0..QH) |h| {
            const qraw = sc.qp[s * QH * 2 * D + h * 2 * D ..][0 .. 2 * D];
            @memcpy(sc.qh, qraw[0..D]);
            rms.rms0(sc.qh, sc.qh, layer.q_norm, d.eps);
            rope(sc.qh, d.rotary_dim, pos, d.theta);

            const khidx = h / d.group();
            var mx: f32 = -std.math.inf(f32);
            for (0..nsel) |j| {
                const kh = cache.kRow(khidx, sc.selected[j]);
                const a = dot(sc.qh, kh) / @sqrt(@as(f32, @floatFromInt(D)));
                sc.attn[j] = a;
                mx = @max(mx, a);
            }
            var den: f32 = 0;
            for (0..nsel) |j| {
                sc.attn[j] = @exp(sc.attn[j] - mx);
                den += sc.attn[j];
            }
            const oh = sc.heads[s * QH * D + h * D ..][0..D];
            @memset(oh, 0);
            for (0..nsel) |j| {
                const wv = sc.attn[j] / den;
                const vh = cache.vRow(khidx, sc.selected[j]);
                for (0..D) |dd| oh[dd] += wv * vh[dd];
            }
            for (0..D) |dd| oh[dd] *= act.sigmoid(qraw[D + dd]);
        }
    }

    layer.o.matmul(out, sc.heads[0 .. S * QH * D], S);
}

// ---- tests -----------------------------------------------------------

const testing = std.testing;
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");

test "QSA on the tiny fixture: finite, deterministic, chunk-invariant" {
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

    // find the full_attention layer
    var qi: ?u32 = null;
    for (m.cfg.is_attn, 0..) |a, i| {
        if (a) {
            qi = @intCast(i);
            break;
        }
    }
    const layer_idx = qi orelse return error.SkipZigTest;

    const d = Dims.of(m.cfg);
    var layer = try Layer.load(gpa, &w, layer_idx);
    defer layer.deinit();

    const T = 7;
    const H = d.hidden;
    var prng = std.Random.DefaultPrng.init(0x9A);
    const r = prng.random();
    const x = try gpa.alloc(f32, T * H);
    defer gpa.free(x);
    for (x) |*v| v.* = r.float(f32) * 2 - 1;

    var sc = try Scratch.init(gpa, d, T, T);
    defer sc.deinit();

    var c1 = try Cache.init(gpa, d, T);
    defer c1.deinit();
    const o1 = try gpa.alloc(f32, T * H);
    defer gpa.free(o1);
    forward(&layer, &c1, d, x, T, 0, o1, &sc);
    for (o1) |v| try testing.expect(std.math.isFinite(v));

    // decode in 3 chunks (2 + 2 + 3) → identical output
    var c2 = try Cache.init(gpa, d, T);
    defer c2.deinit();
    const o2 = try gpa.alloc(f32, T * H);
    defer gpa.free(o2);
    forward(&layer, &c2, d, x[0 .. 2 * H], 2, 0, o2[0 .. 2 * H], &sc);
    forward(&layer, &c2, d, x[2 * H .. 4 * H], 2, 2, o2[2 * H .. 4 * H], &sc);
    forward(&layer, &c2, d, x[4 * H ..], 3, 4, o2[4 * H ..], &sc);
    for (o1, o2) |a, b| try testing.expectApproxEqRel(a, b, 1e-4);

    // fully deterministic on a fresh cache
    var c3 = try Cache.init(gpa, d, T);
    defer c3.deinit();
    const o3 = try gpa.alloc(f32, T * H);
    defer gpa.free(o3);
    forward(&layer, &c3, d, x, T, 0, o3, &sc);
    for (o1, o3) |a, b| try testing.expectEqual(a, b);
}
