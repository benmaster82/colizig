//! Mixture-of-Experts forward + bounded per-layer expert cache.
//!
//! Ported from colibri's `q38_moe_decode`.  Per token:
//!   router → top-k of `experts` → softmax gates (renormalized over the top-k
//!   when `norm_topk`) → each selected expert is a block-FP8 SwiGLU, plus one
//!   always-on shared expert with its own sigmoid gate → weighted sum.
//!
//! Routed experts are streamed: `ExpertCache` is a bounded LRU keyed by expert
//! id, holding decoded-in-place E4M3 matrices.  Phase 4 runs every token through
//! the decode path (grouping tokens by expert for prefill is a Phase 8
//! throughput change and does not affect results).

const std = @import("std");
const Cfg = @import("../model/config.zig").Cfg;
const Weights = @import("../model/weights.zig").Weights;
const NativeMatrix = @import("../model/tensors.zig").NativeMatrix;
const View = @import("../model/tensors.zig").View;
const fp8 = @import("../ops/fp8.zig");
const act = @import("../ops/activation.zig");
const parallel = @import("../runtime/parallel.zig");

pub const Dims = struct {
    hidden: usize,
    experts: usize,
    topk: usize,
    inter: usize,
    shared_inter: usize,
    norm_topk: bool,

    pub fn of(cfg: Cfg) Dims {
        return .{
            .hidden = cfg.hidden,
            .experts = cfg.experts,
            .topk = cfg.topk,
            .inter = cfg.inter,
            .shared_inter = cfg.shared_inter,
            .norm_topk = cfg.norm_topk,
        };
    }
};

// ---- block-FP8 matrix ------------------------------------------------------

pub const Fp8Matrix = struct {
    rows: usize,
    cols: usize,
    /// `[rows*cols]` E4M3 — a **borrowed** slice into a shard mmap (the shards
    /// outlive every `ExpertCache`).  Not copied: a demand load only decodes the
    /// tiny scale table; the E4M3 bytes fault in lazily through the OS page cache
    /// on first `matmul` and are reclaimed under memory pressure, not by us.
    data: []const u8,
    scales: []f32, // [nblk(rows)*nblk(cols)] f32, owned
    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, weight: View, scale: View) !Fp8Matrix {
        if (weight.dtype != .f8_e4m3 or weight.shape.len != 2) return error.BadExpertWeight;
        const rows: usize = @intCast(weight.shape[0]);
        const cols: usize = @intCast(weight.shape[1]);
        if (weight.bytes.len != rows * cols) return error.BadExpertWeight;

        const want_scales = fp8.nblk(rows) * fp8.nblk(cols);
        if (scale.shape.len != 2 or
            @as(usize, @intCast(scale.shape[0])) != fp8.nblk(rows) or
            @as(usize, @intCast(scale.shape[1])) != fp8.nblk(cols))
            return error.BadExpertScale;

        const scales = try gpa.alloc(f32, want_scales);
        errdefer gpa.free(scales);
        try scale.decode(scales);

        return .{ .rows = rows, .cols = cols, .data = weight.bytes, .scales = scales, .allocator = gpa };
    }

    pub fn deinit(self: *Fp8Matrix) void {
        self.allocator.free(self.scales);
        self.* = undefined;
    }

    pub fn byteLen(self: Fp8Matrix) usize {
        return self.data.len + self.scales.len * 4;
    }

    /// y[S, rows] = x[S, cols] @ dequant(self)ᵀ
    pub fn matmul(self: Fp8Matrix, y: []f32, x: []const f32, S: usize) void {
        fp8.matmulFp8(y, x, self.data, self.scales, S, self.cols, self.rows);
    }
};

pub const Expert = struct {
    id: u32,
    gate: Fp8Matrix, // [inter, hidden]
    up: Fp8Matrix, // [inter, hidden]
    down: Fp8Matrix, // [hidden, inter]

    pub fn byteLen(self: Expert) usize {
        return self.gate.byteLen() + self.up.byteLen() + self.down.byteLen();
    }
    pub fn deinit(self: *Expert) void {
        self.gate.deinit();
        self.up.deinit();
        self.down.deinit();
        self.* = undefined;
    }
};

// ---- bounded LRU expert cache --------------------------------------------

pub const CacheStats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    loads: u64 = 0,
    evictions: u64 = 0,
    bytes_resident: u64 = 0,
    /// A demand `get` found the expert already resident *because a prefetch put
    /// it there* (not a prior demand load).
    prefetch_hits: u64 = 0,
    /// A prefetched expert was evicted before any demand access consumed it.
    prefetch_wasted: u64 = 0,
    /// Expert loads on the critical path (a demand miss) — the count form of
    /// `compute_stall_due_to_io` for the MoE.
    demand_loads: u64 = 0,
    /// Expert loads that happened ahead of demand (prefetch).
    prefetch_loads: u64 = 0,
    /// Wall-clock nanoseconds in demand-path expert loads (needs an Io; 0 otherwise).
    demand_ns: u64 = 0,
};

pub const ExpertCache = struct {
    const Slot = struct { expert: ?Expert = null, used: u64 = 0, prefetched: bool = false };

    layer: u32,
    cap: usize,
    slots: []Slot,
    clock: u64 = 0,
    stats: CacheStats = .{},
    allocator: std.mem.Allocator,
    /// Optional — enables wall-clock timing of demand-path expert loads.
    io: ?std.Io = null,

    pub fn init(gpa: std.mem.Allocator, layer: u32, cap: usize) !ExpertCache {
        std.debug.assert(cap >= 1);
        const slots = try gpa.alloc(Slot, cap);
        for (slots) |*s| s.* = .{};
        return .{ .layer = layer, .cap = cap, .slots = slots, .allocator = gpa };
    }

    pub fn deinit(self: *ExpertCache) void {
        for (self.slots) |*s| if (s.expert) |*e| e.deinit();
        self.allocator.free(self.slots);
        self.* = undefined;
    }

    pub fn residentCount(self: ExpertCache) usize {
        var n: usize = 0;
        for (self.slots) |s| {
            if (s.expert != null) n += 1;
        }
        return n;
    }

    fn findSlot(self: *ExpertCache, id: u32) ?usize {
        for (self.slots, 0..) |*s, i| {
            if (s.expert) |e| {
                if (e.id == id) return i;
            }
        }
        return null;
    }

    fn victimSlot(self: *ExpertCache) usize {
        var victim: usize = 0;
        for (self.slots, 0..) |s, i| {
            if (s.expert == null) return i;
            if (s.used < self.slots[victim].used) victim = i;
        }
        // evicting an occupied slot
        if (self.slots[victim].expert) |*e| {
            if (self.slots[victim].prefetched) self.stats.prefetch_wasted += 1;
            self.stats.bytes_resident -= e.byteLen();
            e.deinit();
            self.stats.evictions += 1;
        }
        return victim;
    }

    fn fill(self: *ExpertCache, w: *const Weights, d: Dims, id: u32, prefetched: bool) !usize {
        const victim = self.victimSlot();
        const t0 = if (self.io) |io| std.Io.Timestamp.now(io, .awake) else null;
        const expert = try loadExpert(self.allocator, w, self.layer, d, id);
        if (t0) |ts| {
            if (!prefetched) {
                const dt = ts.durationTo(std.Io.Timestamp.now(self.io.?, .awake)).nanoseconds;
                if (dt > 0) self.stats.demand_ns +|= @intCast(dt);
            }
        }
        if (prefetched) self.stats.prefetch_loads += 1 else self.stats.demand_loads += 1;
        self.slots[victim].expert = expert;
        self.slots[victim].prefetched = prefetched;
        self.clock += 1;
        self.slots[victim].used = self.clock;
        self.stats.loads += 1;
        self.stats.bytes_resident += expert.byteLen();
        return victim;
    }

    /// Fetch expert `id`, loading it on a miss (LRU eviction).  The result is
    /// valid only until the next `get` call on this cache.
    pub fn get(self: *ExpertCache, w: *const Weights, d: Dims, id: u32) !*const Expert {
        if (self.findSlot(id)) |i| {
            self.clock += 1;
            self.slots[i].used = self.clock;
            self.stats.hits += 1;
            if (self.slots[i].prefetched) {
                self.stats.prefetch_hits += 1;
                self.slots[i].prefetched = false; // consumed
            }
            return &self.slots[i].expert.?;
        }
        self.stats.misses += 1;
        const i = try self.fill(w, d, id, false);
        return &self.slots[i].expert.?;
    }

    /// Warm the cache with expert `id` ahead of demand.  A no-op if already
    /// resident.  Does not count as a hit or a miss.
    pub fn prefetch(self: *ExpertCache, w: *const Weights, d: Dims, id: u32) !void {
        if (self.findSlot(id) != null) return;
        _ = try self.fill(w, d, id, true);
    }

    pub fn hitRate(self: ExpertCache) f64 {
        const total = self.stats.hits + self.stats.misses;
        if (total == 0) return 0;
        return @as(f64, @floatFromInt(self.stats.hits)) / @as(f64, @floatFromInt(total));
    }
};

fn loadExpert(gpa: std.mem.Allocator, w: *const Weights, layer: u32, d: Dims, id: u32) !Expert {
    var nb: [160]u8 = undefined;
    var sb: [120]u8 = undefined;

    const one = struct {
        fn m(
            g: std.mem.Allocator,
            weights: *const Weights,
            name_buf: []u8,
            suf_buf: []u8,
            l: u32,
            e: u32,
            proj: []const u8,
        ) !Fp8Matrix {
            const wname = std.fmt.bufPrint(suf_buf, "layers.{d}.mlp.experts.{d}.{s}.weight", .{ l, e, proj }) catch unreachable;
            const wv = weights.viewBySuffix(wname, name_buf) orelse return error.ExpertTensorMissing;
            var sbuf: [160]u8 = undefined;
            const sname = std.fmt.bufPrint(&sbuf, "layers.{d}.mlp.experts.{d}.{s}.weight_scale_inv", .{ l, e, proj }) catch unreachable;
            var snbuf: [200]u8 = undefined;
            const sv = weights.viewBySuffix(sname, &snbuf) orelse return error.ExpertScaleMissing;
            return Fp8Matrix.load(g, wv, sv);
        }
    }.m;

    var gate = try one(gpa, w, &nb, &sb, layer, id, "gate_proj");
    errdefer gate.deinit();
    var up = try one(gpa, w, &nb, &sb, layer, id, "up_proj");
    errdefer up.deinit();
    const down = try one(gpa, w, &nb, &sb, layer, id, "down_proj");

    // shape sanity against config
    if (gate.rows != d.inter or gate.cols != d.hidden or
        up.rows != d.inter or up.cols != d.hidden or
        down.rows != d.hidden or down.cols != d.inter)
        return error.ExpertShapeMismatch;

    return .{ .id = id, .gate = gate, .up = up, .down = down };
}


// ---- resident per-layer MoE weights -------------------------------------

pub const Layer = struct {
    router: NativeMatrix, // [experts, hidden]
    sh_gate_proj: NativeMatrix, // [shared_inter, hidden]
    sh_up_proj: NativeMatrix, // [shared_inter, hidden]
    sh_down_proj: NativeMatrix, // [hidden, shared_inter]
    sh_gate: []f32, // [hidden]
    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, w: *const Weights, layer: u32) !Layer {
        var nb: [128]u8 = undefined;
        var sb: [96]u8 = undefined;
        const S = struct {
            fn s(buf: []u8, l: u32, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "layers.{d}.mlp.{s}", .{ l, suffix }) catch unreachable;
            }
        }.s;

        var self: Layer = undefined;
        self.allocator = gpa;
        self.router = try w.matrixBySuffix(S(&sb, layer, "gate.weight"), &nb);
        errdefer self.router.deinit();
        self.sh_gate_proj = try w.matrixBySuffix(S(&sb, layer, "shared_expert.gate_proj.weight"), &nb);
        errdefer self.sh_gate_proj.deinit();
        self.sh_up_proj = try w.matrixBySuffix(S(&sb, layer, "shared_expert.up_proj.weight"), &nb);
        errdefer self.sh_up_proj.deinit();
        self.sh_down_proj = try w.matrixBySuffix(S(&sb, layer, "shared_expert.down_proj.weight"), &nb);
        errdefer self.sh_down_proj.deinit();
        self.sh_gate = try w.vectorBySuffix(S(&sb, layer, "shared_expert_gate.weight"), &nb);
        return self;
    }

    pub fn deinit(self: *Layer) void {
        self.router.deinit();
        self.sh_gate_proj.deinit();
        self.sh_up_proj.deinit();
        self.sh_down_proj.deinit();
        self.allocator.free(self.sh_gate);
        self.* = undefined;
    }
};

pub const Scratch = struct {
    max_tokens: usize,
    logits: []f32,
    sg: []f32,
    su: []f32,
    sh: []f32,
    shared: []f32,
    eg: []f32,
    eu: []f32,
    eh: []f32,
    eo: []f32,
    idx: []u32,
    gates: []f32,
    // Grouped prefill (S > 1): route every token first, then evaluate each
    // distinct expert once against the batch of tokens that picked it.
    r_idx: []u32, // [max_tokens * topk] expert id per (token, slot)
    r_gate: []f32, // [max_tokens * topk] gate per (token, slot)
    ord: []u32, // [max_tokens * topk] slot order, sorted by expert id
    xb: []f32, // [max_tokens * hidden] gathered token rows for one expert
    bg: []f32, // [max_tokens * inter]
    bu: []f32, // [max_tokens * inter]
    bo: []f32, // [max_tokens * hidden]
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: Dims, max_tokens: usize) !Scratch {
        const T = @max(max_tokens, 1);
        return .{
            .max_tokens = T,
            .logits = try gpa.alloc(f32, d.experts),
            .sg = try gpa.alloc(f32, d.shared_inter),
            .su = try gpa.alloc(f32, d.shared_inter),
            .sh = try gpa.alloc(f32, d.shared_inter),
            .shared = try gpa.alloc(f32, d.hidden),
            .eg = try gpa.alloc(f32, d.topk * d.inter),
            .eu = try gpa.alloc(f32, d.topk * d.inter),
            .eh = try gpa.alloc(f32, d.topk * d.inter),
            .eo = try gpa.alloc(f32, d.topk * d.hidden),
            .idx = try gpa.alloc(u32, d.topk),
            .gates = try gpa.alloc(f32, d.topk),
            .r_idx = try gpa.alloc(u32, T * d.topk),
            .r_gate = try gpa.alloc(f32, T * d.topk),
            .ord = try gpa.alloc(u32, T * d.topk),
            .xb = try gpa.alloc(f32, T * d.hidden),
            .bg = try gpa.alloc(f32, T * d.inter),
            .bu = try gpa.alloc(f32, T * d.inter),
            .bo = try gpa.alloc(f32, T * d.hidden),
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *Scratch) void {
        const g = self.allocator;
        inline for (.{ self.logits, self.sg, self.su, self.sh, self.shared, self.eg, self.eu, self.eh, self.eo }) |b| g.free(b);
        g.free(self.idx);
        g.free(self.gates);
        inline for (.{ self.r_idx, self.r_gate, self.ord, self.xb, self.bg, self.bu, self.bo }) |b| g.free(b);
        self.* = undefined;
    }
};

/// Route + evaluate MoE for `x` (`[S, hidden]`), writing `out` (`[S, hidden]`).
/// If `routed_last` is given (len == topk), it receives the expert ids the LAST
/// token routed to (for prefetch prediction).
///
/// Decode (`S == 1`) runs the straight per-token path.  Prefill (`S > 1`) routes
/// every token first, then evaluates each distinct expert **once** against the
/// batch of tokens that picked it — one pass over the 4.7 MiB expert weights
/// instead of one per token.  The two paths accumulate `out` in a different
/// order, so they differ only in f32 rounding (< 1e-4).
pub fn forward(
    layer: *const Layer,
    cache: *ExpertCache,
    w: *const Weights,
    d: Dims,
    x: []const f32,
    S: usize,
    out: []f32,
    sc: *Scratch,
    routed_last: ?[]u32,
    /// This layer's `[experts]` routed-selection counters (learned priors), or null.
    usage_row: ?[]u64,
) !void {
    std.debug.assert(x.len == S * d.hidden and out.len == S * d.hidden);
    if (S == 1) return forwardDense(layer, cache, w, d, x, S, out, sc, routed_last, usage_row);
    std.debug.assert(S <= sc.max_tokens);
    return forwardGrouped(layer, cache, w, d, x, S, out, sc, routed_last, usage_row);
}

inline fn bumpUsage(usage_row: ?[]u64, ids: []const u32) void {
    if (usage_row) |ur| for (ids) |id| {
        if (id < ur.len) ur[id] +|= 1;
    };
}

/// Fill `sc.idx[0..K]` / `sc.gates[0..K]` with the router's top-k pick for `xs`.
fn route(layer: *const Layer, d: Dims, xs: []const f32, sc: *Scratch) void {
    const E = d.experts;
    const K = d.topk;
    layer.router.matmul(sc.logits[0..E], xs, 1);
    var mx: f32 = sc.logits[0];
    for (sc.logits[0..E]) |v| mx = @max(mx, v);
    var all: f64 = 0;
    for (sc.logits[0..E]) |*v| {
        v.* = @exp(v.* - mx);
        all += v.*;
    }
    var top: f64 = 0;
    for (0..K) |z| {
        var best: usize = 0;
        var bv: f32 = -1;
        for (0..E) |e| {
            var chosen = false;
            for (0..z) |j| chosen = chosen or (sc.idx[j] == e);
            if (!chosen and sc.logits[e] > bv) {
                bv = sc.logits[e];
                best = e;
            }
        }
        sc.idx[z] = @intCast(best);
        top += sc.logits[best];
    }
    const den: f64 = if (d.norm_topk) top else all;
    for (0..K) |z| sc.gates[z] = @floatCast(@as(f64, sc.logits[sc.idx[z]]) / den);
}

/// Shared expert (SwiGLU) with its sigmoid gate → `ys[dd] += gate·shared(xs)`.
fn addSharedExpert(layer: *const Layer, d: Dims, xs: []const f32, ys: []f32, sc: *Scratch) void {
    const H = d.hidden;
    const SI = d.shared_inter;
    layer.sh_gate_proj.matmul(sc.sg[0..SI], xs, 1);
    layer.sh_up_proj.matmul(sc.su[0..SI], xs, 1);
    for (0..SI) |j| sc.sh[j] = act.silu(sc.sg[j]) * sc.su[j];
    layer.sh_down_proj.matmul(sc.shared[0..H], sc.sh[0..SI], 1);
    var sgate: f32 = 0;
    for (0..H) |dd| sgate += xs[dd] * layer.sh_gate[dd];
    sgate = act.sigmoid(sgate);
    for (0..H) |dd| ys[dd] += sgate * sc.shared[dd];
}

fn forwardDense(
    layer: *const Layer,
    cache: *ExpertCache,
    w: *const Weights,
    d: Dims,
    x: []const f32,
    S: usize,
    out: []f32,
    sc: *Scratch,
    routed_last: ?[]u32,
    usage_row: ?[]u64,
) !void {
    const H = d.hidden;
    const K = d.topk;
    const I = d.inter;

    const par = cache.cap >= K and K <= 64;

    for (0..S) |s| {
        const xs = x[s * H ..][0..H];
        const ys = out[s * H ..][0..H];
        @memset(ys, 0);

        route(layer, d, xs, sc);
        bumpUsage(usage_row, sc.idx[0..K]);

        if (par) {
            // Pull the K experts into the cache serially (LRU mutation off the
            // parallel path; safe to hold all K pointers since cap >= K), then
            // evaluate them across worker threads into disjoint `eo` rows, then
            // reduce in top-k order — bit-identical to the serial version.
            var ex: [64]*const Expert = undefined;
            for (0..K) |z| ex[z] = try cache.get(w, d, sc.idx[z]);
            evalExpertsParallel(ex[0..K], xs, sc.gates[0..K], sc, I, H);
            for (0..K) |z| {
                const eo = sc.eo[z * H ..][0..H];
                for (0..H) |dd| ys[dd] += eo[dd];
            }
        } else {
            for (0..K) |z| {
                const e = try cache.get(w, d, sc.idx[z]);
                e.gate.matmul(sc.eg[0..I], xs, 1);
                e.up.matmul(sc.eu[0..I], xs, 1);
                for (0..I) |j| sc.eh[j] = act.silu(sc.eg[j]) * sc.eu[j];
                e.down.matmul(sc.eo[0..H], sc.eh[0..I], 1);
                const g = sc.gates[z];
                for (0..H) |dd| ys[dd] += g * sc.eo[dd];
            }
        }
        addSharedExpert(layer, d, xs, ys, sc);

        if (s + 1 == S) {
            if (routed_last) |rl| @memcpy(rl[0..@min(rl.len, K)], sc.idx[0..@min(rl.len, K)]);
        }
    }
}

/// Evaluate `experts` (SwiGLU, gate-scaled) into `sc.eo[z*H..]` — one row per
/// expert, written disjointly so the fan-out needs no locking.  Each expert's
/// own matmuls stay serial (the fan-out is over experts, see `parallel.chunks`).
const EvalCtx = struct {
    ex: []const *const Expert,
    xs: []const f32,
    gates: []const f32,
    eg: []f32,
    eu: []f32,
    eh: []f32,
    eo: []f32,
    I: usize,
    H: usize,
};

fn evalExpertBody(c: EvalCtx, z0: usize, z1: usize) void {
    var z = z0;
    while (z < z1) : (z += 1) {
        const eg = c.eg[z * c.I ..][0..c.I];
        const eu = c.eu[z * c.I ..][0..c.I];
        const eh = c.eh[z * c.I ..][0..c.I];
        const eo = c.eo[z * c.H ..][0..c.H];
        c.ex[z].gate.matmul(eg, c.xs, 1);
        c.ex[z].up.matmul(eu, c.xs, 1);
        for (0..c.I) |j| eh[j] = act.silu(eg[j]) * eu[j];
        c.ex[z].down.matmul(eo, eh, 1);
        const g = c.gates[z];
        for (0..c.H) |dd| eo[dd] *= g;
    }
}

fn evalExpertsParallel(
    experts: []const *const Expert,
    xs: []const f32,
    gates: []const f32,
    sc: *Scratch,
    I: usize,
    H: usize,
) void {
    parallel.chunks(experts.len, experts.len * 3 * I * H, EvalCtx{
        .ex = experts,
        .xs = xs,
        .gates = gates,
        .eg = sc.eg,
        .eu = sc.eu,
        .eh = sc.eh,
        .eo = sc.eo,
        .I = I,
        .H = H,
    }, evalExpertBody);
}

fn expertSlotLess(rid: []const u32, a: u32, b: u32) bool {
    return rid[a] < rid[b];
}

fn forwardGrouped(
    layer: *const Layer,
    cache: *ExpertCache,
    w: *const Weights,
    d: Dims,
    x: []const f32,
    S: usize,
    out: []f32,
    sc: *Scratch,
    routed_last: ?[]u32,
    usage_row: ?[]u64,
) !void {
    const H = d.hidden;
    const K = d.topk;
    const I = d.inter;

    // Phase 1 — route every token, apply its shared expert.
    for (0..S) |s| {
        const xs = x[s * H ..][0..H];
        const ys = out[s * H ..][0..H];
        @memset(ys, 0);

        route(layer, d, xs, sc);
        bumpUsage(usage_row, sc.idx[0..K]);
        for (0..K) |z| {
            sc.r_idx[s * K + z] = sc.idx[z];
            sc.r_gate[s * K + z] = sc.gates[z];
        }
        addSharedExpert(layer, d, xs, ys, sc);

        if (s + 1 == S) {
            if (routed_last) |rl| @memcpy(rl[0..@min(rl.len, K)], sc.idx[0..@min(rl.len, K)]);
        }
    }

    // Phase 2 — order the (token, slot) pairs by expert id.
    const M = S * K;
    for (0..M) |i| sc.ord[i] = @intCast(i);
    std.sort.pdq(u32, sc.ord[0..M], @as([]const u32, sc.r_idx), expertSlotLess);

    // Phase 3 — one expert at a time, batched over the tokens that chose it.
    var p: usize = 0;
    while (p < M) {
        const eid = sc.r_idx[sc.ord[p]];
        var q = p;
        while (q < M and sc.r_idx[sc.ord[q]] == eid) q += 1;
        const n = q - p;

        for (0..n) |k| {
            const tok = sc.ord[p + k] / K;
            @memcpy(sc.xb[k * H ..][0..H], x[tok * H ..][0..H]);
        }
        const ex = try cache.get(w, d, eid);
        ex.gate.matmul(sc.bg[0 .. n * I], sc.xb[0 .. n * H], n);
        ex.up.matmul(sc.bu[0 .. n * I], sc.xb[0 .. n * H], n);
        for (0..n * I) |j| sc.bg[j] = act.silu(sc.bg[j]) * sc.bu[j];
        ex.down.matmul(sc.bo[0 .. n * H], sc.bg[0 .. n * I], n);

        for (0..n) |k| {
            const slot = sc.ord[p + k];
            const tok = slot / K;
            const g = sc.r_gate[slot];
            const ys = out[tok * H ..][0..H];
            const eo = sc.bo[k * H ..][0..H];
            for (0..H) |dd| ys[dd] += g * eo[dd];
        }
        p = q;
    }
}

// ---- tests -----------------------------------------------------------

const testing = std.testing;
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");

test "MoE forward is finite and deterministic on the tiny fixture" {
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
    const layer_idx: u32 = 0;
    var layer = try Layer.load(gpa, &w, layer_idx);
    defer layer.deinit();

    const T = 4;
    var cache = try ExpertCache.init(gpa, layer_idx, d.experts);
    defer cache.deinit();
    var sc = try Scratch.init(gpa, d, T);
    defer sc.deinit();

    const x = try gpa.alloc(f32, T * d.hidden);
    defer gpa.free(x);
    var prng = std.Random.DefaultPrng.init(0x30E);
    const r = prng.random();
    for (x) |*v| v.* = r.float(f32) * 2 - 1;

    const o1 = try gpa.alloc(f32, T * d.hidden);
    defer gpa.free(o1);
    const o2 = try gpa.alloc(f32, T * d.hidden);
    defer gpa.free(o2);

    try forward(&layer, &cache, &w, d, x, T, o1, &sc, null, null);
    for (o1) |v| try testing.expect(std.math.isFinite(v));
    // grouped path: each distinct routed expert is fetched exactly once
    try testing.expect(cache.stats.hits + cache.stats.misses <= d.experts);
    try testing.expect(cache.stats.hits + cache.stats.misses >= 1);

    var cache2 = try ExpertCache.init(gpa, layer_idx, d.experts);
    defer cache2.deinit();
    try forward(&layer, &cache2, &w, d, x, T, o2, &sc, null, null);
    for (o1, o2) |a, b| try testing.expectApproxEqRel(a, b, 1e-5);

    // grouped prefill ≈ per-token dense path (differ only in f32 add order)
    const od = try gpa.alloc(f32, T * d.hidden);
    defer gpa.free(od);
    var cache3 = try ExpertCache.init(gpa, layer_idx, d.experts);
    defer cache3.deinit();
    for (0..T) |s| {
        try forward(&layer, &cache3, &w, d, x[s * d.hidden ..][0..d.hidden], 1, od[s * d.hidden ..][0..d.hidden], &sc, null, null);
    }
    for (o1, od) |a, b| try testing.expectApproxEqAbs(a, b, 1e-3);

    // cap 1 → each distinct expert evicts the previous
    var tight = try ExpertCache.init(gpa, layer_idx, 1);
    defer tight.deinit();
    try forward(&layer, &tight, &w, d, x, T, o1, &sc, null, null);
    try testing.expect(tight.residentCount() <= 1);
    try testing.expect(tight.stats.evictions > 0);

    // replay with a full warm cache: no new misses, no evictions
    const misses_before = cache.stats.misses;
    try forward(&layer, &cache, &w, d, x, T, o2, &sc, null, null);
    try testing.expectEqual(misses_before, cache.stats.misses);
    try testing.expectEqual(@as(u64, 0), cache.stats.evictions);
}
