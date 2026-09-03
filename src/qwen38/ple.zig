//! PLE — hashed n-gram embedding, injected once (at `ple_layer`, index 1).
//!
//! Ported from colibri's `q38_hash_row` / `q38_ple_row` / `q38_ple`.  Per token
//! the bigram/trigram history is hashed to `ngram_heads` row addresses; each row
//! is `ngram_head_dim` E4M3 values read straight from the (never-resident)
//! `split_ngram_parts`-way partitioned table.  The concatenated rows are
//! projected to key/value, gated against the 4-branch residual, folded through a
//! dilated causal conv, and returned to be added into the residual.
//!
//! Row addresses are a pure function of the token ids, so they are known before
//! any compute — `prefetchRows` computes and reads them all up front.  The
//! bounded async I/O queue is Phase 7; here the table is memory-mapped and a
//! "read" is a page fault serviced by the OS page cache.

const std = @import("std");
const Cfg = @import("../model/config.zig").Cfg;
const Weights = @import("../model/weights.zig").Weights;
const View = @import("../model/tensors.zig").View;
const NativeMatrix = @import("../model/tensors.zig").NativeMatrix;
const fp8 = @import("../ops/fp8.zig");
const act = @import("../ops/activation.zig");
const rms = @import("../ops/rmsnorm.zig");

pub const Dims = struct {
    hidden: usize,
    hc_count: usize,
    hc_width: usize,
    ple_dim: usize,
    ple_convk: usize,
    ngram_size: usize,
    ngram_heads: usize,
    ngram_head_dim: usize,
    heads_per_ngram: usize,
    ple_layer: u32,
    eos_id: i64,
    eps: f32,

    pub fn stateLen(self: Dims) usize {
        return (self.ple_convk - 1) * self.ngram_size;
    }

    pub fn of(cfg: Cfg) Dims {
        return .{
            .hidden = cfg.hidden,
            .hc_count = cfg.hc_count,
            .hc_width = cfg.hc_width,
            .ple_dim = cfg.ple_dim,
            .ple_convk = cfg.ple_convk,
            .ngram_size = cfg.ngram_size,
            .ngram_heads = cfg.ngram_heads,
            .ngram_head_dim = cfg.ngram_head_dim,
            .heads_per_ngram = cfg.heads_per_ngram,
            .ple_layer = cfg.ple_layer,
            .eos_id = cfg.eos_id,
            .eps = cfg.eps,
        };
    }
};

// ---- the streamed n-gram table -----------------------------------------

pub const Table = struct {
    /// One `View` per `split_ngram_parts` shard (bytes into the shard mmap).
    shards: []View,
    /// Cumulative row counts, length `shards.len + 1`.
    part_start: []i64,
    ngram_head_dim: usize,
    heads_per_ngram: usize,
    ngram_heads: usize,
    weight_scale: f32,
    multipliers: [3]u64,
    head_vocab: []i64,
    head_offset: []i64,
    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, w: *const Weights, cfg: Cfg) !Table {
        var nb: [200]u8 = undefined;
        var sb: [160]u8 = undefined;
        const L = cfg.ple_layer;

        const scale_v = w.viewBySuffix(
            fmtL(&sb, L, "ple.ple_embedding.ngram_embedding.weight_scale"),
            &nb,
        );
        var weight_scale: f32 = 1.0;
        if (scale_v) |v| {
            var one: [1]f32 = undefined;
            try v.decode(&one);
            weight_scale = one[0];
        }

        const mult = try readI64s(gpa, w, fmtL(&sb, L, "ple.ple_embedding.layer_multipliers"), &nb, cfg.ngram_size);
        defer gpa.free(mult);
        if (mult.len != 3) return error.BadPleMeta;

        const head_vocab = try readI64s(gpa, w, fmtL(&sb, L, "ple.ple_embedding.ngram_heads_vocab_sizes"), &nb, cfg.ngram_heads);
        errdefer gpa.free(head_vocab);
        const head_offset = try readI64s(gpa, w, fmtL(&sb, L, "ple.ple_embedding.ngram_heads_offsets"), &nb, cfg.ngram_heads);
        errdefer gpa.free(head_offset);
        for (head_vocab, head_offset) |vv, oo| {
            if (vv <= 0 or oo < 0) return error.BadPleMeta;
        }

        // shards: single `ngram_embedding.weight`, else `shard_<p>.weight`.
        var shards: std.ArrayList(View) = .empty;
        errdefer shards.deinit(gpa);
        if (w.viewBySuffix(fmtL(&sb, L, "ple.ple_embedding.ngram_embedding.weight"), &nb)) |single| {
            try shards.append(gpa, single);
        } else {
            for (0..cfg.ngram_parts) |p| {
                const name = std.fmt.bufPrint(&sb, "layers.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.weight", .{ L, p }) catch unreachable;
                const v = w.viewBySuffix(name, &nb) orelse return error.PleShardMissing;
                try shards.append(gpa, v);
            }
        }

        const part_start = try gpa.alloc(i64, shards.items.len + 1);
        errdefer gpa.free(part_start);
        part_start[0] = 0;
        for (shards.items, 0..) |v, i| {
            if (v.shape.len != 2 or @as(usize, @intCast(v.shape[1])) != cfg.ngram_head_dim)
                return error.BadPleShard;
            if (v.dtype != .f8_e4m3 and v.dtype != .f32) return error.BadPleShard;
            part_start[i + 1] = part_start[i] + @as(i64, @intCast(v.shape[0]));
        }

        // the table must cover every reachable row
        var need: i64 = 0;
        for (head_vocab, head_offset) |vv, oo| need = @max(need, oo + vv);
        if (part_start[shards.items.len] < need) return error.PleTableTooSmall;

        return .{
            .shards = try shards.toOwnedSlice(gpa),
            .part_start = part_start,
            .ngram_head_dim = cfg.ngram_head_dim,
            .heads_per_ngram = cfg.heads_per_ngram,
            .ngram_heads = cfg.ngram_heads,
            .weight_scale = weight_scale,
            .multipliers = .{ @bitCast(mult[0]), @bitCast(mult[1]), @bitCast(mult[2]) },
            .head_vocab = head_vocab,
            .head_offset = head_offset,
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *Table) void {
        self.allocator.free(self.shards);
        self.allocator.free(self.part_start);
        self.allocator.free(self.head_vocab);
        self.allocator.free(self.head_offset);
        self.* = undefined;
    }

    pub fn totalRows(self: Table) i64 {
        return self.part_start[self.shards.len];
    }

    /// Deterministic row address for `head` given the current token and its
    /// (up to two) predecessors.  Bigram for `head < heads_per_ngram`, else
    /// trigram.  Matches colibri `q38_hash_row` exactly (u64 wrap, C `%`).
    pub fn hashRow(self: Table, head: usize, cur: i64, p1: i64, p2: i64) i64 {
        var x: u64 = @as(u64, @bitCast(cur)) *% self.multipliers[0];
        x ^= @as(u64, @bitCast(p1)) *% self.multipliers[1];
        if (head >= self.heads_per_ngram) x ^= @as(u64, @bitCast(p2)) *% self.multipliers[2];
        const sx: i64 = @bitCast(x);
        const modv = self.head_vocab[head];
        var r = @rem(sx, modv);
        if (r < 0) r += modv;
        return self.head_offset[head] + r;
    }

    /// Read one row (`out.len == ngram_head_dim`), applying the scalar scale.
    pub fn readRow(self: Table, row: i64, out: []f32) void {
        std.debug.assert(out.len == self.ngram_head_dim);
        std.debug.assert(row >= 0 and row < self.totalRows());
        var p: usize = 0;
        while (p + 1 < self.shards.len and row >= self.part_start[p + 1]) p += 1;
        const local: usize = @intCast(row - self.part_start[p]);
        const v = self.shards[p];
        const nd = self.ngram_head_dim;
        switch (v.dtype) {
            .f8_e4m3 => {
                const base = local * nd;
                for (0..nd) |d| out[d] = fp8.e4m3ToF32(v.bytes[base + d]) * self.weight_scale;
            },
            .f32 => {
                const base = local * nd * 4;
                for (0..nd) |d| out[d] = @bitCast(std.mem.readInt(u32, v.bytes[base + d * 4 ..][0..4], .little));
            },
            else => unreachable,
        }
    }
};

fn fmtL(buf: []u8, layer: u32, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ layer, suffix }) catch unreachable;
}

fn readI64s(gpa: std.mem.Allocator, w: *const Weights, suffix: []const u8, nb: []u8, count: usize) ![]i64 {
    const v = w.viewBySuffix(suffix, nb) orelse return error.BadPleMeta;
    if (v.dtype != .i64 or v.numel() != count or v.bytes.len != count * 8) return error.BadPleMeta;
    const out = try gpa.alloc(i64, count);
    errdefer gpa.free(out);
    for (0..count) |k| out[k] = std.mem.readInt(i64, v.bytes[k * 8 ..][0..8], .little);
    return out;
}

// ---- persistent state --------------------------------------------------

pub const State = struct {
    ring: []f32, // hc_width * stateLen
    history: [2]i64 = .{ 0, 0 },
    history_len: u8 = 0,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: Dims) !State {
        const ring = try gpa.alloc(f32, d.hc_width * d.stateLen());
        @memset(ring, 0);
        return .{ .ring = ring, .allocator = gpa };
    }

    pub fn reset(self: *State) void {
        @memset(self.ring, 0);
        self.history = .{ 0, 0 };
        self.history_len = 0;
    }

    pub fn deinit(self: *State) void {
        self.allocator.free(self.ring);
        self.* = undefined;
    }

    fn predecessors(self: State, eos: i64) struct { p1: i64, p2: i64 } {
        return .{
            .p1 = if (self.history_len >= 1) self.history[self.history_len - 1] else eos,
            .p2 = if (self.history_len >= 2) self.history[self.history_len - 2] else eos,
        };
    }

    fn advance(self: *State, tok: i64, eos: i64) void {
        if (tok == eos) {
            self.history_len = 0;
        } else if (self.history_len == 0) {
            self.history[0] = tok;
            self.history_len = 1;
        } else if (self.history_len == 1) {
            self.history[1] = tok;
            self.history_len = 2;
        } else {
            self.history[0] = self.history[1];
            self.history[1] = tok;
        }
    }
};

// ---- resident PLE weights --------------------------------------------

pub const Layer = struct {
    key: NativeMatrix, // [hc_width, ple_dim]
    value: NativeMatrix, // [hidden, ple_dim]
    norm_key: []f32, // [hc_width]
    norm_query: []f32, // [hc_width]
    norm_conv: []f32, // [hc_width]
    conv: []f32, // [hc_width, ple_convk]
    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, w: *const Weights, layer: u32) !Layer {
        var nb: [128]u8 = undefined;
        var sb: [96]u8 = undefined;
        const S = struct {
            fn s(buf: []u8, l: u32, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "layers.{d}.ple.{s}", .{ l, suffix }) catch unreachable;
            }
        }.s;
        var self: Layer = undefined;
        self.allocator = gpa;
        self.key = try w.matrixBySuffix(S(&sb, layer, "key_proj.weight"), &nb);
        errdefer self.key.deinit();
        self.value = try w.matrixBySuffix(S(&sb, layer, "value_proj.weight"), &nb);
        errdefer self.value.deinit();
        self.norm_key = try w.vectorBySuffix(S(&sb, layer, "norm_key.weight"), &nb);
        errdefer gpa.free(self.norm_key);
        self.norm_query = try w.vectorBySuffix(S(&sb, layer, "norm_query.weight"), &nb);
        errdefer gpa.free(self.norm_query);
        self.norm_conv = try w.vectorBySuffix(S(&sb, layer, "norm_conv.weight"), &nb);
        errdefer gpa.free(self.norm_conv);
        self.conv = try w.vectorBySuffix(S(&sb, layer, "conv1d.weight"), &nb);
        return self;
    }

    pub fn deinit(self: *Layer) void {
        self.key.deinit();
        self.value.deinit();
        self.allocator.free(self.norm_key);
        self.allocator.free(self.norm_query);
        self.allocator.free(self.norm_conv);
        self.allocator.free(self.conv);
        self.* = undefined;
    }
};

pub const Scratch = struct {
    emb: []f32,
    keys: []f32,
    value: []f32,
    kn: []f32,
    qn: []f32,
    gated: []f32,
    norm: []f32,
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: Dims) !Scratch {
        return .{
            .emb = try gpa.alloc(f32, d.ple_dim),
            .keys = try gpa.alloc(f32, d.hc_width),
            .value = try gpa.alloc(f32, d.hidden),
            .kn = try gpa.alloc(f32, d.hc_width),
            .qn = try gpa.alloc(f32, d.hc_width),
            .gated = try gpa.alloc(f32, d.hc_width),
            .norm = try gpa.alloc(f32, d.hc_width),
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *Scratch) void {
        const g = self.allocator;
        inline for (.{ self.emb, self.keys, self.value, self.kn, self.qn, self.gated, self.norm }) |b| g.free(b);
        self.* = undefined;
    }
};

/// Pre-compute and read every n-gram row for `ids[0..S]` up front.  The history
/// window is *simulated*, not mutated.  Result is `[S · ngram_heads · ngram_head_dim]`.
pub fn prefetchRows(
    gpa: std.mem.Allocator,
    table: Table,
    d: Dims,
    ids: []const i64,
    S: usize,
    state: State,
) ![]f32 {
    const nd = d.ngram_head_dim;
    const nh = d.ngram_heads;
    const buf = try gpa.alloc(f32, S * nh * nd);
    errdefer gpa.free(buf);

    var hist = state;
    for (0..S) |s| {
        const pr = hist.predecessors(d.eos_id);
        for (0..nh) |h| {
            const row = table.hashRow(h, ids[s], pr.p1, pr.p2);
            table.readRow(row, buf[(s * nh + h) * nd ..][0..nd]);
        }
        hist.advance(ids[s], d.eos_id);
    }
    return buf;
}

fn readOneRow(table: Table, row: i64, out: []f32) void {
    table.readRow(row, out);
}

/// Like `prefetchRows`, but when `io` is available the `S · ngram_heads`
/// independent row reads fan out concurrently through `std.Io.Group`.  This is
/// the deterministic-prefetch path: every address is a pure function of the
/// token ids, so the reads can start before layer 0 runs.  Bit-identical result.
pub fn prefetchRowsAsync(
    gpa: std.mem.Allocator,
    table: Table,
    d: Dims,
    ids: []const i64,
    S: usize,
    state: State,
    io: ?std.Io,
) ![]f32 {
    const io_ = io orelse return prefetchRows(gpa, table, d, ids, S, state);

    const nd = d.ngram_head_dim;
    const nh = d.ngram_heads;
    const total = S * nh;
    const buf = try gpa.alloc(f32, total * nd);
    errdefer gpa.free(buf);
    const rows = try gpa.alloc(i64, total);
    defer gpa.free(rows);

    var hist = state;
    for (0..S) |s| {
        const pr = hist.predecessors(d.eos_id);
        for (0..nh) |h| rows[s * nh + h] = table.hashRow(h, ids[s], pr.p1, pr.p2);
        hist.advance(ids[s], d.eos_id);
    }

    var group: std.Io.Group = .init;
    for (0..total) |i| {
        group.async(io_, readOneRow, .{ table, rows[i], buf[i * nd ..][0..nd] });
    }
    group.await(io_) catch {};
    return buf;
}

/// Compute the PLE contribution for `ids[0..S]`, advancing `state`, writing
/// `out` (`[S, hc_width]`).  The caller adds `out` into the residual.
/// `prefetch`, if given, is the buffer from `prefetchRows` for the same chunk.
pub fn forward(
    layer: *const Layer,
    table: Table,
    state: *State,
    d: Dims,
    ids: []const i64,
    S: usize,
    hyper: []const f32,
    out: []f32,
    prefetch: ?[]const f32,
    sc: *Scratch,
) void {
    const H = d.hidden;
    const C = d.hc_count;
    const W = d.hc_width;
    const E = d.ple_dim;
    const CK = d.ple_convk;
    const SL = d.stateLen();
    const nh = d.ngram_heads;
    const nd = d.ngram_head_dim;
    std.debug.assert(hyper.len == S * W and out.len == S * W);

    for (0..S) |s| {
        const pr = state.predecessors(d.eos_id);

        if (prefetch) |pf| {
            @memcpy(sc.emb[0 .. nh * nd], pf[s * nh * nd ..][0 .. nh * nd]);
        } else {
            for (0..nh) |h| {
                const row = table.hashRow(h, ids[s], pr.p1, pr.p2);
                table.readRow(row, sc.emb[h * nd ..][0..nd]);
            }
        }

        layer.key.matmul(sc.keys[0..W], sc.emb[0..E], 1);
        layer.value.matmul(sc.value[0..H], sc.emb[0..E], 1);

        const hyper_row = hyper[s * W ..][0..W];
        for (0..C) |b| {
            const off = b * H;
            rms.rms0(sc.kn[off..][0..H], sc.keys[off..][0..H], layer.norm_key[off..][0..H], d.eps);
            rms.rms0(sc.qn[off..][0..H], hyper_row[off..][0..H], layer.norm_query[off..][0..H], d.eps);
            var dot: f32 = 0;
            for (0..H) |dd| dot += sc.kn[off + dd] * sc.qn[off + dd];
            dot /= @sqrt(@as(f32, @floatFromInt(H)));
            const shaped = std.math.copysign(@sqrt(@max(@abs(dot), 1e-6)), dot);
            const g = act.sigmoid(shaped);
            for (0..H) |dd| sc.gated[off + dd] = g * sc.value[dd];
            rms.rms0(sc.norm[off..][0..H], sc.gated[off..][0..H], layer.norm_conv[off..][0..H], d.eps);
        }

        const out_row = out[s * W ..][0..W];
        for (0..W) |dd| {
            var a: f32 = layer.conv[dd * CK + CK - 1] * sc.norm[dd];
            const ring = state.ring[dd * SL ..][0..SL];
            for (0..CK - 1) |k| a += layer.conv[dd * CK + k] * ring[k * d.ngram_size];
            out_row[dd] = sc.gated[dd] + act.silu(a);
            var k: usize = 0;
            while (k + 1 < SL) : (k += 1) ring[k] = ring[k + 1];
            ring[SL - 1] = sc.norm[dd];
        }

        state.advance(ids[s], d.eos_id);
    }
}

// ---- tests -----------------------------------------------------------

const testing = std.testing;
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");

test "PLE on the tiny fixture: prefetch-equivalent, chunk-invariant, finite" {
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
    var table = try Table.load(gpa, &w, m.cfg);
    defer table.deinit();
    var layer = try Layer.load(gpa, &w, d.ple_layer);
    defer layer.deinit();

    // hash address range
    for (0..d.ngram_heads) |h| {
        const row = table.hashRow(h, 5, 2, 9);
        try testing.expect(row >= table.head_offset[h] and row < table.head_offset[h] + table.head_vocab[h]);
        try testing.expect(row >= 0 and row < table.totalRows());
    }

    const T = 5;
    const W = d.hc_width;
    var prng = std.Random.DefaultPrng.init(0x9E);
    const r = prng.random();
    const ids = try gpa.alloc(i64, T);
    defer gpa.free(ids);
    for (ids) |*v| v.* = r.intRangeAtMost(i64, 0, @as(i64, @intCast(m.cfg.vocab - 1)));
    const hyper = try gpa.alloc(f32, T * W);
    defer gpa.free(hyper);
    for (hyper) |*v| v.* = r.float(f32) * 2 - 1;

    var sc = try Scratch.init(gpa, d);
    defer sc.deinit();

    // inline reads
    var s1 = try State.init(gpa, d);
    defer s1.deinit();
    const o1 = try gpa.alloc(f32, T * W);
    defer gpa.free(o1);
    forward(&layer, table, &s1, d, ids, T, hyper, o1, null, &sc);
    for (o1) |v| try testing.expect(std.math.isFinite(v));

    // prefetched reads → identical
    var s2 = try State.init(gpa, d);
    defer s2.deinit();
    const pf = try prefetchRows(gpa, table, d, ids, T, s2);
    defer gpa.free(pf);
    const o2 = try gpa.alloc(f32, T * W);
    defer gpa.free(o2);
    forward(&layer, table, &s2, d, ids, T, hyper, o2, pf, &sc);
    for (o1, o2) |a, bv| try testing.expectEqual(a, bv);

    // chunk-boundary invariance
    var s3 = try State.init(gpa, d);
    defer s3.deinit();
    const o3 = try gpa.alloc(f32, T * W);
    defer gpa.free(o3);
    forward(&layer, table, &s3, d, ids[0..2], 2, hyper[0 .. 2 * W], o3[0 .. 2 * W], null, &sc);
    forward(&layer, table, &s3, d, ids[2..], 3, hyper[2 * W ..], o3[2 * W ..], null, &sc);
    for (o1, o3) |a, cv| try testing.expectApproxEqRel(a, cv, 1e-4);
}
