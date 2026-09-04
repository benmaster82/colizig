//! Learned routed-expert priors, persisted next to the checkpoint.
//!
//! MoE routing is highly skewed: a small, stable set of experts per layer takes
//! most of the traffic across a whole conversation.  `ExpertUsage` counts how
//! often each `(layer, expert)` is routed and writes the totals to
//! `<model_dir>/.colizig_usage`; the next run reads them back and pre-warms each
//! bounded `ExpertCache` with its historically-hottest experts, so the first
//! tokens hit warm slots instead of thrashing the LRU cold.
//!
//! Best-effort throughout: a missing / stale / unwritable file never fails
//! inference, it just means no priors.

const std = @import("std");
const moe = @import("../qwen38/moe.zig");
const parallel = @import("parallel.zig");
const Weights = @import("../model/weights.zig").Weights;

const magic = "CZU1";
pub const filename = ".colizig_usage";

pub const ExpertUsage = struct {
    layers: u32,
    experts: u32,
    counts: []u64, // [layers * experts], owned, LE on disk
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, layers: u32, experts: u32) !ExpertUsage {
        const counts = try gpa.alloc(u64, @as(usize, layers) * experts);
        @memset(counts, 0);
        return .{ .layers = layers, .experts = experts, .counts = counts, .allocator = gpa };
    }

    pub fn deinit(self: *ExpertUsage) void {
        self.allocator.free(self.counts);
        self.* = undefined;
    }

    pub fn row(self: *const ExpertUsage, layer: u32) []u64 {
        return self.counts[@as(usize, layer) * self.experts ..][0..self.experts];
    }

    /// Record that `layer` routed to `expert` (saturating).
    pub fn bump(self: *ExpertUsage, layer: u32, expert: u32) void {
        if (layer >= self.layers or expert >= self.experts) return;
        const c = &self.counts[@as(usize, layer) * self.experts + expert];
        c.* +|= 1;
    }

    /// The `out.len` hottest experts of `layer`, **ascending** by count so a warm
    /// loop that fills a cache in order leaves the hottest with the highest LRU
    /// clock (evicted last).  Returns the filled prefix (may be shorter if fewer
    /// experts were ever seen).
    pub fn topAscending(self: *const ExpertUsage, layer: u32, out: []u32) []u32 {
        const r = self.row(layer);
        var picked: [512]u32 = undefined;
        const want = @min(out.len, @min(@as(usize, self.experts), picked.len));
        var used = std.StaticBitSet(512).initEmpty();
        var n: usize = 0;
        // selection sort of the top `want`: O(want · experts), both ≤ 512.
        while (n < want) : (n += 1) {
            var best: usize = 0;
            var best_c: u64 = 0;
            for (r, 0..) |c, e| {
                if (!used.isSet(e) and c > best_c) {
                    best_c = c;
                    best = e;
                }
            }
            if (best_c == 0) break; // nothing left that was ever routed
            used.set(best);
            picked[n] = @intCast(best);
        }
        for (0..n) |i| out[i] = picked[n - 1 - i]; // descending → ascending
        return out[0..n];
    }

    /// Read `<dir>/.colizig_usage`.  `null` if absent, unreadable, or its
    /// `(layers, experts)` header does not match the model.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, layers: u32, experts: u32) ?ExpertUsage {
        var f = dir.openFile(io, filename, .{}) catch return null;
        defer f.close(io);

        const want: usize = 12 + @as(usize, layers) * experts * 8;
        const size: usize = @intCast((f.stat(io) catch return null).size);
        if (size != want) return null;

        const buf = gpa.alloc(u8, size) catch return null;
        defer gpa.free(buf);
        const got = f.readPositionalAll(io, buf, 0) catch return null;
        if (got != size) return null;
        if (!std.mem.eql(u8, buf[0..4], magic)) return null;
        if (std.mem.readInt(u32, buf[4..8], .little) != layers) return null;
        if (std.mem.readInt(u32, buf[8..12], .little) != experts) return null;

        const u = ExpertUsage.init(gpa, layers, experts) catch return null;
        for (u.counts, 0..) |*c, i| c.* = std.mem.readInt(u64, buf[12 + i * 8 ..][0..8], .little);
        return u;
    }

    /// Merge `self` into `<dir>/.colizig_usage` (load existing totals, add,
    /// rewrite).  Best-effort - silently does nothing on any I/O error.
    pub fn save(self: *const ExpertUsage, io: std.Io, dir: std.Io.Dir) void {
        const gpa = self.allocator;

        // existing totals with a matching header, or all-zero
        var prior: ExpertUsage = undefined;
        var have_prior = false;
        if (load(gpa, io, dir, self.layers, self.experts)) |p| {
            prior = p;
            have_prior = true;
        }
        defer if (have_prior) prior.deinit();

        const buf = gpa.alloc(u8, 12 + self.counts.len * 8) catch return;
        defer gpa.free(buf);
        @memcpy(buf[0..4], magic);
        std.mem.writeInt(u32, buf[4..8], self.layers, .little);
        std.mem.writeInt(u32, buf[8..12], self.experts, .little);
        for (self.counts, 0..) |c, i| {
            const base = if (have_prior) prior.counts[i] else 0;
            std.mem.writeInt(u64, buf[12 + i * 8 ..][0..8], base +| c, .little);
        }

        var wf = dir.createFile(io, filename, .{}) catch return;
        defer wf.close(io);
        wf.writePositionalAll(io, buf, 0) catch {};
    }
};

var g_sink: usize = 0;

/// Pre-fill every bounded `ExpertCache` in `caches` with its layer's hottest
/// experts from `usage`, then fault those experts' E4M3 pages into the OS cache
/// (up to `page_budget` bytes, hottest layers first) so the first tokens neither
/// miss the LRU nor stall on a cold SSD read.  A no-op for caches large enough
/// to hold every expert.  `page_budget == 0` skips the page touch.
pub fn warmCaches(
    usage: *const ExpertUsage,
    caches: []moe.ExpertCache,
    w: *const Weights,
    d: moe.Dims,
    page_budget: u64,
) void {
    var buf: [512]u32 = undefined;
    for (caches, 0..) |*cache, layer| {
        if (cache.cap >= d.experts) continue;
        const cap = @min(cache.cap, buf.len);
        const hot = usage.topAscending(@intCast(layer), buf[0..cap]);
        for (hot) |eid| cache.prefetch(w, d, eid) catch {};
    }
    if (page_budget == 0) return;

    // Collect the E4M3 byte slices of every warmed expert, then fault them in
    // parallel (page faults are I/O-bound, so the workers overlap SSD reads).
    var slices: std.ArrayList([]const u8) = .empty;
    defer slices.deinit(caches[0].allocator);
    var acc: u64 = 0;
    outer: for (caches) |*cache| {
        for (cache.slots) |slot| {
            if (slot.expert) |e| {
                for ([_][]const u8{ e.gate.data, e.up.data, e.down.data }) |s| {
                    if (acc + s.len > page_budget) break :outer;
                    slices.append(caches[0].allocator, s) catch break :outer;
                    acc += s.len;
                }
            }
        }
    }
    const Ctx = struct { s: [][]const u8 };
    parallel.chunks(slices.items.len, acc, Ctx{ .s = slices.items }, struct {
        fn body(c: Ctx, lo: usize, hi: usize) void {
            var x: usize = 0;
            for (c.s[lo..hi]) |bytes| {
                var i: usize = 0;
                while (i < bytes.len) : (i += 4096) x +%= bytes[i];
            }
            _ = @atomicRmw(usize, &g_sink, .Xor, x, .monotonic);
        }
    }.body);
}

// ---- tests ---------------------------------------------------------------

test "usage round-trips and merges through the file" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var u = try ExpertUsage.init(gpa, 3, 8);
    defer u.deinit();
    u.bump(0, 5);
    u.bump(0, 5);
    u.bump(0, 2);
    u.bump(2, 7);
    u.save(io, tmp.dir);

    // reload → same totals
    var v = ExpertUsage.load(gpa, io, tmp.dir, 3, 8) orelse return error.TestUnexpectedResult;
    defer v.deinit();
    try std.testing.expectEqual(@as(u64, 2), v.row(0)[5]);
    try std.testing.expectEqual(@as(u64, 1), v.row(0)[2]);
    try std.testing.expectEqual(@as(u64, 1), v.row(2)[7]);

    // hottest of layer 0, ascending: [2, 5]
    var out: [4]u32 = undefined;
    const hot = v.topAscending(0, out[0..4]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 5 }, hot);

    // save again → totals double
    v.save(io, tmp.dir);
    var w2 = ExpertUsage.load(gpa, io, tmp.dir, 3, 8) orelse return error.TestUnexpectedResult;
    defer w2.deinit();
    try std.testing.expectEqual(@as(u64, 4), w2.row(0)[5]);

    // wrong dims → ignored
    try std.testing.expect(ExpertUsage.load(gpa, io, tmp.dir, 4, 8) == null);
}
