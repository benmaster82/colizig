//! Prefetch prediction (brief §14).
//!
//! `Predictor` is the abstraction: given a layer, produce the resources likely
//! to be needed there, with a confidence and an expected use-distance.
//!
//! `LastTokenPredictor` is the initial concrete predictor: MoE routing has
//! strong token-to-token locality, so the experts a layer used for the previous
//! token are a good guess for the next.  It tracks its own accuracy.

const std = @import("std");

pub const PrefetchPrediction = struct {
    layer: u32,
    expert: u32,
    confidence: f32,
    expected_use_distance: u32,
};

pub const Stats = struct {
    predicted: u64 = 0,
    correct: u64 = 0,

    pub fn accuracy(self: Stats) f64 {
        if (self.predicted == 0) return 0;
        return @as(f64, @floatFromInt(self.correct)) / @as(f64, @floatFromInt(self.predicted));
    }
};

pub const LastTokenPredictor = struct {
    topk: usize,
    /// `last[layer]` = ids routed for the previous token (`have[layer]` gates it).
    last: []u32, // layers * topk
    count: []u8, // valid ids per layer
    stats: Stats = .{},
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, layers: usize, topk: usize) !LastTokenPredictor {
        const last = try gpa.alloc(u32, layers * topk);
        errdefer gpa.free(last);
        const count = try gpa.alloc(u8, layers);
        @memset(count, 0);
        return .{ .topk = topk, .last = last, .count = count, .allocator = gpa };
    }

    pub fn deinit(self: *LastTokenPredictor) void {
        self.allocator.free(self.last);
        self.allocator.free(self.count);
        self.* = undefined;
    }

    pub fn reset(self: *LastTokenPredictor) void {
        @memset(self.count, 0);
        self.stats = .{};
    }

    /// The predicted expert ids for `layer` (empty until the first `record`).
    pub fn predict(self: *const LastTokenPredictor, layer: usize) []const u32 {
        return self.last[layer * self.topk ..][0..self.count[layer]];
    }

    pub fn predictScored(self: *const LastTokenPredictor, layer: usize, out: []PrefetchPrediction) []PrefetchPrediction {
        const ids = self.predict(layer);
        const conf: f32 = @floatCast(@max(0.1, self.stats.accuracy()));
        const n = @min(ids.len, out.len);
        for (0..n) |i| out[i] = .{
            .layer = @intCast(layer),
            .expert = ids[i],
            .confidence = conf,
            .expected_use_distance = 1,
        };
        return out[0..n];
    }

    /// After a layer routed `ids` for the current token: score the previous
    /// prediction and store `ids` for the next token.
    pub fn record(self: *LastTokenPredictor, layer: usize, ids: []const u32) void {
        const prev = self.predict(layer);
        for (ids) |id| {
            self.stats.predicted += 1;
            for (prev) |p| {
                if (p == id) {
                    self.stats.correct += 1;
                    break;
                }
            }
        }
        const n = @min(ids.len, self.topk);
        @memcpy(self.last[layer * self.topk ..][0..n], ids[0..n]);
        self.count[layer] = @intCast(n);
    }
};

// ---- tests -----------------------------------------------------------

test "LastTokenPredictor tracks a stable routing pattern" {
    const gpa = std.testing.allocator;
    var p = try LastTokenPredictor.init(gpa, 3, 2);
    defer p.deinit();

    try std.testing.expectEqual(@as(usize, 0), p.predict(0).len);

    p.record(0, &.{ 5, 9 }); // first token: nothing predicted yet
    try std.testing.expectEqualSlices(u32, &.{ 5, 9 }, p.predict(0));

    p.record(0, &.{ 5, 9 }); // same routing → 2/2 correct
    p.record(0, &.{ 5, 3 }); // 1/2 correct

    try std.testing.expectEqual(@as(u64, 6), p.stats.predicted);
    try std.testing.expectEqual(@as(u64, 3), p.stats.correct);
    try std.testing.expect(p.stats.accuracy() > 0.4 and p.stats.accuracy() < 0.6);
}

test "predictScored fills PrefetchPrediction rows" {
    const gpa = std.testing.allocator;
    var p = try LastTokenPredictor.init(gpa, 2, 3);
    defer p.deinit();
    p.record(1, &.{ 1, 2, 3 });
    var buf: [3]PrefetchPrediction = undefined;
    const preds = p.predictScored(1, &buf);
    try std.testing.expectEqual(@as(usize, 3), preds.len);
    try std.testing.expectEqual(@as(u32, 1), preds[0].layer);
    try std.testing.expectEqual(@as(u32, 2), preds[1].expert);
}
