//! Token sampling: temperature → top-k → top-p (nucleus) → draw.
//!
//! `temperature == 0` (the default) is exact greedy / argmax - deterministic,
//! bit-identical to the old decode loop.  Any of `temperature > 0`, `top_k > 0`,
//! `top_p < 1` switches on stochastic sampling seeded from `seed` (0 → time).

const std = @import("std");

pub const Config = struct {
    temperature: f32 = 0,
    top_k: u32 = 0, // 0 = no limit
    top_p: f32 = 1.0, // 1 = no limit
    seed: u64 = 0, // 0 = seed from the clock
};

pub const Sampler = struct {
    cfg: Config,
    seed_used: u64,
    rng: std.Random.DefaultPrng,
    order: []u32, // vocab, reused each pick
    prob: []f32, // vocab, reused each pick
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, vocab: usize, cfg: Config) !Sampler {
        const seed = if (cfg.seed != 0) cfg.seed else seedFromClock(io);
        return .{
            .cfg = cfg,
            .seed_used = seed,
            .rng = std.Random.DefaultPrng.init(seed),
            .order = try gpa.alloc(u32, vocab),
            .prob = try gpa.alloc(f32, vocab),
            .allocator = gpa,
        };
    }

    pub fn deinit(self: *Sampler) void {
        self.allocator.free(self.order);
        self.allocator.free(self.prob);
        self.* = undefined;
    }

    pub fn greedy(self: Sampler) bool {
        return self.cfg.temperature <= 0 and self.cfg.top_k == 0 and self.cfg.top_p >= 1.0;
    }

    /// Pick the next token id from `logits` (`logits.len == vocab`).
    pub fn pick(self: *Sampler, logits: []const f32) usize {
        if (self.greedy()) return argmax(logits);

        const n = logits.len;
        const temp = @max(self.cfg.temperature, 1e-4);

        // candidate set: top-k by logit (or all)
        for (self.order[0..n], 0..) |*o, i| o.* = @intCast(i);
        var k: usize = n;
        if (self.cfg.top_k != 0 and self.cfg.top_k < n) {
            k = self.cfg.top_k;
            partialTopK(self.order[0..n], logits, k);
        }
        // sort the k candidates by logit, descending
        std.sort.pdq(u32, self.order[0..k], logits, gtByLogit);

        // softmax over the candidates (max-subtracted, temperature-scaled)
        const maxl = logits[self.order[0]];
        var sum: f64 = 0;
        for (self.order[0..k], 0..) |id, j| {
            const p = @exp((logits[id] - maxl) / temp);
            self.prob[j] = @floatCast(p);
            sum += p;
        }

        // top-p: shortest prefix whose cumulative probability reaches p
        var m = k;
        if (self.cfg.top_p < 1.0) {
            const target = self.cfg.top_p * @as(f32, @floatCast(sum));
            var cum: f64 = 0;
            m = 0;
            while (m < k) {
                cum += self.prob[m];
                m += 1;
                if (cum >= target) break;
            }
        }

        // draw from prob[0..m]
        var acc: f64 = 0;
        for (self.prob[0..m]) |p| acc += p;
        const r = self.rng.random().float(f64) * acc;
        var c: f64 = 0;
        for (self.order[0..m], 0..) |id, j| {
            c += self.prob[j];
            if (r <= c) return id;
        }
        return self.order[m - 1];
    }
};

fn argmax(logits: []const f32) usize {
    var best: usize = 0;
    for (logits, 0..) |v, i| {
        if (v > logits[best]) best = i;
    }
    return best;
}

fn gtByLogit(logits: []const f32, a: u32, b: u32) bool {
    return logits[a] > logits[b];
}

/// Partition `idx` so its first `k` entries are the `k` largest by `logits`
/// (unordered).  Simple selection - `k` is tiny next to a decode step.
fn partialTopK(idx: []u32, logits: []const f32, k: usize) void {
    var i: usize = 0;
    while (i < k) : (i += 1) {
        var best = i;
        var j = i + 1;
        while (j < idx.len) : (j += 1) {
            if (logits[idx[j]] > logits[idx[best]]) best = j;
        }
        const t = idx[i];
        idx[i] = idx[best];
        idx[best] = t;
    }
}

/// A fresh seed each run from the platform RNG (non-secure is fine - it only
/// needs to vary).
fn seedFromClock(io: std.Io) u64 {
    var buf: [8]u8 = undefined;
    io.random(&buf);
    return std.mem.readInt(u64, &buf, .little);
}

// ---- tests ---------------------------------------------------------------

test "greedy config picks the argmax deterministically" {
    var s = try Sampler.init(std.testing.allocator, std.testing.io, 6, .{});
    defer s.deinit();
    try std.testing.expect(s.greedy());
    const logits = [_]f32{ 0.1, 3.0, -1.0, 2.9, 0.0, 1.5 };
    try std.testing.expectEqual(@as(usize, 1), s.pick(&logits));
    try std.testing.expectEqual(@as(usize, 1), s.pick(&logits));
}

test "top-k 1 is greedy even with temperature" {
    var s = try Sampler.init(std.testing.allocator, std.testing.io, 6, .{ .temperature = 2.0, .top_k = 1, .seed = 7 });
    defer s.deinit();
    const logits = [_]f32{ 0.1, 3.0, -1.0, 2.9, 0.0, 1.5 };
    for (0..8) |_| try std.testing.expectEqual(@as(usize, 1), s.pick(&logits));
}

test "sampling stays within the top-p nucleus" {
    var s = try Sampler.init(std.testing.allocator, std.testing.io, 8, .{ .temperature = 1.0, .top_p = 0.9, .seed = 42 });
    defer s.deinit();
    // one clearly dominant logit, rest negligible → nucleus is {0}
    const logits = [_]f32{ 20.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0 };
    for (0..16) |_| try std.testing.expectEqual(@as(usize, 0), s.pick(&logits));
}

test "temperature spreads the draw across candidates" {
    var s = try Sampler.init(std.testing.allocator, std.testing.io, 4, .{ .temperature = 1.5, .seed = 123 });
    defer s.deinit();
    const logits = [_]f32{ 1.0, 1.0, 1.0, 1.0 }; // uniform → all four reachable
    var seen = [_]bool{false} ** 4;
    for (0..200) |_| seen[s.pick(&logits)] = true;
    try std.testing.expect(seen[0] and seen[1] and seen[2] and seen[3]);
}
