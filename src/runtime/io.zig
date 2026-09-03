//! Unified I/O scheduler (brief §13) — a bounded, typed, priority work queue.
//!
//! Requests name a `ResourceKey` and carry a `Priority`.  `next()` always
//! returns the highest-priority pending request (HIGH before MEDIUM before LOW,
//! FIFO within a priority), so mandatory work is never starved by speculative
//! work.  The queue is bounded: `submit` returns `false` when full and the
//! caller must fall back to a synchronous read.
//!
//! Phase 7b: the scheduler owns the queue + policy + accounting.  The consumer
//! (model.forward) pops keys and performs the read.  PLE row reads additionally
//! fan out concurrently through `std.Io.Group` (see `ple.prefetchRowsAsync`);
//! true compute/IO overlap for experts needs the evented Io backend and a real
//! slow-disk checkpoint to matter — see docs/IO_SCHEDULER.md.

const std = @import("std");

pub const Kind = enum { ple, expert, qsa };

/// HIGH: needed by the next layer.  MEDIUM: predicted next working set.
/// LOW: speculative.
pub const Priority = enum(u2) { high = 0, medium = 1, low = 2 };

pub const ResourceKey = union(Kind) {
    ple: i64, // n-gram table row
    expert: struct { layer: u32, id: u32 },
    qsa: u32, // layer

    pub fn eql(a: ResourceKey, b: ResourceKey) bool {
        if (@as(Kind, a) != @as(Kind, b)) return false;
        return switch (a) {
            .ple => |r| r == b.ple,
            .expert => |e| e.layer == b.expert.layer and e.id == b.expert.id,
            .qsa => |l| l == b.qsa,
        };
    }
};

pub const Stats = struct {
    submitted: u64 = 0,
    deduped: u64 = 0,
    upgraded: u64 = 0,
    rejected_full: u64 = 0,
    serviced: [3]u64 = .{ 0, 0, 0 }, // by priority
    cancelled: u64 = 0,
    queue_peak: usize = 0,

    pub fn servicedTotal(self: Stats) u64 {
        return self.serviced[0] + self.serviced[1] + self.serviced[2];
    }
};

const Entry = struct {
    key: ResourceKey,
    prio: Priority,
    seq: u64, // submission order, for FIFO-within-priority
};

pub const Scheduler = struct {
    capacity: usize,
    q: std.ArrayList(Entry),
    seq: u64 = 0,
    stats: Stats = .{},
    allocator: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, capacity: usize) Scheduler {
        return .{ .capacity = capacity, .q = .empty, .allocator = gpa };
    }

    pub fn deinit(self: *Scheduler) void {
        self.q.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn len(self: Scheduler) usize {
        return self.q.items.len;
    }

    pub fn has(self: Scheduler, key: ResourceKey) bool {
        for (self.q.items) |e| {
            if (e.key.eql(key)) return true;
        }
        return false;
    }

    /// Enqueue `key` at `prio`.  Dedups; a re-submit at a stronger priority
    /// upgrades the existing entry.  Returns `false` if the bounded queue is
    /// full (caller falls back to a synchronous read).
    pub fn submit(self: *Scheduler, key: ResourceKey, prio: Priority) bool {
        for (self.q.items) |*e| {
            if (e.key.eql(key)) {
                self.stats.deduped += 1;
                if (@intFromEnum(prio) < @intFromEnum(e.prio)) {
                    e.prio = prio;
                    self.stats.upgraded += 1;
                }
                return true;
            }
        }
        if (self.q.items.len >= self.capacity) {
            self.stats.rejected_full += 1;
            return false;
        }
        self.seq += 1;
        self.q.append(self.allocator, .{ .key = key, .prio = prio, .seq = self.seq }) catch {
            self.stats.rejected_full += 1;
            return false;
        };
        self.stats.submitted += 1;
        self.stats.queue_peak = @max(self.stats.queue_peak, self.q.items.len);
        return true;
    }

    /// Pop the highest-priority pending key (FIFO within a priority).
    pub fn next(self: *Scheduler) ?ResourceKey {
        if (self.q.items.len == 0) return null;
        var best: usize = 0;
        for (self.q.items, 0..) |e, i| {
            const b = self.q.items[best];
            if (@intFromEnum(e.prio) < @intFromEnum(b.prio) or
                (e.prio == b.prio and e.seq < b.seq)) best = i;
        }
        const e = self.q.orderedRemove(best);
        self.stats.serviced[@intFromEnum(e.prio)] += 1;
        return e.key;
    }

    /// Remove `key` if queued (e.g. it was just demand-served).
    pub fn cancel(self: *Scheduler, key: ResourceKey) bool {
        for (self.q.items, 0..) |e, i| {
            if (e.key.eql(key)) {
                _ = self.q.orderedRemove(i);
                self.stats.cancelled += 1;
                return true;
            }
        }
        return false;
    }

    pub fn clear(self: *Scheduler) void {
        self.q.clearRetainingCapacity();
    }
};

// ---- tests -----------------------------------------------------------

const testing = std.testing;

test "priority ordering: HIGH drains before MEDIUM before LOW" {
    var s = Scheduler.init(testing.allocator, 16);
    defer s.deinit();

    try testing.expect(s.submit(.{ .expert = .{ .layer = 0, .id = 1 } }, .low));
    try testing.expect(s.submit(.{ .ple = 100 }, .medium));
    try testing.expect(s.submit(.{ .qsa = 3 }, .high));
    try testing.expect(s.submit(.{ .ple = 101 }, .high));

    try testing.expectEqual(ResourceKey{ .qsa = 3 }, s.next().?); // high, first
    try testing.expectEqual(ResourceKey{ .ple = 101 }, s.next().?); // high, second
    try testing.expectEqual(ResourceKey{ .ple = 100 }, s.next().?); // medium
    try testing.expectEqual(ResourceKey{ .expert = .{ .layer = 0, .id = 1 } }, s.next().?);
    try testing.expect(s.next() == null);
}

test "bounded queue rejects when full; dedup and upgrade" {
    var s = Scheduler.init(testing.allocator, 2);
    defer s.deinit();

    try testing.expect(s.submit(.{ .ple = 1 }, .low));
    try testing.expect(s.submit(.{ .ple = 2 }, .low));
    try testing.expect(!s.submit(.{ .ple = 3 }, .low)); // full
    try testing.expectEqual(@as(u64, 1), s.stats.rejected_full);

    try testing.expect(s.submit(.{ .ple = 1 }, .high)); // dedup + upgrade
    try testing.expectEqual(@as(u64, 1), s.stats.upgraded);
    try testing.expectEqual(ResourceKey{ .ple = 1 }, s.next().?); // now first
}

test "mandatory work is never starved by a full backlog of speculative work" {
    var s = Scheduler.init(testing.allocator, 64);
    defer s.deinit();
    for (0..50) |i| _ = s.submit(.{ .expert = .{ .layer = 1, .id = @intCast(i) } }, .low);
    try testing.expect(s.submit(.{ .ple = 999 }, .high));
    // the HIGH request comes out first despite 50 LOW requests ahead of it
    try testing.expectEqual(ResourceKey{ .ple = 999 }, s.next().?);
}

test "cancel removes a queued key" {
    var s = Scheduler.init(testing.allocator, 8);
    defer s.deinit();
    _ = s.submit(.{ .expert = .{ .layer = 2, .id = 7 } }, .medium);
    try testing.expect(s.has(.{ .expert = .{ .layer = 2, .id = 7 } }));
    try testing.expect(s.cancel(.{ .expert = .{ .layer = 2, .id = 7 } }));
    try testing.expect(!s.has(.{ .expert = .{ .layer = 2, .id = 7 } }));
    try testing.expect(!s.cancel(.{ .ple = 0 }));
}
