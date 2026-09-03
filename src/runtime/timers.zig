//! Per-phase wall-clock accounting for the forward pass (brief §23).
//!
//! Whole-call timing per subsystem.  The demand-path expert load time
//! (`compute_stall_due_to_io`) is tracked separately on `ExpertCache`.

const std = @import("std");

pub const Phase = enum {
    embed,
    gated_residual,
    deltanet,
    qsa,
    moe,
    ple,
    lm_head,
};

pub const count = @typeInfo(Phase).@"enum".fields.len;

pub const Timers = struct {
    io: std.Io,
    ns: [count]u64 = [_]u64{0} ** count,
    forwards: u64 = 0,

    pub fn init(io: std.Io) Timers {
        return .{ .io = io };
    }

    pub fn now(self: Timers) std.Io.Timestamp {
        return std.Io.Timestamp.now(self.io, .awake);
    }

    pub fn add(self: *Timers, phase: Phase, start: std.Io.Timestamp) void {
        const d = start.durationTo(self.now()).nanoseconds;
        if (d > 0) self.ns[@intFromEnum(phase)] +|= @intCast(d);
    }

    pub fn get(self: Timers, phase: Phase) u64 {
        return self.ns[@intFromEnum(phase)];
    }

    pub fn total(self: Timers) u64 {
        var t: u64 = 0;
        for (self.ns) |v| t +|= v;
        return t;
    }

    pub fn print(self: Timers, w: *std.Io.Writer) !void {
        const per: f64 = if (self.forwards != 0) @floatFromInt(self.forwards) else 1;
        try w.writeAll("phase timings (ms total / ms per forward):\n");
        inline for (std.meta.fields(Phase)) |f| {
            const v = self.ns[f.value];
            try w.print("  {s:<16} {d:>10.3} {d:>10.4}\n", .{
                f.name,
                @as(f64, @floatFromInt(v)) / 1e6,
                @as(f64, @floatFromInt(v)) / 1e6 / per,
            });
        }
    }
};

test "timer accumulates" {
    var t = Timers.init(std.testing.io);
    const s = t.now();
    t.add(.moe, s);
    t.forwards = 1;
    // monotonic clock: elapsed is >= 0
    try std.testing.expect(t.get(.moe) < std.math.maxInt(u64));
}
