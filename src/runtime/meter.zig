//! A pass-through allocator that tracks current and peak live bytes.
//!
//! This is *our* allocation accounting, not process RSS - but it lines up with
//! the memory-model tiers and lets `benchmark` / `stress` report an honest
//! "peak tracked bytes" without a platform RSS call.

const std = @import("std");

pub const Meter = struct {
    child: std.mem.Allocator,
    current: usize = 0,
    peak: usize = 0,

    pub fn init(child: std.mem.Allocator) Meter {
        return .{ .child = child };
    }

    pub fn allocator(self: *Meter) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn note(self: *Meter, delta: isize) void {
        if (delta >= 0) {
            self.current += @intCast(delta);
        } else {
            self.current -|= @intCast(-delta);
        }
        self.peak = @max(self.peak, self.current);
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *Meter = @ptrCast(@alignCast(ctx));
        const p = self.child.rawAlloc(len, alignment, ra) orelse return null;
        self.note(@intCast(len));
        return p;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *Meter = @ptrCast(@alignCast(ctx));
        if (!self.child.rawResize(buf, alignment, new_len, ra)) return false;
        self.note(@as(isize, @intCast(new_len)) - @as(isize, @intCast(buf.len)));
        return true;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *Meter = @ptrCast(@alignCast(ctx));
        const p = self.child.rawRemap(buf, alignment, new_len, ra) orelse return null;
        self.note(@as(isize, @intCast(new_len)) - @as(isize, @intCast(buf.len)));
        return p;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *Meter = @ptrCast(@alignCast(ctx));
        self.child.rawFree(buf, alignment, ra);
        self.note(-@as(isize, @intCast(buf.len)));
    }
};

test "meter tracks current and peak" {
    var m = Meter.init(std.testing.allocator);
    const a = m.allocator();
    const p1 = try a.alloc(u8, 1000);
    try std.testing.expectEqual(@as(usize, 1000), m.current);
    const p2 = try a.alloc(u8, 500);
    try std.testing.expectEqual(@as(usize, 1500), m.current);
    try std.testing.expectEqual(@as(usize, 1500), m.peak);
    a.free(p1);
    try std.testing.expectEqual(@as(usize, 500), m.current);
    try std.testing.expectEqual(@as(usize, 1500), m.peak);
    a.free(p2);
    try std.testing.expectEqual(@as(usize, 0), m.current);
}
