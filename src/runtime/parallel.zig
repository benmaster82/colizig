//! Opt-in loop parallelism for the hot kernels, via `std.Io.Group`.
//!
//! A process-wide switch set once by the CLI before any forward: the ops are
//! leaves that either have a worker pool available or don't.  Unit tests leave
//! it off, so their results stay deterministic; `forward` / `chat` / `benchmark`
//! turn it on.
//!
//! `chunks` splits `[0, n)` into contiguous ranges and runs `body(ctx, lo, hi)`
//! on each.  `body` must only write disjoint / append-only state across chunks
//! (all current call sites write `y[... + o]` for `o` in their own range).

const std = @import("std");

var g_io: ?std.Io = null;
var g_max_tasks: usize = 1;

/// Below this many scalar ops, fan-out overhead outweighs the work — stay
/// serial.  Tuned so the toy fixture never threads but the real model's
/// projections (millions of MACs) do.
pub const min_work: usize = 96 * 1024;

/// Enable parallelism.  `max_tasks` caps the fan-out (0 → CPU count).
pub fn enable(io: std.Io, max_tasks: usize) void {
    g_io = io;
    g_max_tasks = if (max_tasks == 0) (std.Thread.getCpuCount() catch 1) else max_tasks;
}

pub fn disable() void {
    g_io = null;
    g_max_tasks = 1;
}

pub fn enabled() bool {
    return g_io != null and g_max_tasks > 1;
}

pub fn maxTasks() usize {
    return g_max_tasks;
}

/// Set while a `chunks` body runs on a worker thread, so a nested `chunks` call
/// from inside it stays serial instead of spawning a second fan-out over the
/// same pool (e.g. the MoE loop fans experts, and each expert's matmul must not
/// fan again).
threadlocal var in_worker: bool = false;

/// Run `body(ctx, lo, hi)` over `[0, n)`.  `work` is the approximate total
/// scalar-op count of the whole loop; below `min_work` it runs serially even
/// when parallelism is enabled.
pub fn chunks(n: usize, work: usize, ctx: anytype, comptime body: anytype) void {
    if (n == 0) return;
    const io = g_io orelse {
        body(ctx, 0, n);
        return;
    };
    const tasks = @min(g_max_tasks, n);
    if (tasks <= 1 or work < min_work or in_worker) {
        body(ctx, 0, n);
        return;
    }
    const Wrap = struct {
        fn run(c: @TypeOf(ctx), lo: usize, hi: usize) void {
            in_worker = true;
            defer in_worker = false;
            body(c, lo, hi);
        }
    };
    const csize = (n + tasks - 1) / tasks;
    var group: std.Io.Group = .init;
    var lo: usize = 0;
    while (lo < n) : (lo += csize) {
        const hi = @min(lo + csize, n);
        group.async(io, Wrap.run, .{ ctx, lo, hi });
    }
    group.await(io) catch {};
}

// ---- tests --------------------------------------------------------

test "chunks covers [0,n) exactly once, serial and parallel" {
    const gpa = std.testing.allocator;
    const n = 1000;
    const buf = try gpa.alloc(u32, n);
    defer gpa.free(buf);

    const Ctx = struct { b: []u32 };
    const body = struct {
        fn f(c: Ctx, lo: usize, hi: usize) void {
            for (lo..hi) |i| c.b[i] += 1;
        }
    }.f;

    @memset(buf, 0);
    disable();
    chunks(n, 1 << 20, Ctx{ .b = buf }, body);
    for (buf) |v| try std.testing.expectEqual(@as(u32, 1), v);

    @memset(buf, 0);
    enable(std.testing.io, 8);
    defer disable();
    chunks(n, 1 << 20, Ctx{ .b = buf }, body); // above min_work → really parallel
    for (buf) |v| try std.testing.expectEqual(@as(u32, 1), v);

    @memset(buf, 0);
    chunks(n, 10, Ctx{ .b = buf }, body); // below min_work → serial fallback
    for (buf) |v| try std.testing.expectEqual(@as(u32, 1), v);
}
