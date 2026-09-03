//! Numerically stable softmax (max-subtracted).

const std = @import("std");

/// In-place softmax over `x`.
pub fn softmax(x: []f32) void {
    if (x.len == 0) return;
    var m: f32 = x[0];
    for (x[1..]) |v| m = @max(m, v);
    var sum: f64 = 0;
    for (x) |*v| {
        const e = @exp(v.* - m);
        v.* = e;
        sum += e;
    }
    const inv: f32 = @floatCast(1.0 / sum);
    for (x) |*v| v.* *= inv;
}

/// softmax of `src` written to `dst` (same length).
pub fn softmaxInto(dst: []f32, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    @memcpy(dst, src);
    softmax(dst);
}

test "softmax sums to one and is order preserving" {
    var x = [_]f32{ 1.0, 2.0, 3.0, -1.0 };
    softmax(&x);
    var sum: f32 = 0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    try std.testing.expect(x[2] > x[1] and x[1] > x[0] and x[0] > x[3]);
}

test "softmax is shift invariant" {
    var a = [_]f32{ 0.5, -2.0, 4.0 };
    var b = [_]f32{ 0.5 + 10, -2.0 + 10, 4.0 + 10 };
    softmax(&a);
    softmax(&b);
    for (a, b) |x, y| try std.testing.expectApproxEqAbs(x, y, 1e-6);
}

test "large constant offset does not overflow" {
    var x = [_]f32{ 1000, 1000, 1000, 1000 };
    softmax(&x);
    for (x) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.25), v, 1e-6);
}
