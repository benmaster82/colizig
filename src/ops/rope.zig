//! Rotary position embedding - NeoX split-half form, matching colibri's
//! `q38_rope`.  Only the first `rotary_dim` lanes of a head are rotated
//! (partial rotary factor); the rest pass through.
//!
//! Text-only path: `pos` is the plain token index.  Multimodal mRoPE sectioning
//! is out of scope (see docs/ARCHITECTURE.md).

const std = @import("std");

/// Rotate `x[0..rotary_dim]` in place for absolute position `pos`.
pub fn rope(x: []f32, rotary_dim: usize, pos: usize, theta: f32) void {
    std.debug.assert(rotary_dim % 2 == 0);
    std.debug.assert(rotary_dim <= x.len);
    const half = rotary_dim / 2;
    const p: f32 = @floatFromInt(pos);
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const exp: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rotary_dim));
        const ang = p / std.math.pow(f32, theta, exp);
        const co = @cos(ang);
        const si = @sin(ang);
        const a = x[i];
        const b = x[i + half];
        x[i] = a * co - b * si;
        x[i + half] = b * co + a * si;
    }
}

test "rope at position 0 is the identity" {
    var x = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const orig = x;
    rope(&x, 4, 0, 10000);
    for (x, orig) |v, o| try std.testing.expectApproxEqAbs(o, v, 1e-6);
}

test "rope preserves the norm of each rotated pair and leaves the tail alone" {
    var x = [_]f32{ 0.3, -1.1, 2.0, 0.7, 9.0, -9.0 };
    const orig = x;
    rope(&x, 4, 7, 1_000_000);
    // pair (0,2)
    try std.testing.expectApproxEqAbs(
        orig[0] * orig[0] + orig[2] * orig[2],
        x[0] * x[0] + x[2] * x[2],
        1e-4,
    );
    // pair (1,3)
    try std.testing.expectApproxEqAbs(
        orig[1] * orig[1] + orig[3] * orig[3],
        x[1] * x[1] + x[3] * x[3],
        1e-4,
    );
    // tail untouched
    try std.testing.expectEqual(orig[4], x[4]);
    try std.testing.expectEqual(orig[5], x[5]);
}

test "rotating by +pos then -pos round-trips" {
    var x = [_]f32{ 0.5, 1.5, -2.5, 3.5 };
    const orig = x;
    // rope is a rotation by angle a(pos); apply pos then a matching inverse.
    rope(&x, 4, 3, 10000);
    // inverse rotation: negate the sin terms by rotating with swapped halves
    const half = 2;
    const p: f32 = 3;
    inline for (0..half) |i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / 4.0;
        const ang = p / std.math.pow(f32, 10000, e);
        const co = @cos(ang);
        const si = @sin(ang);
        const a = x[i];
        const b = x[i + half];
        x[i] = a * co + b * si;
        x[i + half] = b * co - a * si;
    }
    for (x, orig) |v, o| try std.testing.expectApproxEqAbs(o, v, 1e-4);
}
