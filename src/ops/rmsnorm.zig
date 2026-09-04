//! RMS normalization, both Qwen4-Exp variants (matching colibri).
//!
//! `rms0`  — zero-centered: the learned scale is `1 + w` (Qwen4-Exp norms).
//! `rmsGated` — inherited from Qwen3-Next's DeltaNet `RMSNormGated`: NOT
//!             zero-centered, and multiplied by a gate branch.
//!
//! Sum-of-squares is accumulated in f64 like the reference.

const std = @import("std");
const act = @import("activation.zig");

/// out_i = x_i * rsqrt(mean(x^2) + eps) * (1 + w_i)
pub fn rms0(out: []f32, x: []const f32, w: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len and x.len == w.len);
    const n = x.len;
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, v) * v;
    // colibri casts the mean to f32 before adding eps — match it.
    const mean: f32 = @floatCast(ss / @as(f64, @floatFromInt(n)));
    const r: f32 = 1.0 / @sqrt(mean + eps);
    for (out, x, w) |*o, v, wi| o.* = v * r * (1.0 + wi);
}

pub fn rms0InPlace(x: []f32, w: []const f32, eps: f32) void {
    rms0(x, x, w, eps);
}

/// Plain RMSNorm: `out_i = x_i * rsqrt(mean(x^2) + eps) * w_i` — the scale is
/// `w`, not `1 + w`. Used by Qwen2/Qwen3 (`Qwen3RMSNorm`), including the
/// per-head QK-norm. `w.len` may be shorter than `x.len` (per-head norm applied
/// to each `w.len`-wide slice) — pass matching lengths.
pub fn rms(out: []f32, x: []const f32, w: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len and x.len == w.len);
    const n = x.len;
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, v) * v;
    const mean: f32 = @floatCast(ss / @as(f64, @floatFromInt(n)));
    const r: f32 = 1.0 / @sqrt(mean + eps);
    for (out, x, w) |*o, v, wi| o.* = v * r * wi;
}

pub fn rmsInPlace(x: []f32, w: []const f32, eps: f32) void {
    rms(x, x, w, eps);
}

/// out_i = x_i * rsqrt(mean(x^2) + eps) * w_i * g(gate_i)
/// where g is sigmoid when `sigmoid_gate`, else silu.
pub fn rmsGated(
    out: []f32,
    x: []const f32,
    gate: []const f32,
    w: []const f32,
    eps: f32,
    sigmoid_gate: bool,
) void {
    std.debug.assert(out.len == x.len and x.len == w.len and x.len == gate.len);
    const n = x.len;
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, v) * v;
    const mean: f32 = @floatCast(ss / @as(f64, @floatFromInt(n)));
    const r: f32 = 1.0 / @sqrt(mean + eps);
    for (out, x, gate, w) |*o, v, gi, wi| {
        const g = if (sigmoid_gate) act.sigmoid(gi) else act.silu(gi);
        o.* = v * r * wi * g;
    }
}

test "rms0 with unit input and zero weight normalizes to unit RMS" {
    var out: [4]f32 = undefined;
    const x = [_]f32{ 1, -1, 1, -1 };
    const w = [_]f32{ 0, 0, 0, 0 };
    rms0(&out, &x, &w, 0);
    for (out) |v| try std.testing.expectApproxEqAbs(@as(f32, 1.0), @abs(v), 1e-6);
}

test "rms0 weight acts as (1 + w) scale" {
    var a: [3]f32 = undefined;
    var b: [3]f32 = undefined;
    const x = [_]f32{ 2, 4, 6 };
    rms0(&a, &x, &[_]f32{ 0, 0, 0 }, 1e-6);
    rms0(&b, &x, &[_]f32{ 1, 1, 1 }, 1e-6);
    for (a, b) |av, bv| try std.testing.expectApproxEqRel(2.0 * av, bv, 1e-5);
}

test "rmsGated folds in the gate branch" {
    var g_out: [3]f32 = undefined;
    var plain: [3]f32 = undefined;
    const x = [_]f32{ 1, 2, 3 };
    const w = [_]f32{ 1, 1, 1 };
    const gate = [_]f32{ 0, 0, 0 }; // sigmoid(0) = 0.5
    rmsGated(&g_out, &x, &gate, &w, 1e-6, true);
    // plain rms with scale w (not 1+w): recompute directly
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, v) * v;
    const r: f32 = @floatCast(1.0 / @sqrt(ss / 3.0 + 1e-6));
    for (&plain, x, w) |*o, v, wi| o.* = v * r * wi;
    for (g_out, plain) |gv, pv| try std.testing.expectApproxEqRel(0.5 * pv, gv, 1e-5);
}
