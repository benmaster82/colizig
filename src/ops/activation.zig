//! Scalar elementwise activations shared by the Qwen4-Exp kernels.
//!
//! Forms match colibri's `q38_sigmoid` / `q38_silu` / `q38_softplus` so numeric
//! results line up with the reference engine.

const std = @import("std");

/// Numerically stable logistic sigmoid.
pub fn sigmoid(x: f32) f32 {
    if (x >= 0) {
        const z = @exp(-x);
        return 1.0 / (1.0 + z);
    }
    const z = @exp(x);
    return z / (1.0 + z);
}

pub fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

/// softplus(x) = log(1 + e^x), with the large-x shortcut colibri uses.
pub fn softplus(x: f32) f32 {
    return if (x > 20.0) x else std.math.log1p(@exp(x));
}

/// tanh approximation of GELU (`gelu_pytorch_tanh`).
pub fn geluTanh(x: f32) f32 {
    const c: f32 = 0.7978845608028654; // sqrt(2/pi)
    const inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

pub fn siluInPlace(v: []f32) void {
    for (v) |*e| e.* = silu(e.*);
}

pub fn sigmoidInPlace(v: []f32) void {
    for (v) |*e| e.* = sigmoid(e.*);
}

test "sigmoid / silu reference points" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), sigmoid(1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(0), 1e-6);
    try std.testing.expectApproxEqAbs(sigmoid(-3), 1.0 - sigmoid(3), 1e-6);
}

test "softplus large-x shortcut is continuous enough" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.6931472), softplus(0), 1e-6);
    try std.testing.expectApproxEqRel(softplus(19.9), @as(f32, 19.9) + std.math.log1p(@exp(@as(f32, -19.9))), 1e-4);
}
