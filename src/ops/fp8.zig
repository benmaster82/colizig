//! OCP FP8 (E4M3) decoding and block-scaled matmul for routed MoE experts.
//!
//! The checkpoint stores each expert projection as E4M3 bytes plus a
//! `weight_scale_inv` tensor of per-128×128-block f32 scales.  Effective weight
//! element `(o, i)` is `e4m3(byte[o,i]) · scale[o/128, i/128]`.  `matmulFp8`
//! folds the scale in per block, matching colibri's fused `matmul_fp8`.

const std = @import("std");
const parallel = @import("../runtime/parallel.zig");
const mm = @import("matmul.zig");

pub const block: usize = 128;

/// Largest `I` (input width) the row-dequant matmul handles on the stack; every
/// routed-expert projection is well under this (`hidden` = 2560).
const max_dequant_row: usize = 8192;

pub fn nblk(x: usize) usize {
    return (x + block - 1) / block;
}

/// E4M3 byte → f32 (no scale). Bias 7, 3-bit mantissa; `0x7F`/`0xFF` are NaN.
///
/// Backed by a 256-entry table computed at comptime: the hot MoE matmul reads
/// millions of weight bytes per expert and a branchy scalar decode there was the
/// dominant cost.  `e4m3_lut` is bit-identical to the scalar `decodeE4M3` below
/// (the `m·2^k` scale is exact for the whole E4M3 exponent range).
pub fn e4m3ToF32(byte: u8) f32 {
    return e4m3_lut[byte];
}

pub const e4m3_lut: [256]f32 = blk: {
    @setEvalBranchQuota(10000);
    var t: [256]f32 = undefined;
    for (&t, 0..) |*slot, b| slot.* = decodeE4M3(@intCast(b));
    break :blk t;
};

fn decodeE4M3(byte: u8) f32 {
    const sign: f32 = if (byte & 0x80 != 0) -1.0 else 1.0;
    const exp: u32 = (byte >> 3) & 0x0f;
    const mant: u32 = byte & 0x07;
    if (exp == 0) {
        if (mant == 0) return sign * 0.0;
        return sign * pow2f(@as(f32, @floatFromInt(mant)) / 8.0, 1 - 7); // subnormal
    }
    if (exp == 0x0f and mant == 0x07) return sign * @as(f32, @bitCast(@as(u32, 0x7fc00000)));
    const m = 1.0 + @as(f32, @floatFromInt(mant)) / 8.0;
    return sign * pow2f(m, @as(i32, @intCast(exp)) - 7);
}

/// `m · 2^k`, evaluated in f64 then narrowed.  Exact for k in the E4M3 range
/// (|k| ≤ 9), and comptime-friendly (no `std.math.ldexp`).
fn pow2f(m: f32, k: i32) f32 {
    var r: f64 = m;
    var i: i32 = 0;
    if (k >= 0) {
        while (i < k) : (i += 1) r *= 2.0;
    } else {
        while (i > k) : (i -= 1) r *= 0.5;
    }
    return @floatCast(r);
}

// ---- SIMD E4M3 row dequant -------------------------------------------------

const lanes: usize = std.simd.suggestVectorLength(u32) orelse 8;
comptime {
    std.debug.assert(block % lanes == 0); // a lane-chunk never straddles a scale block
}

const L = lanes;
const Vu = @Vector(L, u32);
const Vf = @Vector(L, f32);

/// Decode one `L`-lane chunk of E4M3 bytes to f32 and fold in the block scale
/// `sc` (broadcast — the caller guarantees the chunk stays inside one 128-block).
/// The f32 bit pattern is built arithmetically; see `dequantRow`'s doc comment.
inline fn decodeLane(bytes: @Vector(L, u8), sc: Vf) Vf {
    const Vs = @Vector(L, u5);
    const zero: Vu = @splat(0);
    const s7: Vs = @splat(7);
    const s3: Vs = @splat(3);
    const s31: Vs = @splat(31);
    const s23: Vs = @splat(23);
    const s20: Vs = @splat(20);
    const m4: Vu = @splat(0x0f);
    const m3: Vu = @splat(0x07);
    const bias: Vu = @splat(120);
    const sub_step: Vf = @splat(0x1p-9);
    const v15: Vu = @splat(15);
    const v7: Vu = @splat(7);
    const nan_bits: Vu = @splat(0x7fc00000);

    const b: Vu = @as(Vu, bytes);
    const sign32 = (b >> s7) << s31;
    const exp = (b >> s3) & m4;
    const mant = b & m3;
    const normal_bits = sign32 | ((exp + bias) << s23) | (mant << s20);
    const mant_f: Vf = @floatFromInt(mant);
    const subn_bits = @as(Vu, @bitCast(mant_f * sub_step)) | sign32;
    var bits = @select(u32, exp == zero, subn_bits, normal_bits);
    bits = @select(u32, (exp == v15) & (mant == v7), nan_bits, bits);
    return @as(Vf, @bitCast(bits)) * sc;
}

/// Decode a full E4M3 weight row into f32, folding in the per-128-block scale,
/// entirely with `@Vector` ops — no 256-entry gather, no per-byte branch.
///
/// E4M3 is `[sign:1][exp:4][mant:3]`, bias 7.  A **normal** value
/// `(-1)^s · (1 + mant/8) · 2^(exp-7)` maps straight onto the f32 field layout
/// `[sign:1][exp:8][mant:23]`, bias 127:
///
///     f32_bits = (s << 31) | ((exp + 120) << 23) | (mant << 20)
///
/// (`exp + 120` re-biases 7→127; `mant << 20` places the 3 mantissa bits).  This
/// is bit-exact — no rounding, the value is representable.  **Subnormals**
/// (`exp == 0`, mant ≠ 0) are `(-1)^s · mant · 2^-9`, computed as a small normal
/// f32 `float(mant) · 2^-9` with the sign bit OR'd back in.  Zero and the lone
/// NaN (`0x7F` / `0xFF`) fall out of these two paths (NaN is `@select`ed in).
/// Bit-identical to `e4m3_lut` for every value that appears in a real checkpoint.
fn dequantRow(out: []f32, w: []const u8, scale_row: []const f32) void {
    var i: usize = 0;
    while (i + L <= w.len) : (i += L) {
        const sc: Vf = @splat(scale_row[i / block]);
        out[i..][0..L].* = decodeLane(w[i..][0..L].*, sc);
    }
    while (i < w.len) : (i += 1) out[i] = e4m3_lut[w[i]] * scale_row[i / block];
}

/// Fused decode + dot for a single activation row (the S == 1 decode path):
/// decode each E4M3 lane-chunk straight into two `@mulAdd` accumulator chains,
/// skipping the f32 weight-row buffer and its store/reload.  Two accumulators
/// hide the FMA latency; the reduction order differs from the `dequantRow` +
/// `dotF32` path only by f32 rounding (< 1e-4, same as grouped vs dense).
fn dotFp8Row(x: []const f32, w: []const u8, scale_row: []const f32) f32 {
    var acc0: Vf = @splat(0);
    var acc1: Vf = @splat(0);
    var i: usize = 0;
    while (i + 2 * L <= w.len) : (i += 2 * L) {
        const sc0: Vf = @splat(scale_row[i / block]);
        const w0 = decodeLane(w[i..][0..L].*, sc0);
        const x0: Vf = x[i..][0..L].*;
        acc0 = @mulAdd(Vf, x0, w0, acc0);
        const sc1: Vf = @splat(scale_row[(i + L) / block]);
        const w1 = decodeLane(w[i + L ..][0..L].*, sc1);
        const x1: Vf = x[i + L ..][0..L].*;
        acc1 = @mulAdd(Vf, x1, w1, acc1);
    }
    while (i + L <= w.len) : (i += L) {
        const sc: Vf = @splat(scale_row[i / block]);
        const wv = decodeLane(w[i..][0..L].*, sc);
        const xv: Vf = x[i..][0..L].*;
        acc0 = @mulAdd(Vf, xv, wv, acc0);
    }
    var sum: f32 = @reduce(.Add, acc0 + acc1);
    while (i < w.len) : (i += 1) sum += x[i] * (e4m3_lut[w[i]] * scale_row[i / block]);
    return sum;
}

/// y[S, O] = x[S, I] @ (dequant(w))ᵀ, with `w` row-major `[O, I]` E4M3 and
/// `scales` row-major `[nblk(O), nblk(I)]`.
pub fn matmulFp8(
    y: []f32,
    x: []const f32,
    w: []const u8,
    scales: []const f32,
    S: usize,
    I: usize,
    O: usize,
) void {
    const nbi = nblk(I);
    std.debug.assert(x.len == S * I and w.len == O * I and y.len == S * O);
    std.debug.assert(scales.len == nblk(O) * nbi);
    std.debug.assert(I <= max_dequant_row);

    const Ctx = struct {
        y: []f32,
        x: []const f32,
        w: []const u8,
        scales: []const f32,
        S: usize,
        I: usize,
        O: usize,
        nbi: usize,
    };
    const ctx = Ctx{ .y = y, .x = x, .w = w, .scales = scales, .S = S, .I = I, .O = O, .nbi = nbi };

    if (S == 1) {
        // Decode path: fuse the E4M3 decode into the dot — no f32 weight-row
        // buffer, no store/reload (the dequant is ~90 % of this kernel's cost).
        parallel.chunks(O, O * I, ctx, struct {
            fn body(c: Ctx, o0: usize, o1: usize) void {
                var o = o0;
                while (o < o1) : (o += 1) {
                    c.y[o] = dotFp8Row(c.x[0..c.I], c.w[o * c.I ..][0..c.I], c.scales[(o / block) * c.nbi ..][0..c.nbi]);
                }
            }
        }.body);
        return;
    }

    // Prefill path (S > 1): dequant each weight row to f32 once (the cost
    // amortises over S dots), then SIMD-dot it against every input row.
    parallel.chunks(O, O * I * S, ctx, struct {
        fn body(c: Ctx, o0: usize, o1: usize) void {
            var rowbuf: [max_dequant_row]f32 = undefined;
            const wf = rowbuf[0..c.I];
            var o = o0;
            while (o < o1) : (o += 1) {
                dequantRow(wf, c.w[o * c.I ..][0..c.I], c.scales[(o / block) * c.nbi ..][0..c.nbi]);
                for (0..c.S) |s| {
                    c.y[s * c.O + o] = mm.dotF32(c.x[s * c.I ..][0..c.I], wf);
                }
            }
        }
    }.body);
}

// ---- tests -----------------------------------------------------------

/// A handful of E4M3 bytes that decode to exact small values — used by the
/// fixture generator and the tests below.
pub const demo_bytes = [_]u8{ 0x38, 0x34, 0x30, 0x3c, 0xb8, 0xb4, 0x40, 0x2c };
pub const demo_values = [_]f32{ 1.0, 0.75, 0.5, 1.5, -1.0, -0.75, 2.0, 0.375 };

test "e4m3 decodes the demo bytes exactly" {
    for (demo_bytes, demo_values) |b, v| {
        try std.testing.expectEqual(v, e4m3ToF32(b));
    }
    try std.testing.expectEqual(@as(f32, 0), e4m3ToF32(0x00));
}

test "matmulFp8 == reference matmul on dequantized weights" {
    const gpa = std.testing.allocator;
    const S = 3;
    const I = 200; // spans two 128-blocks
    const O = 5;

    var prng = std.Random.DefaultPrng.init(0xF8);
    const rnd = prng.random();

    const x = try gpa.alloc(f32, S * I);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.float(f32) * 2 - 1;

    const w = try gpa.alloc(u8, O * I);
    defer gpa.free(w);
    for (w) |*b| b.* = demo_bytes[rnd.uintLessThan(usize, demo_bytes.len)];

    const nbi = nblk(I);
    const scales = try gpa.alloc(f32, nblk(O) * nbi);
    defer gpa.free(scales);
    for (scales) |*sc| sc.* = 0.5 + rnd.float(f32);

    // reference: dequantize to f32 then plain matmul
    const wf = try gpa.alloc(f32, O * I);
    defer gpa.free(wf);
    for (0..O) |o| for (0..I) |i| {
        wf[o * I + i] = e4m3ToF32(w[o * I + i]) * scales[(o / block) * nbi + i / block];
    };

    const y1 = try gpa.alloc(f32, S * O);
    defer gpa.free(y1);
    const y2 = try gpa.alloc(f32, S * O);
    defer gpa.free(y2);

    matmulFp8(y1, x, w, scales, S, I, O);
    @import("matmul.zig").matmul(y2, x, wf, S, I, O);

    for (y1, y2) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-4);
}

test "matmulFp8 S==1 fused decode path matches the reference" {
    const gpa = std.testing.allocator;
    const I = 384; // three 128-blocks
    const O = 6;

    var prng = std.Random.DefaultPrng.init(0x5115);
    const rnd = prng.random();

    const x = try gpa.alloc(f32, I);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.float(f32) * 2 - 1;

    const w = try gpa.alloc(u8, O * I);
    defer gpa.free(w);
    for (w) |*b| { // any finite E4M3 byte, incl. subnormals and zero
        var v = rnd.int(u8);
        if (v == 0x7f or v == 0xff) v = 0x00;
        b.* = v;
    }

    const nbi = nblk(I);
    const scales = try gpa.alloc(f32, nblk(O) * nbi);
    defer gpa.free(scales);
    for (scales) |*sc| sc.* = 0.25 + rnd.float(f32);

    const wf = try gpa.alloc(f32, O * I);
    defer gpa.free(wf);
    for (0..O) |o| for (0..I) |i| {
        wf[o * I + i] = e4m3ToF32(w[o * I + i]) * scales[(o / block) * nbi + i / block];
    };

    const y1 = try gpa.alloc(f32, O);
    defer gpa.free(y1);
    const y2 = try gpa.alloc(f32, O);
    defer gpa.free(y2);

    matmulFp8(y1, x, w, scales, 1, I, O); // fused S==1 path
    @import("matmul.zig").matmul(y2, x, wf, 1, I, O);

    for (y1, y2) |a, b| try std.testing.expectApproxEqRel(b, a, 1e-4);
}

test "dequantRow (SIMD) is bit-identical to the scalar LUT path" {
    const gpa = std.testing.allocator;
    // width that is not a multiple of the vector length and spans two 128-blocks
    const I = 128 + 77;
    const w = try gpa.alloc(u8, I);
    defer gpa.free(w);
    // exercise every byte value, including subnormals / zero / NaN
    for (w, 0..) |*b, i| b.* = @intCast(i % 256);
    const scale_row = [_]f32{ 0.5, 1.75 };

    const got = try gpa.alloc(f32, I);
    defer gpa.free(got);
    dequantRow(got, w, &scale_row);

    for (0..I) |i| {
        const want = e4m3_lut[w[i]] * scale_row[i / block];
        if (std.math.isNan(want)) {
            try std.testing.expect(std.math.isNan(got[i]));
        } else {
            try std.testing.expectEqual(want, got[i]);
        }
    }
}

test "nblk" {
    try std.testing.expectEqual(@as(usize, 1), nblk(1));
    try std.testing.expectEqual(@as(usize, 1), nblk(128));
    try std.testing.expectEqual(@as(usize, 2), nblk(129));
    try std.testing.expectEqual(@as(usize, 20), nblk(2560));
}
