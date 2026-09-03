//! Dense matrix multiply: `y[S,O] = x[S,I] @ Wᵀ`, with `W` row-major `[O,I]`
//! (each output row `o` is the weight vector dotted with each input row).
//! This matches colibri's `q38_matmul` / `q38_matmul_bf16`.
//!
//! Activations and accumulation are f32; BF16 weights are widened per element
//! (no BF16-rounded dot product), matching the reference.
//!
//! `@Vector` SIMD dot products; the output-row loop parallelizes through
//! `runtime/parallel.zig` when the CLI enables it (each task writes a disjoint
//! set of output columns, so the result is identical to the serial path).

const std = @import("std");
const parallel = @import("../runtime/parallel.zig");

const lanes = std.simd.suggestVectorLength(f32) orelse 8;
const V = @Vector(lanes, f32);
const Vu16 = @Vector(lanes, u16);
const Vu32 = @Vector(lanes, u32);

/// f32 weights.
pub fn matmul(y: []f32, x: []const f32, w: []const f32, S: usize, I: usize, O: usize) void {
    std.debug.assert(x.len == S * I and w.len == O * I and y.len == S * O);
    const Ctx = struct { y: []f32, x: []const f32, w: []const f32, S: usize, I: usize, O: usize };
    parallel.chunks(O, O * I * S, Ctx{ .y = y, .x = x, .w = w, .S = S, .I = I, .O = O }, struct {
        fn body(c: Ctx, o0: usize, o1: usize) void {
            var o = o0;
            while (o < o1) : (o += 1) {
                const wr = c.w[o * c.I ..][0..c.I];
                for (0..c.S) |s| c.y[s * c.O + o] = dotF32(c.x[s * c.I ..][0..c.I], wr);
            }
        }
    }.body);
}

/// BF16 weights (raw u16 bit patterns), f32 activations and accumulation.
pub fn matmulBf16(y: []f32, x: []const f32, w: []const u16, S: usize, I: usize, O: usize) void {
    std.debug.assert(x.len == S * I and w.len == O * I and y.len == S * O);
    const Ctx = struct { y: []f32, x: []const f32, w: []const u16, S: usize, I: usize, O: usize };
    parallel.chunks(O, O * I * S, Ctx{ .y = y, .x = x, .w = w, .S = S, .I = I, .O = O }, struct {
        fn body(c: Ctx, o0: usize, o1: usize) void {
            var o = o0;
            while (o < o1) : (o += 1) {
                const wr = c.w[o * c.I ..][0..c.I];
                for (0..c.S) |s| c.y[s * c.O + o] = dotBf16(c.x[s * c.I ..][0..c.I], wr);
            }
        }
    }.body);
}

// The dot products run four independent `@mulAdd` (FMA) accumulator chains so the
// ~4-cycle FMA latency is hidden by throughput (one serial `acc += va*vb` chain
// stalls at ~1 FMA / 4 cycles).  `@mulAdd` contracts mul+add into a single
// fused instruction — this is also what colibri's `gcc -O3 -march=native` auto-
// vectoriser emits (`-ffp-contract=fast` is on by default), so the result stays
// numerically aligned with the C reference.  All the model's inner widths
// (2560, 640, 256, 128, vocab) are multiples of `unroll*lanes`, so the fast loop
// carries them; the two tail loops cover the fixture's odd shapes.
const unroll = 4;

pub fn dotF32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const n = a.len;
    var acc: [unroll]V = @splat(@as(V, @splat(0)));
    var i: usize = 0;
    while (i + unroll * lanes <= n) : (i += unroll * lanes) {
        inline for (0..unroll) |k| {
            const va: V = a[i + k * lanes ..][0..lanes].*;
            const vb: V = b[i + k * lanes ..][0..lanes].*;
            acc[k] = @mulAdd(V, va, vb, acc[k]);
        }
    }
    while (i + lanes <= n) : (i += lanes) {
        const va: V = a[i..][0..lanes].*;
        const vb: V = b[i..][0..lanes].*;
        acc[0] = @mulAdd(V, va, vb, acc[0]);
    }
    var total: V = @splat(0);
    inline for (0..unroll) |k| total += acc[k];
    var sum: f32 = @reduce(.Add, total);
    while (i < n) : (i += 1) sum = @mulAdd(f32, a[i], b[i], sum);
    return sum;
}

pub fn dotBf16(a: []const f32, b: []const u16) f32 {
    std.debug.assert(a.len == b.len);
    const n = a.len;
    const sh: @Vector(lanes, u5) = @splat(16);
    var acc: [unroll]V = @splat(@as(V, @splat(0)));
    var i: usize = 0;
    while (i + unroll * lanes <= n) : (i += unroll * lanes) {
        inline for (0..unroll) |k| {
            const va: V = a[i + k * lanes ..][0..lanes].*;
            // bf16 is the top 16 bits of an f32 — widen the whole lane and shift.
            const raw: Vu16 = b[i + k * lanes ..][0..lanes].*;
            const wide: V = @bitCast(@as(Vu32, raw) << sh);
            acc[k] = @mulAdd(V, va, wide, acc[k]);
        }
    }
    while (i + lanes <= n) : (i += lanes) {
        const va: V = a[i..][0..lanes].*;
        const raw: Vu16 = b[i..][0..lanes].*;
        const wide: V = @bitCast(@as(Vu32, raw) << sh);
        acc[0] = @mulAdd(V, va, wide, acc[0]);
    }
    var total: V = @splat(0);
    inline for (0..unroll) |k| total += acc[k];
    var sum: f32 = @reduce(.Add, total);
    while (i < n) : (i += 1) sum = @mulAdd(f32, a[i], bf16ToF32(b[i]), sum);
    return sum;
}

pub fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn f32ToBf16(v: f32) u16 {
    // round-to-nearest-even
    const x: u32 = @bitCast(v);
    const rounding_bias: u32 = 0x7fff + ((x >> 16) & 1);
    return @truncate((x + rounding_bias) >> 16);
}

// ---- tests -------------------------------------------------------------

fn naive(y: []f32, x: []const f32, w: []const f32, S: usize, I: usize, O: usize) void {
    for (0..S) |s| for (0..O) |o| {
        var acc: f32 = 0;
        for (0..I) |i| acc += x[s * I + i] * w[o * I + i];
        y[s * O + o] = acc;
    };
}

test "SIMD matmul agrees with the naive triple loop" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();

    const S = 5;
    const I = 37; // deliberately not a multiple of the vector width
    const O = 9;
    const x = try gpa.alloc(f32, S * I);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, O * I);
    defer gpa.free(w);
    const y1 = try gpa.alloc(f32, S * O);
    defer gpa.free(y1);
    const y2 = try gpa.alloc(f32, S * O);
    defer gpa.free(y2);
    for (x) |*v| v.* = rnd.float(f32) * 2 - 1;
    for (w) |*v| v.* = rnd.float(f32) * 2 - 1;

    matmul(y1, x, w, S, I, O);
    naive(y2, x, w, S, I, O);
    for (y1, y2) |a, b| try std.testing.expectApproxEqRel(b, a, 1e-4);
}

test "bf16 matmul matches f32 matmul on bf16-exact values" {
    const gpa = std.testing.allocator;
    const S = 3;
    const I = 20;
    const O = 4;
    const xf = try gpa.alloc(f32, S * I);
    defer gpa.free(xf);
    const wf = try gpa.alloc(f32, O * I);
    defer gpa.free(wf);
    const wb = try gpa.alloc(u16, O * I);
    defer gpa.free(wb);
    var prng = std.Random.DefaultPrng.init(1);
    const rnd = prng.random();
    for (xf) |*v| v.* = bf16ToF32(f32ToBf16(rnd.float(f32)));
    for (wf, wb) |*fv, *bv| {
        const b = f32ToBf16(rnd.float(f32));
        bv.* = b;
        fv.* = bf16ToF32(b);
    }
    const y1 = try gpa.alloc(f32, S * O);
    defer gpa.free(y1);
    const y2 = try gpa.alloc(f32, S * O);
    defer gpa.free(y2);
    matmul(y1, xf, wf, S, I, O);
    matmulBf16(y2, xf, wb, S, I, O);
    for (y1, y2) |a, b| try std.testing.expectApproxEqRel(a, b, 1e-5);
}

test "bf16 round-trip of representable values" {
    for ([_]f32{ 0, 1, -1, 0.5, 2, -8, 1.5 }) |v| {
        try std.testing.expectEqual(v, bf16ToF32(f32ToBf16(v)));
    }
}

test "threaded matmul is bit-identical to serial (large enough to actually fan out)" {
    const gpa = std.testing.allocator;
    const S = 2;
    const I = 256;
    const O = 512; // O*I*S = 256Ki > parallel.min_work
    var prng = std.Random.DefaultPrng.init(0xA11);
    const r = prng.random();
    const x = try gpa.alloc(f32, S * I);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, O * I);
    defer gpa.free(w);
    for (x) |*v| v.* = r.float(f32) * 2 - 1;
    for (w) |*v| v.* = r.float(f32) * 2 - 1;
    const ys = try gpa.alloc(f32, S * O);
    defer gpa.free(ys);
    const yp = try gpa.alloc(f32, S * O);
    defer gpa.free(yp);

    parallel.disable();
    matmul(ys, x, w, S, I, O);
    parallel.enable(std.testing.io, 8);
    defer parallel.disable();
    matmul(yp, x, w, S, I, O);

    for (ys, yp) |a, b| try std.testing.expectEqual(a, b);
}
