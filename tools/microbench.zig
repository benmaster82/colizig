//! Standalone kernel micro-benchmark — isolates the FP8 / BF16 inner loops from
//! all the I/O, threading and cache noise in `benchmark`.
//!
//!   zig run -O ReleaseFast tools/microbench.zig
//!
//! Times, on realistic MoE-expert shapes (single token, S=1):
//!   - the f32 dot product: 1 accumulator vs 4 accumulators + @mulAdd (FMA)
//!   - the E4M3 dequant alone
//!   - the fused dequant+dot vs the split (dequant to a buffer, then dot)

const std = @import("std");

const lanes = std.simd.suggestVectorLength(f32) orelse 8;
const V = @Vector(lanes, f32);

// ---------------- dot products ----------------

fn dot1(a: []const f32, b: []const f32) f32 {
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + lanes <= a.len) : (i += lanes) {
        const va: V = a[i..][0..lanes].*;
        const vb: V = b[i..][0..lanes].*;
        acc += va * vb;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

const UNROLL = 4;
fn dot4(a: []const f32, b: []const f32) f32 {
    var acc: [UNROLL]V = @splat(@as(V, @splat(0)));
    var i: usize = 0;
    while (i + UNROLL * lanes <= a.len) : (i += UNROLL * lanes) {
        inline for (0..UNROLL) |k| {
            const va: V = a[i + k * lanes ..][0..lanes].*;
            const vb: V = b[i + k * lanes ..][0..lanes].*;
            acc[k] = @mulAdd(V, va, vb, acc[k]);
        }
    }
    while (i + lanes <= a.len) : (i += lanes) {
        const va: V = a[i..][0..lanes].*;
        const vb: V = b[i..][0..lanes].*;
        acc[0] = @mulAdd(V, va, vb, acc[0]);
    }
    var tot: V = @splat(0);
    inline for (0..UNROLL) |k| tot += acc[k];
    var sum: f32 = @reduce(.Add, tot);
    while (i < a.len) : (i += 1) sum = @mulAdd(f32, a[i], b[i], sum);
    return sum;
}

// ---------------- E4M3 dequant (copy of ops/fp8.zig) ----------------

const block: usize = 128;
const L = lanes;

fn dequantRow(out: []f32, w: []const u8, scale_row: []const f32) void {
    const Vu = @Vector(L, u32);
    const Vf = @Vector(L, f32);
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

    var i: usize = 0;
    while (i + L <= w.len) : (i += L) {
        const sc: Vf = @splat(scale_row[i / block]);
        const bytes: @Vector(L, u8) = w[i..][0..L].*;
        const b: Vu = @as(Vu, bytes);
        const sign32 = (b >> s7) << s31;
        const exp = (b >> s3) & m4;
        const mant = b & m3;
        const normal_bits = sign32 | ((exp + bias) << s23) | (mant << s20);
        const mant_f: Vf = @floatFromInt(mant);
        const subn_bits = @as(Vu, @bitCast(mant_f * sub_step)) | sign32;
        var bits = @select(u32, exp == zero, subn_bits, normal_bits);
        bits = @select(u32, (exp == v15) & (mant == v7), nan_bits, bits);
        const val: Vf = @bitCast(bits);
        out[i..][0..L].* = val * sc;
    }
}

/// fused: decode L bytes and FMA them straight into 4 accumulators, no wf buffer.
fn dotFp8Fused(x: []const f32, w: []const u8, scale_row: []const f32) f32 {
    const Vu = @Vector(L, u32);
    const Vf = @Vector(L, f32);
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

    var acc: [2]V = @splat(@as(V, @splat(0)));
    var i: usize = 0;
    while (i + 2 * L <= w.len) : (i += 2 * L) {
        inline for (0..2) |k| {
            const off = i + k * L;
            const sc: Vf = @splat(scale_row[off / block]);
            const bytes: @Vector(L, u8) = w[off..][0..L].*;
            const b: Vu = @as(Vu, bytes);
            const sign32 = (b >> s7) << s31;
            const exp = (b >> s3) & m4;
            const mant = b & m3;
            const normal_bits = sign32 | ((exp + bias) << s23) | (mant << s20);
            const mant_f: Vf = @floatFromInt(mant);
            const subn_bits = @as(Vu, @bitCast(mant_f * sub_step)) | sign32;
            var bits = @select(u32, exp == zero, subn_bits, normal_bits);
            bits = @select(u32, (exp == v15) & (mant == v7), nan_bits, bits);
            const wv: V = @as(V, @bitCast(bits)) * sc;
            const xv: V = x[off..][0..L].*;
            acc[k] = @mulAdd(V, xv, wv, acc[k]);
        }
    }
    const sum: f32 = @reduce(.Add, acc[0] + acc[1]);
    return sum;
}

extern "kernel32" fn QueryPerformanceCounter(*u64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(*u64) callconv(.winapi) i32;
var qpf: u64 = 1;
/// nanoseconds
// ---- LUT-based FP8 decode (colibri's approach: 256-entry table + gather) ----

const e4m3_lut: [256]f32 = blk: {
    @setEvalBranchQuota(20000);
    var t: [256]f32 = undefined;
    for (&t, 0..) |*slot, bi| {
        const byte: u8 = @intCast(bi);
        const sgn: f32 = if (byte & 0x80 != 0) -1.0 else 1.0;
        const e: i32 = @intCast((byte >> 3) & 0x0f);
        const mn: u32 = byte & 0x07;
        if (e == 0) {
            slot.* = if (mn == 0) sgn * 0.0 else sgn * (@as(f32, @floatFromInt(mn)) / 8.0) * 0x1p-6;
        } else if (e == 15 and mn == 7) {
            slot.* = std.math.nan(f32);
        } else {
            var v: f64 = 1.0 + @as(f64, @floatFromInt(mn)) / 8.0;
            var k: i32 = e - 7;
            while (k > 0) : (k -= 1) v *= 2;
            while (k < 0) : (k += 1) v *= 0.5;
            slot.* = sgn * @as(f32, @floatCast(v));
        }
    }
    break :blk t;
};

/// scalar LUT loop — let LLVM auto-vectorise (it can emit vgatherdps on AVX2).
fn dotFp8LutScalar(x: []const f32, w: []const u8, scale_row: []const f32) f32 {
    var acc: f32 = 0;
    for (w, 0..) |b, i| acc += x[i] * e4m3_lut[b] * scale_row[i / block];
    return acc;
}

/// explicit manual gather: 8 scalar LUT loads assembled into a vector, then FMA.
fn dotFp8LutGather(x: []const f32, w: []const u8, scale_row: []const f32) f32 {
    var acc: [2]V = @splat(@as(V, @splat(0)));
    var i: usize = 0;
    while (i + 2 * lanes <= w.len) : (i += 2 * lanes) {
        inline for (0..2) |kk| {
            const off = i + kk * lanes;
            var wv: V = undefined;
            inline for (0..lanes) |j| wv[j] = e4m3_lut[w[off + j]];
            const sc: V = @splat(scale_row[off / block]);
            const xv: V = x[off..][0..lanes].*;
            acc[kk] = @mulAdd(V, xv, wv * sc, acc[kk]);
        }
    }
    return @reduce(.Add, acc[0] + acc[1]);
}

fn now() u64 {
    var c: u64 = 0;
    _ = QueryPerformanceCounter(&c);
    return c * 1_000_000_000 / qpf;
}

pub fn main() !void {
    _ = QueryPerformanceFrequency(&qpf);
    const gpa = std.heap.page_allocator;
    const out = std.debug.print;

    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();

    // gate/up expert projection: I=2560, O=640, S=1
    const I = 2560;
    const O = 640;
    const iters = 4000;

    const x = try gpa.alloc(f32, I);
    const wf = try gpa.alloc(f32, O * I);
    const wb = try gpa.alloc(u8, O * I);
    const scales = try gpa.alloc(f32, ((O + 127) / 128) * ((I + 127) / 128));
    const buf = try gpa.alloc(f32, I);
    for (x) |*v| v.* = rnd.float(f32) * 2 - 1;
    for (wf) |*v| v.* = rnd.float(f32) * 2 - 1;
    for (wb) |*v| v.* = rnd.int(u8);
    for (scales) |*v| v.* = 0.01 + rnd.float(f32) * 0.05;

    const flop_per_matmul: f64 = @floatFromInt(2 * O * I);
    var sink: f32 = 0;

    // ---- pure f32 dot, 1 acc vs 4 acc ----
    {
        var t0 = now();
        for (0..iters) |_| {
            var acc: f32 = 0;
            for (0..O) |o| acc += dot1(x, wf[o * I ..][0..I]);
            sink += acc;
        }
        const ns1: f64 = @floatFromInt(now() - t0);
        t0 = now();
        for (0..iters) |_| {
            var acc: f32 = 0;
            for (0..O) |o| acc += dot4(x, wf[o * I ..][0..I]);
            sink += acc;
        }
        const ns4: f64 = @floatFromInt(now() - t0);
        out("f32 matmul  I={d} O={d}\n", .{ I, O });
        out("  dot1 (acc+=va*vb)     {d:.1} us/matmul   {d:.1} GFLOP/s\n", .{ ns1 / iters / 1000, flop_per_matmul * iters / ns1 });
        out("  dot4 (@mulAdd x4)     {d:.1} us/matmul   {d:.1} GFLOP/s\n", .{ ns4 / iters / 1000, flop_per_matmul * iters / ns4 });
    }

    // ---- FP8: dequant-only, split, fused ----
    {
        var t0 = now();
        for (0..iters) |_| {
            for (0..O) |o| dequantRow(buf, wb[o * I ..][0..I], scales[(o / 128) * ((I + 127) / 128) ..]);
            sink += buf[0];
        }
        const nsd: f64 = @floatFromInt(now() - t0);

        t0 = now();
        for (0..iters) |_| {
            var acc: f32 = 0;
            for (0..O) |o| {
                dequantRow(buf, wb[o * I ..][0..I], scales[(o / 128) * ((I + 127) / 128) ..]);
                acc += dot4(x, buf);
            }
            sink += acc;
        }
        const nss: f64 = @floatFromInt(now() - t0);

        t0 = now();
        for (0..iters) |_| {
            var acc: f32 = 0;
            for (0..O) |o| acc += dotFp8Fused(x, wb[o * I ..][0..I], scales[(o / 128) * ((I + 127) / 128) ..]);
            sink += acc;
        }
        const nsf: f64 = @floatFromInt(now() - t0);

        t0 = now();
        for (0..iters) |_| {
            var acc: f32 = 0;
            for (0..O) |o| acc += dotFp8LutScalar(x, wb[o * I ..][0..I], scales[(o / 128) * ((I + 127) / 128) ..]);
            sink += acc;
        }
        const nsl: f64 = @floatFromInt(now() - t0);

        t0 = now();
        for (0..iters) |_| {
            var acc: f32 = 0;
            for (0..O) |o| acc += dotFp8LutGather(x, wb[o * I ..][0..I], scales[(o / 128) * ((I + 127) / 128) ..]);
            sink += acc;
        }
        const nsg: f64 = @floatFromInt(now() - t0);

        out("\nFP8 expert matmul  I={d} O={d}\n", .{ I, O });
        out("  dequant only (arith)   {d:.1} us/matmul\n", .{nsd / iters / 1000});
        out("  split  (arith dequant+dot4)  {d:.1} us/matmul   {d:.1} GFLOP/s\n", .{ nss / iters / 1000, flop_per_matmul * iters / nss });
        out("  fused  (arith decode+FMA)    {d:.1} us/matmul   {d:.1} GFLOP/s\n", .{ nsf / iters / 1000, flop_per_matmul * iters / nsf });
        out("  LUT scalar (auto-vec)        {d:.1} us/matmul   {d:.1} GFLOP/s\n", .{ nsl / iters / 1000, flop_per_matmul * iters / nsl });
        out("  LUT manual gather + FMA      {d:.1} us/matmul   {d:.1} GFLOP/s\n", .{ nsg / iters / 1000, flop_per_matmul * iters / nsg });
    }

    std.mem.doNotOptimizeAway(sink);
}
