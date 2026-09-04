//! Tensor decoding: raw checkpoint bytes → f32 working buffers.
//!
//! A `View` is a window onto a tensor's bytes (into a shard mmap or an owned
//! buffer) plus its dtype and shape.  Nothing is materialized until `decode`
//! / `decodeRow` is called with a caller-provided `[]f32`.
//!
//! Phase 2 decodes the resident dense dtypes: F32, F16, BF16.  FP8 (routed
//! experts, PLE table) needs block/scalar scales and lands with those
//! subsystems - it errors clearly here rather than returning wrong numbers.

const std = @import("std");
const builtin = @import("builtin");
const st = @import("safetensors.zig");
const mm = @import("../ops/matmul.zig");

pub const DType = st.DType;

comptime {
    // Raw checkpoint bytes are little-endian; NativeMatrix copies them verbatim.
    std.debug.assert(builtin.cpu.arch.endian() == .little);
}

pub const DecodeError = error{
    LengthMismatch,
    NotMatrix,
    RowOutOfRange,
    /// FP8 / integer dtypes: decoding needs scales or a dedicated path.
    UnsupportedDType,
};

pub const View = struct {
    bytes: []const u8,
    dtype: DType,
    shape: []const u64,

    pub fn numel(self: View) u64 {
        var n: u64 = 1;
        for (self.shape) |d| n *|= d;
        return n;
    }

    /// Decode the whole tensor into `out` (`out.len` must equal `numel`).
    pub fn decode(self: View, out: []f32) DecodeError!void {
        if (out.len != self.numel()) return error.LengthMismatch;
        try decodeInto(self.dtype, self.bytes, out);
    }

    /// Decode row `row` of a 2-D tensor into `out` (`out.len == shape[1]`).
    pub fn decodeRow(self: View, row: usize, out: []f32) DecodeError!void {
        if (self.shape.len != 2) return error.NotMatrix;
        const rows = self.shape[0];
        const cols = self.shape[1];
        if (row >= rows) return error.RowOutOfRange;
        if (out.len != cols) return error.LengthMismatch;
        const esz = self.dtype.elemSize();
        const start: usize = @intCast(@as(u64, row) * cols * esz);
        const len: usize = @intCast(cols * esz);
        try decodeInto(self.dtype, self.bytes[start .. start + len], out);
    }
};

/// A 2-D weight matrix `[rows, cols]` kept resident in its native dtype
/// (BF16 as `[]u16`, or F32), copied out of the shard mmap into an owned,
/// properly-aligned buffer.  `matmul` computes `y[S,rows] = x[S,cols] @ selfᵀ`.
pub const NativeMatrix = struct {
    rows: usize,
    cols: usize,
    storage: union(enum) {
        f32: []f32,
        bf16: []u16,
    },
    allocator: std.mem.Allocator,

    pub fn load(gpa: std.mem.Allocator, v: View) (DecodeError || std.mem.Allocator.Error)!NativeMatrix {
        if (v.shape.len != 2) return error.NotMatrix;
        const rows: usize = @intCast(v.shape[0]);
        const cols: usize = @intCast(v.shape[1]);
        const n = rows * cols;
        switch (v.dtype) {
            .f32 => {
                const buf = try gpa.alloc(f32, n);
                errdefer gpa.free(buf);
                if (v.bytes.len != n * 4) return error.LengthMismatch;
                @memcpy(std.mem.sliceAsBytes(buf), v.bytes);
                return .{ .rows = rows, .cols = cols, .storage = .{ .f32 = buf }, .allocator = gpa };
            },
            .bf16 => {
                const buf = try gpa.alloc(u16, n);
                errdefer gpa.free(buf);
                if (v.bytes.len != n * 2) return error.LengthMismatch;
                @memcpy(std.mem.sliceAsBytes(buf), v.bytes);
                return .{ .rows = rows, .cols = cols, .storage = .{ .bf16 = buf }, .allocator = gpa };
            },
            else => return error.UnsupportedDType,
        }
    }

    pub fn deinit(self: *NativeMatrix) void {
        switch (self.storage) {
            .f32 => |b| self.allocator.free(b),
            .bf16 => |b| self.allocator.free(b),
        }
        self.* = undefined;
    }

    pub fn byteLen(self: NativeMatrix) usize {
        return switch (self.storage) {
            .f32 => |b| b.len * 4,
            .bf16 => |b| b.len * 2,
        };
    }

    /// y[S, rows] = x[S, cols] @ selfᵀ
    pub fn matmul(self: NativeMatrix, y: []f32, x: []const f32, S: usize) void {
        switch (self.storage) {
            .f32 => |w| mm.matmul(y, x, w, S, self.cols, self.rows),
            .bf16 => |w| mm.matmulBf16(y, x, w, S, self.cols, self.rows),
        }
    }
};

pub fn decodeInto(dtype: DType, src: []const u8, out: []f32) DecodeError!void {
    const esz = dtype.elemSize();
    if (src.len != out.len * esz) return error.LengthMismatch;
    switch (dtype) {
        .f32 => {
            var i: usize = 0;
            while (i < out.len) : (i += 1) {
                out[i] = @bitCast(std.mem.readInt(u32, src[i * 4 ..][0..4], .little));
            }
        },
        .f16 => {
            var i: usize = 0;
            while (i < out.len) : (i += 1) {
                const h: f16 = @bitCast(std.mem.readInt(u16, src[i * 2 ..][0..2], .little));
                out[i] = h;
            }
        },
        .bf16 => {
            var i: usize = 0;
            while (i < out.len) : (i += 1) {
                out[i] = mm.bf16ToF32(std.mem.readInt(u16, src[i * 2 ..][0..2], .little));
            }
        },
        else => return error.UnsupportedDType,
    }
}

/// e4m3 (OCP FP8) decoding lives in `ops/fp8.zig` (used with block scales by the
/// MoE path); re-exported here for convenience.
pub const e4m3ToF32 = @import("../ops/fp8.zig").e4m3ToF32;

// ---- tests -------------------------------------------------------------

test "decode f32 / bf16 / f16 round-trips a known pattern" {
    const vals = [_]f32{ 0.0, 1.0, -2.0, 0.5, 12.0 };

    var f32_bytes: [vals.len * 4]u8 = undefined;
    for (vals, 0..) |v, i| std.mem.writeInt(u32, f32_bytes[i * 4 ..][0..4], @bitCast(v), .little);
    var out: [vals.len]f32 = undefined;
    try decodeInto(.f32, &f32_bytes, &out);
    try std.testing.expectEqualSlices(f32, &vals, &out);

    var bf_bytes: [vals.len * 2]u8 = undefined;
    for (vals, 0..) |v, i| std.mem.writeInt(u16, bf_bytes[i * 2 ..][0..2], mm.f32ToBf16(v), .little);
    try decodeInto(.bf16, &bf_bytes, &out);
    try std.testing.expectEqualSlices(f32, &vals, &out); // all bf16-exact

    var h_bytes: [vals.len * 2]u8 = undefined;
    for (vals, 0..) |v, i| std.mem.writeInt(u16, h_bytes[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little);
    try decodeInto(.f16, &h_bytes, &out);
    try std.testing.expectEqualSlices(f32, &vals, &out);
}

test "decodeRow slices a matrix" {
    // 2x3 bf16 matrix
    const m = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var bytes: [12]u8 = undefined;
    for (m, 0..) |v, i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], mm.f32ToBf16(v), .little);
    const v: View = .{ .bytes = &bytes, .dtype = .bf16, .shape = &.{ 2, 3 } };
    var row: [3]f32 = undefined;
    try v.decodeRow(1, &row);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 4, 5, 6 }, &row);
    try std.testing.expectError(error.RowOutOfRange, v.decodeRow(2, &row));
}

test "unsupported dtype errors instead of guessing" {
    var b: [2]u8 = .{ 0, 0 };
    var out: [2]f32 = undefined;
    try std.testing.expectError(error.UnsupportedDType, decodeInto(.f8_e4m3, b[0..2], out[0..2]));
}
