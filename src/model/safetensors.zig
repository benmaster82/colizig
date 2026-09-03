//! Minimal reader for the `safetensors` container header.
//!
//! Layout: an 8-byte little-endian header length `N`, then `N` bytes of JSON
//! mapping each tensor name to `{ "dtype", "shape", "data_offsets":[begin,end] }`
//! (plus an optional `"__metadata__"` object), then the raw tensor bodies.
//!
//! Phase 1 reads ONLY the header. `absoluteOffset` gives where a tensor's bytes
//! begin in the file; nothing here reads them.

const std = @import("std");

pub const Error = error{
    NotSafetensors,
    HeaderTooLarge,
    BadHeaderJson,
    BadTensorEntry,
    OffsetOutOfRange,
    SizeMismatch,
    TooManyDims,
} || std.mem.Allocator.Error;

/// Cap on the JSON header; real shards are a few MiB of header at most.
pub const max_header_bytes: u64 = 256 << 20;
pub const max_rank = 8;

pub const DType = enum {
    f64,
    f32,
    f16,
    bf16,
    f8_e4m3,
    f8_e5m2,
    i64,
    i32,
    i16,
    i8,
    u8,
    boolean,

    pub fn fromStr(s: []const u8) ?DType {
        const map = .{
            .{ "F64", DType.f64 },   .{ "F32", DType.f32 },         .{ "F16", DType.f16 },
            .{ "BF16", DType.bf16 }, .{ "F8_E4M3", DType.f8_e4m3 }, .{ "F8_E5M2", DType.f8_e5m2 },
            .{ "I64", DType.i64 },   .{ "I32", DType.i32 },         .{ "I16", DType.i16 },
            .{ "I8", DType.i8 },     .{ "U8", DType.u8 },           .{ "BOOL", DType.boolean },
        };
        inline for (map) |pair| {
            if (std.mem.eql(u8, s, pair[0])) return pair[1];
        }
        return null;
    }

    pub fn elemSize(self: DType) u64 {
        return switch (self) {
            .f64, .i64 => 8,
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .f8_e4m3, .f8_e5m2, .i8, .u8, .boolean => 1,
        };
    }

    pub fn label(self: DType) []const u8 {
        return switch (self) {
            .f64 => "F64",
            .f32 => "F32",
            .f16 => "F16",
            .bf16 => "BF16",
            .f8_e4m3 => "F8_E4M3",
            .f8_e5m2 => "F8_E5M2",
            .i64 => "I64",
            .i32 => "I32",
            .i16 => "I16",
            .i8 => "I8",
            .u8 => "U8",
            .boolean => "BOOL",
        };
    }
};

pub const Entry = struct {
    /// Owned by the caller-supplied allocator.
    name: []u8,
    dtype: DType,
    /// `shape[0..rank]` is valid. Owned.
    shape: []u64,
    /// Byte range within the data segment (i.e. relative to the segment start).
    data_begin: u64,
    data_end: u64,

    pub fn numel(self: Entry) u64 {
        var n: u64 = 1;
        for (self.shape) |d| n *|= d;
        return n;
    }

    pub fn byteLen(self: Entry) u64 {
        return self.data_end - self.data_begin;
    }
};

pub const Header = struct {
    /// Length of the JSON header, i.e. bytes 8 .. 8+json_len.
    json_len: u64,
    entries: []Entry,
    metadata_json: ?[]u8,
    allocator: std.mem.Allocator,

    /// Absolute byte offset of a tensor's data within the file.
    pub fn absoluteOffset(self: Header, e: Entry) u64 {
        return 8 + self.json_len + e.data_begin;
    }

    pub fn deinit(self: *Header) void {
        for (self.entries) |e| {
            self.allocator.free(e.name);
            self.allocator.free(e.shape);
        }
        self.allocator.free(self.entries);
        if (self.metadata_json) |m| self.allocator.free(m);
        self.* = undefined;
    }
};

/// Parse the header of an already-open safetensors file. `file_size` bounds the
/// offset checks.
pub fn readHeader(
    gpa: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    file_size: u64,
) Error!Header {
    if (file_size < 8) return error.NotSafetensors;

    var len_buf: [8]u8 = undefined;
    const got = file.readPositionalAll(io, &len_buf, 0) catch return error.NotSafetensors;
    if (got != 8) return error.NotSafetensors;
    const json_len = std.mem.readInt(u64, &len_buf, .little);

    if (json_len == 0 or json_len > max_header_bytes) return error.HeaderTooLarge;
    if (8 + json_len > file_size) return error.OffsetOutOfRange;
    const data_segment_len = file_size - 8 - json_len;

    const json = try gpa.alloc(u8, json_len);
    defer gpa.free(json);
    const jn = file.readPositionalAll(io, json, 8) catch return error.BadHeaderJson;
    if (jn != json_len) return error.BadHeaderJson;

    var parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch return error.BadHeaderJson;
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return error.BadHeaderJson,
    };

    var entries: std.ArrayList(Entry) = .empty;
    errdefer {
        for (entries.items) |e| {
            gpa.free(e.name);
            gpa.free(e.shape);
        }
        entries.deinit(gpa);
    }

    var metadata_json: ?[]u8 = null;
    errdefer if (metadata_json) |m| gpa.free(m);

    var it = obj.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        if (std.mem.eql(u8, key, "__metadata__")) {
            metadata_json = try std.fmt.allocPrint(gpa, "{f}", .{std.json.fmt(kv.value_ptr.*, .{})});
            continue;
        }
        const te = switch (kv.value_ptr.*) {
            .object => |t| t,
            else => return error.BadTensorEntry,
        };

        const dtype_s = switch (te.get("dtype") orelse return error.BadTensorEntry) {
            .string => |s| s,
            else => return error.BadTensorEntry,
        };
        const dtype = DType.fromStr(dtype_s) orelse return error.BadTensorEntry;

        const shape_v = switch (te.get("shape") orelse return error.BadTensorEntry) {
            .array => |a| a,
            else => return error.BadTensorEntry,
        };
        if (shape_v.items.len > max_rank) return error.TooManyDims;
        const shape = try gpa.alloc(u64, shape_v.items.len);
        errdefer gpa.free(shape);
        for (shape_v.items, 0..) |dv, i| {
            shape[i] = switch (dv) {
                .integer => |x| if (x < 0) return error.BadTensorEntry else @intCast(x),
                else => return error.BadTensorEntry,
            };
        }

        const off_v = switch (te.get("data_offsets") orelse return error.BadTensorEntry) {
            .array => |a| a,
            else => return error.BadTensorEntry,
        };
        if (off_v.items.len != 2) return error.BadTensorEntry;
        const begin: u64 = switch (off_v.items[0]) {
            .integer => |x| if (x < 0) return error.BadTensorEntry else @intCast(x),
            else => return error.BadTensorEntry,
        };
        const end: u64 = switch (off_v.items[1]) {
            .integer => |x| if (x < 0) return error.BadTensorEntry else @intCast(x),
            else => return error.BadTensorEntry,
        };
        if (begin > end or end > data_segment_len) return error.OffsetOutOfRange;

        const name = try gpa.dupe(u8, key);
        errdefer gpa.free(name);

        const e: Entry = .{
            .name = name,
            .dtype = dtype,
            .shape = shape,
            .data_begin = begin,
            .data_end = end,
        };
        // numel * elemSize must equal the declared byte range.
        if (e.numel() *| dtype.elemSize() != e.byteLen()) return error.SizeMismatch;

        try entries.append(gpa, e);
    }

    return .{
        .json_len = json_len,
        .entries = try entries.toOwnedSlice(gpa),
        .metadata_json = metadata_json,
        .allocator = gpa,
    };
}

// ---- tests ---------------------------------------------------------------

/// Build a valid in-memory safetensors blob for one tensor of zeros.
fn buildBlob(gpa: std.mem.Allocator, name: []const u8, dtype: []const u8, shape: []const u64, elem: u64) ![]u8 {
    var numel: u64 = 1;
    for (shape) |d| numel *= d;
    const body = numel * elem;

    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(gpa);
    try hdr.print(gpa, "{{\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ name, dtype });
    for (shape, 0..) |d, i| {
        if (i != 0) try hdr.appendSlice(gpa, ",");
        try hdr.print(gpa, "{d}", .{d});
    }
    try hdr.print(gpa, "],\"data_offsets\":[0,{d}]}}}}", .{body});

    const out = try gpa.alloc(u8, 8 + hdr.items.len + body);
    std.mem.writeInt(u64, out[0..8], hdr.items.len, .little);
    @memcpy(out[8 .. 8 + hdr.items.len], hdr.items);
    @memset(out[8 + hdr.items.len ..], 0);
    return out;
}

test "readHeader parses a one-tensor blob from a temp file" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    const blob = try buildBlob(gpa, "model.embed_tokens.weight", "BF16", &.{ 4, 8 }, 2);
    defer gpa.free(blob);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "m.safetensors", .data = blob });

    var file = try tmp.dir.openFile(io, "m.safetensors", .{});
    defer file.close(io);

    var header = try readHeader(gpa, io, file, blob.len);
    defer header.deinit();

    try std.testing.expectEqual(@as(usize, 1), header.entries.len);
    const e = header.entries[0];
    try std.testing.expectEqualStrings("model.embed_tokens.weight", e.name);
    try std.testing.expectEqual(DType.bf16, e.dtype);
    try std.testing.expectEqual(@as(u64, 64), e.byteLen());
    try std.testing.expectEqual(@as(u64, 8 + header.json_len), header.absoluteOffset(e));
}

test "readHeader rejects a truncated size range" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var blob = try buildBlob(gpa, "t", "F32", &.{2}, 4);
    defer gpa.free(blob);
    // Corrupt: shrink the file so the declared body no longer fits.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "bad.safetensors", .data = blob[0 .. blob.len - 3] });
    var file = try tmp.dir.openFile(io, "bad.safetensors", .{});
    defer file.close(io);
    try std.testing.expectError(error.OffsetOutOfRange, readHeader(gpa, io, file, blob.len - 3));
}
