//! Byte-count parsing and human-readable formatting.
//!
//! Sizes on the CLI are written like `8G`, `16GiB`, `512M`, `1024`.  We treat
//! `K/M/G/T` and `KiB/MiB/GiB/TiB` identically as binary (1024-based) units,
//! matching how RAM is actually budgeted; a bare number is bytes.

const std = @import("std");

pub const ParseError = error{ InvalidSize, Overflow };

/// Parse a size string such as "8G", "16GiB", "512M", "1024".
pub fn parseBytes(text: []const u8) ParseError!u64 {
    var s = std.mem.trim(u8, text, " \t");
    if (s.len == 0) return error.InvalidSize;

    var mult: u64 = 1;
    // Strip an optional trailing "iB" or "B".
    if (std.ascii.endsWithIgnoreCase(s, "ib")) {
        s = s[0 .. s.len - 2];
    } else if (std.ascii.endsWithIgnoreCase(s, "b")) {
        s = s[0 .. s.len - 1];
    }
    if (s.len == 0) return error.InvalidSize;

    switch (std.ascii.toLower(s[s.len - 1])) {
        'k' => {
            mult = 1024;
            s = s[0 .. s.len - 1];
        },
        'm' => {
            mult = 1024 * 1024;
            s = s[0 .. s.len - 1];
        },
        'g' => {
            mult = 1024 * 1024 * 1024;
            s = s[0 .. s.len - 1];
        },
        't' => {
            mult = 1024 * 1024 * 1024 * 1024;
            s = s[0 .. s.len - 1];
        },
        else => {},
    }
    s = std.mem.trim(u8, s, " \t");
    if (s.len == 0) return error.InvalidSize;

    const value = std.fmt.parseUnsigned(u64, s, 10) catch return error.InvalidSize;
    return std.math.mul(u64, value, mult) catch error.Overflow;
}

/// A value that formats itself as a binary size, e.g. `9.19 GiB`.
pub const Human = struct {
    bytes: u64,

    pub fn format(self: Human, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const units = [_][]const u8{ "B", "KiB", "MiB", "GiB", "TiB", "PiB" };
        var value: f64 = @floatFromInt(self.bytes);
        var i: usize = 0;
        while (value >= 1024.0 and i + 1 < units.len) : (i += 1) value /= 1024.0;
        if (i == 0) {
            try writer.print("{d} {s}", .{ self.bytes, units[0] });
        } else {
            try writer.print("{d:.2} {s}", .{ value, units[i] });
        }
    }
};

pub fn human(bytes: u64) Human {
    return .{ .bytes = bytes };
}

test parseBytes {
    try std.testing.expectEqual(@as(u64, 1024), try parseBytes("1024"));
    try std.testing.expectEqual(@as(u64, 8 * 1024 * 1024 * 1024), try parseBytes("8G"));
    try std.testing.expectEqual(@as(u64, 16 * 1024 * 1024 * 1024), try parseBytes("16GiB"));
    try std.testing.expectEqual(@as(u64, 512 * 1024 * 1024), try parseBytes("512M"));
    try std.testing.expectEqual(@as(u64, 4 * 1024), try parseBytes("4k"));
    try std.testing.expectError(error.InvalidSize, parseBytes(""));
    try std.testing.expectError(error.InvalidSize, parseBytes("G"));
    try std.testing.expectError(error.InvalidSize, parseBytes("12x"));
}

test Human {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("512 B", try std.fmt.bufPrint(&buf, "{f}", .{human(512)}));
    try std.testing.expectEqualStrings("1.00 KiB", try std.fmt.bufPrint(&buf, "{f}", .{human(1024)}));
    try std.testing.expectEqualStrings("2.00 GiB", try std.fmt.bufPrint(&buf, "{f}", .{human(2 * 1024 * 1024 * 1024)}));
}
