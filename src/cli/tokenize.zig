//! `tokenize` - load a checkpoint's `tokenizer.json` and encode/decode text.
//! Works from just the tokenizer file (no weights needed).

const std = @import("std");
const args = @import("args.zig");
const tok_mod = @import("../qwen38/tokenizer.zig");

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
) !void {
    if (opts.prompt.len == 0) {
        try err.writeAll("tokenize: --prompt \"...\" is required\n");
        return error.MissingModelDir;
    }
    var dir = (if (std.fs.path.isAbsolute(opts.model_dir))
        std.Io.Dir.openDirAbsolute(io, opts.model_dir, .{})
    else
        std.Io.Dir.cwd().openDir(io, opts.model_dir, .{})) catch {
        try err.print("tokenize: cannot open \"{s}\"\n", .{opts.model_dir});
        return error.OpenFailed;
    };
    defer dir.close(io);

    var tk = try tok_mod.Tokenizer.load(gpa, io, dir, err);
    defer tk.deinit();

    const ids = try tk.encode(gpa, opts.prompt);
    defer gpa.free(ids);
    const back = try tk.decode(gpa, ids, false);
    defer gpa.free(back);

    try out.print("vocab entries: {d}\n", .{tk.vocabSize()});
    try out.print("text:    {s}\n", .{opts.prompt});
    try out.print("ids ({d}): ", .{ids.len});
    for (ids, 0..) |id, i| {
        if (i != 0) try out.writeByte(' ');
        try out.print("{d}", .{id});
    }
    try out.print("\ndecoded: {s}\n", .{back});
    try out.print("round-trip: {s}\n", .{if (std.mem.eql(u8, opts.prompt, back)) "OK" else "MISMATCH"});
}
