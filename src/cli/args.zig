//! CLI option parsing for the `inspect` subcommand.

const std = @import("std");
const budget = @import("../runtime/budget.zig");
const units = @import("../util/units.zig");

pub const Options = struct {
    /// Empty when no positional directory was given (only valid if `require_dir` was false).
    model_dir: []const u8 = "",
    budget: budget.Options = .{},
    /// `forward` only: comma-separated raw token ids, and greedy-decode step count.
    tokens: []const u8 = "",
    steps: u32 = 0,
    /// override the per-layer expert-cache capacity (0 = use the plan).
    expert_cap: u32 = 0,
    /// `chat` only: the user message and an optional system prompt.
    prompt: []const u8 = "",
    system: []const u8 = "",
    /// `benchmark` / `stress`: synthetic prompt length.
    prompt_len: u32 = 0,
    /// worker fan-out for the hot kernels: 0 = auto (CPU count), 1 = single-threaded.
    threads: u32 = 0,
    /// `chat` / `forward`: don't read/write `<model_dir>/.colizig_usage`.
    no_usage: bool = false,
    /// Second copy of the checkpoint on another drive; routed-expert reads split across both.
    mirror: []const u8 = "",
    /// `chat` / `forward` sampling: 0 temperature = greedy (deterministic).
    temperature: f32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1.0,
    seed: u64 = 0,
    /// `chat` / `forward` / `benchmark`: route the block-FP8 matmul through the
    /// CUDA backend (`colizig_cuda.dll`) if present; silently CPU otherwise.
    cuda: bool = false,
    /// bring-up aid: recompute each GPU matmul on the CPU and print the divergence.
    cuda_verify: bool = false,
};

pub const Error = error{ MissingModelDir, BadFlag, MissingValue } || std.Io.Writer.Error;

pub fn parse(
    gpa: std.mem.Allocator,
    args: []const []const u8,
    err: *std.Io.Writer,
    require_dir: bool,
) Error!Options {
    _ = gpa;
    var model_dir: ?[]const u8 = null;
    var b: budget.Options = .{};
    var b_tokens: []const u8 = "";
    var b_steps: u32 = 0;
    var b_ecap: u32 = 0;
    var b_prompt: []const u8 = "";
    var b_system: []const u8 = "";
    var b_plen: u32 = 0;
    var b_threads: u32 = 0;
    var b_no_usage: bool = false;
    var b_mirror: []const u8 = "";
    var b_temp: f32 = 0;
    var b_topk: u32 = 0;
    var b_topp: f32 = 1.0;
    var b_seed: u64 = 0;
    var b_cuda: bool = false;
    var b_cuda_verify: bool = false;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--ram-limit")) {
            const v = try value(args, &i, "--ram-limit", err);
            b.ram_limit = units.parseBytes(v) catch {
                try err.print("--ram-limit: not a valid size: \"{s}\"\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--context")) {
            const v = try value(args, &i, "--context", err);
            b.context = std.fmt.parseUnsigned(u32, v, 10) catch {
                try err.print("--context: not a positive integer: \"{s}\"\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--profile")) {
            const v = try value(args, &i, "--profile", err);
            b.profile = budget.Profile.fromStr(v) orelse {
                try err.print("--profile: unknown profile \"{s}\" (tiny|laptop|desktop|gpu)\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--tokens")) {
            b_tokens = try value(args, &i, "--tokens", err);
        } else if (std.mem.eql(u8, a, "--steps")) {
            const v = try value(args, &i, "--steps", err);
            b_steps = std.fmt.parseUnsigned(u32, v, 10) catch {
                try err.print("--steps: not a non-negative integer: \"{s}\"\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--threads")) {
            const v = try value(args, &i, "--threads", err);
            b_threads = std.fmt.parseUnsigned(u32, v, 10) catch {
                try err.print("--threads: not a non-negative integer: \"{s}\"\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--prompt-len")) {
            const v = try value(args, &i, "--prompt-len", err);
            b_plen = std.fmt.parseUnsigned(u32, v, 10) catch {
                try err.print("--prompt-len: not a positive integer: \"{s}\"\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--prompt")) {
            b_prompt = try value(args, &i, "--prompt", err);
        } else if (std.mem.eql(u8, a, "--system")) {
            b_system = try value(args, &i, "--system", err);
        } else if (std.mem.eql(u8, a, "--expert-cap")) {
            const v = try value(args, &i, "--expert-cap", err);
            b_ecap = std.fmt.parseUnsigned(u32, v, 10) catch {
                try err.print("--expert-cap: not a positive integer: \"{s}\"\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--no-usage")) {
            b_no_usage = true;
        } else if (std.mem.eql(u8, a, "--cuda")) {
            b_cuda = true;
        } else if (std.mem.eql(u8, a, "--cuda-verify")) {
            b_cuda = true;
            b_cuda_verify = true;
        } else if (std.mem.eql(u8, a, "--mirror")) {
            b_mirror = try value(args, &i, "--mirror", err);
        } else if (std.mem.eql(u8, a, "--temperature") or std.mem.eql(u8, a, "--temp")) {
            const v = try value(args, &i, "--temperature", err);
            b_temp = std.fmt.parseFloat(f32, v) catch {
                try err.print("--temperature: not a number: \"{s}\"\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--top-k")) {
            const v = try value(args, &i, "--top-k", err);
            b_topk = std.fmt.parseUnsigned(u32, v, 10) catch {
                try err.print("--top-k: not a non-negative integer: \"{s}\"\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--top-p")) {
            const v = try value(args, &i, "--top-p", err);
            b_topp = std.fmt.parseFloat(f32, v) catch {
                try err.print("--top-p: not a number: \"{s}\"\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--seed")) {
            const v = try value(args, &i, "--seed", err);
            b_seed = std.fmt.parseUnsigned(u64, v, 10) catch {
                try err.print("--seed: not a non-negative integer: \"{s}\"\n", .{v});
                return error.BadFlag;
            };
        } else if (std.mem.eql(u8, a, "--gpu")) {
            const v = try value(args, &i, "--gpu", err);
            if (std.mem.eql(u8, v, "none")) {
                b.gpu = .none;
            } else if (std.mem.eql(u8, v, "auto")) {
                b.gpu = .auto;
                try err.writeAll("note: --gpu auto is recorded but Phase 1 has no GPU backend; planning as CPU-only\n");
            } else {
                try err.print("--gpu: expected none|auto, got \"{s}\"\n", .{v});
                return error.BadFlag;
            }
        } else if (std.mem.startsWith(u8, a, "-")) {
            try err.print("unknown flag: {s}\n", .{a});
            return error.BadFlag;
        } else if (model_dir == null) {
            model_dir = a;
        } else {
            try err.print("unexpected extra argument: {s}\n", .{a});
            return error.BadFlag;
        }
    }

    if (require_dir and model_dir == null) {
        try err.writeAll("missing <MODEL_DIR>\n");
        return error.MissingModelDir;
    }
    return .{
        .model_dir = model_dir orelse "",
        .budget = b,
        .tokens = b_tokens,
        .steps = b_steps,
        .expert_cap = b_ecap,
        .prompt = b_prompt,
        .system = b_system,
        .prompt_len = b_plen,
        .threads = b_threads,
        .no_usage = b_no_usage,
        .mirror = b_mirror,
        .temperature = b_temp,
        .top_k = b_topk,
        .top_p = b_topp,
        .seed = b_seed,
        .cuda = b_cuda,
        .cuda_verify = b_cuda_verify,
    };
}

fn value(args: []const []const u8, i: *usize, name: []const u8, err: *std.Io.Writer) Error![]const u8 {
    if (i.* + 1 >= args.len) {
        err.print("{s}: expected a value\n", .{name}) catch {};
        return error.MissingValue;
    }
    i.* += 1;
    return args[i.*];
}

test "parse flags" {
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);
    const o = try parse(std.testing.allocator, &.{
        "/models/q38", "--ram-limit", "16G", "--context", "8192", "--profile", "desktop",
    }, &sink.writer, true);
    try std.testing.expectEqualStrings("/models/q38", o.model_dir);
    try std.testing.expectEqual(@as(?u64, 16 * 1024 * 1024 * 1024), o.budget.ram_limit);
    try std.testing.expectEqual(@as(u32, 8192), o.budget.context);
    try std.testing.expectEqual(budget.Profile.desktop, o.budget.profile);
}

test "missing model dir is an error" {
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);
    try std.testing.expectError(error.MissingModelDir, parse(std.testing.allocator, &[_][]const u8{ "--context", "4096" }, &sink.writer, true));
    // ...but optional when require_dir = false
    const o = try parse(std.testing.allocator, &[_][]const u8{ "--context", "4096" }, &sink.writer, false);
    try std.testing.expectEqualStrings("", o.model_dir);
}
