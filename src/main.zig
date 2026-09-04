//! colizig — experimental Zig inference engine for Qwen3.8-Flash-Next.
//!
//! Phase 1 scope: metadata only. `inspect` reads config.json + the safetensors
//! index and reports the architecture and a memory-budget plan WITHOUT loading
//! any weights. `chat` / `benchmark` / `stress` are declared but explicitly
//! unimplemented — they fail loudly rather than fake a result (brief §29).

const std = @import("std");
const args_mod = @import("cli/args.zig");
const inspect_mod = @import("cli/inspect.zig");
const selftest_mod = @import("cli/selftest.zig");
const forward_mod = @import("cli/forward.zig");
const chat_mod = @import("cli/chat.zig");
const benchmark_mod = @import("cli/benchmark.zig");
const stress_mod = @import("cli/stress.zig");
const tokenize_mod = @import("cli/tokenize.zig");

const usage =
    \\colizig — experimental memory-streaming Zig inference engine for Qwen3.8-Flash-Next
    \\
    \\usage:
    \\  colizig inspect <MODEL_DIR> [options]     print architecture + memory plan (no weights loaded)
    \\  colizig selftest [MODEL_DIR]              bring-up checks: ops kernels, weights, GDN/MoE/PLE/QSA
    \\  colizig forward <MODEL_DIR> --tokens <csv> [--steps N] [--expert-cap K]   end-to-end forward on raw token ids
    \\  colizig chat <MODEL_DIR> --prompt "..." [--system "..."] [--steps N]      one-shot chat (ChatML + greedy decode)
    \\  colizig tokenize <MODEL_DIR> --prompt "..."      encode/decode text with the checkpoint's tokenizer.json
    \\  colizig benchmark <MODEL_DIR> [--prompt-len N] [--steps N] [--expert-cap K]   runtime telemetry
    \\  colizig stress <MODEL_DIR> [--context N] [--steps N] [--ram-limit G]          sweep RAM budgets
    \\
    \\options:
    \\  --ram-limit <size>   resident memory budget (e.g. 8G, 16GiB); default: profile
    \\  --context <n>        context length in tokens; default: 8192
    \\  --profile <name>     tiny | laptop | desktop | gpu ; default: laptop
    \\  --threads <n>        worker fan-out for the kernels; 0 = auto, 1 = single-threaded
    \\  --no-usage          chat/forward: don't read/write <MODEL_DIR>/.colizig_usage (learned expert priors)
    \\  --mirror <dir>      second copy of the checkpoint on another drive; routed-expert reads split across both
    \\  --temperature <f>  chat/forward: sampling temperature; 0 = greedy (default). also --top-k <n>, --top-p <f>, --seed <n>
    \\  --cuda [--vram <size>]  chat/forward/benchmark: run the block-FP8 matmul on the GPU via colizig_cuda.dll; CPU if absent. --vram caps the resident expert cache (default: most of free VRAM)
    \\  --gpu <mode>         none | auto ; memory-plan hint only
    \\  -h, --help           show this message
    \\
;

/// Windows: make the console speak UTF-8 and honour ANSI SGR so the chat's
/// box-drawing + colour render correctly (no-op elsewhere).
fn winConsoleSetup() void {
    if (@import("builtin").os.tag != .windows) return;
    const w = struct {
        extern "kernel32" fn SetConsoleOutputCP(cp: c_uint) callconv(.winapi) c_int;
        extern "kernel32" fn GetStdHandle(id: u32) callconv(.winapi) ?*anyopaque;
        extern "kernel32" fn GetConsoleMode(h: ?*anyopaque, mode: *u32) callconv(.winapi) c_int;
        extern "kernel32" fn SetConsoleMode(h: ?*anyopaque, mode: u32) callconv(.winapi) c_int;
    };
    _ = w.SetConsoleOutputCP(65001);
    const STD_OUTPUT_HANDLE: u32 = @bitCast(@as(i32, -11));
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    const h = w.GetStdHandle(STD_OUTPUT_HANDLE);
    var mode: u32 = 0;
    if (w.GetConsoleMode(h, &mode) != 0) _ = w.SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;

    winConsoleSetup();

    var stdout_buf: [4096]u8 = undefined;
    var stdout_fw: std.Io.File.Writer = .init(.stdout(), io, &stdout_buf);
    const out = &stdout_fw.interface;

    var stderr_buf: [4096]u8 = undefined;
    var stderr_fw: std.Io.File.Writer = .init(.stderr(), io, &stderr_buf);
    const err = &stderr_fw.interface;

    const argv = try init.minimal.args.toSlice(arena);

    run(gpa, io, out, err, argv) catch |e| {
        out.flush() catch {};
        // These are reported to the user by `run` already; exit cleanly with a
        // non-zero code instead of dumping a Zig stack trace.
        switch (e) {
            error.NoCommand,
            error.UnknownCommand,
            error.ContextDoesNotFit,
            error.SelftestFailed,
            => {
                err.flush() catch {};
                std.process.exit(1);
            },
            error.MissingModelDir, error.BadFlag, error.MissingValue => {
                err.flush() catch {};
                std.process.exit(2);
            },
            error.OpenFailed,
            error.NoCheckpoint,
            error.InvalidConfig,
            error.UnsupportedModel,
            error.BadIndexJson,
            error.ShardMissing,
            error.UnknownTensors,
            error.TokenOutOfVocab,
            error.ContextExhausted,
            error.TooManyTokens,
            error.BadShape,
            => {
                err.flush() catch {};
                std.process.exit(3);
            },
            else => {
                err.flush() catch {};
                return e;
            },
        }
    };
    try out.flush();
    try err.flush();
}

fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    argv: []const []const u8,
) !void {
    if (argv.len < 2) {
        try out.writeAll(usage);
        return error.NoCommand;
    }
    const cmd = argv[1];
    if (std.mem.eql(u8, cmd, "-h") or std.mem.eql(u8, cmd, "--help")) {
        try out.writeAll(usage);
        return;
    }

    const rest = argv[2..];
    if (std.mem.eql(u8, cmd, "inspect")) {
        const opts = try args_mod.parse(gpa, rest, err, true);
        try inspect_mod.run(gpa, io, out, err, opts);
        return;
    }

    if (std.mem.eql(u8, cmd, "selftest")) {
        const opts = try args_mod.parse(gpa, rest, err, false);
        try selftest_mod.run(gpa, io, out, err, opts);
        return;
    }

    if (std.mem.eql(u8, cmd, "forward")) {
        const opts = try args_mod.parse(gpa, rest, err, true);
        try forward_mod.run(gpa, io, out, err, opts);
        return;
    }

    if (std.mem.eql(u8, cmd, "chat")) {
        const opts = try args_mod.parse(gpa, rest, err, true);
        try chat_mod.run(gpa, io, out, err, opts);
        return;
    }

    if (std.mem.eql(u8, cmd, "tokenize")) {
        const opts = try args_mod.parse(gpa, rest, err, true);
        try tokenize_mod.run(gpa, io, out, err, opts);
        return;
    }

    if (std.mem.eql(u8, cmd, "benchmark")) {
        const opts = try args_mod.parse(gpa, rest, err, true);
        try benchmark_mod.run(gpa, io, out, err, opts);
        return;
    }

    if (std.mem.eql(u8, cmd, "stress")) {
        const opts = try args_mod.parse(gpa, rest, err, true);
        try stress_mod.run(gpa, io, out, err, opts);
        return;
    }

    try err.print("unknown command: {s}\n\n", .{cmd});
    try out.writeAll(usage);
    return error.UnknownCommand;
}

test {
    std.testing.refAllDecls(@This());
    _ = @import("util/units.zig");
    _ = @import("cli/args.zig");
    _ = @import("cli/inspect.zig");
    _ = @import("cli/selftest.zig");
    _ = @import("cli/forward.zig");
    _ = @import("cli/chat.zig");
    _ = @import("cli/benchmark.zig");
    _ = @import("cli/stress.zig");
    _ = @import("cli/tokenize.zig");
    _ = @import("model/config.zig");
    _ = @import("model/safetensors.zig");
    _ = @import("model/manifest.zig");
    _ = @import("model/tensors.zig");
    _ = @import("model/weights.zig");
    _ = @import("runtime/budget.zig");
    _ = @import("runtime/io.zig");
    _ = @import("runtime/predict.zig");
    _ = @import("runtime/timers.zig");
    _ = @import("runtime/expert_usage.zig");
    _ = @import("runtime/sampler.zig");
    _ = @import("runtime/meter.zig");
    _ = @import("runtime/parallel.zig");
    _ = @import("ops/matmul.zig");
    _ = @import("ops/rmsnorm.zig");
    _ = @import("ops/rope.zig");
    _ = @import("ops/softmax.zig");
    _ = @import("ops/activation.zig");
    _ = @import("ops/fp8.zig");
    _ = @import("backend/gpu.zig");
    _ = @import("qwen38/gdn.zig");
    _ = @import("qwen38/moe.zig");
    _ = @import("qwen38/ple.zig");
    _ = @import("qwen38/qsa.zig");
    _ = @import("qwen38/residual.zig");
    _ = @import("qwen38/model.zig");
    _ = @import("qwen38/tokenizer.zig");
    _ = @import("qwen38/chat_template.zig");
}
