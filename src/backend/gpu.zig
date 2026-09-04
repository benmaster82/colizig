//! Runtime-loaded CUDA backend (Phase 10a).
//!
//! `colizig_cuda.dll` (built by `zig build cuda`, nvcc → MSVC) is loaded when the
//! CLI passes `--cuda`. If the DLL is absent, or has no usable device, or any
//! symbol is missing, the backend stays **unavailable** and every op silently
//! runs on the CPU - CUDA is never required to build or run the engine.
//!
//! Zig 0.16's `std.DynLib` has no Windows implementation, so this uses
//! `LoadLibraryA` / `GetProcAddress` directly; on non-Windows the whole module
//! degrades to an always-unavailable stub.
//!
//! 10a exposes exactly one op: the block-FP8 matmul used by the MoE experts.
//! Every call re-uploads its weights over PCIe (no VRAM weight cache yet - that
//! is 10b); the point of 10a is a correct GPU path and an honest first number.

const std = @import("std");
const builtin = @import("builtin");
const is_windows = builtin.os.tag == .windows;

const InitFn = *const fn () callconv(.c) c_int;
const ShutdownFn = *const fn () callconv(.c) void;
const NameFn = *const fn () callconv(.c) [*:0]const u8;
const SetBudgetFn = *const fn (bytes: u64) callconv(.c) void;
const StatsFn = *const fn (hits: *u64, misses: *u64, uploaded_mib: *u64, resident: *u64) callconv(.c) void;
const MatmulFp8Fn = *const fn (
    y: [*]f32,
    x: [*]const f32,
    w: [*]const u8,
    scales: [*]const f32,
    s: c_int,
    i: c_int,
    o: c_int,
    key: u64,
) callconv(.c) c_int;

const HMODULE = *opaque {};
extern "kernel32" fn LoadLibraryA(lpLibFileName: [*:0]const u8) callconv(.winapi) ?HMODULE;
extern "kernel32" fn GetProcAddress(hModule: HMODULE, lpProcName: [*:0]const u8) callconv(.winapi) ?*const anyopaque;
extern "kernel32" fn FreeLibrary(hLibModule: HMODULE) callconv(.winapi) i32;

var handle: ?HMODULE = null;
var ready: bool = false;
var name_buf: [160]u8 = undefined;
var name_len: usize = 0;

var f_shutdown: ShutdownFn = undefined;
var f_matmul_fp8: MatmulFp8Fn = undefined;
var f_set_budget: SetBudgetFn = undefined;
var f_stats: StatsFn = undefined;

// The DLL keeps one global device context with shared buffers, so calls into it
// must be serialised. `moe.forwardDense` already skips its per-expert thread
// fan-out when the GPU is up, so this is only a safety net - cheap when
// uncontended. (A per-thread / per-expert VRAM context is 10b.)
var call_lock = std.atomic.Value(bool).init(false);

inline fn lock() void {
    while (call_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}
inline fn unlock() void {
    call_lock.store(false, .release);
}

fn sym(h: HMODULE, comptime T: type, name: [*:0]const u8) ?T {
    const p = GetProcAddress(h, name) orelse return null;
    return @ptrCast(@alignCast(p));
}

/// Try to bring the CUDA backend up. Best-effort: on any failure it writes one
/// line to `err` and leaves the backend unavailable. Call once, from a CLI that
/// was given `--cuda`.
pub fn init(err: *std.Io.Writer) void {
    if (ready) return;
    if (!is_windows) {
        err.writeAll("--cuda: only wired for Windows so far - using CPU\n") catch {};
        return;
    }

    const h = LoadLibraryA("colizig_cuda.dll") orelse {
        err.writeAll("--cuda: colizig_cuda.dll not found (build it with `zig build cuda`) - using CPU\n") catch {};
        return;
    };

    const init_fn = sym(h, InitFn, "colizig_cuda_init") orelse return fail(h, err, "colizig_cuda_init");
    const name_fn = sym(h, NameFn, "colizig_cuda_device_name") orelse return fail(h, err, "colizig_cuda_device_name");
    const shutdown_fn = sym(h, ShutdownFn, "colizig_cuda_shutdown") orelse return fail(h, err, "colizig_cuda_shutdown");
    const matmul_fn = sym(h, MatmulFp8Fn, "colizig_cuda_matmul_fp8") orelse return fail(h, err, "colizig_cuda_matmul_fp8");
    const budget_fn = sym(h, SetBudgetFn, "colizig_cuda_set_vram_budget") orelse return fail(h, err, "colizig_cuda_set_vram_budget");
    const stats_fn = sym(h, StatsFn, "colizig_cuda_stats") orelse return fail(h, err, "colizig_cuda_stats");

    const rc = init_fn();
    if (rc != 0) {
        err.print("--cuda: no usable CUDA device (colizig_cuda_init rc={d}) - using CPU\n", .{rc}) catch {};
        _ = FreeLibrary(h);
        return;
    }

    const nm = std.mem.span(name_fn());
    name_len = @min(nm.len, name_buf.len);
    @memcpy(name_buf[0..name_len], nm[0..name_len]);

    f_shutdown = shutdown_fn;
    f_matmul_fp8 = matmul_fn;
    f_set_budget = budget_fn;
    f_stats = stats_fn;
    handle = h;
    ready = true;
    err.print("--cuda: CUDA backend up - {s}\n", .{name_buf[0..name_len]}) catch {};
}

fn fail(h: HMODULE, err: *std.Io.Writer, missing: []const u8) void {
    err.print("--cuda: {s} missing from colizig_cuda.dll - using CPU\n", .{missing}) catch {};
    _ = FreeLibrary(h);
}

pub fn deinit() void {
    if (!ready) return;
    f_shutdown();
    if (handle) |h| _ = FreeLibrary(h);
    handle = null;
    ready = false;
}

pub fn available() bool {
    return ready;
}

/// `--cuda-verify`: after each GPU matmul, recompute on the CPU and print the
/// max abs / rel divergence for the first `verify_budget` calls, then stop the
/// process. A 10a bring-up aid, off by default.
pub var verify: bool = false;
var verify_calls: u64 = 0;
var verify_worst: f32 = 0;

pub fn verifyReport(cpu_max_abs: f32, cpu_max_rel: f32, S: usize, I: usize, O: usize) void {
    verify_calls += 1;
    // Only shout when the divergence is large enough to matter (f32 FP8 dot
    // rounding is ~1e-4 rel); otherwise just keep the running worst.
    if (cpu_max_rel > verify_worst) verify_worst = cpu_max_rel;
    // Shout only on a genuinely large divergence - a big relΔ alone is usually
    // just a near-zero denominator.
    if (cpu_max_abs > 1e-3 and cpu_max_rel > 1e-2)
        std.debug.print("cuda-verify #{d}  S={d} I={d} O={d}   max|Δ|={e:.3}  max relΔ={e:.3}  <-- LARGE\n", .{ verify_calls, S, I, O, cpu_max_abs, cpu_max_rel });
}

pub fn verifySummary() void {
    if (verify) std.debug.print("cuda-verify: {d} GPU matmuls checked, worst relΔ vs CPU = {e:.3}\n", .{ verify_calls, verify_worst });
}

pub fn deviceName() []const u8 {
    return name_buf[0..name_len];
}

/// Bytes of VRAM the weight cache may use (10b). 0 = cache off (every keyed call
/// re-uploads, like 10a). Clamped to free VRAM inside the DLL.
pub fn setVramBudget(bytes: u64) void {
    if (ready) f_set_budget(bytes);
}

pub fn statsLine(w: *std.Io.Writer) void {
    if (!ready) return;
    var hits: u64 = 0;
    var misses: u64 = 0;
    var up_mib: u64 = 0;
    var resident: u64 = 0;
    f_stats(&hits, &misses, &up_mib, &resident);
    const total = hits + misses;
    const pct: f64 = if (total == 0) 0 else 100.0 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(total));
    w.print("  cuda: {d} matmuls  VRAM cache {d:.1}% hit ({d} resident experts)  {d} MiB pushed H2D\n", .{ total, pct, resident, up_mib }) catch {};
}

/// GPU block-FP8 matmul: `y[S,O] = x[S,I] @ dequant(w)ᵀ`. `key` identifies the
/// weight for the VRAM cache (0 = do not cache). Returns true if the GPU handled
/// it; false → the caller must run the CPU path.
pub fn matmulFp8(
    y: []f32,
    x: []const f32,
    w: []const u8,
    scales: []const f32,
    S: usize,
    I: usize,
    O: usize,
    key: u64,
) bool {
    if (!ready) return false;
    lock();
    defer unlock();
    const rc = f_matmul_fp8(y.ptr, x.ptr, w.ptr, scales.ptr, @intCast(S), @intCast(I), @intCast(O), key);
    return rc == 0;
}

test "gpu backend is gracefully unavailable without the DLL" {
    try std.testing.expect(!available());
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    init(&w);
    try std.testing.expect(!available()); // no colizig_cuda.dll in the test env
    try std.testing.expect(!matmulFp8(&.{}, &.{}, &.{}, &.{}, 1, 1, 1, 0));
    setVramBudget(1 << 30); // no-op when unavailable
    deinit();
}
