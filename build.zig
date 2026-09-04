const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // The engine streams a ~170 GB model through numeric kernels — a Debug build
    // is 3-4x slower and only misleads benchmarks. Default to ReleaseFast; pass
    // `-Doptimize=Debug` (or `ReleaseSafe`) explicitly for safety-checked runs.
    const optimize = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Optimization mode (default: ReleaseFast)",
    ) orelse .ReleaseFast;

    // ---- main engine executable -------------------------------------------
    const exe = b.addExecutable(.{
        .name = "colizig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the engine (e.g. `zig build run -- inspect <dir>`)");
    run_step.dependOn(&run_cmd.step);

    // ---- unit tests ------------------------------------------------------
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // ---- tiny synthetic fixture generator -------------------------------
    // (declared before the test wiring so `test` can depend on it)
    const cfg_module = b.createModule(.{
        .root_source_file = b.path("src/model/config.zig"),
        .target = target,
        .optimize = optimize,
    });
    const fixture = b.addExecutable(.{
        .name = "gen-tiny-fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/gen_tiny_fixture.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "config", .module = cfg_module }},
        }),
    });
    const run_fixture = b.addRunArtifact(fixture);
    if (b.args) |args| run_fixture.addArgs(args);
    const fixture_step = b.step("gen-fixture", "Regenerate test/fixtures/tiny");
    fixture_step.dependOn(&run_fixture.step);

    // Tests read test/fixtures/tiny; make sure it is present and current.
    run_tests.step.dependOn(&run_fixture.step);

    // ---- reference oracle (needs python + numpy) ------------------------
    const oracle = b.addSystemCommand(&.{ "python", "tools/reference/build_oracle.py" });
    oracle.step.dependOn(&run_fixture.step);
    const oracle_step = b.step("oracle", "Regenerate test/fixtures/tiny/oracle.json from the NumPy reference");
    oracle_step.dependOn(&oracle.step);

    // ---- optional CUDA backend (needs nvcc + a host C++ compiler) --------
    // `zig build cuda` compiles src/backend/cuda/*.cu into colizig_cuda.dll and
    // installs it next to the engine. The engine loads it at runtime only with
    // `--cuda`; the normal build and `zig build test` never touch this step, so
    // CUDA is not a dependency of colizig.
    {
        const arch = b.option([]const u8, "cuda-arch", "nvcc -arch value for `zig build cuda` (default: native)") orelse "native";
        // `nvcc` from PATH. `%CUDA_PATH%\bin` is on PATH after a normal CUDA
        // install; if not, run `zig build cuda` from an "x64 Native Tools" /
        // CUDA-aware shell, or override with -Dnvcc=<full path>.
        const nvcc = b.option([]const u8, "nvcc", "path to nvcc (default: from PATH)") orelse "nvcc";
        const nvcc_cmd = b.addSystemCommand(&.{ nvcc, "-O3", "-arch", arch, "--shared", "-o" });
        const dll = nvcc_cmd.addOutputFileArg("colizig_cuda.dll");
        nvcc_cmd.addFileArg(b.path("src/backend/cuda/colizig_cuda.cu"));
        nvcc_cmd.addArg("-lcudart");
        const install_dll = b.addInstallBinFile(dll, "colizig_cuda.dll");
        const cuda_step = b.step("cuda", "Build colizig_cuda.dll (nvcc; run from a VS dev prompt if nvcc can't find cl.exe)");
        cuda_step.dependOn(&install_dll.step);
    }
}
