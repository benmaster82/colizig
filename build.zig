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
        .name = "qwen38-zig",
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
}
