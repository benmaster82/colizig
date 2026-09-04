//! Weights - the materialization layer on top of `Manifest`.
//!
//! Opens every shard once and memory-maps it whole (read-only); each tensor is
//! then a zero-copy sub-slice.  If mmap is unavailable the shard is read into an
//! owned buffer instead.  Decoding to f32 happens only when a kernel asks for it
//! (`View.decode` / `View.decodeRow`), never eagerly.
//!
//! Phase 2 exposes `embed` (embedding row lookup) and `lmHead` (final
//! projection) as verified units.  The transformer layers are Phase 3+.

const std = @import("std");
const manifest_mod = @import("manifest.zig");
const tensors = @import("tensors.zig");
const mm_ops = @import("../ops/matmul.zig");
const parallel = @import("../runtime/parallel.zig");

pub const Manifest = manifest_mod.Manifest;
pub const TensorLocation = manifest_mod.TensorLocation;
pub const View = tensors.View;

/// Stack row-buffer cap for the parallel `lmHead` path; larger hidden sizes
/// fall back to a heap buffer + serial loop.
const lm_head_max_row = 8192;

pub const Error = error{
    OpenFailed,
    ShardOpenFailed,
    TensorMissing,
    ShapeMismatch,
} || tensors.DecodeError || std.mem.Allocator.Error || std.Io.Writer.Error;

const Shard = struct {
    file: std.Io.File,
    map: ?std.Io.File.MemoryMap,
    owned: ?[]u8,
    data: []const u8,
};

/// Round-robin cursor for splitting routed-expert reads across the primary and
/// a mirror drive.  Racy by design - both drives hold identical bytes, so an
/// uneven split is the only consequence.
var mirror_rr: usize = 0;
/// Of every 20 expert reads, this many go to the mirror; the rest to the
/// primary.  ~55 % matches a fast-mirror / slower-primary NVMe pair.
const mirror_share: usize = 11;

pub const Weights = struct {
    manifest: *const Manifest,
    shards: []Shard,
    /// Optional second copy of each shard on another drive (`attachMirror`);
    /// `null` per index for a partial mirror or none at all.
    mirror: []?Shard = &.{},
    io: std.Io,
    allocator: std.mem.Allocator,
    /// name → location, so `find` is O(1).  The MoE demand path looks up three
    /// tensors per expert load; a linear scan of the ~150k-tensor manifest there
    /// cost more wall time than the actual weight I/O.
    index: std.StringHashMapUnmanaged(*const TensorLocation) = .empty,

    pub fn open(
        gpa: std.mem.Allocator,
        io: std.Io,
        dir_path: []const u8,
        manifest: *const Manifest,
        err: *std.Io.Writer,
    ) Error!Weights {
        var dir = openDirAny(io, dir_path) catch {
            try err.print("weights: cannot reopen model directory \"{s}\"\n", .{dir_path});
            return error.OpenFailed;
        };
        defer dir.close(io);

        const shards = try gpa.alloc(Shard, manifest.shard_paths.len);
        var opened: usize = 0;
        errdefer {
            for (shards[0..opened]) |*s| closeShard(io, gpa, s);
            gpa.free(shards);
        }

        for (manifest.shard_paths, 0..) |name, i| {
            var file = dir.openFile(io, name, .{}) catch {
                try err.print("weights: cannot open shard {s}\n", .{name});
                return error.ShardOpenFailed;
            };
            const size: usize = @intCast((file.stat(io) catch {
                file.close(io);
                return error.ShardOpenFailed;
            }).size);

            var shard: Shard = .{ .file = file, .map = null, .owned = null, .data = &.{} };
            if (file.createMemoryMap(io, .{
                .len = size,
                .protection = .{ .read = true, .write = false },
                .offset = 0,
            })) |m| {
                var mmap = m;
                mmap.read(io) catch {};
                shard.map = mmap;
                shard.data = mmap.memory[0..size];
            } else |_| {
                const buf = try gpa.alloc(u8, size);
                const n = file.readPositionalAll(io, buf, 0) catch {
                    gpa.free(buf);
                    file.close(io);
                    return error.ShardOpenFailed;
                };
                shard.owned = buf;
                shard.data = buf[0..n];
            }
            shards[i] = shard;
            opened += 1;
        }

        var index: std.StringHashMapUnmanaged(*const TensorLocation) = .empty;
        errdefer index.deinit(gpa);
        try index.ensureTotalCapacity(gpa, @intCast(manifest.tensors.len));
        for (manifest.tensors) |*t| index.putAssumeCapacity(t.name, t);

        return .{
            .manifest = manifest,
            .shards = shards,
            .io = io,
            .allocator = gpa,
            .index = index,
        };
    }

    /// Open a second copy of the checkpoint at `dir_path` and mmap each shard
    /// whose mirror file exists with a matching size.  Routed-expert reads then
    /// alternate between the two drives.  Best-effort: on any problem the affected
    /// shard just isn't mirrored.  Returns the number of shards mirrored.
    pub fn attachMirror(self: *Weights, gpa: std.mem.Allocator, io: std.Io, dir_path: []const u8) usize {
        var dir = openDirAny(io, dir_path) catch return 0;
        defer dir.close(io);

        const mir = gpa.alloc(?Shard, self.shards.len) catch return 0;
        @memset(mir, null);
        var n: usize = 0;
        for (self.manifest.shard_paths, 0..) |name, i| {
            var file = dir.openFile(io, name, .{}) catch continue;
            const size: usize = @intCast((file.stat(io) catch {
                file.close(io);
                continue;
            }).size);
            if (size != self.shards[i].data.len) {
                file.close(io);
                continue;
            }
            if (file.createMemoryMap(io, .{ .len = size, .protection = .{ .read = true, .write = false }, .offset = 0 })) |m| {
                var mmap = m;
                mmap.read(io) catch {};
                mir[i] = .{ .file = file, .map = mmap, .owned = null, .data = mmap.memory[0..size] };
                n += 1;
            } else |_| {
                file.close(io);
            }
        }
        self.mirror = mir;
        return n;
    }

    pub fn deinit(self: *Weights) void {
        self.index.deinit(self.allocator);
        for (self.shards) |*s| closeShard(self.io, self.allocator, s);
        self.allocator.free(self.shards);
        if (self.mirror.len != 0) {
            for (self.mirror) |*ms| if (ms.*) |*s| closeShard(self.io, self.allocator, s);
            self.allocator.free(self.mirror);
        }
        self.* = undefined;
    }

    pub fn usingMmap(self: Weights) bool {
        for (self.shards) |s| {
            if (s.map == null) return false;
        }
        return self.shards.len != 0;
    }

    pub fn find(self: Weights, name: []const u8) ?*const TensorLocation {
        return self.index.get(name);
    }

    pub fn view(self: Weights, loc: *const TensorLocation) View {
        const start: usize = @intCast(loc.offset);
        const end: usize = @intCast(loc.offset + loc.size);
        var data = self.shards[loc.shard].data;
        // routed-expert bytes: split the read across the mirror drive
        if (loc.category == .moe_expert and loc.shard < self.mirror.len) {
            if (self.mirror[loc.shard]) |ms| {
                mirror_rr +%= 1;
                if (mirror_rr % 20 < mirror_share) data = ms.data;
            }
        }
        return .{ .bytes = data[start..end], .dtype = loc.dtype, .shape = loc.shape };
    }

    pub fn viewByName(self: Weights, name: []const u8) ?View {
        const loc = self.find(name) orelse return null;
        return self.view(loc);
    }

    /// Prefixed name lookup: `<text_prefix>.<suffix>` then bare `<suffix>`.
    pub fn viewBySuffix(self: Weights, suffix: []const u8, buf: []u8) ?View {
        const full = std.fmt.bufPrint(buf, "{s}.{s}", .{ self.manifest.text_prefix, suffix }) catch return null;
        return self.viewByName(full) orelse self.viewByName(suffix);
    }

    /// Materialize a whole tensor to a freshly-allocated f32 slice (caller frees).
    pub fn materialize(self: Weights, loc: *const TensorLocation) Error![]f32 {
        return self.materializeView(self.view(loc));
    }

    pub fn materializeView(self: Weights, v: View) Error![]f32 {
        const out = try self.allocator.alloc(f32, @intCast(v.numel()));
        errdefer self.allocator.free(out);
        try v.decode(out);
        return out;
    }

    /// Look up `<text_prefix>.<suffix>` (then bare `<suffix>`) and materialize it
    /// to an owned f32 slice.  For norms, biases, conv kernels.
    pub fn vectorBySuffix(self: Weights, suffix: []const u8, name_buf: []u8) Error![]f32 {
        const v = self.viewBySuffix(suffix, name_buf) orelse return error.TensorMissing;
        return self.materializeView(v);
    }

    /// Look up `<text_prefix>.<suffix>` and load it as a resident native matrix.
    pub fn matrixBySuffix(self: Weights, suffix: []const u8, name_buf: []u8) Error!tensors.NativeMatrix {
        const v = self.viewBySuffix(suffix, name_buf) orelse return error.TensorMissing;
        return tensors.NativeMatrix.load(self.allocator, v);
    }

    // ---- partial forward pieces (Phase 2) ------------------------------

    /// Embedding row for `token_id` → `out` (`out.len == hidden`).
    pub fn embed(self: Weights, token_id: usize, out: []f32) Error!void {
        var buf: [96]u8 = undefined;
        const v = self.viewBySuffix("embed_tokens.weight", &buf) orelse return error.TensorMissing;
        if (v.shape.len != 2 or v.shape[1] != out.len) return error.ShapeMismatch;
        try v.decodeRow(token_id, out);
    }

    /// Final projection: `logits[vocab] = lm_head @ hidden`.  Streams one weight
    /// row at a time so the [vocab, hidden] matrix is never fully materialized.
    /// The per-row decode+dot over the whole vocab dominates a single-token
    /// forward, so it fans out over `runtime/parallel.zig` (each task owns a
    /// disjoint id range and a stack row buffer); identical to the serial path.
    pub fn lmHead(self: Weights, hidden: []const f32, logits: []f32) Error!void {
        const loc = self.find("lm_head.weight") orelse return error.TensorMissing;
        const v = self.view(loc);
        if (v.shape.len != 2 or v.shape[0] != logits.len or v.shape[1] != hidden.len)
            return error.ShapeMismatch;

        // Serial heap-buffer fallback: unsupported dtype (surface the error) or
        // a hidden size that would overflow the parallel path's stack buffer.
        const supported = v.dtype == .f32 or v.dtype == .f16 or v.dtype == .bf16;
        if (!supported or hidden.len > lm_head_max_row) {
            const row = try self.allocator.alloc(f32, hidden.len);
            defer self.allocator.free(row);
            for (logits, 0..) |*o, i| {
                try v.decodeRow(i, row);
                o.* = mm_ops.dotF32(hidden, row);
            }
            return;
        }

        const Ctx = struct { v: View, hidden: []const f32, logits: []f32, cols: usize };
        parallel.chunks(
            logits.len,
            logits.len * hidden.len,
            Ctx{ .v = v, .hidden = hidden, .logits = logits, .cols = hidden.len },
            struct {
                fn body(c: Ctx, r0: usize, r1: usize) void {
                    var rowbuf: [lm_head_max_row]f32 = undefined;
                    const row = rowbuf[0..c.cols];
                    var r = r0;
                    while (r < r1) : (r += 1) {
                        c.v.decodeRow(r, row) catch unreachable;
                        c.logits[r] = mm_ops.dotF32(c.hidden, row);
                    }
                }
            }.body,
        );
    }
};

fn closeShard(io: std.Io, gpa: std.mem.Allocator, s: *Shard) void {
    if (s.map) |*m| m.destroy(io);
    if (s.owned) |b| gpa.free(b);
    s.file.close(io);
}

fn openDirAny(io: std.Io, dir_path: []const u8) !std.Io.Dir {
    if (std.fs.path.isAbsolute(dir_path)) return std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    return std.Io.Dir.cwd().openDir(io, dir_path, .{});
}

// ---- tests -------------------------------------------------------------

test "materialize + embed + lmHead against the tiny fixture" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);

    var m = manifest_mod.open(gpa, io, "test/fixtures/tiny", &sink.writer) catch |e| switch (e) {
        error.OpenFailed, error.NoCheckpoint => return error.SkipZigTest,
        else => return e,
    };
    defer m.deinit();

    var w = try Weights.open(gpa, io, "test/fixtures/tiny", &m, &sink.writer);
    defer w.deinit();

    const H: usize = m.cfg.hidden;
    const V: usize = m.cfg.vocab;

    // embedding row is finite and has the right length
    const e = try gpa.alloc(f32, H);
    defer gpa.free(e);
    try w.embed(3, e);
    for (e) |v| try std.testing.expect(std.math.isFinite(v));

    // lm_head projection: shape-correct and finite
    const logits = try gpa.alloc(f32, V);
    defer gpa.free(logits);
    try w.lmHead(e, logits);
    for (logits) |v| try std.testing.expect(std.math.isFinite(v));

    // lmHead must match a direct materialize+dot on a sample row
    const loc = w.find("lm_head.weight").?;
    const full = try w.materialize(loc);
    defer gpa.free(full);
    const sample = 7;
    var acc: f32 = 0;
    for (0..H) |i| acc += e[i] * full[sample * H + i];
    try std.testing.expectApproxEqRel(acc, logits[sample], 1e-4);

    // a missing tensor errors rather than returning zeros
    try std.testing.expect(w.viewByName("model.language_model.nonexistent") == null);
}
