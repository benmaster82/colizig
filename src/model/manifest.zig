//! ModelManifest — the checkpoint as metadata only.
//!
//! Reads `config.json` and either `model.safetensors.index.json` (sharded) or a
//! lone `model.safetensors`, then parses every shard header (no tensor bodies)
//! into a flat `[]TensorLocation`.  Each tensor is classified by name into a
//! `Category` with an optional layer / expert index, which drives residency
//! accounting and the memory-budget plan.

const std = @import("std");
const config = @import("config.zig");
const st = @import("safetensors.zig");

pub const Cfg = config.Cfg;
pub const DType = st.DType;

pub const Error = error{
    OpenFailed,
    NoCheckpoint,
    BadIndexJson,
    ShardMissing,
    UnknownTensors,
} || config.Error || st.Error || std.Io.Writer.Error;

pub const Category = enum {
    embedding,
    lm_head,
    final_norm,
    gated_residual,
    attn_dense,
    qsa_indexer,
    deltanet,
    router,
    shared_expert,
    moe_expert,
    ple_table,
    ple_meta,
    ple_dense,
    norm,
    vision,
    mtp,
    unknown,

    pub fn residency(self: Category) Residency {
        return switch (self) {
            .moe_expert => .streamable,
            .ple_table => .cold,
            .vision, .mtp => .ignored,
            .unknown => .ignored,
            else => .resident,
        };
    }
};

pub const Residency = enum { resident, streamable, cold, ignored };

pub const TensorLocation = struct {
    /// Owned by the manifest allocator.
    name: []u8,
    /// Index into `Manifest.shard_paths`.
    shard: u32,
    /// Absolute byte offset of the tensor body within its shard file.
    offset: u64,
    /// Byte length of the tensor body.
    size: u64,
    dtype: DType,
    /// Owned. `shape[0..]` is the full shape.
    shape: []u64,
    layer: ?u32,
    expert: ?u32,
    category: Category,

    pub fn numel(self: TensorLocation) u64 {
        var n: u64 = 1;
        for (self.shape) |d| n *|= d;
        return n;
    }
};

pub const Manifest = struct {
    cfg: Cfg,
    tensors: []TensorLocation,
    shard_paths: [][]u8,
    /// Prefix the checkpoint uses for text tensors: "model.language_model" or "model".
    text_prefix: []const u8,
    /// True when the shard files were absent and only `model.safetensors.index.json`
    /// was available: tensor names + categories are known, but `size` / `shape` /
    /// `dtype` / `offset` are placeholders.
    metadata_only: bool = false,
    /// `metadata.total_size` from the index, when present.
    total_size: ?u64 = null,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Manifest) void {
        const gpa = self.allocator;
        for (self.tensors) |t| {
            gpa.free(t.name);
            gpa.free(t.shape);
        }
        gpa.free(self.tensors);
        for (self.shard_paths) |p| gpa.free(p);
        gpa.free(self.shard_paths);
        self.cfg.deinit();
        self.* = undefined;
    }

    pub fn countByCategory(self: Manifest, cat: Category) usize {
        var n: usize = 0;
        for (self.tensors) |t| {
            if (t.category == cat) n += 1;
        }
        return n;
    }

    pub fn countByResidency(self: Manifest, r: Residency) usize {
        var n: usize = 0;
        for (self.tensors) |t| {
            if (t.category.residency() == r) n += 1;
        }
        return n;
    }

    /// Total parameters across every non-ignored tensor.
    pub fn paramCount(self: Manifest) u128 {
        var total: u128 = 0;
        for (self.tensors) |t| {
            if (t.category.residency() == .ignored) continue;
            total += t.numel();
        }
        return total;
    }

    pub fn paramCountByResidency(self: Manifest, r: Residency) u128 {
        var total: u128 = 0;
        for (self.tensors) |t| {
            if (t.category.residency() == r) total += t.numel();
        }
        return total;
    }

    /// Bytes that must be resident, in the checkpoint's native dtype.
    pub fn residentBytes(self: Manifest) u64 {
        var total: u64 = 0;
        for (self.tensors) |t| {
            if (t.category.residency() == .resident) total +|= t.size;
        }
        return total;
    }

    /// Bytes of the PLE n-gram table on disk (never resident).
    pub fn pleTableBytes(self: Manifest) u64 {
        var total: u64 = 0;
        for (self.tensors) |t| {
            if (t.category == .ple_table) total +|= t.size;
        }
        return total;
    }

    /// Bytes of one full set of routed experts (all layers, all experts).
    pub fn routedExpertBytes(self: Manifest) u64 {
        var total: u64 = 0;
        for (self.tensors) |t| {
            if (t.category == .moe_expert) total +|= t.size;
        }
        return total;
    }

    pub fn hasUnknown(self: Manifest) bool {
        for (self.tensors) |t| {
            if (t.category == .unknown) return true;
        }
        return false;
    }
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, dir_path: []const u8, err: *std.Io.Writer) Error!Manifest {
    var dir = openDirAny(io, dir_path) catch |e| {
        try err.print("cannot open model directory \"{s}\": {s}\n", .{ dir_path, @errorName(e) });
        return error.OpenFailed;
    };
    defer dir.close(io);

    var cfg = try config.parseDir(gpa, io, dir, err);
    errdefer cfg.deinit();

    // Discover the shard file list. The safetensors index only tells us which
    // files exist; each shard's own header is the source of truth for which
    // tensors it contains, so we do not need the weight_map cross-reference.
    var shard_names: std.ArrayList([]u8) = .empty;
    errdefer {
        for (shard_names.items) |p| gpa.free(p);
        shard_names.deinit(gpa);
    }

    const has_index = blk: {
        dir.access(io, "model.safetensors.index.json", .{}) catch break :blk false;
        break :blk true;
    };

    // weight_map entries kept for the metadata-only fallback (owned dup names).
    var wm_names: std.ArrayList([]u8) = .empty;
    var wm_shard: std.ArrayList(u32) = .empty;
    defer {
        for (wm_names.items) |p| gpa.free(p);
        wm_names.deinit(gpa);
        wm_shard.deinit(gpa);
    }
    var total_size: ?u64 = null;

    if (has_index) {
        const bytes = dir.readFileAlloc(io, "model.safetensors.index.json", gpa, .limited(256 << 20)) catch {
            try err.writeAll("cannot read model.safetensors.index.json\n");
            return error.BadIndexJson;
        };
        defer gpa.free(bytes);
        var parsed = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch {
            try err.writeAll("model.safetensors.index.json is not valid JSON\n");
            return error.BadIndexJson;
        };
        defer parsed.deinit();
        const root = switch (parsed.value) {
            .object => |o| o,
            else => return error.BadIndexJson,
        };
        if (root.get("metadata")) |md| {
            if (md == .object) {
                if (md.object.get("total_size")) |ts| {
                    if (ts == .integer and ts.integer >= 0) total_size = @intCast(ts.integer);
                }
            }
        }
        const wm = switch (root.get("weight_map") orelse return error.BadIndexJson) {
            .object => |o| o,
            else => return error.BadIndexJson,
        };
        var it = wm.iterator();
        while (it.next()) |kv| {
            const file = switch (kv.value_ptr.*) {
                .string => |s| s,
                else => return error.BadIndexJson,
            };
            const si = try internShard(gpa, &shard_names, file);
            try wm_names.append(gpa, try gpa.dupe(u8, kv.key_ptr.*));
            try wm_shard.append(gpa, si);
        }
        if (shard_names.items.len == 0) return error.BadIndexJson;
    } else {
        dir.access(io, "model.safetensors", .{}) catch {
            try err.print(
                "\"{s}\" has neither model.safetensors.index.json nor model.safetensors\n",
                .{dir_path},
            );
            return error.NoCheckpoint;
        };
        _ = try internShard(gpa, &shard_names, "model.safetensors");
    }

    var tensors: std.ArrayList(TensorLocation) = .empty;
    errdefer {
        for (tensors.items) |t| {
            gpa.free(t.name);
            gpa.free(t.shape);
        }
        tensors.deinit(gpa);
    }

    var text_prefix: []const u8 = "model";

    // If the shard files are absent but the index is present, fall back to a
    // metadata-only manifest: names + categories from the weight_map, no shapes.
    const shards_present = blk: {
        dir.access(io, shard_names.items[0], .{}) catch break :blk false;
        break :blk true;
    };
    if (has_index and !shards_present) {
        for (wm_names.items, wm_shard.items) |name, si| {
            if (std.mem.startsWith(u8, name, "model.language_model.")) text_prefix = "model.language_model";
            const cls = classify(name);
            try tensors.append(gpa, .{
                .name = try gpa.dupe(u8, name),
                .shard = si,
                .offset = 0,
                .size = 0,
                .dtype = .bf16,
                .shape = try gpa.dupe(u64, &.{}),
                .layer = cls.layer,
                .expert = cls.expert,
                .category = cls.category,
            });
        }
        const shard_paths_mo = try shard_names.toOwnedSlice(gpa);
        var m: Manifest = .{
            .cfg = cfg,
            .tensors = try tensors.toOwnedSlice(gpa),
            .shard_paths = shard_paths_mo,
            .text_prefix = text_prefix,
            .metadata_only = true,
            .total_size = total_size,
            .allocator = gpa,
        };
        if (m.hasUnknown()) {
            try err.writeAll("checkpoint contains tensors this engine does not recognize:\n");
            for (m.tensors) |t| {
                if (t.category == .unknown) try err.print("  {s}\n", .{t.name});
            }
            m.deinit();
            return error.UnknownTensors;
        }
        return m;
    }

    // Parse each shard header and build TensorLocation entries.
    for (shard_names.items, 0..) |shard_name, si| {
        var file = dir.openFile(io, shard_name, .{}) catch {
            try err.print("missing shard: {s}\n", .{shard_name});
            return error.ShardMissing;
        };
        defer file.close(io);
        const stat = file.stat(io) catch return error.ShardMissing;
        var header = try st.readHeader(gpa, io, file, stat.size);
        defer header.deinit();

        for (header.entries) |e| {
            if (std.mem.startsWith(u8, e.name, "model.language_model.")) text_prefix = "model.language_model";

            const cls = classify(e.name);
            const loc: TensorLocation = .{
                .name = try gpa.dupe(u8, e.name),
                .shard = @intCast(si),
                .offset = header.absoluteOffset(e),
                .size = e.byteLen(),
                .dtype = e.dtype,
                .shape = try gpa.dupe(u64, e.shape),
                .layer = cls.layer,
                .expert = cls.expert,
                .category = cls.category,
            };
            try tensors.append(gpa, loc);
        }
    }

    // Materialize owned shard paths.
    const shard_paths = try shard_names.toOwnedSlice(gpa);

    var m: Manifest = .{
        .cfg = cfg,
        .tensors = try tensors.toOwnedSlice(gpa),
        .shard_paths = shard_paths,
        .text_prefix = text_prefix,
        .total_size = total_size,
        .allocator = gpa,
    };

    if (m.hasUnknown()) {
        try err.writeAll("checkpoint contains tensors this engine does not recognize:\n");
        for (m.tensors) |t| {
            if (t.category == .unknown) try err.print("  {s}\n", .{t.name});
        }
        m.deinit();
        return error.UnknownTensors;
    }
    return m;
}

fn openDirAny(io: std.Io, dir_path: []const u8) !std.Io.Dir {
    if (std.fs.path.isAbsolute(dir_path)) {
        return std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    }
    return std.Io.Dir.cwd().openDir(io, dir_path, .{});
}

fn internShard(gpa: std.mem.Allocator, list: *std.ArrayList([]u8), name: []const u8) !u32 {
    for (list.items, 0..) |p, i| {
        if (std.mem.eql(u8, p, name)) return @intCast(i);
    }
    try list.append(gpa, try gpa.dupe(u8, name));
    return @intCast(list.items.len - 1);
}

// ---- name classification --------------------------------------------------

const Class = struct {
    category: Category,
    layer: ?u32 = null,
    expert: ?u32 = null,
};

/// Parse a leading run of digits; returns value and the remainder after them.
fn leadingInt(s: []const u8) ?struct { value: u32, rest: []const u8 } {
    var i: usize = 0;
    while (i < s.len and s[i] >= '0' and s[i] <= '9') : (i += 1) {}
    if (i == 0) return null;
    const v = std.fmt.parseUnsigned(u32, s[0..i], 10) catch return null;
    return .{ .value = v, .rest = s[i..] };
}

pub fn classify(name: []const u8) Class {
    if (std.mem.indexOf(u8, name, ".visual.") != null or std.mem.startsWith(u8, name, "visual."))
        return .{ .category = .vision };
    if (std.mem.indexOf(u8, name, ".mtp.") != null or std.mem.startsWith(u8, name, "mtp."))
        return .{ .category = .mtp };
    if (std.mem.eql(u8, name, "lm_head.weight"))
        return .{ .category = .lm_head };

    var rest = name;
    if (std.mem.startsWith(u8, rest, "model.language_model.")) {
        rest = rest["model.language_model.".len..];
    } else if (std.mem.startsWith(u8, rest, "model.")) {
        rest = rest["model.".len..];
    }

    if (std.mem.eql(u8, rest, "embed_tokens.weight")) return .{ .category = .embedding };
    if (std.mem.eql(u8, rest, "norm.weight")) return .{ .category = .final_norm };
    if (std.mem.startsWith(u8, rest, "hyper_connection_mixer")) return .{ .category = .gated_residual };

    if (std.mem.startsWith(u8, rest, "layers.")) {
        const after = rest["layers.".len..];
        const li = leadingInt(after) orelse return .{ .category = .unknown };
        if (li.rest.len == 0 or li.rest[0] != '.') return .{ .category = .unknown };
        const sub = li.rest[1..];
        const layer = li.value;

        if (std.mem.indexOf(u8, sub, "hyper_connection") != null or
            std.mem.startsWith(u8, sub, "attn_hyper") or
            std.mem.startsWith(u8, sub, "mlp_hyper") or
            std.mem.indexOf(u8, sub, "hc_norm") != null)
            return .{ .category = .gated_residual, .layer = layer };

        if (std.mem.startsWith(u8, sub, "self_attn.indexer."))
            return .{ .category = .qsa_indexer, .layer = layer };
        if (std.mem.startsWith(u8, sub, "self_attn."))
            return .{ .category = .attn_dense, .layer = layer };
        if (std.mem.startsWith(u8, sub, "linear_attn."))
            return .{ .category = .deltanet, .layer = layer };

        if (std.mem.eql(u8, sub, "mlp.gate.weight"))
            return .{ .category = .router, .layer = layer };
        if (std.mem.startsWith(u8, sub, "mlp.shared_expert"))
            return .{ .category = .shared_expert, .layer = layer };
        if (std.mem.startsWith(u8, sub, "mlp.experts.")) {
            const e_after = sub["mlp.experts.".len..];
            if (std.mem.startsWith(u8, e_after, "gate_up_proj") or
                std.mem.startsWith(u8, e_after, "down_proj"))
                return .{ .category = .moe_expert, .layer = layer }; // fused, per-expert slices
            const ei = leadingInt(e_after) orelse return .{ .category = .unknown };
            return .{ .category = .moe_expert, .layer = layer, .expert = ei.value };
        }

        if (std.mem.indexOf(u8, sub, "ngram_embedding.weight_scale") != null)
            return .{ .category = .ple_meta, .layer = layer };
        if (std.mem.indexOf(u8, sub, "ngram_embedding.shard_") != null or
            std.mem.endsWith(u8, sub, "ngram_embedding.weight"))
            return .{ .category = .ple_table, .layer = layer };
        if (std.mem.startsWith(u8, sub, "ple.ple_embedding."))
            return .{ .category = .ple_meta, .layer = layer };
        if (std.mem.startsWith(u8, sub, "ple."))
            return .{ .category = .ple_dense, .layer = layer };

        if (std.mem.indexOf(u8, sub, "norm") != null)
            return .{ .category = .norm, .layer = layer };

        return .{ .category = .unknown, .layer = layer };
    }

    return .{ .category = .unknown };
}

// ---- tests --------------------------------------------------------------

test "classify covers the Qwen4-Exp tensor namespace" {
    const cases = .{
        .{ "model.language_model.embed_tokens.weight", Category.embedding, @as(?u32, null), @as(?u32, null) },
        .{ "lm_head.weight", Category.lm_head, null, null },
        .{ "model.norm.weight", Category.final_norm, null, null },
        .{ "model.language_model.hyper_connection_mixer.hc_norm.weight", Category.gated_residual, null, null },
        .{ "model.language_model.layers.7.self_attn.q_proj.weight", Category.attn_dense, @as(?u32, 7), null },
        .{ "model.language_model.layers.7.self_attn.indexer.index_qk_proj.weight", Category.qsa_indexer, 7, null },
        .{ "model.language_model.layers.3.linear_attn.in_proj_qkv.weight", Category.deltanet, 3, null },
        .{ "model.language_model.layers.3.mlp.gate.weight", Category.router, 3, null },
        .{ "model.language_model.layers.3.mlp.shared_expert.gate_proj.weight", Category.shared_expert, 3, null },
        .{ "model.language_model.layers.3.mlp.experts.42.gate_proj.weight", Category.moe_expert, 3, @as(?u32, 42) },
        .{ "model.language_model.layers.3.mlp.experts.gate_up_proj", Category.moe_expert, 3, null },
        .{ "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_5.weight", Category.ple_table, 2, null },
        .{ "model.language_model.layers.2.ple.ple_embedding.layer_multipliers", Category.ple_meta, 2, null },
        .{ "model.language_model.layers.2.ple.key_proj.weight", Category.ple_dense, 2, null },
        .{ "model.visual.blocks.0.attn.qkv.weight", Category.vision, null, null },
        .{ "model.language_model.mtp.layers.0.self_attn.q_proj.weight", Category.mtp, null, null },
    };
    inline for (cases) |c| {
        const got = classify(c[0]);
        try std.testing.expectEqual(c[1], got.category);
        try std.testing.expectEqual(@as(?u32, c[2]), got.layer);
        try std.testing.expectEqual(@as(?u32, c[3]), got.expert);
    }
}

test "open the tiny fixture (needs `zig build gen-fixture`)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);

    var m = open(gpa, io, "test/fixtures/tiny", &sink.writer) catch |e| switch (e) {
        error.OpenFailed, error.NoCheckpoint => return error.SkipZigTest,
        else => return e,
    };
    defer m.deinit();

    try std.testing.expect(!m.hasUnknown());
    try std.testing.expectEqualStrings("model.language_model", m.text_prefix);
    try std.testing.expectEqual(@as(u32, 4), m.cfg.layers);

    // 4 experts x 4 layers x (weight + scale) x 3 projections
    try std.testing.expectEqual(@as(usize, 4 * 4 * 6), m.countByCategory(.moe_expert));
    // 2 PLE table shards on the single PLE layer
    try std.testing.expectEqual(@as(usize, 2), m.countByCategory(.ple_table));
    try std.testing.expect(m.countByCategory(.deltanet) > 0);
    try std.testing.expect(m.countByCategory(.qsa_indexer) > 0);

    // Every expert tensor carries a layer; the per-expert ones carry an id too.
    var saw_expert_id = false;
    for (m.tensors) |t| {
        if (t.category == .moe_expert) {
            try std.testing.expect(t.layer != null);
            if (t.expert) |e| {
                try std.testing.expect(e < 4);
                saw_expert_id = true;
            }
        }
    }
    try std.testing.expect(saw_expert_id);

    try std.testing.expect(m.paramCount() > 0);
    try std.testing.expect(m.residentBytes() > 0);
    try std.testing.expect(m.pleTableBytes() > 0);
}
