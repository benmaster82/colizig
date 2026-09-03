//! Generates `test/fixtures/tiny/` — a structurally faithful but minuscule
//! Qwen4-Exp checkpoint: a correct `config.json`, a `model.safetensors` whose
//! header names and shapes match what `model/manifest.zig` classifies, and a
//! `model.safetensors.index.json`.  Tensor bodies are zero-filled: Phase 1 reads
//! headers only.
//!
//! Run via `zig build gen-fixture`.  Deterministic.

const std = @import("std");
const cfg_mod = @import("config");

const out_dir = "test/fixtures/tiny";

const Dtype = enum {
    bf16,
    f32,
    f8_e4m3,
    i64,

    fn label(d: Dtype) []const u8 {
        return switch (d) {
            .bf16 => "BF16",
            .f32 => "F32",
            .f8_e4m3 => "F8_E4M3",
            .i64 => "I64",
        };
    }
    fn elem(d: Dtype) u64 {
        return switch (d) {
            .i64 => 8,
            .f32 => 4,
            .bf16 => 2,
            .f8_e4m3 => 1,
        };
    }
};

const Tensor = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const u64,
    /// Explicit body bytes (owned by the builder allocator). If null, the body
    /// is synthesized deterministically by `fillBody`.
    body: ?[]const u8 = null,

    fn nbytes(self: Tensor) u64 {
        var n: u64 = 1;
        for (self.shape) |d| n *= d;
        return n * self.dtype.elem();
    }
};

const Builder = struct {
    gpa: std.mem.Allocator,
    list: std.ArrayList(Tensor) = .empty,

    fn add(b: *Builder, name: []const u8, dtype: Dtype, shape: []const u64) !void {
        try b.list.append(b.gpa, .{
            .name = try b.gpa.dupe(u8, name),
            .dtype = dtype,
            .shape = try b.gpa.dupe(u64, shape),
        });
    }

    fn addf(b: *Builder, comptime fmt: []const u8, args: anytype, dtype: Dtype, shape: []const u64) !void {
        const name = try std.fmt.allocPrint(b.gpa, fmt, args);
        defer b.gpa.free(name);
        try b.add(name, dtype, shape);
    }

    fn addRaw(b: *Builder, name: []const u8, dtype: Dtype, shape: []const u64, i64s: []const i64) !void {
        const bytes = try b.gpa.alloc(u8, i64s.len * 8);
        for (i64s, 0..) |v, k| std.mem.writeInt(i64, bytes[k * 8 ..][0..8], v, .little);
        try b.list.append(b.gpa, .{
            .name = try b.gpa.dupe(u8, name),
            .dtype = dtype,
            .shape = try b.gpa.dupe(u64, shape),
            .body = bytes,
        });
    }

    fn addRawf(b: *Builder, comptime fmt: []const u8, args: anytype, dtype: Dtype, shape: []const u64, i64s: []const i64) !void {
        const name = try std.fmt.allocPrint(b.gpa, fmt, args);
        defer b.gpa.free(name);
        try b.addRaw(name, dtype, shape, i64s);
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);
    var c = try cfg_mod.parseSlice(gpa, cfg_mod.tiny_config_json, &sink.writer);
    defer c.deinit();

    var b: Builder = .{ .gpa = gpa };
    defer {
        for (b.list.items) |t| {
            gpa.free(t.name);
            gpa.free(t.shape);
            if (t.body) |bd| gpa.free(bd);
        }
        b.list.deinit(gpa);
    }

    const P = "model.language_model";
    const hidden: u64 = c.hidden;
    const hc_w: u64 = c.hc_width;
    const hc_r: u64 = c.hc_rank;

    // boundaries
    try b.add(P ++ ".embed_tokens.weight", .bf16, &.{ c.vocab, hidden });
    try b.add("lm_head.weight", .bf16, &.{ c.vocab, hidden });
    // final gated-residual mixer (no inject)
    try b.add(P ++ ".hyper_connection_mixer.hc_norm.weight", .f32, &.{hc_w});
    try b.add(P ++ ".hyper_connection_mixer.input_mix_weight_down.weight", .bf16, &.{ hc_r, hc_w });
    try b.add(P ++ ".hyper_connection_mixer.input_mix_weight_up.weight", .bf16, &.{ hc_w, hc_r });

    for (0..c.layers) |li| {
        const L = try std.fmt.allocPrint(gpa, "{s}.layers.{d}", .{ P, li });
        defer gpa.free(L);

        // two gated-residual mixers per layer, each with a block-inject weight
        for ([_][]const u8{ "attn_hyper_connection", "mlp_hyper_connection" }) |kind| {
            try b.addf("{s}.{s}.hc_norm.weight", .{ L, kind }, .f32, &.{hc_w});
            try b.addf("{s}.{s}.input_mix_weight_down.weight", .{ L, kind }, .bf16, &.{ hc_r, hc_w });
            try b.addf("{s}.{s}.input_mix_weight_up.weight", .{ L, kind }, .bf16, &.{ hc_w, hc_r });
            try b.addf("{s}.{s}.block_inject_weight.weight", .{ L, kind }, .bf16, &.{ c.hc_count, hc_w });
        }

        // router + shared expert
        try b.addf("{s}.mlp.gate.weight", .{L}, .bf16, &.{ c.experts, hidden });
        try b.addf("{s}.mlp.shared_expert.gate_proj.weight", .{L}, .bf16, &.{ c.shared_inter, hidden });
        try b.addf("{s}.mlp.shared_expert.up_proj.weight", .{L}, .bf16, &.{ c.shared_inter, hidden });
        try b.addf("{s}.mlp.shared_expert.down_proj.weight", .{L}, .bf16, &.{ hidden, c.shared_inter });
        try b.addf("{s}.mlp.shared_expert_gate.weight", .{L}, .f32, &.{hidden});

        // routed experts, native block-FP8
        const nblk_i = ceilDiv(c.inter, 128);
        const nblk_h = ceilDiv(hidden, 128);
        for (0..c.experts) |e| {
            try b.addf("{s}.mlp.experts.{d}.gate_proj.weight", .{ L, e }, .f8_e4m3, &.{ c.inter, hidden });
            try b.addf("{s}.mlp.experts.{d}.gate_proj.weight_scale_inv", .{ L, e }, .f32, &.{ nblk_i, nblk_h });
            try b.addf("{s}.mlp.experts.{d}.up_proj.weight", .{ L, e }, .f8_e4m3, &.{ c.inter, hidden });
            try b.addf("{s}.mlp.experts.{d}.up_proj.weight_scale_inv", .{ L, e }, .f32, &.{ nblk_i, nblk_h });
            try b.addf("{s}.mlp.experts.{d}.down_proj.weight", .{ L, e }, .f8_e4m3, &.{ hidden, c.inter });
            try b.addf("{s}.mlp.experts.{d}.down_proj.weight_scale_inv", .{ L, e }, .f32, &.{ nblk_h, nblk_i });
        }

        if (c.is_attn[li]) {
            try b.addf("{s}.self_attn.q_proj.weight", .{L}, .bf16, &.{ @as(u64, c.q_heads) * c.head_dim * 2, hidden });
            try b.addf("{s}.self_attn.k_proj.weight", .{L}, .bf16, &.{ @as(u64, c.kv_heads) * c.head_dim, hidden });
            try b.addf("{s}.self_attn.v_proj.weight", .{L}, .bf16, &.{ @as(u64, c.kv_heads) * c.head_dim, hidden });
            try b.addf("{s}.self_attn.o_proj.weight", .{L}, .bf16, &.{ hidden, @as(u64, c.q_heads) * c.head_dim });
            try b.addf("{s}.self_attn.q_norm.weight", .{L}, .f32, &.{c.head_dim});
            try b.addf("{s}.self_attn.k_norm.weight", .{L}, .f32, &.{c.head_dim});
            try b.addf("{s}.self_attn.indexer.index_qk_proj.weight", .{L}, .bf16, &.{ @as(u64, c.idx_qheads + c.idx_kheads) * c.idx_dim, hidden });
            try b.addf("{s}.self_attn.indexer.q_layernorm.weight", .{L}, .f32, &.{c.idx_dim});
            try b.addf("{s}.self_attn.indexer.k_layernorm.weight", .{L}, .f32, &.{c.idx_dim});
        } else {
            const vd: u64 = @as(u64, c.dn_vheads) * c.dn_vdim;
            try b.addf("{s}.linear_attn.in_proj_qkv.weight", .{L}, .bf16, &.{ c.dn_conv_dim, hidden });
            try b.addf("{s}.linear_attn.in_proj_z.weight", .{L}, .bf16, &.{ vd, hidden });
            try b.addf("{s}.linear_attn.in_proj_b.weight", .{L}, .bf16, &.{ c.dn_vheads, hidden });
            try b.addf("{s}.linear_attn.in_proj_a.weight", .{L}, .bf16, &.{ c.dn_vheads, hidden });
            try b.addf("{s}.linear_attn.conv1d.weight", .{L}, .f32, &.{ c.dn_conv_dim, c.dn_convk });
            try b.addf("{s}.linear_attn.dt_bias", .{L}, .f32, &.{c.dn_vheads});
            try b.addf("{s}.linear_attn.A_log", .{L}, .f32, &.{c.dn_vheads});
            try b.addf("{s}.linear_attn.norm.weight", .{L}, .f32, &.{c.dn_vdim});
            try b.addf("{s}.linear_attn.out_proj.weight", .{L}, .bf16, &.{ hidden, vd });
        }

        // PLE on its one layer
        if (li == c.ple_layer) {
            try b.addf("{s}.ple.key_proj.weight", .{L}, .bf16, &.{ hc_w, c.ple_dim });
            try b.addf("{s}.ple.value_proj.weight", .{L}, .bf16, &.{ hidden, c.ple_dim });
            try b.addf("{s}.ple.norm_key.weight", .{L}, .f32, &.{hc_w});
            try b.addf("{s}.ple.norm_query.weight", .{L}, .f32, &.{hc_w});
            try b.addf("{s}.ple.norm_conv.weight", .{L}, .f32, &.{hc_w});
            try b.addf("{s}.ple.conv1d.weight", .{L}, .f32, &.{ hc_w, c.ple_convk });

            // n-gram hash parameters: large odd multipliers, and a vocab/offset
            // partition that exactly fills the table (sum(offset+vocab) == rows).
            const mult = [_]i64{
                @bitCast(@as(u64, 0x9E3779B97F4A7C15)),
                @bitCast(@as(u64, 0xC2B2AE3D27D4EB4F)),
                @bitCast(@as(u64, 0x165667B19E3779F9)),
            };
            const rows_per_part: u64 = 64;
            const total_rows: i64 = @intCast(rows_per_part * c.ngram_parts);
            const per_head: i64 = @divExact(total_rows, @as(i64, @intCast(c.ngram_heads)));
            var vocab: [64]i64 = undefined;
            var offs: [64]i64 = undefined;
            for (0..c.ngram_heads) |h| {
                vocab[h] = per_head;
                offs[h] = @as(i64, @intCast(h)) * per_head;
            }
            try b.addRawf("{s}.ple.ple_embedding.layer_multipliers", .{L}, .i64, &.{c.ngram_size}, &mult);
            try b.addRawf("{s}.ple.ple_embedding.ngram_heads_vocab_sizes", .{L}, .i64, &.{c.ngram_heads}, vocab[0..c.ngram_heads]);
            try b.addRawf("{s}.ple.ple_embedding.ngram_heads_offsets", .{L}, .i64, &.{c.ngram_heads}, offs[0..c.ngram_heads]);
            try b.addf("{s}.ple.ple_embedding.ngram_embedding.weight_scale", .{L}, .f32, &.{1});
            for (0..c.ngram_parts) |p| {
                try b.addf("{s}.ple.ple_embedding.ngram_embedding.shard_{d}.weight", .{ L, p }, .f8_e4m3, &.{ rows_per_part, c.ngram_head_dim });
            }
        }
    }

    try writeFixture(gpa, io, b.list.items);
    try writeTokenizer(gpa, io);

    var log_buf: [512]u8 = undefined;
    var log_fw: std.Io.File.Writer = .init(.stderr(), io, &log_buf);
    try log_fw.interface.print("wrote {s}/ : {d} tensors + tokenizer.json\n", .{ out_dir, b.list.items.len });
    try log_fw.interface.flush();
}

/// A minuscule byte-level BPE tokenizer: a-z / 0-9 / a few punctuation chars,
/// the space (Ġ) and newline (Ċ) byte-unicode chars, a handful of merges that
/// build up "hello"/"world", and the ChatML special tokens.  Every id stays
/// below the tiny config's vocab_size.
fn writeTokenizer(gpa: std.mem.Allocator, io: std.Io) !void {
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(gpa);

    try j.appendSlice(gpa, "{\n  \"model\": { \"type\": \"BPE\",\n    \"vocab\": {");
    var id: u32 = 0;
    var first = true;
    const put = struct {
        fn p(buf: *std.ArrayList(u8), g: std.mem.Allocator, f: *bool, key: []const u8, i: u32) !void {
            if (!f.*) try buf.appendSlice(g, ", ");
            f.* = false;
            try buf.append(g, '"');
            for (key) |ch| {
                if (ch == '"' or ch == '\\') try buf.append(g, '\\');
                try buf.append(g, ch);
            }
            try buf.print(g, "\": {d}", .{i});
        }
    }.p;

    var ch: u8 = 'a';
    while (ch <= 'z') : (ch += 1) {
        try put(&j, gpa, &first, &.{ch}, id);
        id += 1;
    }
    ch = '0';
    while (ch <= '9') : (ch += 1) {
        try put(&j, gpa, &first, &.{ch}, id);
        id += 1;
    }
    for ([_][]const u8{ "\xc4\xa0", "\xc4\x8a", ".", "!", "?", ",", "<", ">", "|" }) |k| { // Ġ Ċ then ascii
        try put(&j, gpa, &first, k, id);
        id += 1;
    }
    // merged tokens
    const merges = [_][]const u8{ "he", "ll", "hell", "hello", "lo", "\xc4\xa0w", "wo", "wor", "worl", "world" };
    for (merges) |m| {
        try put(&j, gpa, &first, m, id);
        id += 1;
    }

    try j.appendSlice(gpa,
        \\},
        \\    "merges": ["h e", "l l", "he ll", "hell o", "l o", "w o", "wo r", "wor l", "worl d"] },
        \\  "added_tokens": [
        \\    { "id": 55, "content": "<|im_start|>", "special": true },
        \\    { "id": 56, "content": "<|im_end|>", "special": true }
        \\  ]
        \\}
    );

    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, out_dir, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = j.items });
}

fn ceilDiv(a: u64, b: u64) u64 {
    return (a + b - 1) / b;
}

fn fnv1a(s: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (s) |c| {
        h ^= c;
        h *%= 0x100000001b3;
    }
    return h;
}

/// A small deterministic value in roughly [-1, 1] for element `k` of tensor `t`.
fn sample(seed: u64, k: usize) f32 {
    var x: u64 = seed ^ (@as(u64, k) *% 0x9e3779b97f4a7c15);
    x ^= x >> 33;
    x *%= 0xff51afd7ed558ccd;
    x ^= x >> 33;
    const frac: f32 = @floatFromInt(x & 0xffffff);
    return (frac / @as(f32, 0x1000000)) * 2.0 - 1.0;
}

fn f32ToBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    const bias: u32 = 0x7fff + ((bits >> 16) & 1);
    return @truncate((bits + bias) >> 16);
}

// E4M3 bytes decoding to exact small values (mirrors ops/fp8.zig demo_bytes).
const e4m3_demo = [_]u8{ 0x38, 0x34, 0x30, 0x3c, 0xb8, 0xb4, 0x40, 0x2c };

fn fillBody(dst: []u8, t: Tensor) void {
    if (t.body) |bd| {
        @memcpy(dst[0..bd.len], bd);
        return;
    }
    const seed = fnv1a(t.name);
    var numel: usize = 1;
    for (t.shape) |d| numel *= @intCast(d);
    // Block scales (weight_scale_inv / weight_scale) must be positive.
    const is_scale = std.mem.endsWith(u8, t.name, "weight_scale_inv") or
        std.mem.endsWith(u8, t.name, "weight_scale");
    switch (t.dtype) {
        .f32 => {
            var k: usize = 0;
            while (k < numel) : (k += 1) {
                const v = if (is_scale) 0.5 + 0.5 * @abs(sample(seed, k)) else sample(seed, k);
                std.mem.writeInt(u32, dst[k * 4 ..][0..4], @bitCast(v), .little);
            }
        },
        .bf16 => {
            var k: usize = 0;
            while (k < numel) : (k += 1) {
                std.mem.writeInt(u16, dst[k * 2 ..][0..2], f32ToBf16(sample(seed, k)), .little);
            }
        },
        .f8_e4m3 => {
            var k: usize = 0;
            while (k < numel) : (k += 1) {
                dst[k] = e4m3_demo[(seed +% k) % e4m3_demo.len];
            }
        },
        .i64 => {}, // left zero — not decoded yet
    }
}

fn writeFixture(gpa: std.mem.Allocator, io: std.Io, tensors: []const Tensor) !void {
    // safetensors header JSON + zero body
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(gpa);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(gpa);

    try header.appendSlice(gpa, "{");
    try index.appendSlice(gpa, "{\n  \"metadata\": {\"total_size\": ");
    const index_total_placeholder = index.items.len;
    _ = index_total_placeholder;

    var cursor: u64 = 0;
    var body_index: std.ArrayList(u8) = .empty; // weight_map body
    defer body_index.deinit(gpa);

    for (tensors, 0..) |t, i| {
        const nb = t.nbytes();
        if (i != 0) try header.appendSlice(gpa, ",");
        try header.print(gpa, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, t.dtype.label() });
        for (t.shape, 0..) |d, k| {
            if (k != 0) try header.appendSlice(gpa, ",");
            try header.print(gpa, "{d}", .{d});
        }
        try header.print(gpa, "],\"data_offsets\":[{d},{d}]}}", .{ cursor, cursor + nb });
        cursor += nb;

        if (i != 0) try body_index.appendSlice(gpa, ",\n");
        try body_index.print(gpa, "    \"{s}\": \"model.safetensors\"", .{t.name});
    }
    try header.appendSlice(gpa, "}");

    const total_body = cursor;

    try index.print(gpa, "{d}}},\n  \"weight_map\": {{\n", .{total_body});
    try index.appendSlice(gpa, body_index.items);
    try index.appendSlice(gpa, "\n  }\n}\n");

    // Assemble model.safetensors
    const st_bytes = try gpa.alloc(u8, 8 + header.items.len + total_body);
    defer gpa.free(st_bytes);
    std.mem.writeInt(u64, st_bytes[0..8], header.items.len, .little);
    @memcpy(st_bytes[8 .. 8 + header.items.len], header.items);
    @memset(st_bytes[8 + header.items.len ..], 0);

    // Deterministic non-zero bodies for the dtypes Phase 2 decodes (F32/F16/
    // BF16); FP8 and I64 tensors stay zero (decoded only in later phases).
    const body = st_bytes[8 + header.items.len ..];
    var off: u64 = 0;
    for (tensors) |t| {
        const nb = t.nbytes();
        fillBody(body[@intCast(off)..][0..@intCast(nb)], t);
        off += nb;
    }

    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, out_dir, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg_mod.tiny_config_json ++ "\n" });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = st_bytes });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}
