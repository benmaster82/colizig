//! Parser and validator for the Qwen4-Exp (`Qwen3.8-Flash-Next`) `config.json`.
//!
//! The released checkpoint is a multimodal wrapper (`model_type: "qwen4_exp"`)
//! whose language model lives under `text_config` (`model_type:
//! "qwen4_exp_text"`).  We parse `text_config` when present, else the root.
//!
//! Like colibri's `q38_load_cfg`, this is deliberately strict: a field whose
//! value this engine does not implement is a hard error, not a silent
//! best-effort.  Nothing here loads weights.

const std = @import("std");
const Value = std.json.Value;

pub const Error = error{
    InvalidConfig,
    UnsupportedModel,
    OutOfMemory,
};

pub const Vision = struct {
    depth: u32,
    hidden: u32,
    heads: u32,
    inter: u32,
    patch: u32,
    merge: u32,
    out_hidden: u32,
    num_pos: u32,
    /// Vision is indexed but never executed in this engine (text-only, as colibri).
    loaded: bool = false,
};

pub const Arch = enum {
    /// Qwen3.8-Flash-Next / Qwen4-Exp: GDN + QSA + PLE + 512-expert MoE + shared.
    qwen4_exp,
    /// Qwen3-MoE (e.g. Qwen3-30B-A3B): plain GQA attention with QK-norm, no
    /// shared expert, no PLE / GDN / hyper-connections. Fields specific to
    /// `qwen4_exp` are left zero.
    qwen3_moe,
};

pub const Cfg = struct {
    arch: Arch = .qwen4_exp,

    // identity (both arches)
    hidden: u32,
    layers: u32,
    vocab: u32,
    max_positions: u32,
    eos_id: i64,
    eps: f32,
    theta: f32,
    partial_rotary: f32 = 1.0,
    rotary_dim: u32,

    // attention (both arches)
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,

    // MoE (both arches; `shared_inter == 0` ⇒ no shared expert)
    experts: u32,
    topk: u32,
    inter: u32,
    shared_inter: u32 = 0,
    norm_topk: bool = true,

    // --- qwen4_exp only (zero for qwen3_moe) ---
    // gated residual ("hyper connections", 4 branches)
    hc_count: u32 = 0,
    hc_rank: u32 = 0,
    hc_width: u32 = 0,
    // lightweight indexer inside QSA layers
    idx_qheads: u32 = 0,
    idx_kheads: u32 = 0,
    idx_dim: u32 = 0,
    idx_budget: u32 = 0,
    idx_ratio: u32 = 0,
    // Gated DeltaNet (linear attention)
    dn_kheads: u32 = 0,
    dn_vheads: u32 = 0,
    dn_kdim: u32 = 0,
    dn_vdim: u32 = 0,
    dn_convk: u32 = 0,
    dn_conv_dim: u32 = 0,
    // PLE / hashed n-gram
    ple_layer: u32 = 0, // 0-based
    ple_dim: u32 = 0,
    ple_convk: u32 = 0,
    ngram_size: u32 = 0,
    heads_per_ngram: u32 = 0,
    ngram_heads: u32 = 0,
    ngram_head_dim: u32 = 0,
    ngram_parts: u32 = 0,

    /// per-layer kind: true = attention (QSA / full), false = Gated DeltaNet.
    /// All-true for qwen3_moe.
    is_attn: []bool,

    vision: ?Vision = null,

    allocator: std.mem.Allocator,

    pub fn deinit(self: *Cfg) void {
        self.allocator.free(self.is_attn);
        self.* = undefined;
    }

    pub fn numAttnLayers(self: Cfg) u32 {
        var n: u32 = 0;
        for (self.is_attn) |a| {
            if (a) n += 1;
        }
        return n;
    }

    pub fn numGdnLayers(self: Cfg) u32 {
        return self.layers - self.numAttnLayers();
    }
};

/// Read `<dir>/config.json` and parse it. `err` receives a human-readable reason
/// on failure.
pub fn parseDir(
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    err: *std.Io.Writer,
) !Cfg {
    const bytes = dir.readFileAlloc(io, "config.json", gpa, .limited(64 << 20)) catch |e| {
        try err.print("cannot read config.json: {s}\n", .{@errorName(e)});
        return error.InvalidConfig;
    };
    defer gpa.free(bytes);
    return parseSlice(gpa, bytes, err);
}

pub fn parseSlice(gpa: std.mem.Allocator, bytes: []const u8, err: *std.Io.Writer) !Cfg {
    var parsed = std.json.parseFromSlice(Value, gpa, bytes, .{}) catch |e| {
        try err.print("config.json is not valid JSON: {s}\n", .{@errorName(e)});
        return error.InvalidConfig;
    };
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |o| o,
        else => {
            try err.writeAll("config.json root is not an object\n");
            return error.InvalidConfig;
        },
    };

    // Prefer the nested language-model config.
    const tc: std.json.ObjectMap = blk: {
        if (root.get("text_config")) |v| switch (v) {
            .object => |o| break :blk o,
            else => {
                try err.writeAll("config.json: text_config must be an object\n");
                return error.InvalidConfig;
            },
        };
        break :blk root;
    };

    const model_type = getStr(tc, "model_type") orelse "";
    if (std.mem.eql(u8, model_type, "qwen3_moe")) return parseQwen3Moe(gpa, tc, err);
    if (!std.mem.eql(u8, model_type, "qwen4_exp_text")) {
        try err.print(
            "unsupported text model_type: \"{s}\" (expected \"qwen4_exp_text\" or \"qwen3_moe\")\n",
            .{model_type},
        );
        return error.UnsupportedModel;
    }

    try requireStr(tc, "hidden_act", "silu", err);
    try requireStr(tc, "output_gate_type", "sigmoid", err);
    try requireBool(tc, "attention_bias", false, err);
    try requireBool(tc, "tie_word_embeddings", false, err);

    const rope: ?std.json.ObjectMap = switch (tc.get("rope_parameters") orelse Value.null) {
        .object => |o| o,
        .null => null,
        else => {
            try err.writeAll("config.json: rope_parameters must be an object\n");
            return error.InvalidConfig;
        },
    };
    if (rope) |rp| try requireStr(rp, "rope_type", "default", err);

    var c: Cfg = undefined;
    c.arch = .qwen4_exp;
    c.allocator = gpa;
    c.vision = null;
    c.is_attn = &.{};
    errdefer if (c.is_attn.len != 0) gpa.free(c.is_attn);

    c.hidden = try reqInt(u32, tc, "hidden_size", err);
    c.layers = try reqInt(u32, tc, "num_hidden_layers", err);
    c.vocab = try reqInt(u32, tc, "vocab_size", err);
    c.max_positions = try reqInt(u32, tc, "max_position_embeddings", err);
    c.eos_id = try reqInt(i64, tc, "eos_token_id", err);
    c.eps = @floatCast(try optFloat(tc, "rms_norm_eps", 1e-6));

    const theta = if (rope) |rp| optFloat(rp, "rope_theta", 10000) else optFloat(tc, "rope_theta", 10000);
    c.theta = @floatCast(try theta);
    const partial = if (rope) |rp| optFloat(rp, "partial_rotary_factor", 1.0) else optFloat(tc, "partial_rotary_factor", 1.0);
    c.partial_rotary = @floatCast(try partial);

    c.hc_count = try optInt(u32, tc, "hc_count", 4);
    c.hc_rank = try optInt(u32, tc, "hc_lowrank", 320);
    c.hc_width = try mul(c.hc_count, c.hidden, "hc_width", err);

    c.q_heads = try reqInt(u32, tc, "num_attention_heads", err);
    c.kv_heads = try reqInt(u32, tc, "num_key_value_heads", err);
    c.head_dim = try reqInt(u32, tc, "head_dim", err);
    // Upstream derives rotary_dim with Python int() — truncation toward zero.
    const rd: f64 = @as(f64, @floatFromInt(c.head_dim)) * @as(f64, c.partial_rotary);
    if (!(rd >= 0) or rd > std.math.maxInt(u32)) {
        try err.writeAll("config.json: derived rotary_dim out of range\n");
        return error.InvalidConfig;
    }
    c.rotary_dim = @intFromFloat(@trunc(rd));

    c.idx_qheads = try reqInt(u32, tc, "indexer_n_heads", err);
    c.idx_kheads = try reqInt(u32, tc, "indexer_kv_heads", err);
    c.idx_dim = try reqInt(u32, tc, "indexer_head_dim", err);
    c.idx_budget = try reqInt(u32, tc, "indexer_budget", err);
    c.idx_ratio = try reqInt(u32, tc, "indexer_compress_ratio", err);

    c.experts = try reqInt(u32, tc, "num_experts", err);
    c.topk = try reqInt(u32, tc, "num_experts_per_tok", err);
    c.inter = try reqInt(u32, tc, "moe_intermediate_size", err);
    c.shared_inter = try reqInt(u32, tc, "shared_expert_intermediate_size", err);
    c.norm_topk = try optBool(tc, "norm_topk_prob", true);

    c.dn_kheads = try reqInt(u32, tc, "linear_num_key_heads", err);
    c.dn_vheads = try reqInt(u32, tc, "linear_num_value_heads", err);
    c.dn_kdim = try reqInt(u32, tc, "linear_key_head_dim", err);
    c.dn_vdim = try reqInt(u32, tc, "linear_value_head_dim", err);
    c.dn_convk = try reqInt(u32, tc, "linear_conv_kernel_dim", err);
    {
        const qk = try mul(try mul(2, c.dn_kheads, "dn_qk", err), c.dn_kdim, "dn_qk", err);
        const v = try mul(c.dn_vheads, c.dn_vdim, "dn_v", err);
        c.dn_conv_dim = try add(qk, v, "dn_conv_dim", err);
    }

    c.ple_dim = try optInt(u32, tc, "ple_embed_dim", c.hidden);
    c.ple_convk = try optInt(u32, tc, "ple_conv_kernel_size", 4);
    c.ngram_size = try optInt(u32, tc, "ngram_size", 3);
    c.heads_per_ngram = try optInt(u32, tc, "heads_per_ngram", 8);
    if (c.ngram_size == 0) {
        try err.writeAll("config.json: ngram_size must be >= 1\n");
        return error.InvalidConfig;
    }
    c.ngram_heads = try mul(c.ngram_size - 1, c.heads_per_ngram, "ngram_heads", err);
    if (c.ngram_heads == 0 or c.ple_dim % c.ngram_heads != 0) {
        try err.writeAll("config.json: ple_embed_dim not divisible by derived ngram_heads\n");
        return error.InvalidConfig;
    }
    c.ngram_head_dim = c.ple_dim / c.ngram_heads;
    c.ngram_parts = try optInt(u32, tc, "split_ngram_parts", 1);

    // ple_layer_ids: exactly one, 1-based in the checkpoint.
    c.ple_layer = blk: {
        const v = tc.get("ple_layer_ids") orelse {
            try err.writeAll("config.json: missing ple_layer_ids\n");
            return error.InvalidConfig;
        };
        const arr = switch (v) {
            .array => |a| a,
            else => {
                try err.writeAll("config.json: ple_layer_ids must be an array\n");
                return error.InvalidConfig;
            },
        };
        if (arr.items.len != 1) {
            try err.writeAll("config.json: exactly one PLE layer is supported\n");
            return error.InvalidConfig;
        }
        const one_based = asInt(i64, arr.items[0]) orelse {
            try err.writeAll("config.json: ple_layer_ids[0] must be an integer\n");
            return error.InvalidConfig;
        };
        if (one_based < 1 or one_based > c.layers) {
            try err.print("config.json: ple_layer_ids[0]={d} out of range 1..{d}\n", .{ one_based, c.layers });
            return error.InvalidConfig;
        }
        break :blk @intCast(one_based - 1);
    };

    // layer_types → is_attn
    {
        const v = tc.get("layer_types") orelse {
            try err.writeAll("config.json: missing layer_types\n");
            return error.InvalidConfig;
        };
        const arr = switch (v) {
            .array => |a| a,
            else => {
                try err.writeAll("config.json: layer_types must be an array\n");
                return error.InvalidConfig;
            },
        };
        if (arr.items.len != c.layers) {
            try err.print("config.json: layer_types has {d} entries, expected num_hidden_layers={d}\n", .{ arr.items.len, c.layers });
            return error.InvalidConfig;
        }
        const is_attn = try gpa.alloc(bool, c.layers);
        errdefer gpa.free(is_attn);
        for (arr.items, 0..) |item, i| {
            const s = switch (item) {
                .string => |t| t,
                else => {
                    try err.print("config.json: layer_types[{d}] is not a string\n", .{i});
                    return error.InvalidConfig;
                },
            };
            if (std.mem.eql(u8, s, "linear_attention")) {
                is_attn[i] = false;
            } else if (std.mem.eql(u8, s, "full_attention") or std.mem.eql(u8, s, "qwen_sparse_attention")) {
                is_attn[i] = true;
            } else {
                try err.print("config.json: unsupported layer type \"{s}\" at index {d}\n", .{ s, i });
                return error.InvalidConfig;
            }
        }
        c.is_attn = is_attn;
    }

    // vision_config: optional; parsed for reporting, never executed.
    if (root.get("vision_config")) |vv| switch (vv) {
        .object => |vo| {
            c.vision = .{
                .depth = try optInt(u32, vo, "depth", 0),
                .hidden = try optInt(u32, vo, "hidden_size", 0),
                .heads = try optInt(u32, vo, "num_heads", 0),
                .inter = try optInt(u32, vo, "intermediate_size", 0),
                .patch = try optInt(u32, vo, "patch_size", 16),
                .merge = try optInt(u32, vo, "spatial_merge_size", 2),
                .out_hidden = try optInt(u32, vo, "out_hidden_size", c.hidden),
                .num_pos = try optInt(u32, vo, "num_position_embeddings", 0),
                .loaded = false,
            };
        },
        else => {},
    };

    try validate(c, err);
    return c;
}

/// Qwen3-MoE (`model_type: "qwen3_moe"`, e.g. Qwen3-30B-A3B). A plain
/// pre-norm transformer: GQA attention with per-head QK RMSNorm and full RoPE,
/// then a top-k MoE with no shared expert. None of the Qwen4-Exp machinery
/// (GDN / QSA indexer / PLE / hyper-connections) is present; those `Cfg` fields
/// stay zero and `is_attn` is all-true.
fn parseQwen3Moe(gpa: std.mem.Allocator, tc: std.json.ObjectMap, err: *std.Io.Writer) !Cfg {
    try requireStr(tc, "hidden_act", "silu", err);
    try requireBool(tc, "attention_bias", false, err);
    try requireBool(tc, "tie_word_embeddings", false, err);
    if (tc.get("rope_scaling")) |v| if (v != .null) {
        try err.writeAll("config.json: rope_scaling / YaRN is not supported (use the base checkpoint)\n");
        return error.UnsupportedModel;
    };
    if (tc.get("mlp_only_layers")) |v| if (v == .array and v.array.items.len != 0) {
        try err.writeAll("config.json: mlp_only_layers (dense layers among the MoE) not supported\n");
        return error.UnsupportedModel;
    };

    const layers = try reqInt(u32, tc, "num_hidden_layers", err);
    const is_attn = try gpa.alloc(bool, layers);
    errdefer gpa.free(is_attn);
    @memset(is_attn, true);

    const head_dim = try reqInt(u32, tc, "head_dim", err);
    const c: Cfg = .{
        .arch = .qwen3_moe,
        .allocator = gpa,
        .is_attn = is_attn,
        .hidden = try reqInt(u32, tc, "hidden_size", err),
        .layers = layers,
        .vocab = try reqInt(u32, tc, "vocab_size", err),
        .max_positions = try reqInt(u32, tc, "max_position_embeddings", err),
        .eos_id = try reqInt(i64, tc, "eos_token_id", err),
        .eps = @floatCast(try optFloat(tc, "rms_norm_eps", 1e-6)),
        .theta = @floatCast(try optFloat(tc, "rope_theta", 1000000)),
        .q_heads = try reqInt(u32, tc, "num_attention_heads", err),
        .kv_heads = try reqInt(u32, tc, "num_key_value_heads", err),
        .head_dim = head_dim,
        .rotary_dim = head_dim, // full RoPE
        .experts = try reqInt(u32, tc, "num_experts", err),
        .topk = try reqInt(u32, tc, "num_experts_per_tok", err),
        .inter = try reqInt(u32, tc, "moe_intermediate_size", err),
        .norm_topk = try optBool(tc, "norm_topk_prob", true),
    };

    try validate(c, err);
    return c;
}

pub fn validate(c: Cfg, err: *std.Io.Writer) !void {
    if (c.arch == .qwen3_moe) return validateQwen3Moe(c, err);
    return validateQwen4Exp(c, err);
}

fn validateQwen3Moe(c: Cfg, err: *std.Io.Writer) !void {
    const need = struct {
        fn f(ok: bool, w: *std.Io.Writer, comptime msg: []const u8) !void {
            if (!ok) {
                try w.writeAll("config.json: " ++ msg ++ " — refusing\n");
                return error.InvalidConfig;
            }
        }
    }.f;
    try need(c.hidden > 0 and c.hidden <= 65536, err, "hidden_size out of range");
    try need(c.layers > 0 and c.layers <= 512 and c.is_attn.len == c.layers, err, "num_hidden_layers out of range");
    try need(c.vocab > 0, err, "vocab_size out of range");
    try need(c.max_positions > 0, err, "max_position_embeddings out of range");
    try need(std.math.isFinite(c.eps) and c.eps > 0, err, "rms_norm_eps invalid");
    try need(std.math.isFinite(c.theta) and c.theta > 0, err, "rope_theta invalid");
    try need(c.eos_id >= 0 and c.eos_id < c.vocab, err, "eos_token_id outside vocabulary");
    try need(c.q_heads > 0 and c.kv_heads > 0 and c.q_heads % c.kv_heads == 0, err, "attention head counts invalid");
    try need(c.head_dim > 0 and c.head_dim % 2 == 0, err, "head_dim invalid");
    try need(c.experts > 0 and c.experts <= 4096, err, "num_experts out of range");
    try need(c.topk > 0 and c.topk <= c.experts, err, "num_experts_per_tok invalid");
    try need(c.inter > 0, err, "moe_intermediate_size invalid");
}

fn validateQwen4Exp(c: Cfg, err: *std.Io.Writer) !void {
    const need = struct {
        fn f(ok: bool, w: *std.Io.Writer, comptime msg: []const u8) !void {
            if (!ok) {
                try w.writeAll("config.json: " ++ msg ++ " — refusing\n");
                return error.InvalidConfig;
            }
        }
    }.f;

    try need(c.hidden > 0 and c.hidden <= 65536, err, "hidden_size out of range");
    try need(c.layers > 0 and c.layers <= 512, err, "num_hidden_layers out of range");
    try need(c.vocab > 0, err, "vocab_size out of range");
    try need(c.max_positions > 0, err, "max_position_embeddings out of range");
    try need(std.math.isFinite(c.eps) and c.eps > 0, err, "rms_norm_eps invalid");
    try need(std.math.isFinite(c.theta) and c.theta > 0, err, "rope_theta invalid");
    try need(c.eos_id >= 0 and c.eos_id < c.vocab, err, "eos_token_id outside vocabulary");
    try need(c.hc_count > 1 and c.hc_count <= 16 and c.hc_rank > 0, err, "gated-residual dimensions invalid");
    try need(c.q_heads > 0 and c.kv_heads > 0 and c.q_heads % c.kv_heads == 0, err, "attention head counts invalid");
    try need(c.head_dim > 0 and c.rotary_dim > 0 and c.rotary_dim % 2 == 0 and c.rotary_dim <= c.head_dim, err, "RoPE dimensions invalid");
    try need(c.idx_qheads > 0 and c.idx_kheads == 1 and c.idx_dim >= c.rotary_dim, err, "indexer dimensions invalid");
    try need(c.idx_ratio > 0 and c.idx_budget > 0 and c.idx_budget % c.idx_ratio == 0, err, "indexer budget invalid");
    try need(c.experts > 0 and c.experts <= 4096, err, "num_experts out of range");
    try need(c.topk > 0 and c.topk <= c.experts, err, "num_experts_per_tok invalid");
    try need(c.inter > 0 and c.shared_inter > 0, err, "MoE widths invalid");
    try need(c.dn_vheads > 0 and c.dn_kheads > 0 and c.dn_vheads % c.dn_kheads == 0, err, "DeltaNet head counts invalid");
    try need(c.dn_kdim > 0 and c.dn_vdim > 0 and c.dn_convk >= 2 and c.dn_conv_dim > 0, err, "DeltaNet dimensions invalid");
    try need(c.ngram_size == 3, err, "only ngram_size == 3 is supported");
    try need(c.ngram_heads > 0 and c.ngram_heads <= 64, err, "derived ngram_heads out of range");
    try need(c.ngram_head_dim > 0 and c.ngram_head_dim <= 512, err, "derived ngram_head_dim out of range");
    try need(c.ple_convk >= 2, err, "ple_conv_kernel_size invalid");
    try need(c.ngram_parts > 0 and c.ngram_parts <= 512, err, "split_ngram_parts out of range");
    try need(c.ple_layer < c.layers, err, "PLE layer index out of range");

    if (c.vision) |v| {
        try need(v.out_hidden == c.hidden, err, "vision out_hidden_size != text hidden_size");
        try need(v.heads == 0 or v.hidden % v.heads == 0, err, "vision hidden_size not divisible by num_heads");
    }
}

// ---- small JSON helpers ------------------------------------------------------

fn getStr(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    return switch (o.get(key) orelse return null) {
        .string => |s| s,
        else => null,
    };
}

fn asInt(comptime T: type, v: Value) ?T {
    return switch (v) {
        .integer => |i| std.math.cast(T, i),
        .float => |f| if (std.math.isFinite(f) and @trunc(f) == f and f >= minFloat(T) and f <= maxFloat(T))
            @intFromFloat(f)
        else
            null,
        .number_string => |s| std.fmt.parseInt(T, s, 10) catch null,
        else => null,
    };
}

fn minFloat(comptime T: type) f64 {
    return @floatFromInt(std.math.minInt(T));
}
fn maxFloat(comptime T: type) f64 {
    return @floatFromInt(std.math.maxInt(T));
}

fn asFloat(v: Value) ?f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch null,
        else => null,
    };
}

fn reqInt(comptime T: type, o: std.json.ObjectMap, key: []const u8, err: *std.Io.Writer) !T {
    const v = o.get(key) orelse {
        try err.print("config.json: missing integer field \"{s}\"\n", .{key});
        return error.InvalidConfig;
    };
    return asInt(T, v) orelse {
        try err.print("config.json: field \"{s}\" is not an integer in range\n", .{key});
        return error.InvalidConfig;
    };
}

fn optInt(comptime T: type, o: std.json.ObjectMap, key: []const u8, default: T) !T {
    const v = o.get(key) orelse return default;
    return asInt(T, v) orelse error.InvalidConfig;
}

fn optFloat(o: std.json.ObjectMap, key: []const u8, default: f64) !f64 {
    const v = o.get(key) orelse return default;
    return asFloat(v) orelse error.InvalidConfig;
}

fn optBool(o: std.json.ObjectMap, key: []const u8, default: bool) !bool {
    return switch (o.get(key) orelse return default) {
        .bool => |b| b,
        else => error.InvalidConfig,
    };
}

fn requireStr(o: std.json.ObjectMap, key: []const u8, expected: []const u8, err: *std.Io.Writer) !void {
    const got = getStr(o, key) orelse {
        try err.print("config.json: \"{s}\" must be the string \"{s}\"\n", .{ key, expected });
        return error.InvalidConfig;
    };
    if (!std.mem.eql(u8, got, expected)) {
        try err.print("config.json: unsupported {s}=\"{s}\" (expected \"{s}\")\n", .{ key, got, expected });
        return error.InvalidConfig;
    }
}

fn requireBool(o: std.json.ObjectMap, key: []const u8, expected: bool, err: *std.Io.Writer) !void {
    const got = switch (o.get(key) orelse Value.null) {
        .bool => |b| b,
        .null => {
            try err.print("config.json: \"{s}\" must be explicitly {}\n", .{ key, expected });
            return error.InvalidConfig;
        },
        else => {
            try err.print("config.json: \"{s}\" must be a boolean\n", .{key});
            return error.InvalidConfig;
        },
    };
    if (got != expected) {
        try err.print("config.json: unsupported {s}={} (expected {})\n", .{ key, got, expected });
        return error.InvalidConfig;
    }
}

fn mul(a: u32, b: u32, comptime name: []const u8, err: *std.Io.Writer) !u32 {
    return std.math.mul(u32, a, b) catch {
        try err.writeAll("config.json: derived " ++ name ++ " overflows u32\n");
        return error.InvalidConfig;
    };
}

fn add(a: u32, b: u32, comptime name: []const u8, err: *std.Io.Writer) !u32 {
    return std.math.add(u32, a, b) catch {
        try err.writeAll("config.json: derived " ++ name ++ " overflows u32\n");
        return error.InvalidConfig;
    };
}

/// Tiny `qwen3_moe` config for `test/fixtures/tiny-qwen3/`.
pub const tiny_qwen3_config_json =
    \\{
    \\  "model_type": "qwen3_moe",
    \\  "hidden_size": 32, "num_hidden_layers": 4, "vocab_size": 64,
    \\  "max_position_embeddings": 4096, "eos_token_id": 2,
    \\  "hidden_act": "silu", "attention_bias": false, "tie_word_embeddings": false,
    \\  "rms_norm_eps": 0.000001, "rope_theta": 1000000,
    \\  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
    \\  "num_experts": 4, "num_experts_per_tok": 2,
    \\  "moe_intermediate_size": 16, "norm_topk_prob": true,
    \\  "decoder_sparse_step": 1, "mlp_only_layers": []
    \\}
;

// ---- tests -----------------------------------------------------------------

test "tiny qwen3_moe config parses" {
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);
    var c = try parseSlice(std.testing.allocator, tiny_qwen3_config_json, &sink.writer);
    defer c.deinit();
    try std.testing.expectEqual(Arch.qwen3_moe, c.arch);
    try std.testing.expectEqual(@as(u32, 16), c.head_dim);
    try std.testing.expectEqual(@as(u32, 16), c.rotary_dim); // full rope
    try std.testing.expectEqual(@as(u32, 0), c.shared_inter); // no shared expert
    try std.testing.expectEqual(@as(u32, 4), c.layers);
    for (c.is_attn) |a| try std.testing.expect(a);
}

/// A minimal but structurally faithful tiny config, mirroring colibri's
/// `test_qwen38_config.c`.  Kept in sync with `tools/gen_tiny_fixture.zig`.
pub const tiny_config_json =
    \\{
    \\  "model_type": "qwen4_exp",
    \\  "text_config": {
    \\    "model_type": "qwen4_exp_text",
    \\    "hidden_size": 32, "num_hidden_layers": 4, "vocab_size": 64,
    \\    "max_position_embeddings": 4096, "eos_token_id": 2,
    \\    "hidden_act": "silu", "output_gate_type": "sigmoid",
    \\    "attention_bias": false, "tie_word_embeddings": false,
    \\    "rms_norm_eps": 0.000001,
    \\    "rope_parameters": { "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25 },
    \\    "hc_count": 4, "hc_lowrank": 8,
    \\    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
    \\    "indexer_n_heads": 1, "indexer_kv_heads": 1, "indexer_head_dim": 4,
    \\    "indexer_budget": 4, "indexer_compress_ratio": 2,
    \\    "num_experts": 4, "num_experts_per_tok": 2,
    \\    "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16,
    \\    "norm_topk_prob": true,
    \\    "linear_num_key_heads": 1, "linear_num_value_heads": 2,
    \\    "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
    \\    "ple_embed_dim": 32, "ple_conv_kernel_size": 4,
    \\    "ngram_size": 3, "heads_per_ngram": 2, "split_ngram_parts": 2,
    \\    "ple_layer_ids": [2],
    \\    "layer_types": ["linear_attention", "full_attention", "linear_attention", "linear_attention"]
    \\  }
    \\}
;

fn parseTiny(gpa: std.mem.Allocator) !Cfg {
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);
    return parseSlice(gpa, tiny_config_json, &sink.writer);
}

test "tiny config parses with expected derived values" {
    var c = try parseTiny(std.testing.allocator);
    defer c.deinit();
    try std.testing.expectEqual(@as(u32, 32), c.hidden);
    try std.testing.expectEqual(@as(u32, 4), c.layers);
    try std.testing.expectEqual(@as(u32, 2), c.rotary_dim); // int(8 * 0.25)
    try std.testing.expectEqual(@as(u32, 4), c.ngram_heads); // (3-1)*2
    try std.testing.expectEqual(@as(u32, 8), c.ngram_head_dim); // 32 / 4
    try std.testing.expectEqual(@as(u32, 128), c.hc_width); // 4 * 32
    try std.testing.expectEqual(@as(u32, 1), c.ple_layer); // 1-based [2] -> 0-based 1
    try std.testing.expect(c.is_attn[1]); // PLE lands on the attention layer
    try std.testing.expectEqual(@as(u32, 1), c.numAttnLayers());
    try std.testing.expectEqual(@as(u32, 3), c.numGdnLayers());
}

test "strict fields are rejected" {
    const gpa = std.testing.allocator;
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);
    const cases = [_][]const u8{
        // wrong hidden_act
        \\{"model_type":"qwen4_exp_text","hidden_size":32,"num_hidden_layers":1,"vocab_size":8,"max_position_embeddings":8,"eos_token_id":2,"hidden_act":"gelu","output_gate_type":"sigmoid","attention_bias":false,"tie_word_embeddings":false,"rms_norm_eps":1e-6,"rope_parameters":{"rope_type":"default","partial_rotary_factor":0.25},"hc_count":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"indexer_n_heads":1,"indexer_kv_heads":1,"indexer_head_dim":4,"indexer_budget":4,"indexer_compress_ratio":2,"num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":4,"shared_expert_intermediate_size":4,"linear_num_key_heads":1,"linear_num_value_heads":1,"linear_key_head_dim":8,"linear_value_head_dim":2,"linear_conv_kernel_dim":2,"ple_embed_dim":8,"ngram_size":3,"heads_per_ngram":1,"split_ngram_parts":1,"ple_layer_ids":[1],"layer_types":["linear_attention"]}
        ,
        // wrong model_type
        \\{"model_type":"llama","hidden_size":32}
        ,
        // attention_bias true
        \\{"model_type":"qwen4_exp_text","hidden_size":32,"num_hidden_layers":1,"vocab_size":8,"max_position_embeddings":8,"eos_token_id":2,"hidden_act":"silu","output_gate_type":"sigmoid","attention_bias":true,"tie_word_embeddings":false,"rms_norm_eps":1e-6,"rope_parameters":{"rope_type":"default","partial_rotary_factor":0.25},"hc_count":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"indexer_n_heads":1,"indexer_kv_heads":1,"indexer_head_dim":4,"indexer_budget":4,"indexer_compress_ratio":2,"num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":4,"shared_expert_intermediate_size":4,"linear_num_key_heads":1,"linear_num_value_heads":1,"linear_key_head_dim":8,"linear_value_head_dim":2,"linear_conv_kernel_dim":2,"ple_embed_dim":8,"ngram_size":3,"heads_per_ngram":1,"split_ngram_parts":1,"ple_layer_ids":[1],"layer_types":["linear_attention"]}
        ,
    };
    for (cases) |j| {
        const r = parseSlice(gpa, j, &sink.writer);
        if (r) |c_ok| {
            var c_mut = c_ok;
            c_mut.deinit();
            return error.TestUnexpectedResult; // should have been rejected
        } else |_| {}
    }
}
