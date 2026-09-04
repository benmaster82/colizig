//! MemoryManager - budget planning half (Phase 1).
//!
//! Given the architecture, the measured resident-weight size, a RAM budget and a
//! context length, compute where every resident byte goes and pick the largest
//! routed-expert cache that still fits.  If nothing fits, say so - the engine
//! must never silently exceed its budget (brief §8, §17).
//!
//! The runtime tier/state machine (COLD/PREFETCHED/RESIDENT/IN_USE/EVICTABLE,
//! actual eviction, VRAM staging) is declared here but not exercised until the
//! inference kernels land.

const std = @import("std");
const Cfg = @import("../model/config.zig").Cfg;

pub const Tier = enum { ssd, ram, vram };
pub const State = enum { cold, prefetched, resident, in_use, evictable };

pub const Gpu = enum { none, auto };

pub const Profile = enum {
    tiny,
    laptop,
    desktop,
    gpu,

    pub fn ramBudget(self: Profile) u64 {
        const g = 1024 * 1024 * 1024;
        return switch (self) {
            .tiny => 8 * g,
            .laptop => 16 * g,
            .desktop => 32 * g,
            .gpu => 64 * g,
        };
    }

    pub fn vramBudget(self: Profile) u64 {
        const g = 1024 * 1024 * 1024;
        return switch (self) {
            .tiny => 0,
            .laptop => 6 * g,
            .desktop => 12 * g,
            .gpu => 24 * g,
        };
    }

    pub fn fromStr(s: []const u8) ?Profile {
        return std.meta.stringToEnum(Profile, s);
    }
};

pub const Options = struct {
    profile: Profile = .laptop,
    ram_limit: ?u64 = null,
    context: u32 = 8192,
    gpu: Gpu = .none,
};

/// Candidate per-layer expert-cache capacities, largest first.
pub const cap_ladder = [_]u32{ 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4 };

/// Empirical bounded working-set peak, calibrated to colibri's reported
/// ~1.1 GiB (documented in docs/MEMORY_BUDGET.md).
pub const scratch_bytes: u64 = 1152 * 1024 * 1024;

pub const Plan = struct {
    ram_budget: u64,
    vram_budget: u64,
    context: u32,
    layers: u32,

    resident_weights: u64,
    fp8_scale_bank: u64,
    gdn_state: u64,
    ple_state: u64,
    per_token_bytes: u64,
    context_state: u64,
    scratch: u64,

    per_expert_bytes: u64,
    expert_cap: u32,
    expert_cache: u64,

    /// Everything that must stay resident with the chosen cap.
    total_resident: u64,
    /// Resident total independent of the expert cache.
    fixed_resident: u64,
    fits: bool,
    reason: []const u8, // "" when it fits

    pub fn print(self: Plan, w: *std.Io.Writer) !void {
        const h = @import("../util/units.zig").human;
        try w.print(
            \\Qwen3.8 Memory Plan
            \\-------------------
            \\  RAM budget:        {f}
            \\  context:           {d} tokens
            \\  resident weights:  {f}   (native BF16)
            \\  FP8 scale bank:    {f}   (decoded per-expert scales)
            \\  GDN state:         {f}
            \\  PLE conv state:    {f}
            \\  context / KV bank: {f}   ({f} per token x {d})
            \\  scratch (peak):    {f}
            \\  ======== private resident (must fit):  {f}
            \\  expert stream:     {f}   (cap {d}/layer x {f}/expert x {d} layers)
            \\                     reclaimable OS page cache - E4M3 pages fault
            \\                     from the shard mmap on demand, not malloc'd
            \\  -------- recommended RAM (private + hot experts): {f}
            \\
        , .{
            h(self.ram_budget),
            self.context,
            h(self.resident_weights),
            h(self.fp8_scale_bank),
            h(self.gdn_state),
            h(self.ple_state),
            h(self.context_state),
            h(self.per_token_bytes),
            self.context,
            h(self.scratch),
            h(self.fixed_resident),
            h(self.expert_cache),
            self.expert_cap,
            h(self.per_expert_bytes),
            self.layers,
            h(self.total_resident),
        });
        if (!self.fits) {
            try w.print("  !! DOES NOT FIT: {s}\n", .{self.reason});
        }
    }
};

fn ceilDiv(a: u64, b: u64) u64 {
    return (a + b - 1) / b;
}

/// Estimate the BF16-resident dense weight bytes from the config alone (for the
/// metadata-only path, when shard headers are unavailable).  Everything except
/// the routed experts and the n-gram table is resident in BF16 (2 B/elem) - see
/// the checkpoint's `modules_to_not_convert`.
pub fn estimateResidentBytes(cfg: Cfg) u64 {
    const H: u64 = cfg.hidden;
    const W: u64 = cfg.hc_width;
    const R: u64 = cfg.hc_rank;
    const C: u64 = cfg.hc_count;
    const D: u64 = cfg.head_dim;

    const gr = W + 2 * R * W + C * W; // one gated-residual mixer (norm + down + up + inject)
    const moe_common = @as(u64, cfg.experts) * H // router
    + 2 * cfg.shared_inter * H + H * cfg.shared_inter + H; // shared expert + gate

    const qsa_layer = 2 * @as(u64, cfg.q_heads) * D * H // q_proj (gated: ·2)
    + 2 * @as(u64, cfg.kv_heads) * D * H // k, v
    + H * @as(u64, cfg.q_heads) * D // o
    + 2 * D // q/k norm
    + (@as(u64, cfg.idx_qheads) + cfg.idx_kheads) * cfg.idx_dim * H // indexer qk
    + 2 * cfg.idx_dim; // indexer norms

    const V: u64 = @as(u64, cfg.dn_vheads) * cfg.dn_vdim;
    const gdn_layer = @as(u64, cfg.dn_conv_dim) * H // in_proj_qkv
    + V * H // in_proj_z
    + 2 * @as(u64, cfg.dn_vheads) * H // in_proj_a, in_proj_b
    + @as(u64, cfg.dn_conv_dim) * cfg.dn_convk // conv1d
    + 2 * cfg.dn_vheads + cfg.dn_vdim // dt_bias, A_log, norm
    + H * V; // out_proj

    const ple_dense = W * cfg.ple_dim // key_proj
    + H * cfg.ple_dim // value_proj
    + 3 * W // norm_key/query/conv
    + W * cfg.ple_convk; // conv1d

    var elems: u64 = 0;
    elems += 2 * @as(u64, cfg.vocab) * H; // embed_tokens + lm_head
    elems += gr; // final mixer
    elems += @as(u64, cfg.layers) * (2 * gr + moe_common);
    elems += @as(u64, cfg.numAttnLayers()) * qsa_layer;
    elems += @as(u64, cfg.numGdnLayers()) * gdn_layer;
    elems += ple_dense;
    return elems * 2; // BF16
}

/// `resident_weights` is the measured native-dtype size of all resident tensors
/// (from `Manifest.residentBytes()`), or an estimate for planning.
pub fn plan(cfg: Cfg, resident_weights: u64, opts: Options) Plan {
    const layers: u64 = cfg.layers;

    // FP8 block scales for every expert, resident so a miss is a single read.
    const nblk_inter = ceilDiv(cfg.inter, 128);
    const nblk_hidden = ceilDiv(cfg.hidden, 128);
    const fp8_scale_bank = cfg.experts * 3 * nblk_inter * nblk_hidden * 4 * layers;

    // One routed expert's E4M3 bytes (gate + up + down).  These are never
    // malloc'd: the cache borrows the shard mmap and the pages fault in on
    // demand, so this is reclaimable page-cache pressure, not private RAM.
    const per_expert_bytes: u64 =
        2 * @as(u64, cfg.inter) * cfg.hidden + @as(u64, cfg.hidden) * cfg.inter;

    // Gated DeltaNet recurrent + causal-conv state, per GDN layer.
    const gdn_per_layer: u64 =
        @as(u64, cfg.dn_vheads) * cfg.dn_kdim * cfg.dn_vdim * 4 +
        @as(u64, cfg.dn_conv_dim) * (cfg.dn_convk - 1) * 4;
    const gdn_state = gdn_per_layer * cfg.numGdnLayers();

    // PLE causal-conv ring buffer (one PLE layer).
    const ple_state: u64 = @as(u64, cfg.hc_width) * (cfg.ple_convk - 1) * cfg.ngram_size * 4;

    // QSA K/V + indexer key, f32, per attention layer per token.
    const per_token_bytes: u64 =
        @as(u64, cfg.numAttnLayers()) * (2 * @as(u64, cfg.kv_heads) * cfg.head_dim + cfg.idx_dim) * 4;
    const context_state = per_token_bytes * opts.context;

    const ram_budget = opts.ram_limit orelse opts.profile.ramBudget();
    const vram_budget = opts.profile.vramBudget();

    const fixed_resident =
        resident_weights +| fp8_scale_bank +| gdn_state +| ple_state +| context_state +| scratch_bytes;

    var chosen_cap: u32 = 0;
    var expert_cache: u64 = 0;
    var fits = false;
    var reason: []const u8 = "";

    if (fixed_resident > ram_budget) {
        reason = "the private resident set (weights + KV + scratch) exceeds the RAM budget; raise --ram-limit or lower --context";
    } else {
        // The private set fits, so the model runs.  Pick the largest expert-cache
        // cap whose reclaimable stream also stays under budget; if even the
        // smallest doesn't, keep the smallest - the OS page cache just recycles
        // E4M3 pages more aggressively (slower, still correct).
        const min_cap = cap_ladder[cap_ladder.len - 1];
        chosen_cap = min_cap;
        expert_cache = @as(u64, min_cap) * per_expert_bytes * layers;
        for (cap_ladder) |cap| {
            const stream = @as(u64, cap) * per_expert_bytes * layers;
            if (fixed_resident +| stream <= ram_budget) {
                chosen_cap = cap;
                expert_cache = stream;
                break;
            }
        }
        fits = true;
    }

    return .{
        .ram_budget = ram_budget,
        .vram_budget = vram_budget,
        .context = opts.context,
        .layers = cfg.layers,
        .resident_weights = resident_weights,
        .fp8_scale_bank = fp8_scale_bank,
        .gdn_state = gdn_state,
        .ple_state = ple_state,
        .per_token_bytes = per_token_bytes,
        .context_state = context_state,
        .scratch = scratch_bytes,
        .per_expert_bytes = per_expert_bytes,
        .expert_cap = chosen_cap,
        .expert_cache = expert_cache,
        .total_resident = fixed_resident +| expert_cache,
        .fixed_resident = fixed_resident,
        .fits = fits,
        .reason = reason,
    };
}

// ---- tests --------------------------------------------------------------

const config = @import("../model/config.zig");

fn tinyCfg(gpa: std.mem.Allocator) !Cfg {
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);
    return config.parseSlice(gpa, config.tiny_config_json, &sink.writer);
}

test "more context never lowers the resident target" {
    var c = try tinyCfg(std.testing.allocator);
    defer c.deinit();
    const a = plan(c, 1 << 20, .{ .context = 1024, .ram_limit = 1 << 30 });
    const b = plan(c, 1 << 20, .{ .context = 8192, .ram_limit = 1 << 30 });
    try std.testing.expect(b.context_state > a.context_state);
    try std.testing.expect(b.fixed_resident >= a.fixed_resident);
}

test "cap selection shrinks as the budget shrinks" {
    var c = try tinyCfg(std.testing.allocator);
    defer c.deinit();
    const big = plan(c, 1 << 20, .{ .ram_limit = 8 << 30, .context = 2048 });
    const small = plan(c, 1 << 20, .{ .ram_limit = 2 << 30, .context = 2048 });
    try std.testing.expect(big.fits and small.fits);
    try std.testing.expect(big.expert_cap >= small.expert_cap);
}

test "absurd context at a tiny budget fails gracefully" {
    var c = try tinyCfg(std.testing.allocator);
    defer c.deinit();
    const p = plan(c, 4 << 30, .{ .ram_limit = 1 << 30, .context = 262144 });
    try std.testing.expect(!p.fits);
    try std.testing.expect(p.reason.len > 0);
    try std.testing.expectEqual(@as(u32, 0), p.expert_cap);
}
