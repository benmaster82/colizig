//! `selftest` — bring-up diagnostics for the Phase 2 primitives.
//!
//! Runs numeric self-consistency checks on the `ops/` kernels (no model needed),
//! and — if a model directory is given — exercises the weight-materialization
//! path: mmap, embedding lookup, and the LM-head projection.
//!
//! This is NOT an inference run. The transformer layers do not exist yet; this
//! only proves the plumbing and the kernels.

const std = @import("std");
const args = @import("args.zig");
const matmul = @import("../ops/matmul.zig");
const rmsnorm = @import("../ops/rmsnorm.zig");
const rope = @import("../ops/rope.zig");
const softmax = @import("../ops/softmax.zig");
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");
const gdn = @import("../qwen38/gdn.zig");
const moe = @import("../qwen38/moe.zig");
const ple = @import("../qwen38/ple.zig");
const qsa = @import("../qwen38/qsa.zig");
const tok_mod = @import("../qwen38/tokenizer.zig");
const template = @import("../qwen38/chat_template.zig");

const Ctx = struct {
    out: *std.Io.Writer,
    failures: u32 = 0,

    fn check(c: *Ctx, name: []const u8, ok: bool) !void {
        try c.out.print("  [{s}] {s}\n", .{ if (ok) "ok " else "FAIL", name });
        if (!ok) c.failures += 1;
    }
};

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
) !void {
    var c: Ctx = .{ .out = out };

    try out.writeAll("ops kernels:\n");
    try opsChecks(gpa, &c);

    if (opts.model_dir.len != 0) {
        try out.writeAll("weights:\n");
        try weightChecks(gpa, io, &c, opts.model_dir, err);
    } else {
        try out.writeAll("weights: (skipped — no model dir given)\n");
    }

    try out.print("\n{d} check(s) failed\n", .{c.failures});
    if (c.failures != 0) return error.SelftestFailed;
}

fn opsChecks(gpa: std.mem.Allocator, c: *Ctx) !void {
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();

    // matmul: SIMD path vs naive triple loop
    {
        const S = 4;
        const I = 53;
        const O = 11;
        const x = try gpa.alloc(f32, S * I);
        defer gpa.free(x);
        const w = try gpa.alloc(f32, O * I);
        defer gpa.free(w);
        const y1 = try gpa.alloc(f32, S * O);
        defer gpa.free(y1);
        const y2 = try gpa.alloc(f32, S * O);
        defer gpa.free(y2);
        for (x) |*v| v.* = rnd.float(f32) * 2 - 1;
        for (w) |*v| v.* = rnd.float(f32) * 2 - 1;
        matmul.matmul(y1, x, w, S, I, O);
        for (0..S) |s| for (0..O) |o| {
            var acc: f32 = 0;
            for (0..I) |i| acc += x[s * I + i] * w[o * I + i];
            y2[s * O + o] = acc;
        };
        var ok = true;
        for (y1, y2) |a, b| ok = ok and @abs(a - b) <= 1e-3 * (1 + @abs(b));
        try c.check("matmul SIMD == naive", ok);
    }

    // rmsnorm: zero weight, symmetric input → unit-RMS output
    {
        var x = [_]f32{ 3, -3, 3, -3, 3, -3 };
        var o: [6]f32 = undefined;
        rmsnorm.rms0(&o, &x, &[_]f32{ 0, 0, 0, 0, 0, 0 }, 0);
        var ss: f32 = 0;
        for (o) |v| ss += v * v;
        try c.check("rms0 normalizes to unit RMS", @abs(ss / 6 - 1) < 1e-5);
    }

    // rope: position 0 is identity; norm preserved at a real position
    {
        var a = [_]f32{ 0.7, -1.3, 2.1, 0.2 };
        const a0 = a;
        rope.rope(&a, 4, 0, 1e7);
        var id = true;
        for (a, a0) |v, o| id = id and @abs(v - o) < 1e-6;
        rope.rope(&a, 4, 5, 1e7);
        const n0 = a0[0] * a0[0] + a0[2] * a0[2];
        const n1 = a[0] * a[0] + a[2] * a[2];
        try c.check("rope identity@0 and norm-preserving", id and @abs(n0 - n1) < 1e-3);
    }

    // softmax: sums to 1, shift invariant
    {
        var p = [_]f32{ 0.2, 5.0, -3.0, 1.1 };
        var q = [_]f32{ 0.2 + 50, 5.0 + 50, -3.0 + 50, 1.1 + 50 };
        softmax.softmax(&p);
        softmax.softmax(&q);
        var sum: f32 = 0;
        for (p) |v| sum += v;
        var same = true;
        for (p, q) |x, y| same = same and @abs(x - y) < 1e-5;
        try c.check("softmax sums to 1 and is shift invariant", @abs(sum - 1) < 1e-6 and same);
    }
}

fn weightChecks(
    gpa: std.mem.Allocator,
    io: std.Io,
    c: *Ctx,
    dir_path: []const u8,
    err: *std.Io.Writer,
) !void {
    var m = manifest_mod.open(gpa, io, dir_path, err) catch |e| {
        try c.check("open manifest", false);
        return e;
    };
    defer m.deinit();
    try c.check("open manifest", true);

    var w = weights_mod.Weights.open(gpa, io, dir_path, &m, err) catch |e| {
        try c.check("open + map shards", false);
        return e;
    };
    defer w.deinit();
    try c.out.print("  [ok ] shards: {s}\n", .{if (w.usingMmap()) "memory-mapped" else "buffered (mmap unavailable)"});

    const H: usize = m.cfg.hidden;
    const V: usize = m.cfg.vocab;

    const emb = try gpa.alloc(f32, H);
    defer gpa.free(emb);
    w.embed(1, emb) catch {
        try c.check("embed token 1", false);
        return;
    };
    var emb_finite = true;
    var emb_norm: f32 = 0;
    for (emb) |v| {
        emb_finite = emb_finite and std.math.isFinite(v);
        emb_norm += v * v;
    }
    emb_norm = @sqrt(emb_norm);
    try c.check("embed token 1 → finite hidden vector", emb_finite);

    const logits = try gpa.alloc(f32, V);
    defer gpa.free(logits);
    w.lmHead(emb, logits) catch {
        try c.check("lm_head projection", false);
        return;
    };
    var lg_finite = true;
    var argmax: usize = 0;
    for (logits, 0..) |v, i| {
        lg_finite = lg_finite and std.math.isFinite(v);
        if (v > logits[argmax]) argmax = i;
    }
    try c.check("lm_head projection → finite logits", lg_finite);

    // cross-check one logit against a direct materialize + dot
    const loc = w.find("lm_head.weight").?;
    const full = try w.materialize(loc);
    defer gpa.free(full);
    const probe = @min(@as(usize, 5), V - 1);
    var ref: f32 = 0;
    for (0..H) |i| ref += emb[i] * full[probe * H + i];
    try c.check("lm_head streamed == materialized", @abs(ref - logits[probe]) <= 1e-3 * (1 + @abs(ref)));

    try c.out.print(
        "  embed |v|={d:.4}, logits argmax={d}, logit[0]={d:.4}\n",
        .{ emb_norm, argmax, logits[0] },
    );

    try gdnChecks(gpa, c, &m, &w);
    try moeChecks(gpa, c, &m, &w);
    try pleChecks(gpa, c, &m, &w);
    try qsaChecks(gpa, c, &m, &w);
    try tokenizerChecks(gpa, io, c, dir_path, err);
}

fn tokenizerChecks(
    gpa: std.mem.Allocator,
    io: std.Io,
    c: *Ctx,
    dir_path: []const u8,
    err: *std.Io.Writer,
) !void {
    var dir = (if (std.fs.path.isAbsolute(dir_path))
        std.Io.Dir.openDirAbsolute(io, dir_path, .{})
    else
        std.Io.Dir.cwd().openDir(io, dir_path, .{})) catch {
        try c.out.writeAll("  (cannot reopen dir for tokenizer)\n");
        return;
    };
    defer dir.close(io);

    var tk = tok_mod.Tokenizer.load(gpa, io, dir, err) catch {
        try c.out.writeAll("  (no tokenizer.json)\n");
        return;
    };
    defer tk.deinit();
    try c.check("load tokenizer.json", true);

    const text = "hello world";
    const ids = try tk.encode(gpa, text);
    defer gpa.free(ids);
    const back = try tk.decode(gpa, ids, true);
    defer gpa.free(back);
    try c.check("tokenizer encode→decode round-trips", std.mem.eql(u8, text, back));

    const msgs = [_]template.Message{.{ .role = "user", .content = "hi" }};
    const rendered = try template.render(gpa, &msgs, true, true);
    defer gpa.free(rendered);
    const rids = try tk.encode(gpa, rendered);
    defer gpa.free(rids);
    const has_im_start = tk.specialId("<|im_start|>") != null;
    var found = false;
    if (tk.specialId("<|im_start|>")) |s| {
        for (rids) |id| {
            if (id == s) found = true;
        }
    }
    try c.check("ChatML template tokenizes with special tokens", has_im_start and found);

    try c.out.print("  tokenizer: {d} vocab entries; \"{s}\" -> {d} ids\n", .{ tk.vocabSize(), text, ids.len });
}

fn qsaChecks(
    gpa: std.mem.Allocator,
    c: *Ctx,
    m: *const manifest_mod.Manifest,
    w: *const weights_mod.Weights,
) !void {
    var qi: ?u32 = null;
    for (m.cfg.is_attn, 0..) |a, i| {
        if (a) {
            qi = @intCast(i);
            break;
        }
    }
    const layer_idx = qi orelse {
        try c.out.writeAll("  (no QSA layer in this checkpoint)\n");
        return;
    };

    const d = qsa.Dims.of(m.cfg);
    var layer = qsa.Layer.load(gpa, w, layer_idx) catch {
        try c.check("load QSA layer weights", false);
        return;
    };
    defer layer.deinit();
    try c.check("load QSA layer weights", true);

    const T = 8;
    const H = d.hidden;
    var prng = std.Random.DefaultPrng.init(0x5A5A);
    const r = prng.random();
    const x = try gpa.alloc(f32, T * H);
    defer gpa.free(x);
    for (x) |*v| v.* = r.float(f32) * 2 - 1;

    var sc = try qsa.Scratch.init(gpa, d, T, T);
    defer sc.deinit();

    var c1 = try qsa.Cache.init(gpa, d, T);
    defer c1.deinit();
    const o1 = try gpa.alloc(f32, T * H);
    defer gpa.free(o1);
    qsa.forward(&layer, &c1, d, x, T, 0, o1, &sc);
    var finite = true;
    for (o1) |v| finite = finite and std.math.isFinite(v);
    try c.check("QSA forward → finite output", finite);

    // prefill (T) vs decode in chunks (3 + 5) → same output
    var c2 = try qsa.Cache.init(gpa, d, T);
    defer c2.deinit();
    const o2 = try gpa.alloc(f32, T * H);
    defer gpa.free(o2);
    qsa.forward(&layer, &c2, d, x[0 .. 3 * H], 3, 0, o2[0 .. 3 * H], &sc);
    qsa.forward(&layer, &c2, d, x[3 * H ..], 5, 3, o2[3 * H ..], &sc);
    var same = true;
    for (o1, o2) |a, b| same = same and @abs(a - b) <= 1e-3 * (1 + @abs(a));
    try c.check("QSA prefill == chunked decode (KV cache replay)", same);

    try c.out.print("  QSA layer {d}: {d} q / {d} kv heads, head_dim {d}, indexer budget {d}, block {d}; cache {d} B/token\n", .{
        layer_idx,                                     d.q_heads,    d.kv_heads,
        d.head_dim,                                    d.idx_budget, d.idx_ratio,
        (2 * d.kv_heads * d.head_dim + d.idx_dim) * 4,
    });
}

fn pleChecks(
    gpa: std.mem.Allocator,
    c: *Ctx,
    m: *const manifest_mod.Manifest,
    w: *const weights_mod.Weights,
) !void {
    const d = ple.Dims.of(m.cfg);
    var table = ple.Table.load(gpa, w, m.cfg) catch {
        try c.check("load PLE n-gram table", false);
        return;
    };
    defer table.deinit();
    var layer = ple.Layer.load(gpa, w, d.ple_layer) catch {
        try c.check("load PLE layer weights", false);
        return;
    };
    defer layer.deinit();
    try c.check("load PLE table + layer weights", true);

    // every hashed address lands inside its head's slice and the table
    var addr_ok = true;
    for (0..d.ngram_heads) |h| {
        const row = table.hashRow(h, 11, 4, 7);
        addr_ok = addr_ok and row >= table.head_offset[h] and
            row < table.head_offset[h] + table.head_vocab[h] and
            row >= 0 and row < table.totalRows();
    }
    try c.check("PLE hash addresses are in range", addr_ok);

    const T = 6;
    const W = d.hc_width;
    var prng = std.Random.DefaultPrng.init(0xB1A5);
    const r = prng.random();
    const ids = try gpa.alloc(i64, T);
    defer gpa.free(ids);
    for (ids) |*v| v.* = r.intRangeAtMost(i64, 0, @as(i64, @intCast(m.cfg.vocab - 1)));
    const hyper = try gpa.alloc(f32, T * W);
    defer gpa.free(hyper);
    for (hyper) |*v| v.* = r.float(f32) * 2 - 1;

    var sc = try ple.Scratch.init(gpa, d);
    defer sc.deinit();

    var s1 = try ple.State.init(gpa, d);
    defer s1.deinit();
    const o1 = try gpa.alloc(f32, T * W);
    defer gpa.free(o1);
    ple.forward(&layer, table, &s1, d, ids, T, hyper, o1, null, &sc);
    var finite = true;
    for (o1) |v| finite = finite and std.math.isFinite(v);
    try c.check("PLE forward → finite output", finite);

    var s2 = try ple.State.init(gpa, d);
    defer s2.deinit();
    const pf = try ple.prefetchRows(gpa, table, d, ids, T, s2);
    defer gpa.free(pf);
    const o2 = try gpa.alloc(f32, T * W);
    defer gpa.free(o2);
    ple.forward(&layer, table, &s2, d, ids, T, hyper, o2, pf, &sc);
    var pf_same = true;
    for (o1, o2) |a, b| pf_same = pf_same and a == b;
    try c.check("PLE prefetched reads == inline reads (bit-identical)", pf_same);

    var s3 = try ple.State.init(gpa, d);
    defer s3.deinit();
    const o3 = try gpa.alloc(f32, T * W);
    defer gpa.free(o3);
    ple.forward(&layer, table, &s3, d, ids[0..3], 3, hyper[0 .. 3 * W], o3[0 .. 3 * W], null, &sc);
    ple.forward(&layer, table, &s3, d, ids[3..], 3, hyper[3 * W ..], o3[3 * W ..], null, &sc);
    var chunk_same = true;
    for (o1, o3) |a, b| chunk_same = chunk_same and @abs(a - b) <= 1e-4 * (1 + @abs(a));
    try c.check("PLE chunk boundaries don't change results", chunk_same);

    try c.out.print(
        "  PLE layer {d}: {d} table shards, {d} rows, {d} reads x {d} B/token, scale {d:.4}\n",
        .{ d.ple_layer, table.shards.len, table.totalRows(), d.ngram_heads, d.ngram_head_dim, table.weight_scale },
    );
}

fn moeChecks(
    gpa: std.mem.Allocator,
    c: *Ctx,
    m: *const manifest_mod.Manifest,
    w: *const weights_mod.Weights,
) !void {
    const d = moe.Dims.of(m.cfg);
    var layer = moe.Layer.load(gpa, w, 0, d) catch {
        try c.check("load MoE layer weights", false);
        return;
    };
    defer layer.deinit();
    try c.check("load MoE layer weights", true);

    const T = 5;
    var sc = try moe.Scratch.init(gpa, d, T);
    defer sc.deinit();

    const H = d.hidden;
    var prng = std.Random.DefaultPrng.init(0x3ED1);
    const r = prng.random();
    const x = try gpa.alloc(f32, T * H);
    defer gpa.free(x);
    for (x) |*v| v.* = r.float(f32) * 2 - 1;

    const o1 = try gpa.alloc(f32, T * H);
    defer gpa.free(o1);
    const o2 = try gpa.alloc(f32, T * H);
    defer gpa.free(o2);

    var full = try moe.ExpertCache.init(gpa, 0, d.experts);
    defer full.deinit();
    moe.forward(&layer, &full, w, d, x, T, o1, &sc, null, null) catch {
        try c.check("MoE forward", false);
        return;
    };
    var finite = true;
    for (o1) |v| finite = finite and std.math.isFinite(v);
    try c.check("MoE forward → finite output", finite);
    try c.check("MoE fetches each distinct routed expert once", full.stats.hits + full.stats.misses >= 1 and full.stats.hits + full.stats.misses <= d.experts);

    // determinism with a fresh cache
    var full2 = try moe.ExpertCache.init(gpa, 0, d.experts);
    defer full2.deinit();
    try moe.forward(&layer, &full2, w, d, x, T, o2, &sc, null, null);
    var same = true;
    for (o1, o2) |a, b| same = same and @abs(a - b) <= 1e-4 * (1 + @abs(a));
    try c.check("MoE deterministic across caches", same);

    // replay is all hits
    const misses = full.stats.misses;
    try moe.forward(&layer, &full, w, d, x, T, o2, &sc, null, null);
    try c.check("warm expert cache: replay is all hits", full.stats.misses == misses and full.stats.evictions == 0);

    // tight cache thrashes
    const cap: usize = @max(@as(usize, 1), d.topk - 1);
    var tight = try moe.ExpertCache.init(gpa, 0, cap);
    defer tight.deinit();
    try moe.forward(&layer, &tight, w, d, x, T, o2, &sc, null, null);
    try c.check("tight expert cache evicts under pressure", tight.stats.evictions > 0 and tight.residentCount() <= cap);

    try c.out.print(
        "  MoE layer 0: {d} experts, top-{d} + shared; full-cache resident {d} B, hit rate {d:.0}%\n",
        .{
            d.experts,
            d.topk,
            full.stats.bytes_resident,
            100.0 * @as(f64, @floatFromInt(full.stats.hits)) /
                @as(f64, @floatFromInt(@max(@as(u64, 1), full.stats.hits + full.stats.misses))),
        },
    );
}

fn gdnChecks(
    gpa: std.mem.Allocator,
    c: *Ctx,
    m: *const manifest_mod.Manifest,
    w: *const weights_mod.Weights,
) !void {
    // first Gated DeltaNet (non-attention) layer
    var gi: ?u32 = null;
    for (m.cfg.is_attn, 0..) |a, i| {
        if (!a) {
            gi = @intCast(i);
            break;
        }
    }
    const layer_idx = gi orelse {
        try c.out.writeAll("  (no GDN layer in this checkpoint)\n");
        return;
    };

    const d = gdn.Dims.of(m.cfg);
    var layer = gdn.Layer.load(gpa, w, layer_idx) catch {
        try c.check("load GDN layer weights", false);
        return;
    };
    defer layer.deinit();
    try c.check("load GDN layer weights", true);

    const T = 6;
    const H = d.hidden;
    var prng = std.Random.DefaultPrng.init(0xDE17A);
    const r = prng.random();
    const x = try gpa.alloc(f32, T * H);
    defer gpa.free(x);
    for (x) |*v| v.* = r.float(f32) * 2 - 1;

    var st = try gdn.GdnState.init(gpa, d);
    defer st.deinit();
    var sc = try gdn.Scratch.init(gpa, d, T);
    defer sc.deinit();

    const whole = try gpa.alloc(f32, T * H);
    defer gpa.free(whole);
    gdn.forward(&layer, &st, d, x, T, whole, &sc);
    var finite = true;
    for (whole) |v| finite = finite and std.math.isFinite(v);
    try c.check("GDN forward → finite output", finite);

    // chunk-boundary invariance
    var st2 = try gdn.GdnState.init(gpa, d);
    defer st2.deinit();
    const part = try gpa.alloc(f32, T * H);
    defer gpa.free(part);
    gdn.forward(&layer, &st2, d, x[0 .. 3 * H], 3, part[0 .. 3 * H], &sc);
    gdn.forward(&layer, &st2, d, x[3 * H ..], 3, part[3 * H ..], &sc);
    var same = true;
    for (whole, part) |a, b| same = same and @abs(a - b) <= 1e-3 * (1 + @abs(a));
    try c.check("GDN chunk boundaries don't change results", same);

    // zero input from a fresh state → zero output
    var st3 = try gdn.GdnState.init(gpa, d);
    defer st3.deinit();
    const zin = try gpa.alloc(f32, 2 * H);
    defer gpa.free(zin);
    @memset(zin, 0);
    const zout = try gpa.alloc(f32, 2 * H);
    defer gpa.free(zout);
    gdn.forward(&layer, &st3, d, zin, 2, zout, &sc);
    var zero = true;
    for (zout) |v| zero = zero and @abs(v) < 1e-5;
    try c.check("GDN zero-input → zero-output", zero);

    try c.out.print("  GDN layer {d}: {d} key / {d} value heads, state {d} KiB/layer\n", .{
        layer_idx, d.kheads, d.vheads, (st.rec.len + st.ring.len) * 4 / 1024,
    });
}
