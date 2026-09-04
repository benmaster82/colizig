//! `chat` for `model_type: qwen3_moe`. Dispatched from chat.zig. A plain
//! streaming REPL (no ColiZig styler box): ChatML framing, per-token detok
//! deltas, per-reply tok/s. `--prompt` = one-shot.

const std = @import("std");
const args = @import("args.zig");
const units = @import("../util/units.zig");
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");
const q3 = @import("../qwen3moe/model.zig");
const tok_mod = @import("../qwen38/tokenizer.zig");
const tmpl = @import("../qwen38/chat_template.zig");
const parallel = @import("../runtime/parallel.zig");
const usage_mod = @import("../runtime/expert_usage.zig");
const sampler_mod = @import("../runtime/sampler.zig");
const gpu = @import("../backend/gpu.zig");

const Timestamp = std.Io.Timestamp;

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
) !void {
    parallel.enable(io, opts.threads);
    defer parallel.disable();
    if (opts.cuda) gpu.init(err);
    if (opts.cuda) gpu.setVramBudget(if (opts.vram != 0) opts.vram else 16 << 30);
    defer gpu.deinit();

    var dir = openDirAny(io, opts.model_dir) catch {
        try err.print("chat: cannot open model directory \"{s}\"\n", .{opts.model_dir});
        return error.OpenFailed;
    };
    defer dir.close(io);

    var m = try manifest_mod.open(gpa, io, opts.model_dir, err);
    defer m.deinit();
    var w = try weights_mod.Weights.open(gpa, io, opts.model_dir, &m, err);
    defer w.deinit();
    var tk = try tok_mod.Tokenizer.load(gpa, io, dir, err);
    defer tk.deinit();

    const interactive = opts.prompt.len == 0;
    const max_new: usize = if (opts.steps != 0) opts.steps else if (interactive) 512 else 128;
    const ctx: usize = if (opts.budget.context != 0) opts.budget.context else 8192;
    const cap: usize = if (opts.expert_cap != 0) opts.expert_cap else @min(128, m.cfg.experts);

    try out.print("  loading Qwen3-MoE: {d} layers, {d} experts, expert cap {d}/layer ...\n", .{ m.cfg.layers, m.cfg.experts, cap });
    try out.flush();

    var model = try q3.Model.load(gpa, &w);
    defer model.deinit();
    var state = try q3.State.init(gpa, &model, ctx, cap, io);
    defer state.deinit();
    var sc = try q3.Scratch.init(gpa, &model, @min(ctx, 512), ctx);
    defer sc.deinit();

    const ul: u32 = @intCast(m.cfg.layers);
    const ue: u32 = @intCast(m.cfg.experts);
    var usage: ?q3.ExpertUsage = if (opts.no_usage) null else (usage_mod.ExpertUsage.load(gpa, io, dir, ul, ue) orelse (q3.ExpertUsage.init(gpa, ul, ue) catch null));
    defer if (usage) |*u| u.deinit();
    defer if (usage) |*u| u.save(io, dir);
    if (usage) |*u| {
        const per_expert = 3 * @as(u64, m.cfg.inter) * m.cfg.hidden + m.cfg.hidden * m.cfg.inter;
        usage_mod.warmCaches(u, state.experts, &w, model.moe_dims, @as(u64, cap) * per_expert * m.cfg.layers);
    }

    var sampler = try sampler_mod.Sampler.init(gpa, io, m.cfg.vocab, .{
        .temperature = opts.temperature,
        .top_k = opts.top_k,
        .top_p = opts.top_p,
        .seed = opts.seed,
    });
    defer sampler.deinit();

    const logits = try gpa.alloc(f32, m.cfg.vocab);
    defer gpa.free(logits);

    var history: std.ArrayList(tmpl.Message) = .empty;
    defer {
        for (history.items) |msg| gpa.free(msg.content);
        history.deinit(gpa);
    }
    if (opts.system.len != 0)
        try history.append(gpa, .{ .role = "system", .content = try gpa.dupe(u8, opts.system) });

    var think = true;

    var line_buf: [8192]u8 = undefined;
    var in = std.Io.File.stdin().readerStreaming(io, &line_buf);

    while (true) {
        var user_text: []const u8 = undefined;
        if (interactive) {
            try out.writeAll("\n\x1b[38;2;247;164;29m>\x1b[0m ");
            try out.flush();
            const raw = in.interface.takeDelimiter('\n') catch break orelse break;
            const line = std.mem.trim(u8, if (std.mem.startsWith(u8, raw, "\xEF\xBB\xBF")) raw[3..] else raw, " \r\t");
            if (line.len == 0) continue;
            if (std.mem.eql(u8, line, "/exit") or std.mem.eql(u8, line, "/quit")) break;
            if (std.mem.eql(u8, line, "/reset")) {
                for (history.items) |msg| gpa.free(msg.content);
                history.clearRetainingCapacity();
                if (opts.system.len != 0) try history.append(gpa, .{ .role = "system", .content = try gpa.dupe(u8, opts.system) });
                state.reset();
                try out.writeAll("  (context cleared)\n");
                continue;
            }
            if (std.mem.eql(u8, line, "/think")) {
                think = !think;
                try out.print("  (thinking {s})\n", .{if (think) "on" else "off"});
                continue;
            }
            user_text = line;
        } else {
            user_text = opts.prompt;
        }

        try history.append(gpa, .{ .role = "user", .content = try gpa.dupe(u8, user_text) });
        const prompt_text = try tmpl.render(gpa, history.items, true, think);
        defer gpa.free(prompt_text);

        const ids = try tk.encode(gpa, prompt_text);
        defer gpa.free(ids);
        const prompt_i64 = try gpa.alloc(i64, ids.len);
        defer gpa.free(prompt_i64);
        for (ids, prompt_i64) |s, *d| d.* = s;

        if (state.pos + ids.len + max_new > ctx) {
            try out.writeAll("  (context full — /reset)\n");
            _ = history.pop();
            if (!interactive) break;
            continue;
        }

        const t0 = Timestamp.now(io, .awake);
        try q3.forward(&model, &state, &sc, prompt_i64, logits, .{ .io = io, .usage = if (usage) |*u| u else null });
        const prefill_ns = ns(io, t0);

        var reply: std.ArrayList(u32) = .empty;
        defer reply.deinit(gpa);
        var printed: usize = 0;
        const td0 = Timestamp.now(io, .awake);
        var one: [1]i64 = undefined;
        try out.writeAll("\n");
        while (reply.items.len < max_new) {
            const next = sampler.pick(logits);
            if (@as(i64, @intCast(next)) == m.cfg.eos_id) break;
            try reply.append(gpa, @intCast(next));
            const full = try tk.decode(gpa, reply.items, true);
            defer gpa.free(full);
            if (full.len > printed) {
                try out.writeAll(full[printed..]);
                try out.flush();
                printed = full.len;
            }
            one[0] = @intCast(next);
            try q3.forward(&model, &state, &sc, &one, logits, .{ .io = io, .usage = if (usage) |*u| u else null });
        }
        const decode_ns = ns(io, td0);

        const reply_text = try tk.decode(gpa, reply.items, true);
        try history.append(gpa, .{ .role = "assistant", .content = reply_text });

        const tps: f64 = if (decode_ns == 0) 0 else @as(f64, @floatFromInt(reply.items.len)) * 1e9 / @as(f64, @floatFromInt(decode_ns));
        try out.print("\n\x1b[2m[{d} tok · prefill {d} tok/{d:.1}s · {d:.2} tok/s]\x1b[0m\n", .{
            reply.items.len, ids.len, @as(f64, @floatFromInt(prefill_ns)) / 1e9, tps,
        });
        try out.flush();

        if (!interactive) break;
    }
    gpu.statsLine(out);
    try out.flush();
}

fn ns(io: std.Io, from: Timestamp) u64 {
    const d = from.durationTo(Timestamp.now(io, .awake)).nanoseconds;
    return if (d > 0) @intCast(d) else 0;
}

fn openDirAny(io: std.Io, path: []const u8) !std.Io.Dir {
    return if (std.fs.path.isAbsolute(path))
        std.Io.Dir.openDirAbsolute(io, path, .{})
    else
        std.Io.Dir.cwd().openDir(io, path, .{});
}
