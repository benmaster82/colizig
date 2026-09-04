//! `chat` — ColiZig: ChatML chat over the real tokenizer + forward.
//!
//!   * `--prompt "..."`  → one-shot: render, prefill, stream the reply, print.
//!   * no `--prompt`     → interactive multi-turn REPL, colibri-style: a header
//!     with the ColiZig logo, then per turn a greyed "thinking" box and the
//!     answer in Qwen's colour, each token streamed as it is produced and a
//!     `tok/s` line per reply.  The KV / DeltaNet / PLE state and the warm
//!     expert cache carry across turns; `/reset` clears the conversation,
//!     `/think` toggles reasoning, `/exit` quits.

const std = @import("std");
const args = @import("args.zig");
const units = @import("../util/units.zig");
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");
const budget = @import("../runtime/budget.zig");
const model_mod = @import("../qwen38/model.zig");
const tok_mod = @import("../qwen38/tokenizer.zig");
const template = @import("../qwen38/chat_template.zig");
const parallel = @import("../runtime/parallel.zig");
const gpu = @import("../backend/gpu.zig");
const usage_mod = @import("../runtime/expert_usage.zig");
const sampler_mod = @import("../runtime/sampler.zig");

const Timestamp = std.Io.Timestamp;

/// Palette: Qwen's violet for the assistant, Zig's amber for the user, grey for
/// reasoning.  Truecolour SGR — every modern terminal handles it.
const C = struct {
    const rst = "\x1b[0m";
    const b = "\x1b[1m";
    const dim = "\x1b[2m";
    const qwen = "\x1b[38;2;133;98;245m"; // Qwen violet
    const zig = "\x1b[38;2;247;164;29m"; // Zig amber (#F7A41D)
    const gray = "\x1b[38;2;143;143;156m";
    const dgray = "\x1b[38;2;95;95;108m";
};

// ---- colibri's hummingbird sprite, recoloured ------------------------------
// Same 15×10 pixel grid and half-block renderer as colibri's `sprite_lines`;
// the palette is remapped to Qwen violet (crest/body) + Zig amber (wing/tail/
// beak).  `.` is transparent.
const sprite = [_][]const u8{
    "....MMM.........",
    "...MMMMM..w.....",
    "....MMMM.ww.....",
    "OOOOTTeTCC......",
    "....TTTTTCC.....",
    ".....TTTTCC.....",
    "......TTCC......",
    ".......TC.......",
    "........C.......",
    "................",
};

fn pixel(ch: u8) ?[3]u8 {
    return switch (ch) {
        'M' => .{ 173, 150, 255 }, // crest  — light Qwen violet
        'T' => .{ 120, 90, 235 }, //  body   — Qwen violet
        'C', 'w' => .{ 247, 164, 29 }, // wing/tail — Zig amber
        'O' => .{ 255, 190, 70 }, //  beak   — bright amber
        'e' => .{ 235, 235, 245 }, // eye    — near-white
        else => null,
    };
}

fn spriteLine(out: *std.Io.Writer, pair: usize) !void {
    const top = sprite[pair * 2];
    const bot = if (pair * 2 + 1 < sprite.len) sprite[pair * 2 + 1] else "               ";
    var x: usize = 0;
    while (x < top.len) : (x += 1) {
        const ct = pixel(top[x]);
        const cb = pixel(bot[x]);
        if (ct == null and cb == null) {
            try out.writeAll("\x1b[0m ");
        } else if (cb == null) {
            const c = ct.?;
            try out.print("\x1b[38;2;{d};{d};{d}m\x1b[49m▀", .{ c[0], c[1], c[2] });
        } else if (ct == null) {
            const c = cb.?;
            try out.print("\x1b[38;2;{d};{d};{d}m\x1b[49m▄", .{ c[0], c[1], c[2] });
        } else {
            const f = ct.?;
            const g = cb.?;
            try out.print("\x1b[38;2;{d};{d};{d}m\x1b[48;2;{d};{d};{d}m▀", .{ f[0], f[1], f[2], g[0], g[1], g[2] });
        }
    }
    try out.writeAll("\x1b[0m");
}

/// Streams the decoded reply with reasoning greyed inside a box and the answer
/// in Qwen's colour.  Fed the growing full text; prints only what is newly safe
/// (a `</think>` marker is never split across a flush).
const Styler = struct {
    out: *std.Io.Writer,
    printed: usize = 0,
    in_think: bool,
    think_open: bool = false,
    answer_open: bool = false,

    fn init(out: *std.Io.Writer, thinking: bool) Styler {
        return .{ .out = out, .in_think = thinking };
    }

    fn feed(self: *Styler, full: []const u8, final: bool) !void {
        const close = "</think>";
        const hold: usize = if (final) 0 else close.len;
        if (full.len < self.printed + hold) return;
        const end = full.len - hold;

        var i = self.printed;
        while (i < end) {
            if (self.in_think) {
                if (std.mem.startsWith(u8, full[i..], close)) {
                    if (self.think_open) try self.out.writeAll("\n  " ++ C.gray ++ "└─" ++ C.rst ++ "\n");
                    self.in_think = false;
                    i += close.len;
                    while (i < end and (full[i] == ' ' or full[i] == '\n' or full[i] == '\r')) i += 1;
                    continue;
                }
                const ch = full[i];
                if (!self.think_open) {
                    if (ch == ' ' or ch == '\n' or ch == '\r') {
                        i += 1;
                        continue;
                    }
                    try self.out.writeAll("\n  " ++ C.gray ++ "┌ thinking\n  │ ");
                    self.think_open = true;
                }
                if (ch == '\n') {
                    try self.out.writeAll("\n  " ++ C.gray ++ "│ ");
                } else {
                    try self.out.writeByte(ch);
                }
                i += 1;
            } else {
                if (!self.answer_open) {
                    const ch = full[i];
                    if (ch == ' ' or ch == '\n' or ch == '\r') {
                        i += 1;
                        continue;
                    }
                    try self.out.writeAll("\n  " ++ C.b ++ C.qwen ++ "◆ " ++ "Coli" ++ C.zig ++ "Zig" ++ C.rst ++ "\n  " ++ C.qwen);
                    self.answer_open = true;
                }
                try self.out.writeByte(full[i]);
                i += 1;
            }
        }
        self.printed = end;
        try self.out.flush();
    }

    fn finish(self: *Styler) !void {
        if (self.in_think and self.think_open) try self.out.writeAll(C.dim ++ " …" ++ C.rst ++ "\n  " ++ C.gray ++ "└─");
        try self.out.writeAll(C.rst);
        try self.out.flush();
    }
};

/// Everything the decode loop needs, assembled once and reused every turn.
const Session = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    tk: *tok_mod.Tokenizer,
    model: *model_mod.Model,
    state: *model_mod.State,
    sc: *model_mod.Scratch,
    fopts: model_mod.Opts,
    sampler: *sampler_mod.Sampler,
    logits: []f32,
    vocab: u32,
    context: u32,
    prefill_chunk: usize,
    eos_id: i64,
    im_end_id: i64,
    max_new: usize,

    fn prefill(s: *Session, ids: []const i64) !void {
        var off: usize = 0;
        while (off < ids.len) {
            const n = @min(s.prefill_chunk, ids.len - off);
            try model_mod.forward(s.model, s.state, s.sc, ids[off .. off + n], s.logits, s.fopts);
            off += n;
        }
    }

    /// Decode up to `max_new` tokens (greedy or sampled), streaming through `styler`.
    fn generate(s: *Session, gen: *std.ArrayList(u32), styler: *Styler) !struct { n: usize, secs: f64 } {
        gen.clearRetainingCapacity();
        var one: [1]i64 = undefined;
        const t0 = Timestamp.now(s.io, .awake);
        var n: usize = 0;
        while (n < s.max_new) : (n += 1) {
            const best = s.sampler.pick(s.logits);
            const bi: i64 = @intCast(best);
            if (bi == s.im_end_id or bi == s.eos_id) break;
            if (best >= s.vocab) break;
            try gen.append(s.gpa, @intCast(best));

            const full = try s.tk.decode(s.gpa, gen.items, false);
            defer s.gpa.free(full);
            try styler.feed(full, false);

            one[0] = bi;
            if (s.state.pos + 1 > s.context) break;
            try model_mod.forward(s.model, s.state, s.sc, one[0..1], s.logits, s.fopts);
        }
        const final = try s.tk.decode(s.gpa, gen.items, false);
        defer s.gpa.free(final);
        try styler.feed(final, true);
        try styler.finish();
        const dt = t0.durationTo(Timestamp.now(s.io, .awake)).nanoseconds;
        return .{ .n = n, .secs = @as(f64, @floatFromInt(dt)) / 1e9 };
    }
};

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
    if (opts.mirror.len != 0) {
        const nm = w.attachMirror(gpa, io, opts.mirror);
        try out.print("  " ++ C.dim ++ "mirror: {d}/{d} shards from {s}" ++ C.rst ++ "\n", .{ nm, w.shards.len, opts.mirror });
        try out.flush();
    }
    var tk = try tok_mod.Tokenizer.load(gpa, io, dir, err);
    defer tk.deinit();

    if (tk.vocabSize() > m.cfg.vocab) {
        try err.print("chat: tokenizer vocab {d} exceeds model vocab {d}\n", .{ tk.vocabSize(), m.cfg.vocab });
        return error.BadShape;
    }

    const interactive = opts.prompt.len == 0;
    const max_new: usize = if (opts.steps != 0) opts.steps else if (interactive) 512 else 64;

    var bopts = opts.budget;
    if (!interactive and bopts.context < opts.prompt.len / 2 + max_new) {
        bopts.context = @intCast(opts.prompt.len / 2 + max_new + 64);
    }
    const plan = budget.plan(m.cfg, m.residentBytes(), bopts);
    if (!plan.fits) {
        try err.print("chat: {s}\n", .{plan.reason});
        return error.ContextDoesNotFit;
    }
    const cap: usize = if (opts.expert_cap != 0) opts.expert_cap else plan.expert_cap;
    const prefill_chunk: usize = @min(@as(usize, plan.context), 512);

    try out.print("  " ++ C.dim ++ "loading {d} layers, {f} resident ..." ++ C.rst ++ "\n", .{
        m.cfg.layers, units.human(m.residentBytes()),
    });
    try out.flush();

    var model = try model_mod.Model.load(gpa, &w);
    defer model.deinit();

    var state = try model_mod.State.init(gpa, &model, plan.context, cap, io);
    defer state.deinit();
    var sc = try model_mod.Scratch.init(gpa, &model, prefill_chunk, plan.context);
    defer sc.deinit();
    var sched = model_mod.Scheduler.init(gpa, 4096);
    defer sched.deinit();
    var predictor = try model_mod.Predictor.init(gpa, m.cfg.layers, m.cfg.topk);
    defer predictor.deinit();

    // learned expert priors — load, warm the caches, save (merged) on exit
    const ul: u32 = @intCast(m.cfg.layers);
    const ue: u32 = @intCast(m.cfg.experts);
    var usage: ?model_mod.ExpertUsage = if (opts.no_usage) null else (usage_mod.ExpertUsage.load(gpa, io, dir, ul, ue) orelse (model_mod.ExpertUsage.init(gpa, ul, ue) catch null));
    defer if (usage) |*u| u.deinit();
    defer if (usage) |*u| u.save(io, dir);
    if (usage) |*u| {
        const stream = @as(u64, cap) * plan.per_expert_bytes * m.cfg.layers;
        const headroom = plan.ram_budget -| plan.fixed_resident;
        try out.print("  " ++ C.dim ++ "warming expert cache from learned priors ..." ++ C.rst ++ "\n", .{});
        try out.flush();
        usage_mod.warmCaches(u, state.experts, &w, model.moe_dims, @min(stream, headroom));
    }

    const logits = try gpa.alloc(f32, m.cfg.vocab);
    defer gpa.free(logits);

    var sampler = try sampler_mod.Sampler.init(gpa, io, m.cfg.vocab, .{
        .temperature = opts.temperature,
        .top_k = opts.top_k,
        .top_p = opts.top_p,
        .seed = opts.seed,
    });
    defer sampler.deinit();
    if (!sampler.greedy())
        try out.print("  " ++ C.dim ++ "sampling: temp {d:.2} · top-k {d} · top-p {d:.2} · seed {d}" ++ C.rst ++ "\n", .{ opts.temperature, opts.top_k, opts.top_p, sampler.seed_used });

    var sess: Session = .{
        .gpa = gpa,
        .io = io,
        .out = out,
        .tk = &tk,
        .model = &model,
        .state = &state,
        .sc = &sc,
        .fopts = .{ .io = io, .scheduler = &sched, .predictor = &predictor, .usage = if (usage) |*u| u else null },
        .sampler = &sampler,
        .logits = logits,
        .vocab = m.cfg.vocab,
        .context = plan.context,
        .prefill_chunk = prefill_chunk,
        .eos_id = m.cfg.eos_id,
        .im_end_id = if (tk.specialId("<|im_end|>")) |x| @intCast(x) else -1,
        .max_new = max_new,
    };

    if (!interactive) return oneShot(&sess, gpa, out, err, opts);
    return repl(&sess, gpa, io, out, err, opts, &m, plan);
}

fn banner(out: *std.Io.Writer, io: std.Io, m: *const manifest_mod.Manifest, ctx: u32) !void {
    const tty = std.Io.File.stdout().isTty(io) catch true;
    const ascii = [_][]const u8{ "  (\\   ", "   )·>  ", "  / \\   ", "        ", "        " };
    const disk = m.total_size orelse 0;

    try out.writeByte('\n');
    var pair: usize = 0;
    while (pair < 5) : (pair += 1) {
        try out.writeAll("  ");
        if (tty) try spriteLine(out, pair) else try out.writeAll(ascii[pair]);
        try out.writeAll("   ");
        switch (pair) {
            0 => try out.writeAll(C.b ++ C.qwen ++ "Coli" ++ C.zig ++ "Zig" ++ C.rst ++ "  " ++ C.dim ++ "colizig" ++ C.rst),
            1 => try out.writeAll(C.dim ++ "tiny engine, immense model" ++ C.rst),
            2 => if (disk != 0)
                try out.print(C.gray ++ "Qwen3.8-Flash-Next · 176B · {f} on disk" ++ C.rst, .{units.human(disk)})
            else
                try out.writeAll(C.gray ++ "Qwen3.8-Flash-Next · 176B" ++ C.rst),
            3 => try out.print(C.dgray ++ "chat · {d} layers · {f} resident · ctx {d}" ++ C.rst, .{ m.cfg.layers, units.human(m.residentBytes()), ctx }),
            else => {},
        }
        try out.writeByte('\n');
    }
    try out.writeAll("  " ++ C.dgray);
    var i: usize = 0;
    while (i < 58) : (i += 1) try out.writeAll("─");
    try out.writeAll(C.rst ++ "\n\n  " ++
        C.dim ++ "message + Enter  ·  " ++ C.rst ++ C.gray ++ "/think" ++ C.dim ++ " reasoning  ·  " ++
        C.rst ++ C.gray ++ "/reset" ++ C.dim ++ " new  ·  " ++ C.rst ++ C.gray ++ "/exit" ++ C.dim ++ " quit" ++ C.rst ++ "\n");
    try out.flush();
}

fn footer(out: *std.Io.Writer, ntok: usize, prompt_tok: usize, pf_ms: f64, secs: f64) !void {
    const tps = if (secs > 0) @as(f64, @floatFromInt(ntok)) / secs else 0;
    try out.print("\n\n  " ++ C.gray ++ C.dim ++ "{d} tok · prefill {d}/{d:.0}ms · {d:.1}s · {d:.2} tok/s" ++ C.rst ++ "\n", .{ ntok, prompt_tok, pf_ms, secs, tps });
    try out.flush();
}

fn oneShot(
    sess: *Session,
    gpa: std.mem.Allocator,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
) !void {
    var msgs: std.ArrayList(template.Message) = .empty;
    defer msgs.deinit(gpa);
    if (opts.system.len != 0) try msgs.append(gpa, .{ .role = "system", .content = opts.system });
    try msgs.append(gpa, .{ .role = "user", .content = opts.prompt });
    const prompt_text = try template.render(gpa, msgs.items, true, true);
    defer gpa.free(prompt_text);

    const ids = try encodeToI64(sess, gpa, prompt_text, err);
    defer gpa.free(ids);

    try out.print("\n  " ++ C.b ++ C.zig ++ "▸ you" ++ C.rst ++ "\n  " ++ C.zig ++ "{s}" ++ C.rst ++ "\n", .{opts.prompt});
    try out.flush();

    const pf0 = Timestamp.now(sess.io, .awake);
    try sess.prefill(ids);
    const pf_ms = @as(f64, @floatFromInt(pf0.durationTo(Timestamp.now(sess.io, .awake)).nanoseconds)) / 1e6;

    var gen: std.ArrayList(u32) = .empty;
    defer gen.deinit(gpa);
    var styler = Styler.init(out, true);
    const r = try sess.generate(&gen, &styler);
    try footer(out, gen.items.len, ids.len, pf_ms, r.secs);
}

fn repl(
    sess: *Session,
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
    m: *const manifest_mod.Manifest,
    plan: budget.Plan,
) !void {
    var in_buf: [8192]u8 = undefined;
    var stdin_r = std.Io.File.stdin().readerStreaming(io, &in_buf);
    const stdin = &stdin_r.interface;

    var gen: std.ArrayList(u32) = .empty;
    defer gen.deinit(gpa);
    var frag: std.ArrayList(u8) = .empty;
    defer frag.deinit(gpa);

    try banner(out, io, m, plan.context);

    var turn: usize = 0;
    var thinking = true;
    while (true) {
        try out.writeAll("\n  " ++ C.b ++ C.zig ++ "▸ you" ++ C.rst ++ "  " ++ C.zig);
        try out.flush();

        const raw = (stdin.takeDelimiter('\n') catch |e| switch (e) {
            error.StreamTooLong => {
                try out.writeAll(C.rst ++ "  " ++ C.dim ++ "(line too long, ignored)" ++ C.rst ++ "\n");
                continue;
            },
            error.ReadFailed => break,
        }) orelse break;
        try out.writeAll(C.rst);
        var line = std.mem.trim(u8, raw, " \t\r\n");
        if (std.mem.startsWith(u8, line, "\xEF\xBB\xBF")) line = line[3..]; // stray UTF-8 BOM
        if (line.len == 0) continue;
        if (std.mem.eql(u8, line, "/exit") or std.mem.eql(u8, line, "/quit")) break;
        if (std.mem.eql(u8, line, "/reset")) {
            sess.state.reset();
            if (sess.fopts.predictor) |p| p.reset();
            turn = 0;
            try out.writeAll("  " ++ C.dim ++ "(conversation cleared)" ++ C.rst ++ "\n");
            continue;
        }
        if (std.mem.eql(u8, line, "/think")) {
            thinking = !thinking;
            try out.print("  " ++ C.dim ++ "(reasoning {s})" ++ C.rst ++ "\n", .{if (thinking) "on" else "off"});
            continue;
        }

        // ChatML fragment for this turn.  The previous assistant turn is in the
        // KV cache but its closing <|im_end|> is not — emit it here first.
        frag.clearRetainingCapacity();
        if (turn == 0) {
            if (opts.system.len != 0)
                try frag.print(gpa, "<|im_start|>system\n{s}<|im_end|>\n", .{opts.system});
        } else {
            try frag.appendSlice(gpa, "<|im_end|>\n");
        }
        try frag.print(gpa, "<|im_start|>user\n{s}<|im_end|>\n{s}", .{ line, template.assistantOpen(thinking) });

        const ids = encodeToI64(sess, gpa, frag.items, err) catch |e| switch (e) {
            error.TokenOutOfVocab => {
                try out.writeAll("  " ++ C.dim ++ "(input produced an out-of-vocab token, turn skipped)" ++ C.rst ++ "\n");
                continue;
            },
            else => return e,
        };
        defer gpa.free(ids);

        if (sess.state.pos + ids.len + 1 > sess.context) {
            try out.print("  " ++ C.dim ++ "(context full: {d}/{d} — /reset)" ++ C.rst ++ "\n", .{ sess.state.pos, sess.context });
            continue;
        }

        const pf0 = Timestamp.now(io, .awake);
        sess.prefill(ids) catch |e| switch (e) {
            error.ContextExhausted, error.TooManyTokens => {
                try out.writeAll("  " ++ C.dim ++ "(turn too long for the context — /reset)" ++ C.rst ++ "\n");
                continue;
            },
            else => return e,
        };
        const pf_ms = @as(f64, @floatFromInt(pf0.durationTo(Timestamp.now(io, .awake)).nanoseconds)) / 1e6;

        var styler = Styler.init(out, thinking);
        const r = try sess.generate(&gen, &styler);
        try footer(out, gen.items.len, ids.len, pf_ms, r.secs);
        turn += 1;
    }
    try out.writeAll("\n  " ++ C.dim ++ "bye" ++ C.rst ++ "\n");
}

/// Encode `text` and convert to `[]i64`, rejecting any id outside the model vocab.
fn encodeToI64(sess: *Session, gpa: std.mem.Allocator, text: []const u8, err: *std.Io.Writer) ![]i64 {
    const ids = try sess.tk.encode(gpa, text);
    defer gpa.free(ids);
    const out = try gpa.alloc(i64, ids.len);
    errdefer gpa.free(out);
    for (ids, out) |id, *dst| {
        if (id >= sess.vocab) {
            try err.print("chat: token id {d} outside model vocab {d}\n", .{ id, sess.vocab });
            return error.TokenOutOfVocab;
        }
        dst.* = id;
    }
    return out;
}

fn openDirAny(io: std.Io, dir_path: []const u8) !std.Io.Dir {
    if (std.fs.path.isAbsolute(dir_path)) return std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    return std.Io.Dir.cwd().openDir(io, dir_path, .{});
}
