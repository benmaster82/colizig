//! `serve <MODEL_DIR> [--port N] [--host H]` — a minimal OpenAI-compatible HTTP
//! endpoint. One connection at a time (local use), the model stays warm, every
//! request is stateless (send the full message history, as the OpenAI API does).
//!
//!   POST /v1/chat/completions   {messages, max_tokens, temperature, top_p, stream}
//!   GET  /v1/models
//!   GET  /                      health text
//!
//! With `"stream": true` the reply is Server-Sent Events
//! (`data: {choices:[{delta:{content}}]}` … `data: [DONE]`).

const std = @import("std");
const args = @import("args.zig");
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");
const tok_mod = @import("../qwen38/tokenizer.zig");
const tmpl = @import("../qwen38/chat_template.zig");
const sampler_mod = @import("../runtime/sampler.zig");
const parallel = @import("../runtime/parallel.zig");
const gpu = @import("../backend/gpu.zig");
const q4 = @import("../qwen38/model.zig");
const q3 = @import("../qwen3moe/model.zig");
const net = std.Io.net;
const http = std.http;

pub fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    opts: args.Options,
) !void {
    var m0 = try manifest_mod.open(gpa, io, opts.model_dir, err);
    const arch = m0.cfg.arch;
    m0.deinit();
    return switch (arch) {
        .qwen3_moe => serveGeneric(q3, gpa, io, out, err, opts),
        .qwen4_exp => serveGeneric(q4, gpa, io, out, err, opts),
    };
}

fn serveGeneric(
    comptime Mdl: type,
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

    var dir = try openDirAny(io, opts.model_dir);
    defer dir.close(io);
    var m = try manifest_mod.open(gpa, io, opts.model_dir, err);
    defer m.deinit();
    var w = try weights_mod.Weights.open(gpa, io, opts.model_dir, &m, err);
    defer w.deinit();
    var tk = try tok_mod.Tokenizer.load(gpa, io, dir, err);
    defer tk.deinit();

    const ctx: usize = if (opts.budget.context != 0) opts.budget.context else 8192;
    const cap: usize = if (opts.expert_cap != 0) opts.expert_cap else @min(128, m.cfg.experts);

    try out.print("loading {s} ({d} layers, expert cap {d}) ...\n", .{ @tagName(m.cfg.arch), m.cfg.layers, cap });
    try out.flush();

    var model = try Mdl.Model.load(gpa, &w);
    defer model.deinit();
    var state = try Mdl.State.init(gpa, &model, ctx, cap, io);
    defer state.deinit();
    var sc = try Mdl.Scratch.init(gpa, &model, @min(ctx, 1024), ctx);
    defer sc.deinit();

    const logits = try gpa.alloc(f32, m.cfg.vocab);
    defer gpa.free(logits);

    const port: u16 = opts.port;
    const host = opts.host;
    var addr = net.IpAddress.parseIp4(host, port) catch net.IpAddress.parseIp4("127.0.0.1", port) catch unreachable;
    var server = try net.IpAddress.listen(&addr, io, .{ .reuse_address = true });
    defer server.socket.close(io);

    try out.print("colizig serve — http://{s}:{d}/v1/chat/completions\n", .{ host, port });
    try out.flush();

    var conn_rbuf: [64 * 1024]u8 = undefined;
    var conn_wbuf: [64 * 1024]u8 = undefined;

    while (true) {
        var stream = server.accept(io) catch continue;
        defer stream.close(io);
        var sr = stream.reader(io, &conn_rbuf);
        var sw = stream.writer(io, &conn_wbuf);
        var hs = http.Server.init(&sr.interface, &sw.interface);

        while (hs.reader.state == .ready) {
            var req = hs.receiveHead() catch break;
            handle(Mdl, gpa, io, &req, &tk, &model, &state, &sc, logits, m.cfg, opts, ctx) catch |e| {
                err.print("serve: request error: {s}\n", .{@errorName(e)}) catch {};
                break;
            };
        }
    }
}

fn handle(
    comptime Mdl: type,
    gpa: std.mem.Allocator,
    io: std.Io,
    req: *http.Server.Request,
    tk: *tok_mod.Tokenizer,
    model: *Mdl.Model,
    state: *Mdl.State,
    sc: *Mdl.Scratch,
    logits: []f32,
    cfg: anytype,
    opts: args.Options,
    ctx: usize,
) !void {
    const target = req.head.target;
    const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];

    if (req.head.method == .GET and (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/health"))) {
        try req.respond("colizig serve — POST /v1/chat/completions\n", .{});
        return;
    }
    if (req.head.method == .GET and std.mem.eql(u8, path, "/v1/models")) {
        try req.respond(
            \\{"object":"list","data":[{"id":"colizig","object":"model","owned_by":"local"}]}
        , .{ .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
        return;
    }
    if (!(req.head.method == .POST and std.mem.eql(u8, path, "/v1/chat/completions"))) {
        try req.respond("not found\n", .{ .status = .not_found });
        return;
    }

    // ---- read + parse the JSON body ----
    var body_buf: [256 * 1024]u8 = undefined;
    const reader = req.readerExpectNone(&.{});
    const want: usize = @min(req.head.content_length orelse body_buf.len, body_buf.len);
    const n = try reader.readSliceShort(body_buf[0..want]);
    const body = body_buf[0..n];

    var parsed = std.json.parseFromSlice(std.json.Value, gpa, body, .{}) catch {
        try req.respond("{\"error\":\"bad json\"}", .{ .status = .bad_request, .extra_headers = &.{jsonct} });
        return;
    };
    defer parsed.deinit();
    const root = if (parsed.value == .object) parsed.value.object else {
        try req.respond("{\"error\":\"body must be an object\"}", .{ .status = .bad_request, .extra_headers = &.{jsonct} });
        return;
    };

    const stream_mode = if (root.get("stream")) |v| (v == .bool and v.bool) else false;
    const max_new: usize = blk: {
        if (root.get("max_tokens")) |v| if (v == .integer and v.integer > 0) break :blk @min(@as(usize, @intCast(v.integer)), 4096);
        break :blk 512;
    };
    var temp: f32 = opts.temperature;
    var top_p: f32 = opts.top_p;
    if (root.get("temperature")) |v| temp = jf32(v, temp);
    if (root.get("top_p")) |v| top_p = jf32(v, top_p);

    // ---- messages → ChatML ----
    var msgs: std.ArrayList(tmpl.Message) = .empty;
    defer msgs.deinit(gpa);
    if (root.get("messages")) |mv| if (mv == .array) {
        for (mv.array.items) |it| {
            if (it != .object) continue;
            const role = if (it.object.get("role")) |r| (if (r == .string) r.string else "user") else "user";
            const content = if (it.object.get("content")) |cc| (if (cc == .string) cc.string else "") else "";
            try msgs.append(gpa, .{ .role = role, .content = content });
        }
    };
    if (msgs.items.len == 0) {
        try req.respond("{\"error\":\"no messages\"}", .{ .status = .bad_request, .extra_headers = &.{jsonct} });
        return;
    }

    const prompt_text = try tmpl.render(gpa, msgs.items, true, true);
    defer gpa.free(prompt_text);
    const ids = try tk.encode(gpa, prompt_text);
    defer gpa.free(ids);
    if (ids.len + max_new + 8 > ctx) {
        try req.respond("{\"error\":\"context too long\"}", .{ .status = .bad_request, .extra_headers = &.{jsonct} });
        return;
    }
    const prompt_i64 = try gpa.alloc(i64, ids.len);
    defer gpa.free(prompt_i64);
    for (ids, prompt_i64) |s, *d| d.* = s;

    var sampler = try sampler_mod.Sampler.init(gpa, io, cfg.vocab, .{
        .temperature = temp,
        .top_k = opts.top_k,
        .top_p = top_p,
        .seed = opts.seed,
    });
    defer sampler.deinit();

    // ---- generate ----
    state.reset();
    try Mdl.forward(model, state, sc, prompt_i64, logits, Mdl.Opts{ .io = io });

    var reply: std.ArrayList(u32) = .empty;
    defer reply.deinit(gpa);

    if (stream_mode) {
        var sbuf: [16 * 1024]u8 = undefined;
        var bw = try req.respondStreaming(&sbuf, .{ .respond_options = .{ .extra_headers = &.{
            .{ .name = "content-type", .value = "text/event-stream" },
            .{ .name = "cache-control", .value = "no-cache" },
        } } });
        var printed: usize = 0;
        var one: [1]i64 = undefined;
        while (reply.items.len < max_new) {
            const next = sampler.pick(logits);
            if (@as(i64, @intCast(next)) == cfg.eos_id) break;
            try reply.append(gpa, @intCast(next));
            const full = try tk.decode(gpa, reply.items, true);
            defer gpa.free(full);
            if (full.len > printed) {
                try sseDelta(&bw.writer, gpa, full[printed..]);
                try bw.flush();
                printed = full.len;
            }
            one[0] = @intCast(next);
            try Mdl.forward(model, state, sc, &one, logits, Mdl.Opts{ .io = io });
        }
        try bw.writer.writeAll("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
        try bw.end();
        return;
    }

    var one: [1]i64 = undefined;
    while (reply.items.len < max_new) {
        const next = sampler.pick(logits);
        if (@as(i64, @intCast(next)) == cfg.eos_id) break;
        try reply.append(gpa, @intCast(next));
        one[0] = @intCast(next);
        try Mdl.forward(model, state, sc, &one, logits, Mdl.Opts{ .io = io });
    }
    const text = try tk.decode(gpa, reply.items, true);
    defer gpa.free(text);

    var resp: std.ArrayList(u8) = .empty;
    defer resp.deinit(gpa);
    try resp.appendSlice(gpa, "{\"id\":\"colizig\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    try jsonString(&resp, gpa, text);
    try resp.print(gpa, "}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d}}}}}", .{ ids.len, reply.items.len });
    try req.respond(resp.items, .{ .extra_headers = &.{jsonct} });
}

const jsonct: http.Header = .{ .name = "content-type", .value = "application/json" };

fn sseDelta(w: *std.Io.Writer, gpa: std.mem.Allocator, chunk: []const u8) !void {
    var s: std.ArrayList(u8) = .empty;
    defer s.deinit(gpa);
    try s.appendSlice(gpa, "data: {\"choices\":[{\"delta\":{\"content\":");
    try jsonString(&s, gpa, chunk);
    try s.appendSlice(gpa, "}}]}\n\n");
    try w.writeAll(s.items);
}

fn jsonString(buf: *std.ArrayList(u8), gpa: std.mem.Allocator, s: []const u8) !void {
    try buf.append(gpa, '"');
    for (s) |c| switch (c) {
        '"' => try buf.appendSlice(gpa, "\\\""),
        '\\' => try buf.appendSlice(gpa, "\\\\"),
        '\n' => try buf.appendSlice(gpa, "\\n"),
        '\r' => try buf.appendSlice(gpa, "\\r"),
        '\t' => try buf.appendSlice(gpa, "\\t"),
        0...8, 11, 12, 14...31 => try buf.print(gpa, "\\u{x:0>4}", .{c}),
        else => try buf.append(gpa, c),
    };
    try buf.append(gpa, '"');
}

fn jf32(v: std.json.Value, dflt: f32) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => dflt,
    };
}

fn openDirAny(io: std.Io, path: []const u8) !std.Io.Dir {
    return if (std.fs.path.isAbsolute(path))
        std.Io.Dir.openDirAbsolute(io, path, .{})
    else
        std.Io.Dir.cwd().openDir(io, path, .{});
}
