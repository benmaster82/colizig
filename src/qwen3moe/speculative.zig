//! Speculative decoding for Qwen3-MoE, greedy-only (see docs/SPECULATIVE.md
//! for the full algorithm, the accept/correction/bonus math, and why
//! Qwen4-Exp isn't wired yet).
//!
//! `model.zig` is completely untouched: `speculativeStep` runs the existing
//! `forward()` unmodified over `[last_confirmed] ++ draft`, then verifies
//! every batch position by re-reading `Scratch.h` (public, retained after
//! `forward()` returns) - one extra RMSNorm + `lmHead` call per position,
//! all stateless. The only sequence state Qwen3-MoE carries is `attn.Cache`,
//! which is absolute-position indexed and `len`-truncatable - rolling back a
//! rejected draft suffix costs nothing beyond decrementing `state.pos` and
//! every layer's `cache.len`.

const std = @import("std");
const model_mod = @import("model.zig");
const rms = @import("../ops/rmsnorm.zig").rms;

pub const Model = model_mod.Model;
pub const State = model_mod.State;
pub const Scratch = model_mod.Scratch;

/// Run one speculative round: batch `[last_confirmed] ++ draft` through a
/// single `forward()` call, verify each position's greedy argmax against the
/// next draft token, and roll the KV cache back to the longest confirmed
/// prefix. Writes the newly-confirmed tokens into `out` (must be at least
/// `draft.len + 1` long: the confirmed draft prefix, plus exactly one
/// correction - where the draft first went wrong - or, if every draft token
/// was right, one bonus token that comes free from the same verification
/// pass) and returns how many were written (always `>= 1`).
///
/// `logits` is scratch, length `model.cfg.vocab` (the caller's existing
/// per-step logits buffer is fine to reuse - its contents are undefined on
/// return). Bit-identical to running `forward()` one token at a time and
/// taking the argmax each step (proven by the unit test below): the verify
/// loop reconstructs, off the *same* `Scratch.h` rows the batched forward
/// already computed, exactly the per-position prediction a token-by-token
/// loop would have made.
pub fn speculativeStep(
    gpa: std.mem.Allocator,
    model: *const Model,
    state: *State,
    sc: *Scratch,
    last_confirmed: i64,
    draft: []const i64,
    logits: []f32,
    out: []i64,
) !usize {
    const k = draft.len;
    std.debug.assert(out.len >= k + 1);
    std.debug.assert(logits.len == model.cfg.vocab);

    var ids_buf: [65]i64 = undefined;
    std.debug.assert(k + 1 <= ids_buf.len);
    ids_buf[0] = last_confirmed;
    @memcpy(ids_buf[1 .. k + 1], draft);
    const ids = ids_buf[0 .. k + 1];

    const H = model.cfg.hidden;
    const pos_before = state.pos;
    try model_mod.forward(model, state, sc, ids, logits, .{});

    const row = try gpa.alloc(f32, H);
    defer gpa.free(row);

    // Walk every batch position's own prediction (what should follow its
    // input token) against the next draft token. `forward()` already
    // applied the final RMSNorm **in place** to the LAST row of `sc.h` and
    // ran `lmHead` on it (into `logits`) - reuse that directly for position
    // `k`. Every earlier row is still pre-final-norm (forward() only norms
    // the last one), so those need the norm applied to a copy before
    // `lmHead` - re-norming the last row a second time would silently
    // double-apply RMSNorm and desync from a token-by-token decode.
    var accepted: usize = 0;
    var correction: i64 = undefined;
    while (true) {
        if (accepted == k) {
            correction = @intCast(std.mem.indexOfMax(f32, logits));
            break;
        }
        @memcpy(row, sc.h[accepted * H ..][0..H]);
        rms(row, row, model.final_norm, model.cfg.eps);
        try model.weights.lmHead(row, logits);
        correction = @intCast(std.mem.indexOfMax(f32, logits));
        if (correction != draft[accepted]) break;
        accepted += 1;
    }

    const valid = accepted + 1;
    @memcpy(out[0..accepted], draft[0..accepted]);
    out[accepted] = correction;

    state.pos = pos_before + valid;
    for (state.kv) |*c| c.len = state.pos;
    return valid;
}

// ---- tests ------------------------------------------------------------

const testing = std.testing;
const manifest_mod = @import("../model/manifest.zig");
const weights_mod = @import("../model/weights.zig");
const lookup_draft = @import("../runtime/lookup_draft.zig");

fn openTiny(gpa: std.mem.Allocator, io: std.Io, sink: *std.Io.Writer) !struct { m: manifest_mod.Manifest, w: weights_mod.Weights } {
    var m = try manifest_mod.open(gpa, io, "test/fixtures/tiny-qwen3", sink);
    errdefer m.deinit();
    const w = try weights_mod.Weights.open(gpa, io, "test/fixtures/tiny-qwen3", &m, sink);
    return .{ .m = m, .w = w };
}

test "speculative decode matches token-by-token greedy decode exactly" {
    const gpa = testing.allocator;
    const io = std.testing.io;
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);

    var opened = openTiny(gpa, io, &sink.writer) catch |e| switch (e) {
        error.OpenFailed, error.NoCheckpoint => return error.SkipZigTest,
        else => return e,
    };
    defer opened.m.deinit();
    var w = opened.w;
    defer w.deinit();

    var model = try Model.load(gpa, &w);
    defer model.deinit();

    const V = model.cfg.vocab;
    const ctx = 24;
    const prompt = [_]i64{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const want_n = 6;

    // ground truth: plain greedy, one token at a time
    var st_ref = try State.init(gpa, &model, ctx, model.cfg.experts, io);
    defer st_ref.deinit();
    var sc_ref = try Scratch.init(gpa, &model, prompt.len, ctx); // generateGreedy prefills in one shot
    defer sc_ref.deinit();
    var ref_out: [want_n]i64 = undefined;
    const ref_n = try model_mod.generateGreedy(&model, &st_ref, &sc_ref, &prompt, &ref_out, .{});

    // speculative: draft depths 0 (degenerates to plain steps), 2, 4 must all
    // reproduce the exact same token sequence, however many rounds it takes.
    for ([_]u32{ 0, 2, 4 }) |k| {
        var st = try State.init(gpa, &model, ctx, model.cfg.experts, io);
        defer st.deinit();
        // sized for both the prompt prefill (S = prompt.len-1) and every
        // speculative round (S = k+1), same as chat.zig reuses one Scratch
        // for its prefill chunks and its per-token decode calls.
        var sc = try Scratch.init(gpa, &model, @max(prompt.len - 1, k + 1), ctx);
        defer sc.deinit();
        const logits = try gpa.alloc(f32, V);
        defer gpa.free(logits);

        // prefill the prompt (all but the last token), matching how chat.zig
        // primes state before the speculative round loop takes over.
        try model_mod.forward(&model, &st, &sc, prompt[0 .. prompt.len - 1], logits, .{});

        var history: std.ArrayList(i64) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, &prompt);

        var out_buf: [17]i64 = undefined;
        var last: i64 = prompt[prompt.len - 1];
        var got: [ref_out.len]i64 = undefined;
        var n: usize = 0;
        while (n < ref_n) {
            const d = lookup_draft.draft(history.items, k) orelse @as([]const i64, &.{});
            const valid = try speculativeStep(gpa, &model, &st, &sc, last, d, logits, &out_buf);
            for (out_buf[0..valid]) |tok| {
                if (n >= ref_n) break;
                got[n] = tok;
                try history.append(gpa, tok);
                last = tok;
                n += 1;
            }
        }
        try testing.expectEqualSlices(i64, ref_out[0..ref_n], got[0..n]);
    }
}
