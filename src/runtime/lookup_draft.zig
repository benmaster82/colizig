//! Prompt-lookup ("n-gram self-speculation") drafting: no second model, no
//! training, no extra weights. Search the token history for the most recent
//! earlier occurrence of the trailing n-gram and propose whatever token(s)
//! followed it as the draft continuation. Classic win on repetitive content
//! (pasted code/text the model echoes or lightly edits) - see
//! docs/SPECULATIVE.md. Pairs with `qwen3moe/speculative.zig`, which verifies
//! the draft in one batched forward and rolls back on mismatch; this module
//! knows nothing about models, forwards, or caches.

const std = @import("std");

/// Trailing-window size used to look up a repeat. Small on purpose - a
/// 3-token match is already a strong enough signal on natural text/code, and
/// keeping it fixed keeps the module dependency-free (no config threading).
const NGRAM: usize = 3;

/// Returns up to `k` tokens that followed the most recent earlier occurrence
/// of `history`'s trailing `NGRAM`-token window, searching strictly before
/// that window, or `null` if there's no earlier occurrence (or `history` is
/// too short, or `k == 0`). The returned slice aliases `history` - it is only
/// valid until the caller next mutates `history`.
pub fn draft(history: []const i64, k: u32) ?[]const i64 {
    if (k == 0 or history.len < NGRAM) return null;
    const needle = history[history.len - NGRAM ..];

    // Backward linear scan: history is a single chat turn's worth of tokens
    // (hundreds to a few thousand), so this is microseconds next to a
    // forward pass - see docs/SPECULATIVE.md for why a hashmap index wasn't
    // worth the complexity here.
    var i: usize = history.len - NGRAM;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(i64, history[i .. i + NGRAM], needle)) {
            const start = i + NGRAM;
            // never draft past the start of the trailing window itself - past
            // that point we'd just be "predicting" tokens we already typed.
            const end = @min(start + k, history.len - NGRAM);
            if (end <= start) return null;
            return history[start..end];
        }
    }
    return null;
}

const testing = std.testing;

test "no match below the n-gram floor" {
    try testing.expect(draft(&.{ 1, 2 }, 4) == null);
}

test "no match when k is zero" {
    try testing.expect(draft(&.{ 1, 2, 3, 1, 2, 3 }, 0) == null);
}

test "no match when the trailing n-gram never repeats" {
    try testing.expect(draft(&.{ 1, 2, 3, 4, 5 }, 4) == null);
}

test "finds the most recent earlier occurrence and proposes what followed it" {
    // trailing window [7,8,9] repeats an earlier [7,8,9] followed by [10,11]
    const h = [_]i64{ 7, 8, 9, 10, 11, 7, 8, 9 };
    const d = draft(&h, 2).?;
    try testing.expectEqualSlices(i64, &.{ 10, 11 }, d);
}

test "clips the draft to what actually followed, even if k asks for more" {
    const h = [_]i64{ 1, 2, 3, 9, 1, 2, 3 };
    const d = draft(&h, 4).?;
    try testing.expectEqualSlices(i64, &.{9}, d);
}

test "picks the MOST RECENT earlier occurrence, not the first" {
    const h = [_]i64{ 1, 2, 3, 100, 0, 1, 2, 3, 200, 1, 2, 3 };
    const d = draft(&h, 2).?;
    try testing.expectEqualSlices(i64, &.{200}, d);
}
