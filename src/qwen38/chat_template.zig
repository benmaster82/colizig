//! ChatML prompt framing (`<|im_start|>role\ncontent<|im_end|>\n`).
//!
//! This is the core of Qwen's chat template.  The upstream
//! `chat_template.jinja` also injects a default system message, tool
//! declarations, and reasoning-effort / thinking blocks — not replicated here.

const std = @import("std");

pub const Message = struct {
    role: []const u8, // "system" | "user" | "assistant"
    content: []const u8,
};

/// The opening `<think>` framing Qwen3.8's `chat_template.jinja` pre-fills after
/// `<|im_start|>assistant\n`: `<think>\n` when reasoning is on (the default),
/// an empty `<think>\n\n</think>\n\n` block when it is off.
pub fn assistantOpen(think: bool) []const u8 {
    return if (think) "<|im_start|>assistant\n<think>\n" else "<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

/// Render `messages` in ChatML.  With `add_generation_prompt`, append an open
/// assistant turn (`assistantOpen(think)`) for the model to continue.
pub fn render(
    gpa: std.mem.Allocator,
    messages: []const Message,
    add_generation_prompt: bool,
    think: bool,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (messages) |m| {
        try out.print(gpa, "<|im_start|>{s}\n{s}<|im_end|>\n", .{ m.role, m.content });
    }
    if (add_generation_prompt) try out.appendSlice(gpa, assistantOpen(think));
    return out.toOwnedSlice(gpa);
}

test "render produces ChatML with a thinking generation prompt" {
    const gpa = std.testing.allocator;
    const msgs = [_]Message{
        .{ .role = "system", .content = "be nice" },
        .{ .role = "user", .content = "hi" },
    };
    const s = try render(gpa, &msgs, true, true);
    defer gpa.free(s);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nbe nice<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
        s,
    );
    const s2 = try render(gpa, &msgs, true, false);
    defer gpa.free(s2);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nbe nice<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        s2,
    );
}
