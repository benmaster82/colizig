//! Byte-level BPE tokenizer (GPT-2 / Qwen family) loaded from `tokenizer.json`.
//!
//! Pipeline: split on special tokens → per-chunk pre-tokenization (an
//! approximation of Qwen's split regex) → byte→unicode remap → BPE merges →
//! vocab lookup.  `decode` reverses it.
//!
//! The pre-tokenizer covers ASCII and treats most non-ASCII codepoints as
//! letters; word boundaries in exotic scripts may differ from the reference
//! `tokenizers` library.  When the real checkpoint is available, validate
//! against known text↔id pairs (there is no reference to check against here).

const std = @import("std");

pub const Error = error{
    TokenizerFileMissing,
    BadTokenizerJson,
    UnknownToken,
} || std.mem.Allocator.Error || std.Io.Writer.Error;

pub const Special = struct {
    id: u32,
    content: []u8, // owned
    /// `special: true` in tokenizer.json - dropped by `decode(skip_special=true)`.
    /// `false` (e.g. `<think>`) still splits the stream atomically during encode
    /// but is kept when decoding.
    special: bool = true,
};

pub const Tokenizer = struct {
    /// token bytes (byte-unicode form) → id
    vocab: std.StringHashMapUnmanaged(u32),
    /// id → token bytes (byte-unicode form), owned
    id_to_tok: [][]u8,
    /// "<a> <b>" → merge rank
    ranks: std.StringHashMapUnmanaged(u32),
    specials: []Special,
    /// GPT-2 byte→codepoint map and its inverse
    b2u: [256]u21,
    u2b: std.AutoHashMapUnmanaged(u21, u8),
    gpa: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,

    pub fn deinit(self: *Tokenizer) void {
        const gpa = self.gpa;
        self.arena.deinit();
        gpa.destroy(self.arena);
        self.* = undefined;
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, err: *std.Io.Writer) Error!Tokenizer {
        const bytes = dir.readFileAlloc(io, "tokenizer.json", gpa, .limited(64 << 20)) catch {
            try err.writeAll("tokenizer: cannot read tokenizer.json\n");
            return error.TokenizerFileMissing;
        };
        defer gpa.free(bytes);
        return parse(gpa, bytes, err);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, err: *std.Io.Writer) Error!Tokenizer {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        arena.* = .init(gpa);
        errdefer {
            arena.deinit();
            gpa.destroy(arena);
        }
        const a = arena.allocator();

        var parsed = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch {
            try err.writeAll("tokenizer.json is not valid JSON\n");
            return error.BadTokenizerJson;
        };
        defer parsed.deinit();
        const root = switch (parsed.value) {
            .object => |o| o,
            else => return error.BadTokenizerJson,
        };
        const model = switch (root.get("model") orelse return error.BadTokenizerJson) {
            .object => |o| o,
            else => return error.BadTokenizerJson,
        };

        var self: Tokenizer = .{
            .vocab = .empty,
            .id_to_tok = &.{},
            .ranks = .empty,
            .specials = &.{},
            .b2u = undefined,
            .u2b = .empty,
            .gpa = gpa,
            .arena = arena,
        };
        buildByteMap(&self.b2u);
        for (self.b2u, 0..) |cp, b| try self.u2b.put(a, cp, @intCast(b));

        // vocab
        const vocab_obj = switch (model.get("vocab") orelse return error.BadTokenizerJson) {
            .object => |o| o,
            else => return error.BadTokenizerJson,
        };
        var max_id: u32 = 0;
        var vit = vocab_obj.iterator();
        while (vit.next()) |kv| {
            const id: u32 = switch (kv.value_ptr.*) {
                .integer => |x| if (x < 0) return error.BadTokenizerJson else @intCast(x),
                else => return error.BadTokenizerJson,
            };
            max_id = @max(max_id, id);
        }
        // added / special tokens contribute ids too
        var specials: std.ArrayList(Special) = .empty;
        if (root.get("added_tokens")) |atv| if (atv == .array) {
            for (atv.array.items) |item| {
                if (item != .object) continue;
                const o = item.object;
                const id: u32 = switch (o.get("id") orelse continue) {
                    .integer => |x| if (x < 0) continue else @intCast(x),
                    else => continue,
                };
                const content = switch (o.get("content") orelse continue) {
                    .string => |s| s,
                    else => continue,
                };
                max_id = @max(max_id, id);
                // Every added token splits the stream atomically during encode
                // (HF matches them before BPE); only `special: true` ones are
                // dropped by `decode(skip_special=true)`.
                const is_special = if (o.get("special")) |sv| (sv == .bool and sv.bool) else false;
                try specials.append(a, .{ .id = id, .content = try a.dupe(u8, content), .special = is_special });
            }
        };

        self.id_to_tok = try a.alloc([]u8, max_id + 1);
        @memset(self.id_to_tok, &.{});
        vit = vocab_obj.iterator();
        while (vit.next()) |kv| {
            const tok = try a.dupe(u8, kv.key_ptr.*);
            const id: u32 = @intCast(kv.value_ptr.*.integer);
            try self.vocab.put(a, tok, id);
            self.id_to_tok[id] = tok;
        }
        for (specials.items) |sp| {
            if (self.id_to_tok[sp.id].len == 0) self.id_to_tok[sp.id] = sp.content;
            try self.vocab.put(a, sp.content, sp.id);
        }
        self.specials = try specials.toOwnedSlice(a);

        // merges: ["a b", ...] or [["a","b"], ...]
        if (model.get("merges")) |mv| if (mv == .array) {
            for (mv.array.items, 0..) |item, rank| {
                var left: []const u8 = "";
                var right: []const u8 = "";
                switch (item) {
                    .string => |s| {
                        const sp = std.mem.indexOfScalar(u8, s, ' ') orelse continue;
                        left = s[0..sp];
                        right = s[sp + 1 ..];
                    },
                    .array => |pair| {
                        if (pair.items.len != 2 or pair.items[0] != .string or pair.items[1] != .string) continue;
                        left = pair.items[0].string;
                        right = pair.items[1].string;
                    },
                    else => continue,
                }
                const key = try std.fmt.allocPrint(a, "{s} {s}", .{ left, right });
                try self.ranks.put(a, key, @intCast(rank));
            }
        };

        return self;
    }

    pub fn vocabSize(self: Tokenizer) usize {
        return self.id_to_tok.len;
    }

    pub fn specialId(self: Tokenizer, content: []const u8) ?u32 {
        for (self.specials) |sp| {
            if (std.mem.eql(u8, sp.content, content)) return sp.id;
        }
        return null;
    }

    /// Encode `text` to token ids (caller owns the returned slice).
    pub fn encode(self: *Tokenizer, gpa: std.mem.Allocator, text: []const u8) Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        try self.encodeInto(gpa, &out, text);
        return out.toOwnedSlice(gpa);
    }

    pub fn encodeInto(self: *Tokenizer, gpa: std.mem.Allocator, out: *std.ArrayList(u32), text: []const u8) Error!void {
        // 1. split on special-token literals (longest first)
        var rest = text;
        while (rest.len != 0) {
            var hit: ?struct { start: usize, sp: Special } = null;
            for (self.specials) |sp| {
                if (sp.content.len == 0) continue;
                if (std.mem.indexOf(u8, rest, sp.content)) |at| {
                    if (hit == null or at < hit.?.start) hit = .{ .start = at, .sp = sp };
                }
            }
            const cut = if (hit) |h| h.start else rest.len;
            if (cut != 0) try self.encodePlain(gpa, out, rest[0..cut]);
            if (hit) |h| {
                try out.append(gpa, h.sp.id);
                rest = rest[h.start + h.sp.content.len ..];
            } else break;
        }
    }

    fn encodePlain(self: *Tokenizer, gpa: std.mem.Allocator, out: *std.ArrayList(u32), text: []const u8) Error!void {
        var it = PreTokenizer{ .s = text };
        var scratch: std.ArrayList(u8) = .empty;
        defer scratch.deinit(gpa);
        while (it.next()) |word| {
            // byte → unicode remap
            scratch.clearRetainingCapacity();
            for (word) |b| {
                var buf: [4]u8 = undefined;
                const n = std.unicode.utf8Encode(self.b2u[b], &buf) catch continue;
                try scratch.appendSlice(gpa, buf[0..n]);
            }
            try self.bpe(gpa, out, scratch.items);
        }
    }

    fn bpe(self: *Tokenizer, gpa: std.mem.Allocator, out: *std.ArrayList(u32), word: []const u8) Error!void {
        // split into codepoint pieces
        var pieces: std.ArrayList([]const u8) = .empty;
        defer pieces.deinit(gpa);
        var ui = std.unicode.Utf8Iterator{ .bytes = word, .i = 0 };
        while (ui.nextCodepointSlice()) |cp| try pieces.append(gpa, cp);
        if (pieces.items.len == 0) return;

        var key_buf: std.ArrayList(u8) = .empty;
        defer key_buf.deinit(gpa);

        while (pieces.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_at: usize = 0;
            for (0..pieces.items.len - 1) |i| {
                key_buf.clearRetainingCapacity();
                try key_buf.appendSlice(gpa, pieces.items[i]);
                try key_buf.append(gpa, ' ');
                try key_buf.appendSlice(gpa, pieces.items[i + 1]);
                if (self.ranks.get(key_buf.items)) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best_at = i;
                    }
                }
            }
            if (best_rank == std.math.maxInt(u32)) break;
            // merge pieces[best_at] and pieces[best_at+1] - they are adjacent in `word`
            const a_start = @intFromPtr(pieces.items[best_at].ptr) - @intFromPtr(word.ptr);
            const b_end = @intFromPtr(pieces.items[best_at + 1].ptr) + pieces.items[best_at + 1].len - @intFromPtr(word.ptr);
            pieces.items[best_at] = word[a_start..b_end];
            _ = pieces.orderedRemove(best_at + 1);
        }

        for (pieces.items) |p| {
            const id = self.vocab.get(p) orelse {
                // fall back to per-codepoint (byte) ids so encoding never fails
                var ci = std.unicode.Utf8Iterator{ .bytes = p, .i = 0 };
                while (ci.nextCodepointSlice()) |c| {
                    if (self.vocab.get(c)) |cid| try out.append(gpa, cid);
                }
                continue;
            };
            try out.append(gpa, id);
        }
    }

    /// Decode ids to UTF-8 text (caller owns the returned slice).
    /// `skip_special` drops special-token ids from the output.
    pub fn decode(self: *Tokenizer, gpa: std.mem.Allocator, ids: []const u32, skip_special: bool) Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        for (ids) |id| {
            if (id >= self.id_to_tok.len) continue;
            if (skip_special and self.isSpecial(id)) continue;
            const tok = self.id_to_tok[id];
            // map each codepoint back to a byte
            var ui = std.unicode.Utf8Iterator{ .bytes = tok, .i = 0 };
            while (ui.nextCodepoint()) |cp| {
                if (self.u2b.get(cp)) |b| {
                    try out.append(gpa, b);
                } else {
                    // not a byte-unicode char (e.g. a special token's literal text)
                    var buf: [4]u8 = undefined;
                    const n = std.unicode.utf8Encode(cp, &buf) catch continue;
                    try out.appendSlice(gpa, buf[0..n]);
                }
            }
        }
        return out.toOwnedSlice(gpa);
    }

    fn isSpecial(self: Tokenizer, id: u32) bool {
        for (self.specials) |sp| {
            if (sp.id == id) return sp.special;
        }
        return false;
    }
};

// ---- GPT-2 byte ↔ unicode map ---------------------------------------

fn buildByteMap(map: *[256]u21) void {
    var used = [_]bool{false} ** 256;
    // printable ranges keep their own codepoint
    inline for (.{ .{ '!', '~' }, .{ 0xA1, 0xAC }, .{ 0xAE, 0xFF } }) |r| {
        var b: u16 = r[0];
        while (b <= r[1]) : (b += 1) {
            map[b] = @intCast(b);
            used[b] = true;
        }
    }
    var n: u21 = 0;
    var b: u16 = 0;
    while (b < 256) : (b += 1) {
        if (!used[b]) {
            map[b] = 256 + n;
            n += 1;
        }
    }
}

// ---- pre-tokenizer (Qwen split, approximated) ----------------------

const PreTokenizer = struct {
    s: []const u8,
    i: usize = 0,

    fn cpAt(self: *const PreTokenizer, at: usize) ?struct { cp: u21, len: usize } {
        if (at >= self.s.len) return null;
        const l = std.unicode.utf8ByteSequenceLength(self.s[at]) catch return .{ .cp = self.s[at], .len = 1 };
        if (at + l > self.s.len) return .{ .cp = self.s[at], .len = 1 };
        const cp = std.unicode.utf8Decode(self.s[at .. at + l]) catch return .{ .cp = self.s[at], .len = 1 };
        return .{ .cp = cp, .len = l };
    }

    fn next(self: *PreTokenizer) ?[]const u8 {
        if (self.i >= self.s.len) return null;
        const start = self.i;

        // contractions: 's 't 're 've 'm 'll 'd  (case-insensitive)
        if (self.s[self.i] == '\'') {
            for ([_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" }) |c| {
                if (self.i + c.len <= self.s.len and std.ascii.eqlIgnoreCase(self.s[self.i .. self.i + c.len], c)) {
                    self.i += c.len;
                    return self.s[start..self.i];
                }
            }
        }

        const first = self.cpAt(self.i).?;

        // optional leading space, then a run
        var j = self.i;
        var lead_space = false;
        if (first.cp == ' ') {
            // " ?X" only if followed by letters or punctuation, else it's whitespace
            const after = self.cpAt(self.i + 1);
            if (after != null and !isSpace(after.?.cp)) {
                lead_space = true;
                j = self.i + 1;
            }
        }

        const anchor = self.cpAt(j) orelse {
            self.i = self.s.len;
            return self.s[start..self.i];
        };

        if (isLetter(anchor.cp) or (lead_space and isLetter(anchor.cp))) {
            j += anchor.len;
            while (self.cpAt(j)) |c| {
                if (!isLetter(c.cp)) break;
                j += c.len;
            }
            self.i = j;
            return self.s[start..self.i];
        }
        if (isDigit(anchor.cp)) {
            // Qwen splits digits one at a time
            self.i = j + anchor.len;
            return self.s[start..self.i];
        }
        if (!isSpace(anchor.cp)) {
            // punctuation run (+ trailing newlines)
            j += anchor.len;
            while (self.cpAt(j)) |c| {
                if (isSpace(c.cp) or isLetter(c.cp) or isDigit(c.cp)) break;
                j += c.len;
            }
            while (self.cpAt(j)) |c| {
                if (c.cp != '\n' and c.cp != '\r') break;
                j += c.len;
            }
            self.i = j;
            return self.s[start..self.i];
        }

        // whitespace run
        j = self.i;
        while (self.cpAt(j)) |c| {
            if (!isSpace(c.cp)) break;
            j += c.len;
        }
        // \s+(?!\S): keep one trailing space attached to the next word
        if (j - self.i > 1 and j < self.s.len) j -= 1;
        self.i = if (j == self.i) self.i + first.len else j;
        return self.s[start..self.i];
    }
};

fn isSpace(cp: u21) bool {
    return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0B or cp == 0x0C or cp == 0xA0;
}
fn isDigit(cp: u21) bool {
    return cp >= '0' and cp <= '9';
}
fn isLetter(cp: u21) bool {
    if (cp < 0x80) return (cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z');
    // non-ASCII: treat as a letter unless it's a known punctuation/symbol block
    return !(cp >= 0x2000 and cp <= 0x206F) and // general punctuation
        !(cp >= 0x3000 and cp <= 0x303F) and // CJK symbols/punct
        !(cp >= 0xFF00 and cp <= 0xFF0F) and !(cp >= 0xFF1A and cp <= 0xFF20);
}

// ---- tests --------------------------------------------------------

const testing = std.testing;

const tiny_tok_json =
    \\{ "model": { "type": "BPE",
    \\  "vocab": { "h":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"he":8,"lo":9,"hel":10,"hello":11,"Ġw":12,"world":13 },
    \\  "merges": ["h e","l o","he l","hel lo","Ġ w","w o","wo r","wor l","worl d"] },
    \\  "added_tokens": [ {"id":20,"content":"<|im_start|>","special":true}, {"id":21,"content":"<|im_end|>","special":true} ]
    \\}
;

test "byte↔unicode map is a bijection" {
    var m: [256]u21 = undefined;
    buildByteMap(&m);
    var seen = std.AutoHashMap(u21, void).init(testing.allocator);
    defer seen.deinit();
    for (m) |cp| {
        try testing.expect(!seen.contains(cp));
        try seen.put(cp, {});
    }
}

test "BPE encode/decode round-trips and merges apply" {
    const gpa = testing.allocator;
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);
    var tk = try Tokenizer.parse(gpa, tiny_tok_json, &sink.writer);
    defer tk.deinit();

    const ids = try tk.encode(gpa, "hello world");
    defer gpa.free(ids);
    // "hello" merges to a single token; " world" -> "Ġw" + "orld"? merges give Ġw, then w o r l d chain not all present
    try testing.expect(ids.len >= 2);
    try testing.expectEqual(@as(u32, 11), ids[0]); // "hello"

    const back = try tk.decode(gpa, ids, true);
    defer gpa.free(back);
    try testing.expectEqualStrings("hello world", back);
}

test "special tokens split the stream and survive round-trip" {
    const gpa = testing.allocator;
    var nul: [0]u8 = .{};
    var sink: std.Io.Writer.Discarding = .init(&nul);
    var tk = try Tokenizer.parse(gpa, tiny_tok_json, &sink.writer);
    defer tk.deinit();

    try testing.expectEqual(@as(?u32, 20), tk.specialId("<|im_start|>"));

    const ids = try tk.encode(gpa, "<|im_start|>hello<|im_end|>");
    defer gpa.free(ids);
    try testing.expectEqual(@as(u32, 20), ids[0]);
    try testing.expectEqual(@as(u32, 21), ids[ids.len - 1]);

    const shown = try tk.decode(gpa, ids, false);
    defer gpa.free(shown);
    try testing.expectEqualStrings("<|im_start|>hello<|im_end|>", shown);
    const clean = try tk.decode(gpa, ids, true);
    defer gpa.free(clean);
    try testing.expectEqualStrings("hello", clean);
}

test "pre-tokenizer splits words, digits, and punctuation" {
    var it = PreTokenizer{ .s = "ab 12!" };
    try testing.expectEqualStrings("ab", it.next().?);
    try testing.expectEqualStrings(" 1", it.next().?);
    try testing.expectEqualStrings("2", it.next().?);
    try testing.expectEqualStrings("!", it.next().?);
    try testing.expect(it.next() == null);
}
