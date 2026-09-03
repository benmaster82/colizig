#!/usr/bin/env python3
"""Independent NumPy reference forward for Qwen3.8-Flash-Next (Qwen4-Exp).

A second implementation of the same math the Zig engine ports from colibri,
written in a deliberately dense / loopy style so a structural bug in either side
shows up as a logit mismatch.  It is *not* the upstream `Qwen4ExpForCausalLM`
(that needs the released checkpoint + `transformers`); it validates the Zig
engine against an independent port of the same specification (brief §25).

Reads a checkpoint directory (config.json + model.safetensors), runs the forward
on given token ids, prints JSON: {final_logits, greedy_ids, ...}.

Requires: numpy, safetensors (only for reading; a builtin reader is used).
"""

import json
import math
import struct
import sys

import numpy as np

MASK64 = (1 << 64) - 1


def e4m3_decode(b):
    b = b.astype(np.uint16)
    sign = np.where(b & 0x80 != 0, -1.0, 1.0).astype(np.float64)
    exp = ((b >> 3) & 0x0F).astype(np.int64)
    mant = (b & 0x07).astype(np.int64)
    out = np.zeros(b.shape, dtype=np.float64)
    sub = exp == 0
    nan = (exp == 0x0F) & (mant == 0x07)
    norm = ~sub & ~nan
    out[sub] = (mant[sub] / 8.0) * (2.0 ** (1 - 7))
    m = 1.0 + mant[norm] / 8.0
    out[norm] = m * (2.0 ** (exp[norm].astype(np.float64) - 7))
    out[nan] = np.nan
    return (sign * out).astype(np.float32)


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        blob = f.read()
    tensors = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        raw = blob[b:e]
        dt, shape = meta["dtype"], meta["shape"]
        if dt == "F32":
            arr = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif dt == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            arr = (u << 16).view(np.float32)
        elif dt == "F8_E4M3":
            arr = e4m3_decode(np.frombuffer(raw, dtype=np.uint8))
        elif dt == "I64":
            arr = np.frombuffer(raw, dtype="<i8")
        else:
            raise ValueError(f"dtype {dt} for {name}")
        tensors[name] = arr.reshape(shape)
    return tensors


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(x)))


def rms0(x, w, eps):
    mean = np.float32(np.mean(x.astype(np.float64) ** 2))
    r = np.float32(1.0 / math.sqrt(mean + eps))
    return x * r * (1.0 + w)


def rms_gated(x, gate, w, eps, sigmoid_gate=True):
    mean = np.float32(np.mean(x.astype(np.float64) ** 2))
    r = np.float32(1.0 / math.sqrt(mean + eps))
    g = sigmoid(gate) if sigmoid_gate else silu(gate)
    return x * r * w * g


def rope(x, rotary_dim, pos, theta):
    x = x.copy()
    half = rotary_dim // 2
    for i in range(half):
        ang = pos / (theta ** ((2 * i) / rotary_dim))
        co, si = math.cos(ang), math.sin(ang)
        a, b = x[i], x[i + half]
        x[i] = a * co - b * si
        x[i + half] = b * co + a * si
    return x


def fp8_block_matmul(x, w, scales):
    """x[I] @ dequant(w[O,I])^T -> y[O];  scales[nblk(O), nblk(I)]."""
    O, I = w.shape
    y = np.zeros(O, dtype=np.float64)
    for o in range(O):
        acc = 0.0
        for b0 in range(0, I, 128):
            b1 = min(b0 + 128, I)
            sc = scales[o // 128, b0 // 128]
            acc += float(np.dot(x[b0:b1], w[o, b0:b1])) * float(sc)
        y[o] = acc
    return y.astype(np.float32)


class Model:
    def __init__(self, cdir):
        cfg = json.load(open(f"{cdir}/config.json"))
        tc = cfg.get("text_config", cfg)
        self.tc = tc
        self.T = load_safetensors(f"{cdir}/model.safetensors")
        self.prefix = "model.language_model" if any(
            k.startswith("model.language_model.") for k in self.T
        ) else "model"
        self.H = tc["hidden_size"]
        self.L = tc["num_hidden_layers"]
        self.C = tc.get("hc_count", 4)
        self.R = tc.get("hc_lowrank", 320)
        self.W = self.C * self.H
        self.eps = float(tc.get("rms_norm_eps", 1e-6))
        rp = tc.get("rope_parameters", {})
        self.theta = float(rp.get("rope_theta", tc.get("rope_theta", 10000)))
        self.partial = float(rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)))
        self.qh = tc["num_attention_heads"]
        self.kvh = tc["num_key_value_heads"]
        self.hd = tc["head_dim"]
        self.rotary = int(self.hd * self.partial)
        self.iq = tc["indexer_n_heads"]
        self.ik = tc["indexer_kv_heads"]
        self.idim = tc["indexer_head_dim"]
        self.ibud = tc["indexer_budget"]
        self.irat = tc["indexer_compress_ratio"]
        self.E = tc["num_experts"]
        self.topk = tc["num_experts_per_tok"]
        self.inter = tc["moe_intermediate_size"]
        self.sinter = tc["shared_expert_intermediate_size"]
        self.norm_topk = bool(tc.get("norm_topk_prob", True))
        self.dnk = tc["linear_num_key_heads"]
        self.dnv = tc["linear_num_value_heads"]
        self.dkd = tc["linear_key_head_dim"]
        self.dvd = tc["linear_value_head_dim"]
        self.dck = tc["linear_conv_kernel_dim"]
        self.dcd = 2 * self.dnk * self.dkd + self.dnv * self.dvd
        self.pdim = tc.get("ple_embed_dim", self.H)
        self.pck = tc.get("ple_conv_kernel_size", 4)
        self.ngram = tc.get("ngram_size", 3)
        self.hpn = tc.get("heads_per_ngram", 8)
        self.nheads = (self.ngram - 1) * self.hpn
        self.ndim = self.pdim // self.nheads
        self.nparts = tc.get("split_ngram_parts", 1)
        self.ple_layer = tc["ple_layer_ids"][0] - 1
        self.layer_types = tc["layer_types"]
        self.eos = tc["eos_token_id"]

    def g(self, name):
        v = self.T.get(f"{self.prefix}.{name}")
        return v if v is not None else self.T[name]

    # ---- gated residual ---------------------------------------------
    def gr_read(self, base, hyper, want_inject):
        H, C, R, W, eps = self.H, self.C, self.R, self.W, self.eps
        norm = np.zeros(W, dtype=np.float32)
        for b in range(C):
            norm[b * H:(b + 1) * H] = rms0(hyper[b * H:(b + 1) * H], self.g(f"{base}.hc_norm.weight")[b * H:(b + 1) * H], eps)
        low = norm @ self.g(f"{base}.input_mix_weight_down.weight").T
        low = silu(low / C)
        mix = low @ self.g(f"{base}.input_mix_weight_up.weight").T
        mixed = np.zeros(H, dtype=np.float32)
        for d in range(H):
            v = 0.0
            for b in range(C):
                v += sigmoid(mix[b * H + d]) * norm[b * H + d]
            mixed[d] = v / C
        inject = None
        if want_inject:
            inject = norm @ self.g(f"{base}.block_inject_weight.weight").T
            inject = 2.0 * sigmoid(inject / C)
        return mixed, inject, norm

    def gr_apply(self, hyper, block, inject):
        H, C = self.H, self.C
        for b in range(C):
            hyper[b * H:(b + 1) * H] += inject[b] * block

    # ---- gated deltanet -------------------------------------------
    def gdn(self, i, x, state):
        H, VH, KH = self.H, self.dnv, self.dnk
        KD, VD, CD, CK = self.dkd, self.dvd, self.dcd, self.dck
        K = KH * KD
        rep = VH // KH
        p = f"layers.{i}.linear_attn"
        qkv = x @ self.g(f"{p}.in_proj_qkv.weight").T
        z = x @ self.g(f"{p}.in_proj_z.weight").T
        bb = x @ self.g(f"{p}.in_proj_b.weight").T
        aa = x @ self.g(f"{p}.in_proj_a.weight").T
        conv_w = self.g(f"{p}.conv1d.weight").reshape(CD, CK)
        ring = state["ring"]
        conv = np.zeros(CD, dtype=np.float32)
        for d in range(CD):
            val = conv_w[d, CK - 1] * qkv[d]
            for tap in range(CK - 1):
                val += conv_w[d, tap] * ring[d, tap]
            conv[d] = silu(val)
            ring[d, :CK - 2] = ring[d, 1:CK - 1]
            ring[d, CK - 2] = qkv[d]
        qi, ki, vi = conv[:K], conv[K:2 * K], conv[2 * K:2 * K + VH * VD]
        dt_bias = self.g(f"{p}.dt_bias.weight") if f"{self.prefix}.{p}.dt_bias.weight" in self.T or f"{p}.dt_bias.weight" in self.T else self.g(f"{p}.dt_bias")
        a_log = self.g(f"{p}.A_log")
        dn_norm = self.g(f"{p}.norm.weight")
        core = np.zeros(VH * VD, dtype=np.float32)
        rec = state["rec"]
        for h in range(VH):
            q = qi[(h // rep) * KD:(h // rep) * KD + KD].astype(np.float32).copy()
            k = ki[(h // rep) * KD:(h // rep) * KD + KD].astype(np.float32).copy()
            qsum = 1e-6 + float(np.dot(q.astype(np.float64), q.astype(np.float64)))
            ksum = 1e-6 + float(np.dot(k.astype(np.float64), k.astype(np.float64)))
            q = q * np.float32(1.0 / math.sqrt(qsum) / math.sqrt(KD))
            k = k * np.float32(1.0 / math.sqrt(ksum))
            v = vi[h * VD:h * VD + VD]
            alpha = math.exp(-math.exp(a_log[h]) * softplus(aa[h] + dt_bias[h]))
            beta = sigmoid(bb[h])
            st = rec[h]
            st *= alpha
            prev = k @ st
            delta = (v - prev) * beta
            st += np.outer(k, delta)
            core[h * VD:h * VD + VD] = q @ st
        norm = np.zeros(VH * VD, dtype=np.float32)
        for h in range(VH):
            norm[h * VD:h * VD + VD] = rms_gated(core[h * VD:h * VD + VD], z[h * VD:h * VD + VD], dn_norm, self.eps, True)
        return norm @ self.g(f"{p}.out_proj.weight").T

    # ---- qwen sparse attention -----------------------------------
    def qsa(self, i, x, pos, cache):
        H, QH, KVH, D = self.H, self.qh, self.kvh, self.hd
        IQ, ID, Rr = self.iq, self.idim, self.irat
        p = f"layers.{i}.self_attn"
        qp = x @ self.g(f"{p}.q_proj.weight").T
        kp = x @ self.g(f"{p}.k_proj.weight").T
        vp = x @ self.g(f"{p}.v_proj.weight").T
        ip = x @ self.g(f"{p}.indexer.index_qk_proj.weight").T
        qn = self.g(f"{p}.q_norm.weight")
        kn = self.g(f"{p}.k_norm.weight")
        iqn = self.g(f"{p}.indexer.q_layernorm.weight")
        ikn = self.g(f"{p}.indexer.k_layernorm.weight")
        for h in range(KVH):
            kh = rms0(kp[h * D:h * D + D], kn, self.eps)
            kh = rope(kh, self.rotary, pos, self.theta)
            cache["k"].setdefault(h, {})[pos] = kh
            cache["v"].setdefault(h, {})[pos] = vp[h * D:h * D + D].copy()
        cache["ik"][pos] = ip[IQ * ID:IQ * ID + ID].copy()
        visible = pos + 1
        blocks = visible // Rr
        tail = blocks * Rr
        qidx = np.zeros((IQ, ID), dtype=np.float32)
        for h in range(IQ):
            qidx[h] = rope(rms0(ip[h * ID:h * ID + ID], iqn, self.eps), self.rotary, pos, self.theta)
        take = min(blocks, self.ibud // Rr)
        scored = []
        for b in range(blocks):
            pool = np.zeros(ID, dtype=np.float32)
            for r in range(Rr):
                pool += cache["ik"][b * Rr + r] / Rr
            pool = rope(rms0(pool, ikn, self.eps), self.rotary, b * Rr, self.theta)
            s = 0.0
            for h in range(IQ):
                a = float(np.dot(qidx[h], pool))
                if a > 0:
                    s += a
            scored.append((s / math.sqrt(ID), b))
        scored.sort(key=lambda t: (-t[0], t[1]))
        selected = []
        for z in range(take):
            for r in range(Rr):
                selected.append(scored[z][1] * Rr + r)
        selected += list(range(tail, visible))
        heads = np.zeros(QH * D, dtype=np.float32)
        for h in range(QH):
            qraw = qp[h * 2 * D:h * 2 * D + 2 * D]
            qhead = rope(rms0(qraw[:D], qn, self.eps), self.rotary, pos, self.theta)
            khidx = h // (QH // KVH)
            sc = np.array([float(np.dot(qhead, cache["k"][khidx][j])) / math.sqrt(D) for j in selected], dtype=np.float64)
            sc = np.exp(sc - sc.max())
            sc /= sc.sum()
            oh = np.zeros(D, dtype=np.float64)
            for w, j in zip(sc, selected):
                oh += w * cache["v"][khidx][j]
            oh = oh.astype(np.float32) * sigmoid(qraw[D:2 * D])
            heads[h * D:h * D + D] = oh
        return heads @ self.g(f"{p}.o_proj.weight").T

    # ---- moe -----------------------------------------------------
    def moe(self, i, x):
        H, E, K = self.H, self.E, self.topk
        p = f"layers.{i}.mlp"
        logits = x @ self.g(f"{p}.gate.weight").T
        ex = np.exp(logits - logits.max())
        allsum = ex.sum()
        idx = []
        for _ in range(K):
            best = -1
            bv = -1.0
            for e in range(E):
                if e in idx:
                    continue
                if ex[e] > bv:
                    bv = ex[e]
                    best = e
            idx.append(best)
        top = sum(ex[e] for e in idx)
        den = top if self.norm_topk else allsum
        gates = [ex[e] / den for e in idx]
        # shared expert
        sg = x @ self.g(f"{p}.shared_expert.gate_proj.weight").T
        su = x @ self.g(f"{p}.shared_expert.up_proj.weight").T
        sh = silu(sg) * su
        shared = sh @ self.g(f"{p}.shared_expert.down_proj.weight").T
        sgate = sigmoid(float(np.dot(x, self.g(f"{p}.shared_expert_gate.weight"))))
        y = np.zeros(H, dtype=np.float32)
        for e, gv in zip(idx, gates):
            eg = self.expert_matmul(i, e, "gate_proj", x)
            eu = self.expert_matmul(i, e, "up_proj", x)
            eh = silu(eg) * eu
            eo = self.expert_matmul(i, e, "down_proj", eh)
            y += gv * eo
        y += sgate * shared
        return y, idx

    def expert_matmul(self, i, e, proj, x):
        p = f"layers.{i}.mlp.experts.{e}.{proj}"
        w = self.g(f"{p}.weight")
        sk = f"{self.prefix}.{p}.weight_scale_inv"
        scales = self.T.get(sk)
        if scales is None:
            scales = self.T.get(f"{p}.weight_scale_inv")
        return fp8_block_matmul(x.astype(np.float32), w.astype(np.float32), scales)

    # ---- ple ---------------------------------------------------
    def hash_row(self, mults, vocab, off, head, cur, p1, p2):
        x = (int(cur) * int(mults[0])) & MASK64
        x ^= (int(p1) * int(mults[1])) & MASK64
        if head >= self.hpn:
            x ^= (int(p2) * int(mults[2])) & MASK64
        sx = x if x < (1 << 63) else x - (1 << 64)
        m = int(vocab[head])
        r = int(math.fmod(sx, m))
        if r < 0:
            r += m
        return int(off[head]) + r

    def ple(self, ids, hyper_all, state):
        i = self.ple_layer
        p = f"layers.{i}.ple"
        H, C, W, PE = self.H, self.C, self.W, self.pdim
        mults = self.g(f"{p}.ple_embedding.layer_multipliers")
        vocab = self.g(f"{p}.ple_embedding.ngram_heads_vocab_sizes")
        off = self.g(f"{p}.ple_embedding.ngram_heads_offsets")
        wscale = 1.0
        ws = self.T.get(f"{self.prefix}.{p}.ple_embedding.ngram_embedding.weight_scale")
        if ws is not None:
            wscale = float(ws.reshape(-1)[0])
        # shard tables
        shards = []
        starts = [0]
        s = 0
        while True:
            t = self.T.get(f"{self.prefix}.{p}.ple_embedding.ngram_embedding.shard_{s}.weight")
            if t is None:
                break
            shards.append(t)
            starts.append(starts[-1] + t.shape[0])
            s += 1
        key_w = self.g(f"{p}.key_proj.weight")
        val_w = self.g(f"{p}.value_proj.weight")
        nk = self.g(f"{p}.norm_key.weight")
        nq = self.g(f"{p}.norm_query.weight")
        nc = self.g(f"{p}.norm_conv.weight")
        conv_w = self.g(f"{p}.conv1d.weight").reshape(W, self.pck)
        SL = (self.pck - 1) * self.ngram
        ring = state["pring"]
        hist = state["hist"]
        S = len(ids)
        out = np.zeros((S, W), dtype=np.float32)
        for si in range(S):
            p1 = hist[-1] if len(hist) >= 1 else self.eos
            p2 = hist[-2] if len(hist) >= 2 else self.eos
            emb = np.zeros(PE, dtype=np.float32)
            for h in range(self.nheads):
                row = self.hash_row(mults, vocab, off, h, ids[si], p1, p2)
                sh = 0
                while sh + 1 < len(starts) - 1 and row >= starts[sh + 1]:
                    sh += 1
                local = row - starts[sh]
                emb[h * self.ndim:h * self.ndim + self.ndim] = shards[sh][local] * wscale
            keys = emb @ key_w.T
            value = emb @ val_w.T
            gated = np.zeros(W, dtype=np.float32)
            norm = np.zeros(W, dtype=np.float32)
            for b in range(C):
                knb = rms0(keys[b * H:b * H + H], nk[b * H:b * H + H], self.eps)
                qnb = rms0(hyper_all[si, b * H:b * H + H], nq[b * H:b * H + H], self.eps)
                dot = float(np.dot(knb, qnb)) / math.sqrt(H)
                shaped = math.copysign(math.sqrt(max(abs(dot), 1e-6)), dot)
                gv = sigmoid(shaped)
                gated[b * H:b * H + H] = gv * value
                norm[b * H:b * H + H] = rms0(gated[b * H:b * H + H], nc[b * H:b * H + H], self.eps)
            for d in range(W):
                a = conv_w[d, self.pck - 1] * norm[d]
                for k in range(self.pck - 1):
                    a += conv_w[d, k] * ring[d, k * self.ngram]
                out[si, d] = gated[d] + silu(a)
                ring[d, :SL - 1] = ring[d, 1:SL]
                ring[d, SL - 1] = norm[d]
            t = ids[si]
            if t == self.eos:
                hist.clear()
            else:
                hist.append(int(t))
                if len(hist) > 2:
                    hist.pop(0)
        return out

    # ---- full forward ----------------------------------------
    def forward(self, ids):
        S = len(ids)
        embed = self.g("embed_tokens.weight")
        hyper = np.zeros((S, self.W), dtype=np.float32)
        for s, t in enumerate(ids):
            row = embed[t].astype(np.float32)
            for b in range(self.C):
                hyper[s, b * self.H:(b + 1) * self.H] = row

        gdn_state = {i: {"rec": np.zeros((self.dnv, self.dkd, self.dvd), dtype=np.float32),
                         "ring": np.zeros((self.dcd, self.dck - 1), dtype=np.float32)}
                     for i in range(self.L) if self.layer_types[i] == "linear_attention"}
        qsa_cache = {i: {"k": {}, "v": {}, "ik": {}}
                     for i in range(self.L) if self.layer_types[i] != "linear_attention"}
        ple_state = {"pring": np.zeros((self.W, (self.pck - 1) * self.ngram), dtype=np.float32), "hist": []}

        for i in range(self.L):
            if i == self.ple_layer:
                hyper += self.ple(ids, hyper, ple_state)
            for s in range(S):
                mixed, inject, _ = self.gr_read(f"layers.{i}.attn_hyper_connection", hyper[s], True)
                if self.layer_types[i] == "linear_attention":
                    block = self.gdn(i, mixed, gdn_state[i])
                else:
                    block = self.qsa(i, mixed, s, qsa_cache[i])
                self.gr_apply(hyper[s], block, inject)
                mixed, inject, _ = self.gr_read(f"layers.{i}.mlp_hyper_connection", hyper[s], True)
                block, _ = self.moe(i, mixed)
                self.gr_apply(hyper[s], block, inject)

        mixed, _, _ = self.gr_read("hyper_connection_mixer", hyper[S - 1], False)
        logits = mixed @ self.T["lm_head.weight"].astype(np.float32).T
        return logits


def main():
    if len(sys.argv) < 3:
        print("usage: qwen38_ref.py <checkpoint_dir> <id,id,...>", file=sys.stderr)
        sys.exit(2)
    cdir = sys.argv[1]
    ids = [int(x) for x in sys.argv[2].replace(",", " ").split()]
    m = Model(cdir)
    logits = m.forward(ids)
    greedy = [int(np.argmax(logits))]
    out = {
        "token_ids": ids,
        "final_logits": [float(x) for x in logits],
        "greedy_next": greedy[0],
        "argmax": int(np.argmax(logits)),
    }
    json.dump(out, sys.stdout)
    print()


if __name__ == "__main__":
    main()
