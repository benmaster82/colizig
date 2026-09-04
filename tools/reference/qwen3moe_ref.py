#!/usr/bin/env python3
"""Independent NumPy reference forward for Qwen3-MoE (e.g. Qwen3-30B-A3B).

A second implementation of the same math `src/qwen3moe/` ports, written loopy so
a structural bug on either side shows up as a logit mismatch. Not the upstream
`Qwen3MoeForCausalLM` - validates the Zig engine against an independent port of
the same spec. Reads a checkpoint dir (config.json + model.safetensors).

Shares the E4M3 decode, the block-FP8 matmul and RoPE with qwen38_ref.py.
"""

import json
import math

import numpy as np

from qwen38_ref import e4m3_decode, fp8_block_matmul, load_safetensors, rope, silu  # noqa: F401


def rms(x, w, eps):
    """Plain RMSNorm (scale = w, not 1 + w) - Qwen2/Qwen3."""
    mean = np.float32(np.mean(x.astype(np.float64) ** 2))
    r = np.float32(1.0 / math.sqrt(mean + eps))
    return (x * r * w).astype(np.float32)


class Model:
    def __init__(self, cdir):
        cfg = json.load(open(f"{cdir}/config.json"))
        assert cfg["model_type"] == "qwen3_moe", cfg["model_type"]
        self.T = load_safetensors(f"{cdir}/model.safetensors")
        self.H = cfg["hidden_size"]
        self.L = cfg["num_hidden_layers"]
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.theta = float(cfg.get("rope_theta", 1_000_000))
        self.qh = cfg["num_attention_heads"]
        self.kvh = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.grp = self.qh // self.kvh
        self.E = cfg["num_experts"]
        self.topk = cfg["num_experts_per_tok"]
        self.norm_topk = bool(cfg.get("norm_topk_prob", True))
        self.vocab = cfg["vocab_size"]

    def g(self, name):
        return self.T[f"model.{name}"] if f"model.{name}" in self.T else self.T[name]

    def fp8(self, x, prefix):
        return fp8_block_matmul(
            x.astype(np.float64),
            self.g(f"{prefix}.weight"),
            self.g(f"{prefix}.weight_scale_inv"),
        )

    def attn(self, i, n, kv, pos):
        p = f"layers.{i}.self_attn"
        hd, grp = self.hd, self.grp
        q = self.fp8(n, f"{p}.q_proj")
        k = self.fp8(n, f"{p}.k_proj")
        v = self.fp8(n, f"{p}.v_proj")
        qn = self.g(f"{p}.q_norm.weight")
        kn = self.g(f"{p}.k_norm.weight")
        qh = []
        for h in range(self.qh):
            v0 = rope(rms(q[h * hd:(h + 1) * hd], qn, self.eps), hd, pos, self.theta)
            qh.append(v0)
        for h in range(self.kvh):
            k[h * hd:(h + 1) * hd] = rope(rms(k[h * hd:(h + 1) * hd], kn, self.eps), hd, pos, self.theta)
        kv["k"].append(k.copy())
        kv["v"].append(v.copy())

        scale = 1.0 / math.sqrt(hd)
        ao = np.zeros(self.qh * hd, dtype=np.float32)
        L = len(kv["k"])
        for h in range(self.qh):
            kvh = h // grp
            scores = np.zeros(L, dtype=np.float32)
            for t in range(L):
                kt = kv["k"][t][kvh * hd:(kvh + 1) * hd]
                scores[t] = float(np.dot(qh[h], kt)) * scale
            mx = scores.max()
            e = np.exp(scores - mx)
            w = e / float(e.sum())
            oh = np.zeros(hd, dtype=np.float64)
            for t in range(L):
                oh += w[t] * kv["v"][t][kvh * hd:(kvh + 1) * hd]
            ao[h * hd:(h + 1) * hd] = oh
        return self.fp8(ao, f"{p}.o_proj")

    def moe(self, i, n):
        p = f"layers.{i}.mlp"
        logits = np.asarray(self.g(f"{p}.gate.weight"), dtype=np.float32) @ n
        m = logits.max()
        pr = np.exp(logits - m)
        idx = np.argsort(-logits)[: self.topk]
        top = float(pr[idx].sum())
        allsum = float(pr.sum())
        den = top if self.norm_topk else allsum
        acc = np.zeros(self.H, dtype=np.float64)
        for e in idx:
            gp = self.fp8(n, f"{p}.experts.{e}.gate_proj")
            up = self.fp8(n, f"{p}.experts.{e}.up_proj")
            hh = silu(gp) * up
            oe = self.fp8(hh, f"{p}.experts.{e}.down_proj")
            acc += (float(pr[e]) / den) * oe
        return acc.astype(np.float32)

    def forward(self, ids):
        emb = np.asarray(self.g("embed_tokens.weight"), dtype=np.float32)
        hs = [emb[t].astype(np.float32).copy() for t in ids]
        kvs = [{"k": [], "v": []} for _ in range(self.L)]
        for pos, _ in enumerate(ids):
            h = hs[pos]
            for i in range(self.L):
                n = rms(h, self.g(f"layers.{i}.input_layernorm.weight"), self.eps)
                h = h + self.attn(i, n, kvs[i], pos)
                n2 = rms(h, self.g(f"layers.{i}.post_attention_layernorm.weight"), self.eps)
                h = h + self.moe(i, n2)
            hs[pos] = h
        last = rms(hs[-1], self.g("norm.weight"), self.eps)
        lm = np.asarray(self.g("lm_head.weight"), dtype=np.float32)  # [vocab, H]
        return (lm @ last).astype(np.float32)
