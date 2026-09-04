#!/usr/bin/env python3
"""Write test/fixtures/tiny/oracle.json from the NumPy reference forward.

Run after `zig build gen-fixture`:

    python tools/reference/build_oracle.py

The Zig test `model.zig : "matches the NumPy reference oracle"` picks it up
automatically (and is skipped when the file is absent, so `zig build test`
never requires Python).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import qwen38_ref  # noqa: E402
import qwen3moe_ref  # noqa: E402

TOKEN_SETS = [
    [1, 2, 3, 4, 5],
    [7, 7, 2, 0, 9, 3],
    [40],
]

TARGETS = [
    (os.path.join("test", "fixtures", "tiny"), qwen38_ref.Model, "tools/reference/qwen38_ref.py"),
    (os.path.join("test", "fixtures", "tiny-qwen3"), qwen3moe_ref.Model, "tools/reference/qwen3moe_ref.py"),
]


def build(fixture, ModelCls, generator):
    if not os.path.isdir(fixture):
        print(f"{fixture} not found — run `zig build gen-fixture` first", file=sys.stderr)
        return False
    m = ModelCls(fixture)
    cases = []
    for ids in TOKEN_SETS:
        logits = m.forward(ids)
        cases.append({
            "token_ids": ids,
            "final_logits": [float(x) for x in logits],
            "argmax": int(np.argmax(logits)),
        })
    out = {
        "generator": generator,
        "note": "independent NumPy port of the same spec; not the upstream HF model",
        "cases": cases,
    }
    with open(os.path.join(fixture, "oracle.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {fixture}/oracle.json : {len(cases)} cases")
    return True


def main():
    ok = True
    for fixture, ModelCls, generator in TARGETS:
        ok = build(fixture, ModelCls, generator) and ok
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
