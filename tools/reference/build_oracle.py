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
from qwen38_ref import Model  # noqa: E402

FIXTURE = os.path.join("test", "fixtures", "tiny")
TOKEN_SETS = [
    [1, 2, 3, 4, 5],
    [7, 7, 2, 0, 9, 3],
    [40],
]


def main():
    if not os.path.isdir(FIXTURE):
        print(f"{FIXTURE} not found — run `zig build gen-fixture` first", file=sys.stderr)
        sys.exit(1)
    m = Model(FIXTURE)
    cases = []
    for ids in TOKEN_SETS:
        logits = m.forward(ids)
        cases.append({
            "token_ids": ids,
            "final_logits": [float(x) for x in logits],
            "argmax": int(np.argmax(logits)),
        })
    out = {
        "generator": "tools/reference/qwen38_ref.py",
        "note": "independent NumPy port of the same spec; not the upstream Qwen4ExpForCausalLM",
        "cases": cases,
    }
    with open(os.path.join(FIXTURE, "oracle.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {FIXTURE}/oracle.json : {len(cases)} cases")


if __name__ == "__main__":
    main()
