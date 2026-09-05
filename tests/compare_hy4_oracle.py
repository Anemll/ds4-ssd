#!/usr/bin/env python3
"""Compare a no-warmup llama-cli oracle capture with native HY4 JSONL.

Both runs must use the same model, raw prompt, F32 caches, greedy sampling,
and eight routed experts. Source --no-warmup makes eval_index 0 the prompt.
The native session harness emits additional lifecycle checks after decode.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path, help="llama-cli oracle manifest.json")
parser.add_argument("native", type=Path, help="test_hy4_session JSONL output")
parser.add_argument("--atol", type=float, default=0.001)
args = parser.parse_args()
ref = json.loads(args.reference.read_text())
rows = [json.loads(line) for line in args.native.read_text().splitlines() if line.strip()]
prompt = next(row for row in rows if row.get("stage") == "raw_prompt")
assert prompt["token_ids"] == ref["prompt_ids"], "prompt token IDs differ"
native = [row for row in rows if row.get("stage") == "prefill" or
          row.get("stage", "").startswith("decode_")]
assert ref["logits"] and len(native) >= len(ref["logits"]), "missing evaluation records"
maximum = 0.0
values = 0
for index, expected in enumerate(ref["logits"]):
    actual = native[index]
    assert expected["eval_index"] == index, "reference must use --no-warmup"
    top = actual["top16"]
    count = len(expected["token_ids"])
    assert count <= len(top), "reference top-k exceeds native capture"
    ids = [item["id"] for item in top[:count]]
    assert ids == expected["token_ids"], f"top IDs differ at evaluation {index}"
    for item, logit in zip(top, expected["logits"]):
        delta = abs(item["logit"] - logit)
        assert delta <= args.atol, f"logit mismatch eval={index} token={item['id']}: {delta}"
        maximum = max(maximum, delta)
        values += 1
print(f"PASS: {len(ref['logits'])} HY4 greedy predictions, {values} top-logit IDs; "
      f"max_abs_delta={maximum:.9g}, tolerance={args.atol:g}")
