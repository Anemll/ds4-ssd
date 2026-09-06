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
parser.add_argument("--native-reference", action="store_true",
                    help="reference is another native JSONL capture (CPU/Metal A/B)")
args = parser.parse_args()
rows = [json.loads(line) for line in args.native.read_text().splitlines() if line.strip()]
prompt = next(row for row in rows if row.get("stage") == "raw_prompt")
def evaluations(records):
    return [row for row in records if "top16" in row and
            (row.get("stage") == "prefill" or row.get("stage", "").startswith("decode_"))]

native = evaluations(rows)
if args.native_reference:
    reference_rows = [json.loads(line) for line in args.reference.read_text().splitlines() if line.strip()]
    expected_rows = evaluations(reference_rows)
    assert len(native) == len(expected_rows) and native, "evaluation counts differ"
    allocations = [next(row for row in records if row.get("stage") == "allocation")
                   for records in (rows, reference_rows)]
    assert allocations[0]["ctx"] == allocations[1]["ctx"], "native context sizes differ"
    for records in (rows, reference_rows):
        assert records[-1].get("result") == "PASS", "incomplete or failed native run"
        allocation = next(row for row in records if row.get("stage") == "allocation")
        assert allocation["slots"] == 8 and allocation["capacity"] == 8, "requires eight slots"
    assert next(row for row in rows if row.get("stage") == "greedy_tokens")["token_ids"] == \
           next(row for row in reference_rows if row.get("stage") == "greedy_tokens")["token_ids"], "generated tokens differ"
    for actual, expected in zip(native, expected_rows):
        assert (actual["stage"], actual["position"], actual["token_ids"], actual["argmax"]) == \
               (expected["stage"], expected["position"], expected["token_ids"], expected["argmax"]), "native token history/prediction differs"
    ref = {
        "prompt_ids": next(row for row in reference_rows if row.get("stage") == "raw_prompt")["token_ids"],
        "logits": [{"eval_index": i, "token_ids": [v["id"] for v in row["top16"]],
                    "logits": [v["logit"] for v in row["top16"]]} for i, row in enumerate(expected_rows)],
    }
else:
    ref = json.loads(args.reference.read_text())
assert prompt["token_ids"] == ref["prompt_ids"], "prompt token IDs differ"
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
