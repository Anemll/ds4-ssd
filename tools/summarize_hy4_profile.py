#!/usr/bin/env python3
"""Summarize decode-only DS4_HY4_PROFILE JSON and optional Flash-MoE layer logs."""
import argparse
import json
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("log", type=Path)
parser.add_argument("--decode-start", type=int, required=True,
                    help="prefilled transcript length from the agent trace")
parser.add_argument("--slots", type=int, help="assert the measured slot capacity")
args = parser.parse_args()
rows, layers = [], []
pattern = re.compile(r"Flash-MoE layer=(\d+) sync=([\d.]+) ms remap/install=([\d.]+) ms misses=(\d+)")
for line in args.log.read_text().splitlines():
    match = pattern.search(line)
    if match:
        layers.append(dict(layer=int(match[1]), sync_ms=float(match[2]),
                           install_ms=float(match[3]), misses=int(match[4])))
    if line.startswith("HY4_PROFILE "):
        row = json.loads(line[len("HY4_PROFILE "):])
        row["layer_logs"] = layers
        layers = []
        rows.append(row)
rows = [row for row in rows if row["pos"] >= args.decode_start]
if not rows:
    parser.error("no completed decode token profiles at/after --decode-start")
for row in rows:
    if row["topk"] != 8 or (args.slots is not None and row["slots"] != args.slots):
        parser.error("unexpected expert count or slot capacity")
    if row["hits"] + row["misses"] != 77 * 8:
        parser.error("incomplete routed cache-reference accounting")
    if row["routed_path"] == "per_expert" and (
            row["routed_quant_dispatches"] != 77 * 8 * 3 or
            row["routed_swiglu_dispatches"] != 77 * 8):
        parser.error("incomplete per-expert dispatch accounting")
    fused=row.get("routed_fused_dispatches",0)
    if row["routed_path"] == "fused_top8" and (fused != 77*2 or
            any(row[key] for key in ("routed_quant_dispatches", "routed_swiglu_dispatches", "routed_reduce_dispatches"))):
        parser.error("incomplete fused top-8 dispatch accounting")
    if fused%2 or row["routed_quant_dispatches"]//24+fused//2 != 77:
        parser.error("incomplete routed layer accounting")
    row["routed_fused_dispatches"]=fused
metrics = ("wall_ms", "worker_cpu_ms", "gpu_ms", "attn_gpu_ms", "ffn_gpu_ms",
           "router_gpu_ms", "ihc_pre_cpu_ms", "ihc_post_cpu_ms", "ihc_head_cpu_ms",
           "router_install_wall_ms", "hits", "misses", "installed_bytes",
           "routed_quant_dispatches", "routed_swiglu_dispatches", "routed_reduce_dispatches", "routed_fused_dispatches")
summary = {key: statistics.mean(row[key] for row in rows) for key in metrics}
summary.update(completed_decode_tokens=len(rows), first_position=rows[0]["pos"],
               last_position=rows[-1]["pos"], slots=sorted({row["slots"] for row in rows}),
               topk=8, routed_paths=sorted({row["routed_path"] for row in rows}), hit_rate=sum(row["hits"] for row in rows) / (len(rows) * 77 * 8))
if all(len(row["layer_logs"]) == 77 for row in rows):
    summary["install_wall_ms"] = statistics.mean(
        sum(layer["install_ms"] for layer in row["layer_logs"]) for row in rows)
    summary["router_sync_wall_ms"] = statistics.mean(
        sum(layer["sync_ms"] for layer in row["layer_logs"]) for row in rows)
summary["timing_note"] = "Per-token means. GPU phase rows are included in gpu_ms; iHC CPU rows are included in worker_cpu_ms. Install wall includes host work. Do not add overlapping clocks."
print(json.dumps(summary, indent=2))
