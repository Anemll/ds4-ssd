#!/usr/bin/env python3
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def pct(values, p):
    if not values:
        return 0
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def load_hist(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "seq": int(row["seq"]),
                "layer": int(row["layer"]),
                "tokens": int(row["tokens"]),
                "refs": int(row["refs"]),
                "unique": int(row["unique"]),
                "expert": int(row["expert"]),
                "expert_refs": int(row["expert_refs"]),
            })
    return rows


def load_qhybrid(paths):
    by_amx = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if "qhybrid" not in row["backend"]:
                    continue
                by_amx[int(row["amx_batch"])].append({
                    "backend": row["backend"],
                    "gpu_batch": int(row["gpu_batch"]),
                    "amx_batch": int(row["amx_batch"]),
                    "ms": float(row["ms"]),
                    "gpu_seq_ms": float(row["gpu_seq_ms"]),
                    "speedup": float(row["speedup_vs_gpu_seq"]),
                })
    return by_amx


def summarize_hist(path):
    rows = load_hist(path)
    refs = [r["expert_refs"] for r in rows]
    by_layer = defaultdict(list)
    seqs = {r["seq"] for r in rows}
    by_layer_expert = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r["expert_refs"])
        by_layer_expert[(r["layer"], r["expert"])].append(r["expert_refs"])

    bins = Counter()
    for v in refs:
        if v < 8:
            bins["01-07"] += 1
        elif v < 16:
            bins["08-15"] += 1
        elif v < 32:
            bins["16-31"] += 1
        elif v < 64:
            bins["32-63"] += 1
        elif v < 128:
            bins["64-127"] += 1
        else:
            bins["128+"] += 1

    total_refs = sum(refs)
    print(f"\n{path.name}")
    approx_chunks = (max(seqs) / len(by_layer)) if rows and by_layer else 0.0
    print(f"  rows={len(rows)} layer_calls={len(seqs)} layers={len(by_layer)} chunks~={approx_chunks:.1f} total_refs={total_refs}")
    print(
        "  expert_refs: "
        f"min={min(refs) if refs else 0} "
        f"p50={pct(refs, 50):.1f} p75={pct(refs, 75):.1f} "
        f"p90={pct(refs, 90):.1f} p95={pct(refs, 95):.1f} "
        f"p99={pct(refs, 99):.1f} max={max(refs) if refs else 0}"
    )
    print(
        "  bins: "
        + " ".join(f"{name}={bins[name]}" for name in ["01-07", "08-15", "16-31", "32-63", "64-127", "128+"])
    )
    eligible16 = sum(1 for v in refs if 16 <= v < 32)
    eligible32 = sum(1 for v in refs if 32 <= v < 64)
    large = sum(1 for v in refs if v >= 64)
    print(f"  AMX-shape candidates: B16={eligible16} B32={eligible32} large_gpu={large}")

    layer_max = [max(vs) for vs in by_layer.values() if vs]
    print(
        "  layer max expert_refs: "
        f"p50={pct(layer_max, 50):.1f} p90={pct(layer_max, 90):.1f} "
        f"p99={pct(layer_max, 99):.1f} max={max(layer_max) if layer_max else 0}"
    )
    repeated = {k: vs for k, vs in by_layer_expert.items() if len(vs) > 1}
    repeated_rows = sum(len(vs) for vs in repeated.values())
    repeated_refs = sum(sum(vs) for vs in repeated.values())
    max_repeats = max((len(vs) for vs in by_layer_expert.values()), default=0)
    print(
        "  chunk-cache locality: "
        f"unique_layer_experts={len(by_layer_expert)} repeated_layer_experts={len(repeated)} "
        f"repeated_rows={repeated_rows} ({(100.0 * repeated_rows / len(rows)) if rows else 0.0:.1f}%) "
        f"repeated_refs={repeated_refs} ({(100.0 * repeated_refs / total_refs) if total_refs else 0.0:.1f}%) "
        f"max_repeats={max_repeats}"
    )
    by_layer_totals = defaultdict(Counter)
    for (layer, expert), vs in by_layer_expert.items():
        by_layer_totals[layer][expert] = sum(vs)
    cover_parts = []
    for k in [1, 2, 4, 8, 16, 32]:
        covered = 0
        for layer, totals in by_layer_totals.items():
            covered += sum(v for _, v in totals.most_common(k))
        cache_gib = len(by_layer_totals) * k * 48.0 / 1024.0
        cover_parts.append(f"top{k}/layer={100.0 * covered / total_refs:.1f}%@{cache_gib:.1f}GiB")
    print("  dense-cache coverage: " + " ".join(cover_parts))


def summarize_qhybrid(by_amx):
    if not by_amx:
        return
    print("\nqhybrid crossover")
    for amx_batch in sorted(by_amx):
        rows = sorted(by_amx[amx_batch], key=lambda r: (r["backend"], r["gpu_batch"]))
        for mode in ["cached", "bgdequant"]:
            mode_rows = [r for r in rows if mode in r["backend"]]
            if not mode_rows:
                continue
            win = next((r for r in mode_rows if r["speedup"] >= 1.05), None)
            best = max(mode_rows, key=lambda r: r["speedup"])
            if win:
                print(
                    f"  AMX B{amx_batch} {mode}: first >=1.05x at GPU B{win['gpu_batch']} "
                    f"({win['speedup']:.3f}x); best GPU B{best['gpu_batch']} ({best['speedup']:.3f}x)"
                )
            else:
                print(
                    f"  AMX B{amx_batch} {mode}: no >=1.05x point; "
                    f"best GPU B{best['gpu_batch']} ({best['speedup']:.3f}x)"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hist", nargs="*", type=Path)
    ap.add_argument("--qhybrid", nargs="*", type=Path, default=[])
    args = ap.parse_args()

    for path in args.hist:
        summarize_hist(path)
    summarize_qhybrid(load_qhybrid(args.qhybrid))


if __name__ == "__main__":
    main()
