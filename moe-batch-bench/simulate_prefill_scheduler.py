#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_latency_table(paths, backend_prefix=None, dtype_contains=None):
    table = {}
    for path in paths:
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if backend_prefix and not row["backend"].startswith(backend_prefix):
                    continue
                if dtype_contains and dtype_contains not in row["dtype"]:
                    continue
                table[int(row["batch"])] = float(row["ms"])
    return dict(sorted(table.items()))


def interp(table, x):
    if not table:
        raise SystemExit("empty latency table")
    xs = sorted(table)
    if x <= xs[0]:
        return table[xs[0]]
    if x >= xs[-1]:
        # Extrapolate with the last slope instead of clamping large hot experts.
        x0, x1 = xs[-2], xs[-1]
        y0, y1 = table[x0], table[x1]
        return y1 + (x - x1) * (y1 - y0) / (x1 - x0)
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = table[x0], table[x1]
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    return table[xs[-1]]


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


def build_topk_cache(rows, topk):
    if topk <= 0:
        return set()
    totals = defaultdict(Counter)
    for r in rows:
        totals[r["layer"]][r["expert"]] += r["expert_refs"]
    cached = set()
    for layer, c in totals.items():
        for expert, _ in c.most_common(topk):
            cached.add((layer, expert))
    return cached


def build_first_chunk_cache(rows, topk):
    if topk <= 0:
        return set()
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["seq"]].append(r)
    first_by_layer = {}
    for seq in sorted(by_seq):
        layer = by_seq[seq][0]["layer"]
        if layer not in first_by_layer:
            first_by_layer[layer] = by_seq[seq]
    cached = set()
    for layer, layer_rows in first_by_layer.items():
        c = Counter()
        for r in layer_rows:
            c[r["expert"]] += r["expert_refs"]
        for expert, _ in c.most_common(topk):
            cached.add((layer, expert))
    return cached


def simulate(rows, gpu_table, amx_table, topk, amx_min, amx_max, policy, dequant_ms):
    if policy == "oracle-preload":
        target_cache = build_topk_cache(rows, topk)
        materialized = set(target_cache)
    elif policy == "oracle-fill":
        target_cache = build_topk_cache(rows, topk)
        materialized = set()
    elif policy == "first-fill":
        target_cache = build_first_chunk_cache(rows, topk)
        materialized = set()
    elif policy == "dynamic-fill":
        target_cache = {(r["layer"], r["expert"]) for r in rows}
        materialized = set()
    else:
        raise SystemExit(f"unknown policy: {policy}")

    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["seq"]].append(r)

    baseline_ms = 0.0
    hybrid_ms = 0.0
    amx_rows = 0
    amx_refs = 0
    gpu_rows = 0
    gpu_refs = 0
    layer_call_count = 0
    fill_count = 0
    fill_ms = 0.0

    for seq in sorted(by_seq):
        layer_call_count += 1
        gpu_ms = 0.0
        amx_ms = 0.0
        fills_this_call = 0
        for r in by_seq[seq]:
            b = r["expert_refs"]
            baseline_ms += interp(gpu_table, b)
            key = (r["layer"], r["expert"])
            is_cached = key in materialized
            if is_cached and amx_min <= b <= amx_max:
                amx_ms += interp(amx_table, b)
                amx_rows += 1
                amx_refs += b
            else:
                gpu_ms += interp(gpu_table, b)
                gpu_rows += 1
                gpu_refs += b
            if key in target_cache and key not in materialized:
                fills_this_call += 1
                materialized.add(key)
        if fills_this_call:
            fill_count += fills_this_call
            fill_ms += fills_this_call * dequant_ms
            gpu_ms += fills_this_call * dequant_ms
        hybrid_ms += max(gpu_ms, amx_ms)

    tokens = sum(r["tokens"] for r in by_seq.values() for r in r[:1])
    # Each seq is one layer call; tokens repeat per layer. Use unique chunk token
    # counts once per 43-layer cycle when possible.
    layers = len({r["layer"] for r in rows}) or 43
    tokens = sum(by_seq[seq][0]["tokens"] for seq in sorted(by_seq) if (seq - 1) % layers == 0)
    return {
        "topk": topk,
        "amx_min": amx_min,
        "amx_max": amx_max,
        "policy": policy,
        "tokens": tokens,
        "layer_calls": layer_call_count,
        "target_cache": len(target_cache),
        "cache_gib": len(target_cache) * 48.0 / 1024.0,
        "fill_count": fill_count,
        "fill_ms": fill_ms,
        "baseline_ms": baseline_ms,
        "hybrid_ms": hybrid_ms,
        "speedup": baseline_ms / hybrid_ms if hybrid_ms else 0.0,
        "baseline_tps": tokens * 1000.0 / baseline_ms if baseline_ms else 0.0,
        "hybrid_tps": tokens * 1000.0 / hybrid_ms if hybrid_ms else 0.0,
        "amx_rows": amx_rows,
        "amx_refs": amx_refs,
        "gpu_rows": gpu_rows,
        "gpu_refs": gpu_refs,
    }


def print_result(name, r):
    cache = f"{r['policy']}:top{r['topk']}" if r["topk"] else r["policy"]
    print(
        f"{name},{cache},B{r['amx_min']}-{r['amx_max']},"
        f"{r['baseline_tps']:.2f},{r['hybrid_tps']:.2f},{r['speedup']:.3f},"
        f"{r['target_cache']},{r['cache_gib']:.2f},{r['fill_count']},{r['fill_ms']:.1f},"
        f"{r['amx_rows']},{r['amx_refs']},{r['gpu_rows']},{r['gpu_refs']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hist", nargs="+", type=Path)
    ap.add_argument("--gpu", nargs="+", type=Path, required=True)
    ap.add_argument("--amx", nargs="+", type=Path, required=True)
    ap.add_argument("--dequant-ms", type=float, default=0.96)
    args = ap.parse_args()

    gpu = read_latency_table(args.gpu, "gpu_ds4_moe_block")
    amx = read_latency_table(args.amx, "amx_bnns_quant", "cached")
    print("prompt,cache,amx_range,baseline_tps_est,hybrid_tps_est,speedup,target_cache,cache_gib,fill_count,fill_ms,amx_rows,amx_refs,gpu_rows,gpu_refs")
    for path in args.hist:
        rows = load_hist(path)
        name = path.stem.replace("prompt_hist_", "")
        for policy in ["first-fill", "oracle-fill", "oracle-preload"]:
            for topk in [4, 8, 16, 32]:
                for amx_min, amx_max in [(16, 31), (32, 63), (32, 127), (16, 127)]:
                    r = simulate(rows, gpu, amx, topk, amx_min, amx_max, policy, args.dequant_ms)
                    print_result(name, r)
        for amx_min, amx_max in [(16, 31), (32, 63), (32, 127), (16, 127)]:
            r = simulate(rows, gpu, amx, 0, amx_min, amx_max, "dynamic-fill", args.dequant_ms)
            print_result(name, r)


if __name__ == "__main__":
    main()
