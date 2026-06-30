#!/usr/bin/env python3
"""
dspark_condition_parse.py — parse a condition-sweep OUT_DIR and compare kernel
variants across the context axis.

Reads <variant>.r<rep>.log files (DS4_DSPARK_BLOCK_TIMING output), extracts the
per-block (position, verify_ms) curve for each variant, bins by context
position, takes the median across reps, and prints:
  - aggregate gen t/s / verify / acceptance / tau per variant
  - a verify-ms-vs-position table (variants side by side)
  - delta% vs a baseline variant per position bin
  - the break-even position where each variant crosses the baseline

Position is cumulative committed tokens (the spec block emits `committed` tokens
per block), so it tracks the real context the verifier attends.

Usage: dspark_condition_parse.py <out_dir> [baseline_variant] [bin_width]
"""
import re, sys, os, glob, statistics
from collections import defaultdict

BLOCK_RE = re.compile(r'committed=(\d+).*?verify=([\d.]+)\s*ms')
GEN_RE   = re.compile(r'generation:\s*([\d.]+)\s*t/s')
PERF_RE  = re.compile(r'verify=([\d.]+)\s*proposed.*?block=([\d.]+)\s*ms.*?tau=([\d.]+)')
VERIFYMS_RE = re.compile(r'block=[\d.]+\s*ms\s*\(draft=[\d.]+\s*verify=([\d.]+)')
ACC_RE   = re.compile(r'acceptance:\s*([\d.]+)%')

def parse_log(path):
    """Return (curve[(pos,verify_ms)], aggregate dict)."""
    pos = 0
    curve = []
    agg = {}
    for line in open(path, errors='replace'):
        m = BLOCK_RE.search(line)
        if m:
            committed = int(m.group(1)); v = float(m.group(2))
            curve.append((pos, v)); pos += committed
        g = GEN_RE.search(line)
        if g: agg['gen'] = float(g.group(1))
        vm = VERIFYMS_RE.search(line)
        if vm: agg['verify'] = float(vm.group(1))
        p = PERF_RE.search(line)
        if p: agg['tau'] = float(p.group(3))
        a = ACC_RE.search(line)
        if a: agg['acc'] = float(a.group(1))
    return curve, agg

def variant_of(fname):
    # <variant>.r<rep>.log
    base = os.path.basename(fname)
    return re.sub(r'\.r\d+\.log$', '', base)

def binned_median(curves, width):
    """curves: list of curve lists (one per rep). -> {bin: median_verify}."""
    buckets = defaultdict(list)
    for curve in curves:
        for p, v in curve:
            buckets[p // width].append(v)
    return {k: statistics.median(vs) for k, vs in buckets.items()}

def main():
    if len(sys.argv) < 2:
        print("usage: dspark_condition_parse.py <out_dir> [baseline] [bin_width]"); sys.exit(1)
    out_dir = sys.argv[1]
    baseline = sys.argv[2] if len(sys.argv) > 2 else 'strict'
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    logs = sorted(glob.glob(os.path.join(out_dir, '*.log')))
    by_variant_curves = defaultdict(list)
    by_variant_agg = defaultdict(list)
    for log in logs:
        v = variant_of(log)
        curve, agg = parse_log(log)
        if curve: by_variant_curves[v].append(curve)
        if agg: by_variant_agg[v].append(agg)

    variants = sorted(by_variant_curves)
    if not variants:
        print(f"no parseable logs in {out_dir}"); sys.exit(1)

    # Aggregate table
    print("=== aggregate (median across reps) ===")
    print(f"{'variant':<16} {'gen t/s':>8} {'verify':>8} {'acc%':>6} {'tau':>5}")
    def med(vals, key):
        xs = [a[key] for a in vals if key in a]
        return statistics.median(xs) if xs else float('nan')
    for v in variants:
        a = by_variant_agg.get(v, [])
        print(f"{v:<16} {med(a,'gen'):>8.2f} {med(a,'verify'):>8.2f} {med(a,'acc'):>6.1f} {med(a,'tau'):>5.2f}")

    # Per-position binned verify-ms
    binned = {v: binned_median(by_variant_curves[v], width) for v in variants}
    all_bins = sorted(set().union(*[set(b) for b in binned.values()]))

    print(f"\n=== verify ms by context position (bin={width}, median) ===")
    header = f"{'pos':>7}" + "".join(f"{v[:12]:>13}" for v in variants)
    print(header)
    for k in all_bins:
        row = f"{k*width:>7}"
        for v in variants:
            row += f"{binned[v].get(k, float('nan')):>13.1f}"
        print(row)

    if baseline in binned:
        print(f"\n=== delta% vs '{baseline}' (negative = faster) ===")
        others = [v for v in variants if v != baseline]
        print(f"{'pos':>7}" + "".join(f"{v[:12]:>13}" for v in others))
        crossed = {v: None for v in others}
        for k in all_bins:
            base_v = binned[baseline].get(k)
            row = f"{k*width:>7}"
            for v in others:
                cv = binned[v].get(k)
                if base_v and cv is not None:
                    d = 100.0 * (cv - base_v) / base_v
                    row += f"{d:>+12.1f}%"
                    if crossed[v] is None and d < 0:
                        crossed[v] = k * width
                else:
                    row += f"{'-':>13}"
            print(row)
        print(f"\n=== break-even (first position where variant beats '{baseline}') ===")
        for v in others:
            where = crossed[v]
            print(f"  {v:<16} {'always faster (from pos '+str(where)+')' if where==0 else ('@ pos ~'+str(where) if where is not None else 'never faster in range')}")

if __name__ == '__main__':
    main()
