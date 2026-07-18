#!/usr/bin/env python3
"""Compare matching raw-f32 full-vocabulary logit dumps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float64, copy=False)
    m = float(np.max(y))
    return y - (m + math.log(float(np.exp(y - m).sum())))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--per-case", type=Path)
    args = ap.parse_args()

    ref_files = {p.name: p for p in args.reference.glob("*.f32")}
    cand_files = {p.name: p for p in args.candidate.glob("*.f32")}
    names = sorted(ref_files.keys() & cand_files.keys())
    if not names:
        raise SystemExit("no matching .f32 files")

    rows: list[dict[str, float | int | str]] = []
    for name in names:
        ref = np.fromfile(ref_files[name], dtype=np.float32)
        cand = np.fromfile(cand_files[name], dtype=np.float32)
        if ref.size == 0 or ref.shape != cand.shape:
            raise SystemExit(f"shape mismatch for {name}: {ref.shape} vs {cand.shape}")
        if not np.isfinite(ref).all() or not np.isfinite(cand).all():
            raise SystemExit(f"non-finite logits in {name}")

        logp = log_softmax(ref)
        logq = log_softmax(cand)
        p = np.exp(logp)
        q = np.exp(logq)
        mix = 0.5 * (p + q)
        logm = np.log(np.maximum(mix, np.finfo(np.float64).tiny))
        delta = cand.astype(np.float64) - ref.astype(np.float64)
        top_ref = int(np.argmax(ref))
        top_cand = int(np.argmax(cand))
        top5_ref = set(np.argpartition(ref, -5)[-5:].tolist())
        top5_cand = set(np.argpartition(cand, -5)[-5:].tolist())
        rows.append(
            {
                "case": name.removesuffix(".f32"),
                "kl": float(np.sum(p * (logp - logq))),
                "js": float(0.5 * np.sum(p * (logp - logm)) + 0.5 * np.sum(q * (logq - logm))),
                "rms": float(np.sqrt(np.mean(delta * delta))),
                "max_abs": float(np.max(np.abs(delta))),
                "corr": float(np.corrcoef(ref.astype(np.float64), cand.astype(np.float64))[0, 1]),
                "top1": int(top_ref == top_cand),
                "top5_overlap": len(top5_ref & top5_cand),
            }
        )

    if args.per_case:
        args.per_case.parent.mkdir(parents=True, exist_ok=True)
        with args.per_case.open("w", encoding="utf-8") as fp:
            fp.write("case\tkl_ref_candidate\tjs\trms\tmax_abs\tcorr\ttop1\ttop5_overlap\n")
            for row in rows:
                fp.write(
                    f"{row['case']}\t{row['kl']:.9g}\t{row['js']:.9g}\t"
                    f"{row['rms']:.9g}\t{row['max_abs']:.9g}\t{row['corr']:.9g}\t"
                    f"{row['top1']}\t{row['top5_overlap']}\n"
                )

    print(f"cases\t{len(rows)}")
    for key in ("kl", "js", "rms", "max_abs"):
        vals = [float(row[key]) for row in rows]
        print(f"{key}_mean\t{np.mean(vals):.9g}")
        print(f"{key}_median\t{np.median(vals):.9g}")
        print(f"{key}_p95\t{percentile(vals, 95):.9g}")
        print(f"{key}_max\t{max(vals):.9g}")
    print(f"corr_mean\t{np.mean([float(row['corr']) for row in rows]):.9g}")
    print(f"corr_min\t{min(float(row['corr']) for row in rows):.9g}")
    print(f"top1_match_rate\t{np.mean([int(row['top1']) for row in rows]):.9g}")
    print(f"top5_overlap_mean\t{np.mean([int(row['top5_overlap']) for row in rows]):.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
