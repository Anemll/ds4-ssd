#!/usr/bin/env python3
import argparse
import re
import subprocess
import time
from pathlib import Path


def run_cmd(name, cmd, out_path):
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - t0
    out_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed with rc={proc.returncode}; see {out_path}")
    return elapsed, proc.stdout


def run_concurrent(gpu_cmd, ane_cmd, gpu_out, ane_out):
    t0 = time.perf_counter()
    gpu = subprocess.Popen(gpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ane = subprocess.Popen(ane_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    gpu_stdout, _ = gpu.communicate()
    ane_stdout, _ = ane.communicate()
    elapsed = time.perf_counter() - t0
    gpu_out.write_text(gpu_stdout)
    ane_out.write_text(ane_stdout)
    if gpu.returncode != 0:
        raise SystemExit(f"concurrent gpu failed with rc={gpu.returncode}; see {gpu_out}")
    if ane.returncode != 0:
        raise SystemExit(f"concurrent ane failed with rc={ane.returncode}; see {ane_out}")
    return elapsed, gpu_stdout, ane_stdout


def parse_gpu_ms(text):
    rows = []
    for line in text.splitlines():
        if line.startswith("gpu_ds4_moe_block,"):
            parts = line.split(",")
            rows.append(float(parts[6]))
    return rows[-1] if rows else None


def parse_ane_ms(text):
    matches = re.findall(r"eval \(B=.*?\):\s+([0-9.]+) ms/iter", text)
    return float(matches[-1]) if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-batch", type=int, default=2048)
    ap.add_argument("--ane-batch", type=int, default=64)
    ap.add_argument("--gpu-iters", type=int, default=500)
    ap.add_argument("--ane-iters", type=int, default=250)
    ap.add_argument("--out-dir", type=Path, default=Path("moe-batch-bench"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = args.out_dir
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    gpu_cmd = [
        str(root / "moe-batch-bench" / "moe-batch-bench"),
        "--backend", "ds4",
        "--batches", str(args.gpu_batch),
        "--iters", str(args.gpu_iters),
    ]
    ane_cmd = [
        str(root / "moe-batch-bench" / "ane_ds4_mlp_inmem_bench_packed_split3"),
        "-bench-shape", "7168", "18432", "3", str(args.ane_batch), "5", str(args.ane_iters),
    ]

    gpu_s, gpu_text = run_cmd("solo gpu", gpu_cmd, out / "ane_gpu_probe_gpu_solo.log")
    ane_s, ane_text = run_cmd("solo ane", ane_cmd, out / "ane_gpu_probe_ane_solo.log")
    both_s, gpu_both_text, ane_both_text = run_concurrent(
        gpu_cmd,
        ane_cmd,
        out / "ane_gpu_probe_gpu_concurrent.log",
        out / "ane_gpu_probe_ane_concurrent.log",
    )

    gpu_ms = parse_gpu_ms(gpu_text)
    ane_ms = parse_ane_ms(ane_text)
    gpu_both_ms = parse_gpu_ms(gpu_both_text)
    ane_both_ms = parse_ane_ms(ane_both_text)
    expected_seq = gpu_s + ane_s
    overlap = expected_seq / both_s if both_s else 0.0
    ideal = max(gpu_s, ane_s)
    ideal_eff = ideal / both_s if both_s else 0.0

    summary = (
        "ANE+GPU concurrency probe\n"
        f"gpu_cmd={' '.join(gpu_cmd)}\n"
        f"ane_cmd={' '.join(ane_cmd)}\n"
        f"solo_gpu_wall_s={gpu_s:.3f} gpu_ms={gpu_ms}\n"
        f"solo_ane_wall_s={ane_s:.3f} ane_ms={ane_ms}\n"
        f"concurrent_wall_s={both_s:.3f} gpu_concurrent_ms={gpu_both_ms} ane_concurrent_ms={ane_both_ms}\n"
        f"sequential_wall_s={expected_seq:.3f}\n"
        f"overlap_speedup_vs_sequential={overlap:.3f}\n"
        f"efficiency_vs_ideal_max_wall={ideal_eff:.3f}\n"
    )
    (out / "ane_gpu_probe_summary.txt").write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
