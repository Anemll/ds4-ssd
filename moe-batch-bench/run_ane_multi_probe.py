#!/usr/bin/env python3
import argparse
import re
import subprocess
import time
from pathlib import Path


def run_one(cmd, log):
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - t0
    log.write_text(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"command failed rc={proc.returncode}; see {log}")
    return elapsed, proc.stdout


def run_many(cmds, logs):
    t0 = time.perf_counter()
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for cmd in cmds]
    outputs = []
    for proc, log in zip(procs, logs):
        text, _ = proc.communicate()
        log.write_text(text)
        outputs.append(text)
        if proc.returncode != 0:
            raise SystemExit(f"command failed rc={proc.returncode}; see {log}")
    return time.perf_counter() - t0, outputs


def parse_ms(text):
    matches = re.findall(r"eval \(B=.*?\):\s+([0-9.]+) ms/iter", text)
    return float(matches[-1]) if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ane-batch", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out-dir", type=Path, default=Path("moe-batch-bench"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = args.out_dir
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        str(root / "moe-batch-bench" / "ane_ds4_mlp_inmem_bench_packed_split3"),
        "-bench-shape", "7168", "18432", "3", str(args.ane_batch), "5", str(args.iters),
    ]
    solo_s, solo_text = run_one(base_cmd, out / "ane_multi_probe_solo.log")
    cmds = [base_cmd for _ in range(args.jobs)]
    logs = [out / f"ane_multi_probe_parallel_{i}.log" for i in range(args.jobs)]
    parallel_s, outputs = run_many(cmds, logs)
    solo_ms = parse_ms(solo_text)
    parallel_ms = [parse_ms(text) for text in outputs]
    seq_s = solo_s * args.jobs
    speedup = seq_s / parallel_s if parallel_s else 0.0
    summary = (
        "ANE multi-job probe\n"
        f"jobs={args.jobs} ane_batch={args.ane_batch} iters={args.iters}\n"
        f"solo_wall_s={solo_s:.3f} solo_ms={solo_ms}\n"
        f"parallel_wall_s={parallel_s:.3f} parallel_ms={parallel_ms}\n"
        f"sequential_wall_s={seq_s:.3f}\n"
        f"speedup_vs_sequential={speedup:.3f}\n"
    )
    (out / "ane_multi_probe_summary.txt").write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
