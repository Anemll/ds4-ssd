#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PROMPT = (
    "Continue the sequence forever. Output only comma-separated integers "
    "starting at 1. Do not explain. Do not stop until the token limit."
)


TURN_STATS_RE = re.compile(
    r"ds4-agent: turn-stats round=(?P<round>\d+) generated=(?P<generated>\d+) "
    r"max=(?P<max>\d+) decode_s=(?P<decode_s>[0-9.]+) "
    r"gen_tps=(?P<gen_tps>[0-9.]+) ctx=(?P<ctx>\d+) stop=(?P<stop>\S+)"
)


def cases_for_suite(suite):
    base = [
        ("baseline_slot32", {}, []),
        ("slot16", {}, ["--moe-slot-bank", "16"]),
        ("slot24", {}, ["--moe-slot-bank", "24"]),
        ("slot48", {}, ["--moe-slot-bank", "48"]),
        ("slot64", {}, ["--moe-slot-bank", "64"]),
        ("ssd16", {}, ["--ssd-cache", "16GB"]),
        ("ssd24", {}, ["--ssd-cache", "24GB"]),
        ("ssd32", {}, ["--ssd-cache", "32GB"]),
        ("ssd48", {}, ["--ssd-cache", "48GB"]),
        ("ssd64", {}, ["--ssd-cache", "64GB"]),
        ("ssd_auto", {}, ["--ssd-cache", "auto"]),
        ("ssd48_no_auto_per_slot", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "48GB"]),
        ("ssd64_no_auto_per_slot", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "64GB"]),
        ("mixed_slots6_grouped", {"DS4_FLASH_MOE_MIXED_SLOTS6_GROUPED": "1"}, []),
        ("decode_prefetch", {"DS4_FLASH_MOE_DECODE_PREFETCH": "1"}, []),
        ("decode_prefetch_max4", {"DS4_FLASH_MOE_DECODE_PREFETCH": "1", "DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS": "4"}, []),
        ("io_split1", {}, ["--moe-cache-io-split", "1"]),
        ("io_split8", {}, ["--moe-cache-io-split", "8"]),
        ("direct_slot_pread0", {"DS4_FLASH_MOE_DIRECT_SLOT_PREAD": "0"}, []),
    ]
    if suite == "coarse":
        return base
    if suite == "focus":
        return [
            ("ssd48_no_auto", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "48GB"]),
            ("ssd56_no_auto", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "56GB"]),
            ("ssd64_no_auto", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "64GB"]),
            ("ssd72_no_auto", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "72GB"]),
            ("ssd80_no_auto", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "80GB"]),
            ("ssd64_no_auto_io2", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "64GB", "--moe-cache-io-split", "2"]),
            ("ssd64_no_auto_io8", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "64GB", "--moe-cache-io-split", "8"]),
            ("ssd64_no_auto_io16", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "64GB", "--moe-cache-io-split", "16"]),
            ("ssd64_no_auto_prefetch", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_DECODE_PREFETCH": "1"}, ["--ssd-cache", "64GB"]),
            ("ssd64_no_auto_prefetch_max2", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_DECODE_PREFETCH": "1", "DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS": "2"}, ["--ssd-cache", "64GB"]),
            ("ssd64_no_auto_shared_down0", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_DECODE_PREFETCH_SHARED_DOWN": "0"}, ["--ssd-cache", "64GB"]),
            ("ssd64_no_auto_direct_pread0", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_DIRECT_SLOT_PREAD": "0"}, ["--ssd-cache", "64GB"]),
            ("ssd64_no_auto_slotwise", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_SLOTWISE_DECODE": "1"}, ["--ssd-cache", "64GB"]),
            ("ssd64_no_auto_baked_slot", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_SLOTWISE_DECODE": "1", "DS4_FLASH_MOE_BAKED_SLOT_DECODE": "1"}, ["--ssd-cache", "64GB"]),
        ]
    if suite == "final":
        return [
            ("ssd64_no_auto", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1"}, ["--ssd-cache", "64GB"]),
            ("ssd64_no_auto_slotwise", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_SLOTWISE_DECODE": "1"}, ["--ssd-cache", "64GB"]),
            ("ssd64_no_auto_baked_slot", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_SLOTWISE_DECODE": "1", "DS4_FLASH_MOE_BAKED_SLOT_DECODE": "1"}, ["--ssd-cache", "64GB"]),
            ("ssd64_no_auto_slotwise_io8", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_SLOTWISE_DECODE": "1"}, ["--ssd-cache", "64GB", "--moe-cache-io-split", "8"]),
            ("ssd64_no_auto_slotwise_io16", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_SLOTWISE_DECODE": "1"}, ["--ssd-cache", "64GB", "--moe-cache-io-split", "16"]),
            ("ssd64_no_auto_slotwise_prefetch", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_SLOTWISE_DECODE": "1", "DS4_FLASH_MOE_DECODE_PREFETCH": "1"}, ["--ssd-cache", "64GB"]),
            ("ssd64_no_auto_slotwise_shared_down0", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_SLOTWISE_DECODE": "1", "DS4_FLASH_MOE_DECODE_PREFETCH_SHARED_DOWN": "0"}, ["--ssd-cache", "64GB"]),
            ("ssd72_mixed_forced", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO": "1"}, ["--ssd-cache", "72GB"]),
            ("ssd72_mixed_forced_slotwise", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO": "1", "DS4_FLASH_MOE_SLOTWISE_DECODE": "1"}, ["--ssd-cache", "72GB"]),
            ("ssd80_mixed_forced", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO": "1"}, ["--ssd-cache", "80GB"]),
            ("ssd80_mixed_forced_slotwise", {"DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS": "1", "DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO": "1", "DS4_FLASH_MOE_SLOTWISE_DECODE": "1"}, ["--ssd-cache", "80GB"]),
        ]
    raise ValueError(f"unknown suite: {suite}")


def parse_first(pattern, text):
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def run_case(args, out_dir, name, case_env, extra_args, repeat):
    log_path = out_dir / f"{repeat:02d}_{name}.log"
    env = os.environ.copy()
    env.update({
        "DS4_LOCK_FILE": f"/tmp/ds4-agent-m5max-{os.getpid()}-{repeat}-{name}.lock",
        "DS4_AGENT_TURN_STATS": "1",
        "DS4_MXFP4_NATIVE": "1",
        "DS4_FLASH_MOE_ANE_PREFILL": "1",
        "DS4_METAL_PREFILL_CHUNK": "4096",
        "DS4_FLASH_MOE_RESIDENCY_STATS": "0",
    })
    env.update(case_env)
    cmd = [
        "./ds4-agent",
        "--non-interactive",
        "-m",
        args.model,
        "--ctx",
        str(args.ctx),
        "-n",
        str(args.tokens),
        "--temp",
        "0",
        "--nothink",
        *extra_args,
        "-p",
        args.prompt,
    ]
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=args.cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=args.timeout,
    )
    wall_s = time.monotonic() - t0
    log_path.write_text(proc.stdout)
    stats = None
    for match in TURN_STATS_RE.finditer(proc.stdout):
        stats = match.groupdict()
    slots = parse_first(r"slot banks allocated: layers=\d+ slots=(\d+)", proc.stdout)
    layout = parse_first(r"slot bank layout: ([^\n]+)", proc.stdout)
    decode_io = parse_first(r"ds4: decode  I/O: ([^\n]+)", proc.stdout)
    path = parse_first(r"prefill compute: routed experts = ([^|]+)", proc.stdout).strip()
    native = "decode2=pair/sum6" in proc.stdout
    engaged = "two-dispatch plane decode arm engaged" in proc.stdout
    row = {
        "name": name,
        "repeat": repeat,
        "returncode": proc.returncode,
        "wall_s": f"{wall_s:.3f}",
        "generated": stats["generated"] if stats else "",
        "max": stats["max"] if stats else "",
        "decode_s": stats["decode_s"] if stats else "",
        "gen_tps": stats["gen_tps"] if stats else "",
        "ctx_end": stats["ctx"] if stats else "",
        "stop": stats["stop"] if stats else "",
        "slots": slots,
        "layout": layout,
        "path": path,
        "decode2": "1" if native else "0",
        "two_dispatch": "1" if engaged else "0",
        "decode_io": decode_io,
        "env": " ".join(f"{k}={v}" for k, v in sorted(case_env.items())),
        "args": " ".join(extra_args),
        "log": str(log_path),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["coarse", "focus", "final"], default="coarse")
    parser.add_argument("--tokens", type=int, default=600)
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--model", default="/Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("bench-results") / f"ds4-agent-m5max-{args.suite}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    fieldnames = [
        "name", "repeat", "returncode", "wall_s", "generated", "max",
        "decode_s", "gen_tps", "ctx_end", "stop", "slots", "layout", "path",
        "decode2", "two_dispatch", "decode_io", "env", "args", "log",
    ]
    rows = []
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for repeat in range(args.repeats):
            for name, env, extra in cases_for_suite(args.suite):
                print(f"RUN {name} repeat={repeat}", flush=True)
                row = run_case(args, out_dir, name, env, extra, repeat)
                rows.append(row)
                writer.writerow(row)
                fp.flush()
                print(
                    f"DONE {name}: tps={row['gen_tps']} generated={row['generated']} "
                    f"slots={row['slots']} layout={row['layout']} stop={row['stop']}",
                    flush=True,
                )
    ranked = sorted(
        [r for r in rows if r["gen_tps"]],
        key=lambda r: float(r["gen_tps"]),
        reverse=True,
    )
    print(f"\nresults: {csv_path}")
    for row in ranked[:10]:
        print(
            f"{row['gen_tps']:>8}  {row['name']:<28} slots={row['slots']:<4} "
            f"layout={row['layout']} args={row['args']} env={row['env']}"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.TimeoutExpired as exc:
        print(f"timeout: {exc}", file=sys.stderr)
        sys.exit(124)
