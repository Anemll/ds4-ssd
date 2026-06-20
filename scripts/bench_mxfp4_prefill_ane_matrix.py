#!/usr/bin/env python3
"""Run MXFP4 sidecar prefill chunk/ANE matrix with ds4-bench."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import pathlib
import statistics
import subprocess
import sys
from dataclasses import dataclass


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = pathlib.Path("/Users/anemll/Models/DSv4-Flash-MXFP4-native-flash")
DEFAULT_PROMPTS = [
    ROOT / "tests/long_context_story_prompt.txt",
    ROOT / "tests/long_context_security_prompt.txt",
    ROOT / "speed-bench/promessi_sposi.txt",
]
DEFAULT_CHUNKS = [1024, 2048, 4096, 8192, 16384]


@dataclass(frozen=True)
class RunSpec:
    prompt: pathlib.Path
    chunk: int
    ane: str

    @property
    def prompt_id(self) -> str:
        return self.prompt.stem.replace(".", "_").replace("-", "_")

    @property
    def run_id(self) -> str:
        return f"{self.prompt_id}__chunk{self.chunk}__ane_{self.ane}"


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part, 10))
    if not out:
        raise argparse.ArgumentTypeError("empty list")
    return out


def parse_prompt_list(values: list[str] | None) -> list[pathlib.Path]:
    if not values:
        return DEFAULT_PROMPTS
    prompts: list[pathlib.Path] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                prompts.append(pathlib.Path(part).expanduser())
    return prompts


def read_existing_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def run_one(args: argparse.Namespace, spec: RunSpec, csv_path: pathlib.Path, log_path: pathlib.Path) -> int:
    env = os.environ.copy()
    env["DS4_METAL_PREFILL_CHUNK"] = str(spec.chunk)
    if spec.ane == "on":
        env["DS4_FLASH_MOE_ANE_PREFILL"] = "1"
    else:
        env["DS4_FLASH_MOE_ANE_PREFILL"] = "0"

    cmd = [
        str(args.bench),
        "-m",
        str(args.model),
        "--moe-slot-bank",
        str(args.moe_slot_bank),
        "--prompt-file",
        str(spec.prompt),
        "--ctx-start",
        str(args.ctx_start),
        "--ctx-max",
        str(args.ctx_max),
        "--step-mul",
        str(args.step_mul),
        "--full-prefill-each-frontier",
        "--gen-tokens",
        str(args.gen_tokens),
        "--csv",
        str(csv_path),
    ]

    with log_path.open("wb") as log:
        log.write(("command: " + " ".join(cmd) + "\n").encode())
        log.write(f"DS4_METAL_PREFILL_CHUNK={spec.chunk}\n".encode())
        log.write(f"DS4_FLASH_MOE_ANE_PREFILL={env['DS4_FLASH_MOE_ANE_PREFILL']}\n".encode())
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def collect_rows(specs: list[RunSpec], raw_dir: pathlib.Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in specs:
        csv_path = raw_dir / f"{spec.run_id}.csv"
        for row in read_existing_csv(csv_path):
            out = {
                "prompt": spec.prompt_id,
                "prompt_file": str(spec.prompt),
                "ane": spec.ane,
                "prefill_chunk": str(spec.chunk),
            }
            out.update(row)
            rows.append(out)
    return rows


def write_combined(rows: list[dict[str, str]], path: pathlib.Path) -> None:
    fields = [
        "prompt",
        "prompt_file",
        "ane",
        "prefill_chunk",
        "ctx_tokens",
        "prefill_tokens",
        "prefill_tps",
        "gen_tokens",
        "gen_tps",
        "kvcache_bytes",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary(rows: list[dict[str, str]], path: pathlib.Path) -> None:
    by_prompt_ctx: dict[tuple[str, int], list[dict[str, str]]] = {}
    by_ctx_config: dict[tuple[int, int, str], list[float]] = {}
    for row in rows:
        ctx = int(row["ctx_tokens"])
        chunk = int(row["prefill_chunk"])
        ane = row["ane"]
        prompt = row["prompt"]
        tps = float(row["prefill_tps"])
        by_prompt_ctx.setdefault((prompt, ctx), []).append(row)
        by_ctx_config.setdefault((ctx, chunk, ane), []).append(tps)

    lines: list[str] = []
    lines.append("# MXFP4 Prefill Chunk / ANE Matrix")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Average Winners")
    lines.append("")
    lines.append("| ctx_tokens | best_chunk | best_ane | mean_prefill_tps | samples |")
    lines.append("|---:|---:|:---:|---:|---:|")
    for ctx in sorted({key[0] for key in by_ctx_config}):
        candidates = []
        for (cand_ctx, chunk, ane), values in by_ctx_config.items():
            if cand_ctx != ctx:
                continue
            candidates.append((statistics.mean(values), chunk, ane, len(values)))
        if not candidates:
            continue
        mean_tps, chunk, ane, count = max(candidates, key=lambda item: item[0])
        lines.append(f"| {ctx} | {chunk} | {ane} | {mean_tps:.2f} | {count} |")

    lines.append("")
    lines.append("## Per-Prompt Winners")
    lines.append("")
    lines.append("| prompt | ctx_tokens | best_chunk | best_ane | prefill_tps |")
    lines.append("|---|---:|---:|:---:|---:|")
    for (prompt, ctx), candidates in sorted(by_prompt_ctx.items(), key=lambda item: (item[0][0], item[0][1])):
        best = max(candidates, key=lambda row: float(row["prefill_tps"]))
        lines.append(
            f"| {prompt} | {ctx} | {best['prefill_chunk']} | {best['ane']} | "
            f"{float(best['prefill_tps']):.2f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--bench", type=pathlib.Path, default=ROOT / "ds4-bench")
    parser.add_argument("--prompt", action="append", help="Prompt file. May be repeated or comma-separated.")
    parser.add_argument("--chunks", type=parse_int_list, default=DEFAULT_CHUNKS)
    parser.add_argument("--ctx-start", type=int, default=1024)
    parser.add_argument("--ctx-max", type=int, default=16384)
    parser.add_argument("--step-mul", type=float, default=2.0)
    parser.add_argument("--gen-tokens", type=int, default=1)
    parser.add_argument("--moe-slot-bank", type=int, default=96)
    parser.add_argument("--out-dir", type=pathlib.Path)
    parser.add_argument("--force", action="store_true", help="Rerun even when raw CSV already exists.")
    args = parser.parse_args(argv)

    prompts = [path.resolve() for path in parse_prompt_list(args.prompt)]
    missing = [str(path) for path in [args.model, args.bench, *prompts] if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 2

    if args.out_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = ROOT / "bench-results" / f"mxfp4-prefill-ane-{stamp}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    log_dir = args.out_dir / "logs"
    raw_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    specs = [
        RunSpec(prompt=prompt, chunk=chunk, ane=ane)
        for prompt in prompts
        for chunk in args.chunks
        for ane in ("off", "on")
    ]

    failures = 0
    for index, spec in enumerate(specs, 1):
        csv_path = raw_dir / f"{spec.run_id}.csv"
        log_path = log_dir / f"{spec.run_id}.log"
        if csv_path.exists() and csv_path.stat().st_size > 0 and not args.force:
            print(f"[{index}/{len(specs)}] skip {spec.run_id}")
            continue
        print(f"[{index}/{len(specs)}] run {spec.run_id}", flush=True)
        rc = run_one(args, spec, csv_path, log_path)
        if rc != 0:
            failures += 1
            print(f"  failed rc={rc}; see {log_path}", file=sys.stderr)

    rows = collect_rows(specs, raw_dir)
    write_combined(rows, args.out_dir / "combined.csv")
    write_summary(rows, args.out_dir / "summary.md")
    print(f"wrote {args.out_dir / 'combined.csv'}")
    print(f"wrote {args.out_dir / 'summary.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
