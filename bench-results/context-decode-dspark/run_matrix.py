#!/usr/bin/env python3
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODEL = "/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major"
DRAFT = "/Users/anemll/Models/DSv4-Flash-DSpark-draft"
PREFIXES = [1024, 2048, 4096, 8192, 16384, 32768]
DECODE = 2000
CTX = int(os.environ.get("DS4_BENCH_CTX", "40000"))
PROMPT_TAIL = (
    "\n\nTask: Make a game of Tetris in C++ for macOS. "
    "Include the full source code and explanatory notes."
)
PROMPT_BASE_TOKENS = 38


RE_PREFILL = re.compile(
    r"ds4: prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s "
    r"\((\d+) tokens in ([0-9.]+)s\)"
)
RE_TTF = re.compile(
    r"ds4: ttf: first-output=([0-9.]+)s total=([0-9.]+)s "
    r"\(engine=([0-9.]+)s session=([0-9.]+)s prompt=([0-9.]+)s "
    r"prefill=([0-9.]+)s decode=([0-9.]+)s\)"
)
RE_DSPARK = re.compile(
    r"ds4: dspark perf: draft=([0-9.]+) tok/s, verify=([0-9.]+) proposed tok/s, "
    r"verify-accepted=([0-9.]+) tok/s, block=([0-9.]+) ms .* tau=([0-9.]+), blocks=(\d+)"
)
RE_INPUT = re.compile(r"processing (\d+) input tokens:")


def prompt_for_tokens(n: int) -> str:
    if n < PROMPT_BASE_TOKENS:
        raise ValueError(f"target prefix must be at least {PROMPT_BASE_TOKENS} tokens")
    return "Context padding follows:" + (" x" * (n - PROMPT_BASE_TOKENS)) + PROMPT_TAIL


def parse_metrics(text: str) -> dict:
    row = {}
    m = RE_PREFILL.search(text)
    if m:
        row.update({
            "prefill_tps": m.group(1),
            "generation_tps": m.group(2),
            "generated_tokens": m.group(3),
            "decode_s": m.group(4),
        })
    m = RE_TTF.search(text)
    if m:
        row.update({
            "first_output_s": m.group(1),
            "total_s": m.group(2),
            "engine_s": m.group(3),
            "session_s": m.group(4),
            "prompt_s": m.group(5),
            "prefill_s": m.group(6),
            "ttf_decode_s": m.group(7),
        })
    m = RE_DSPARK.search(text)
    if m:
        row.update({
            "dspark_draft_tps": m.group(1),
            "dspark_verify_proposed_tps": m.group(2),
            "dspark_verify_accepted_tps": m.group(3),
            "dspark_block_ms": m.group(4),
            "dspark_tau": m.group(5),
            "dspark_blocks": m.group(6),
        })
    inputs = RE_INPUT.findall(text)
    if inputs:
        row["input_tokens_reported"] = inputs[-1]
    return row


def run_case(kind: str, prefix: int) -> dict:
    prompt_path = OUT / f"prompt-{prefix}.txt"
    prompt_path.write_text(prompt_for_tokens(prefix), encoding="utf-8")
    log_path = OUT / f"{kind}-p{prefix}-d{DECODE}-ctx{CTX}.log"

    env = os.environ.copy()
    env["DS4_DSPARK_PERF"] = "1"
    env["DS4_DSPARK_FRONTIER_DRAFT"] = "1"

    cmd = [
        str(ROOT / "ds4"),
        "-m", MODEL,
        "-c", str(CTX),
        "--temp", "0",
        "--nothink",
        "--resident",
        "-n", str(DECODE),
        "--prompt-file", str(prompt_path),
    ]
    if kind == "dspark":
        cmd[3:3] = ["--draft", "dspark", "--draft-path", DRAFT]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - t0
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")

    row = {
        "kind": kind,
        "ctx": str(CTX),
        "target_prefix_tokens": str(prefix),
        "decode_target_tokens": str(DECODE),
        "returncode": str(proc.returncode),
        "wall_s": f"{elapsed:.3f}",
        "log": str(log_path.relative_to(ROOT)),
    }
    row.update(parse_metrics(proc.stdout))
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    fields = [
        "kind", "ctx", "target_prefix_tokens", "input_tokens_reported",
        "decode_target_tokens", "generated_tokens", "generation_tps",
        "decode_s", "prefill_tps", "prefill_s", "first_output_s", "total_s",
        "engine_s", "session_s", "wall_s",
        "dspark_draft_tps", "dspark_verify_proposed_tps",
        "dspark_verify_accepted_tps", "dspark_block_ms", "dspark_tau",
        "dspark_blocks", "returncode", "log",
    ]
    for prefix in PREFIXES:
        for kind in ("baseline", "dspark"):
            print(f"RUN kind={kind} prefix={prefix} decode={DECODE} ctx={CTX}", flush=True)
            row = run_case(kind, prefix)
            rows.append(row)
            print(
                "DONE kind={kind} prefix={target_prefix_tokens} gen_tps={generation_tps} "
                "decode_s={decode_s} rc={returncode}".format(**{
                    **{k: "" for k in fields},
                    **row,
                }),
                flush=True,
            )
            with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
    with (OUT / "results.md").open("w", encoding="utf-8") as fp:
        fp.write("| kind | prefix | input | gen t/s | decode s | prefill t/s | dspark tau | blocks |\n")
        fp.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            fp.write(
                f"| {row.get('kind','')} | {row.get('target_prefix_tokens','')} | "
                f"{row.get('input_tokens_reported','')} | {row.get('generation_tps','')} | "
                f"{row.get('decode_s','')} | {row.get('prefill_tps','')} | "
                f"{row.get('dspark_tau','')} | {row.get('dspark_blocks','')} |\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
