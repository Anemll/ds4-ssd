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
MODEL = "/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
PREFIXES = [1024, 2048, 4096, 8192, 16384, 32768]
DECODE = 2000
CTX = 40000

RE_PREFILL = re.compile(
    r"ds4: prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s "
    r"\((\d+) tokens in ([0-9.]+)s\)"
)


def parse_metrics(text: str) -> dict:
    row = {}
    m = RE_PREFILL.search(text)
    if m:
        row["prefill_tps"] = m.group(1)
        row["generation_tps"] = m.group(2)
        row["generated_tokens"] = m.group(3)
        row["decode_s"] = m.group(4)
    return row


def run_case(prefix: int) -> dict:
    prompt_path = OUT / f"prompt-{prefix}.txt"
    log_path = OUT / f"current-monolithic-p{prefix}-d{DECODE}-ctx{CTX}.log"
    env = os.environ.copy()
    env["DS4_DSPARK_PERF"] = "1"
    env["DS4_DSPARK_FRONTIER_DRAFT"] = "1"
    cmd = [
        str(ROOT / "ds4"),
        "--model", MODEL,
        "-c", str(CTX),
        "--temp", "0",
        "--nothink",
        "-n", str(DECODE),
        "--prompt-file", str(prompt_path),
    ]
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
        "kind": "current_monolithic",
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
    fields = [
        "kind", "ctx", "target_prefix_tokens", "decode_target_tokens",
        "generated_tokens", "generation_tps", "decode_s", "prefill_tps",
        "wall_s", "returncode", "log",
    ]
    rows = []
    for prefix in PREFIXES:
        print(f"RUN current_monolithic prefix={prefix}", flush=True)
        row = run_case(prefix)
        rows.append(row)
        print(
            f"DONE current_monolithic prefix={prefix} "
            f"gen_tps={row.get('generation_tps','')} decode_s={row.get('decode_s','')} "
            f"rc={row['returncode']}",
            flush=True,
        )
        with (OUT / "current_monolithic_synthetic.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
