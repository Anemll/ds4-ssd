#!/usr/bin/env python3
"""Compare generated sample directories from generate_samples."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp, delimiter="\t"):
            rows[row["id"]] = row
    return rows


def char_lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def token_lcp(a: str, b: str) -> int:
    aw = a.split()
    bw = b.split()
    n = min(len(aw), len(bw))
    for i in range(n):
        if aw[i] != bw[i]:
            return i
    return n


def max_repeated_line_run(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    best = cur = 0
    prev = None
    for line in lines:
        if line == prev:
            cur += 1
        else:
            cur = 1
            prev = line
        best = max(best, cur)
    return best


def load_text(run_dir: Path, case_id: str) -> str:
    return (run_dir / "outputs" / f"{case_id}.txt").read_text(encoding="utf-8", errors="replace")


def main() -> int:
    if len(sys.argv) < 4:
        print(
            f"usage: {sys.argv[0]} BASE_RUN CANDIDATE_RUN OUT_PREFIX [CANDIDATE_RUN OUT_PREFIX ...]",
            file=sys.stderr,
        )
        return 2

    base_dir = Path(sys.argv[1])
    base = load_summary(base_dir / "summary.tsv")
    args = sys.argv[2:]
    if len(args) % 2 != 0:
        raise SystemExit("candidate arguments must be RUN_DIR OUT_PREFIX pairs")

    for run_arg, out_arg in zip(args[0::2], args[1::2]):
        run_dir = Path(run_arg)
        out_prefix = Path(out_arg)
        cand = load_summary(run_dir / "summary.tsv")
        ids = sorted(set(base) & set(cand))
        if not ids:
            raise SystemExit(f"no common cases for {run_dir}")

        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        per_case_path = out_prefix.with_suffix(".per_case.tsv")
        summary_path = out_prefix.with_suffix(".summary.tsv")

        byte_equal = 0
        char_lcp_total = 0
        token_lcp_total = 0
        base_tokens = cand_tokens = 0
        base_decode = cand_decode = 0.0
        cand_draft_slots = cand_draft_accepted = 0
        max_line_run = 0

        with per_case_path.open("w", newline="", encoding="utf-8") as fp:
            fields = [
                "id",
                "byte_equal",
                "base_generated",
                "candidate_generated",
                "base_tps",
                "candidate_tps",
                "speedup",
                "candidate_acceptance_pct",
                "char_lcp",
                "token_lcp",
                "candidate_max_repeated_line_run",
                "base_output",
                "candidate_output",
            ]
            wr = csv.DictWriter(fp, delimiter="\t", fieldnames=fields)
            wr.writeheader()
            for case_id in ids:
                btxt = load_text(base_dir, case_id)
                ctxt = load_text(run_dir, case_id)
                beq = btxt.encode("utf-8") == ctxt.encode("utf-8")
                byte_equal += int(beq)
                clcp = char_lcp(btxt, ctxt)
                tlcp = token_lcp(btxt, ctxt)
                line_run = max_repeated_line_run(ctxt)
                char_lcp_total += clcp
                token_lcp_total += tlcp
                max_line_run = max(max_line_run, line_run)

                b = base[case_id]
                c = cand[case_id]
                btok = int(b["generated_tokens"])
                ctok = int(c["generated_tokens"])
                bsec = float(b["decode_s"])
                csec = float(c["decode_s"])
                base_tokens += btok
                cand_tokens += ctok
                base_decode += bsec
                cand_decode += csec
                cand_draft_slots += int(c["draft_slots"])
                cand_draft_accepted += int(c["draft_accepted"])
                btps = float(b["gen_tps"])
                ctps = float(c["gen_tps"])
                wr.writerow(
                    {
                        "id": case_id,
                        "byte_equal": int(beq),
                        "base_generated": btok,
                        "candidate_generated": ctok,
                        "base_tps": f"{btps:.6f}",
                        "candidate_tps": f"{ctps:.6f}",
                        "speedup": f"{(ctps / btps) if btps else 0.0:.6f}",
                        "candidate_acceptance_pct": c["acceptance_pct"],
                        "char_lcp": clcp,
                        "token_lcp": tlcp,
                        "candidate_max_repeated_line_run": line_run,
                        "base_output": b["output_file"],
                        "candidate_output": c["output_file"],
                    }
                )

        base_eff = base_tokens / base_decode if base_decode > 0 else 0.0
        cand_eff = cand_tokens / cand_decode if cand_decode > 0 else 0.0
        cand_acc = (
            100.0 * cand_draft_accepted / cand_draft_slots if cand_draft_slots else 0.0
        )
        with summary_path.open("w", encoding="utf-8") as fp:
            fp.write("metric\tvalue\n")
            fp.write(f"cases\t{len(ids)}\n")
            fp.write(f"byte_equal_cases\t{byte_equal}\n")
            fp.write(f"byte_diff_cases\t{len(ids) - byte_equal}\n")
            fp.write(f"base_effective_tps\t{base_eff:.6f}\n")
            fp.write(f"candidate_effective_tps\t{cand_eff:.6f}\n")
            fp.write(f"speedup\t{(cand_eff / base_eff) if base_eff else 0.0:.6f}\n")
            fp.write(f"base_tokens\t{base_tokens}\n")
            fp.write(f"candidate_tokens\t{cand_tokens}\n")
            fp.write(f"candidate_acceptance_pct\t{cand_acc:.6f}\n")
            fp.write(f"avg_char_lcp\t{char_lcp_total / len(ids):.3f}\n")
            fp.write(f"avg_token_lcp\t{token_lcp_total / len(ids):.3f}\n")
            fp.write(f"max_repeated_line_run\t{max_line_run}\n")

        print(f"wrote {summary_path}")
        print(f"wrote {per_case_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
