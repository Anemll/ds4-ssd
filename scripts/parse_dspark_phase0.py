#!/usr/bin/env python3
"""Parse DSpark Phase-0 benchmark logs into a TSV table."""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Iterable, List


GEN_RE = re.compile(
    r"generation:\s+([0-9.]+)\s+t/s\s+\((\d+)\s+tokens in ([0-9.]+)s\)"
)
ACCEPT_RE = re.compile(
    r"dspark acceptance:\s+([0-9.]+)%\s+\((\d+)/(\d+) draft tokens\)"
)
PERF_RE = re.compile(
    r"dspark perf:\s+draft=([0-9.]+)\s+tok/s,\s+"
    r"verify=([0-9.]+)\s+proposed tok/s,\s+"
    r"verify-accepted=([0-9.]+)\s+tok/s,\s+block=([0-9.]+)\s+ms\s+"
    r"\(draft=([0-9.]+)\s+verify=([0-9.]+)\s+overhead=([0-9.]+)\s+"
    r"commit=([0-9.]+),\s+tau=([0-9.]+),\s+blocks=(\d+)\)"
)
DECODE_EQ_RE = re.compile(
    r"dspark decode-eq:\s+draft=([0-9.]+)\s+verify=([0-9.]+)\s+"
    r"overhead=([0-9.]+)\s+block=([0-9.]+)\s+"
    r"baseline_decode=([0-9.]+)\s+ms\s+tau=([0-9.]+)"
)
DISPATCH_RE = re.compile(r"dspark verify dispatch-est .*")
OVERLAP_RE = re.compile(r"dspark routed overlap .*")
SHARED_PREFIX_RE = re.compile(
    r"dspark shared-prefix profile\s+block=(?P<block>\d+)\s+"
    r"path=(?P<path>[A-Za-z0-9_.()/-]+)\s+n=(?P<n>\d+)\s+"
    r"layers=(?P<layers>\d+)\s+calls=(?P<calls>\d+)\s+"
    r"indexed=(?P<indexed>\d+)\s+avg_rows=(?P<avg_rows>[0-9.]+)\s+"
    r"raw_keys\s+row=(?P<raw_row_keys>\d+)\s+"
    r"shared_est=(?P<raw_shared_est>\d+)\s+"
    r"saved=(?P<raw_saved>\d+)\s+reuse=(?P<raw_reuse>[0-9.]+)x\s+"
    r"common_row_keys=(?P<raw_common_row_keys>\d+)\s+"
    r"comp_keys\s+row=(?P<comp_row_keys>\d+)\s+"
    r"shared_est=(?P<comp_shared_est>\d+)\s+"
    r"saved=(?P<comp_saved>\d+)\s+reuse=(?P<comp_reuse>[0-9.]+)x\s+"
    r"common_row_keys=(?P<comp_common_row_keys>\d+)"
)
ATTN_ROWS_SHAPE_RE = re.compile(
    r"dspark attn rows shape\s+block=(?P<block>\d+)\s+"
    r"calls=(?P<calls>\d+)\s+indexed=(?P<indexed>\d+)\s+"
    r"rows=(?P<rows>\d+)\s+max_n=(?P<max_n>\d+)\s+"
    r"raw_only=(?P<raw_only>\d+)\s+mixed=(?P<mixed>\d+)\s+"
    r"raw_same_count=(?P<raw_same_count>\d+)\s+"
    r"raw_same_start=(?P<raw_same_start>\d+)\s+"
    r"comp_same_count=(?P<comp_same_count>\d+)\s+"
    r"raw_keys\s+row=(?P<raw_row_keys>\d+)\s+"
    r"shared_est=(?P<raw_shared_est>\d+)\s+"
    r"reuse=(?P<raw_reuse>[0-9.]+)x\s+"
    r"(?:raw_intersection=(?P<raw_intersection>\d+)\s+"
    r"raw_lane_aligned=(?P<raw_lane_aligned>\d+)\s+"
    r"raw_lane=(?P<raw_lane>[0-9.]+)%\s+"
    r"raw_lane_runs=(?P<raw_lane_runs>\d+)\s+"
    r"raw_lane_avg_run=(?P<raw_lane_avg_run>[0-9.]+)\s+"
    r"raw_lane_max_run=(?P<raw_lane_max_run>\d+)\s+)?"
    r"comp_keys\s+row=(?P<comp_row_keys>\d+)\s+"
    r"shared_est=(?P<comp_shared_est>\d+)\s+"
    r"reuse=(?P<comp_reuse>[0-9.]+)x\s+"
    r"(?:comp_common=(?P<comp_common>\d+)\s+"
    r"comp_lane_aligned=(?P<comp_lane_aligned>\d+)\s+"
    r"comp_lane=(?P<comp_lane>[0-9.]+)%\s+"
    r"comp_lane_runs=(?P<comp_lane_runs>\d+)\s+"
    r"comp_lane_avg_run=(?P<comp_lane_avg_run>[0-9.]+)\s+"
    r"comp_lane_max_run=(?P<comp_lane_max_run>\d+)\s+)?"
    r"zero_comp_rows=(?P<zero_comp_rows>\d+)\s+"
    r"pad_rows=(?P<pad_rows>\d+)\s+"
    r"ring_rows=(?P<ring_rows>\d+)\s+"
    r"raw_minmax=(?P<raw_min>\d+)/(?P<raw_max>\d+)\s+"
    r"comp_minmax=(?P<comp_min>\d+)/(?P<comp_max>\d+)"
)
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_>=.]+)=([0-9.]+x?|[0-9.]+%?|[A-Za-z0-9_.-]+)")
TARGET_FORWARD_BACKEND_RE = re.compile(
    r"target-forward unified N<=5 backend active:\s+([A-Za-z0-9_.()/-]+)\s+n=(\d+)"
)
TARGET_FORWARD_PROFILE_RE = re.compile(
    r"target-forward unified profile n=(\d+)\s+backend=([A-Za-z0-9_.()/-]+)\s+"
    r"ok=(\d+)\s+total=([-0-9.]+)\s+ms"
)


COLUMNS = [
    "label",
    "budget",
    "path",
    "generation_tps",
    "generated_tokens",
    "decode_s",
    "acceptance_pct",
    "accepted",
    "draft_slots",
    "draft_tps",
    "verify_proposed_tps",
    "verify_accepted_tps",
    "block_ms",
    "draft_ms",
    "verify_ms",
    "overhead_ms",
    "commit_ms",
    "tau",
    "blocks",
    "decode_eq_draft",
    "decode_eq_verify",
    "decode_eq_overhead",
    "decode_eq_block",
    "baseline_decode_ms",
    "unified_backend",
    "unified_active_n",
    "tfwd_decode_calls",
    "tfwd_decode_avg_ms",
    "tfwd_verify_calls",
    "tfwd_verify_avg_ms",
    "tfwd_verify_min_ms",
    "tfwd_verify_max_ms",
    "tfwd_failed_calls",
    "tfwd_n1_avg_ms",
    "tfwd_n2_avg_ms",
    "tfwd_n3_avg_ms",
    "tfwd_n4_avg_ms",
    "tfwd_n5_avg_ms",
    "disp_n",
    "disp_total",
    "disp_attn",
    "disp_kv",
    "disp_comp",
    "disp_index",
    "disp_heads",
    "disp_output_hc",
    "disp_ffn_pre",
    "disp_router",
    "disp_routed_moe",
    "disp_ordered_sum",
    "disp_shared",
    "disp_post_hc",
    "disp_target_hidden",
    "disp_head",
    "disp_topk",
    "disp_read",
    "route_n",
    "route_slots",
    "route_unique",
    "route_reuse",
    "route_dup_pct",
    "route_pair_overlap",
    "route_max_multiplicity",
    "route_unique_min",
    "route_unique_max",
    "route_reuse125_layers",
    "route_reuse150_layers",
    "sp_block",
    "sp_path",
    "sp_n",
    "sp_layers",
    "sp_calls",
    "sp_indexed",
    "sp_avg_rows",
    "sp_raw_row_keys",
    "sp_raw_shared_est",
    "sp_raw_saved",
    "sp_raw_reuse",
    "sp_raw_common_row_keys",
    "sp_comp_row_keys",
    "sp_comp_shared_est",
    "sp_comp_saved",
    "sp_comp_reuse",
    "sp_comp_common_row_keys",
    "ar_block",
    "ar_calls",
    "ar_indexed",
    "ar_rows",
    "ar_max_n",
    "ar_raw_only",
    "ar_mixed",
    "ar_raw_same_count",
    "ar_raw_same_start",
    "ar_comp_same_count",
    "ar_raw_row_keys",
    "ar_raw_shared_est",
    "ar_raw_reuse",
    "ar_raw_intersection",
    "ar_raw_lane_aligned",
    "ar_raw_lane",
    "ar_raw_lane_runs",
    "ar_raw_lane_avg_run",
    "ar_raw_lane_max_run",
    "ar_comp_row_keys",
    "ar_comp_shared_est",
    "ar_comp_reuse",
    "ar_comp_common",
    "ar_comp_lane_aligned",
    "ar_comp_lane",
    "ar_comp_lane_runs",
    "ar_comp_lane_avg_run",
    "ar_comp_lane_max_run",
    "ar_zero_comp_rows",
    "ar_pad_rows",
    "ar_ring_rows",
    "ar_raw_min",
    "ar_raw_max",
    "ar_comp_min",
    "ar_comp_max",
]


def clean_value(value: str) -> str:
    if value.endswith("x") or value.endswith("%"):
        return value[:-1]
    return value


def parse_key_values(line: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for key, value in KEY_VALUE_RE.findall(line):
        values[key] = clean_value(value)
    return values


def fmt_avg(values: List[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.3f}"


def infer_budget(path: str) -> str:
    base = os.path.basename(path)
    for pattern in (
        r"budget([0-9]+)",
        r"active([0-9]+)",
        r"verify([0-9]+)",
        r"_b([0-9]+)",
    ):
        match = re.search(pattern, base)
        if match:
            return match.group(1)
    return ""


def parse_log(path: str, label: str | None = None, budget: str | None = None) -> Dict[str, str]:
    row = {key: "" for key in COLUMNS}
    row["path"] = path
    row["label"] = label or os.path.splitext(os.path.basename(path))[0]
    row["budget"] = budget if budget is not None else infer_budget(path)
    best_dispatch_n = -1
    best_overlap_n = -1
    tfwd_by_n: Dict[int, List[float]] = {}
    tfwd_verify_values: List[float] = []
    tfwd_decode_values: List[float] = []
    tfwd_failed_calls = 0

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc

    for line in lines:
        match = GEN_RE.search(line)
        if match:
            row["generation_tps"] = match.group(1)
            row["generated_tokens"] = match.group(2)
            row["decode_s"] = match.group(3)

        match = ACCEPT_RE.search(line)
        if match:
            row["acceptance_pct"] = match.group(1)
            row["accepted"] = match.group(2)
            row["draft_slots"] = match.group(3)

        match = PERF_RE.search(line)
        if match:
            (
                row["draft_tps"],
                row["verify_proposed_tps"],
                row["verify_accepted_tps"],
                row["block_ms"],
                row["draft_ms"],
                row["verify_ms"],
                row["overhead_ms"],
                row["commit_ms"],
                row["tau"],
                row["blocks"],
            ) = match.groups()

        match = DECODE_EQ_RE.search(line)
        if match:
            (
                row["decode_eq_draft"],
                row["decode_eq_verify"],
                row["decode_eq_overhead"],
                row["decode_eq_block"],
                row["baseline_decode_ms"],
                _tau,
            ) = match.groups()

        match = TARGET_FORWARD_BACKEND_RE.search(line)
        if match:
            row["unified_backend"] = match.group(1)
            row["unified_active_n"] = match.group(2)

        match = TARGET_FORWARD_PROFILE_RE.search(line)
        if match:
            n_rows = int(match.group(1))
            backend = match.group(2)
            ok = match.group(3) == "1"
            total_ms = float(match.group(4))
            if not ok:
                tfwd_failed_calls += 1
                continue
            if total_ms < 0.0:
                continue
            tfwd_by_n.setdefault(n_rows, []).append(total_ms)
            if n_rows == 1 and backend == "decode":
                tfwd_decode_values.append(total_ms)
            elif n_rows > 1:
                tfwd_verify_values.append(total_ms)
                if not row["unified_backend"]:
                    row["unified_backend"] = backend
                if not row["unified_active_n"]:
                    row["unified_active_n"] = str(n_rows)

        if DISPATCH_RE.search(line):
            values = parse_key_values(line)
            dispatch_n = int(float(values.get("n", "-1")))
            if dispatch_n < best_dispatch_n:
                continue
            best_dispatch_n = dispatch_n
            for key in (
                "n",
                "total",
                "attn",
                "kv",
                "comp",
                "index",
                "heads",
                "output_hc",
                "ffn_pre",
                "router",
                "routed_moe",
                "ordered_sum",
                "shared",
                "post_hc",
                "target_hidden",
                "head",
                "topk",
                "read",
            ):
                if key in values:
                    row[f"disp_{key}" if key != "n" else "disp_n"] = values[key]

        if OVERLAP_RE.search(line):
            values = parse_key_values(line)
            overlap_n = int(float(values.get("n", "-1")))
            if overlap_n < best_overlap_n:
                continue
            best_overlap_n = overlap_n
            key_map = {
                "n": "route_n",
                "slots": "route_slots",
                "unique": "route_unique",
                "reuse": "route_reuse",
                "dup": "route_dup_pct",
                "pair_overlap": "route_pair_overlap",
                "max_multiplicity": "route_max_multiplicity",
                "unique_min": "route_unique_min",
                "unique_max": "route_unique_max",
                "reuse>=1.25": "route_reuse125_layers",
                "reuse>=1.50": "route_reuse150_layers",
            }
            for key, dest in key_map.items():
                if key in values:
                    row[dest] = values[key]

        match = SHARED_PREFIX_RE.search(line)
        if match:
            for key, value in match.groupdict().items():
                row[f"sp_{key}"] = value

        match = ATTN_ROWS_SHAPE_RE.search(line)
        if match:
            for key, value in match.groupdict().items():
                if value is not None:
                    row[f"ar_{key}"] = value

    if tfwd_decode_values:
        row["tfwd_decode_calls"] = str(len(tfwd_decode_values))
        row["tfwd_decode_avg_ms"] = fmt_avg(tfwd_decode_values)
    if tfwd_verify_values:
        row["tfwd_verify_calls"] = str(len(tfwd_verify_values))
        row["tfwd_verify_avg_ms"] = fmt_avg(tfwd_verify_values)
        row["tfwd_verify_min_ms"] = f"{min(tfwd_verify_values):.3f}"
        row["tfwd_verify_max_ms"] = f"{max(tfwd_verify_values):.3f}"
    if tfwd_failed_calls:
        row["tfwd_failed_calls"] = str(tfwd_failed_calls)
    for n_rows in range(1, 6):
        avg = fmt_avg(tfwd_by_n.get(n_rows, []))
        if avg:
            row[f"tfwd_n{n_rows}_avg_ms"] = avg

    return row


def print_tsv(rows: Iterable[Dict[str, str]]) -> None:
    print("\t".join(COLUMNS))
    for row in rows:
        print("\t".join(row.get(column, "") for column in COLUMNS))


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="stderr log files to parse")
    parser.add_argument("--label", help="label to use when parsing a single log")
    parser.add_argument("--budget", help="budget to use when parsing a single log")
    args = parser.parse_args(argv)

    if (args.label or args.budget) and len(args.logs) != 1:
        parser.error("--label/--budget only apply when one log is provided")

    rows = [parse_log(path, args.label, args.budget) for path in args.logs]
    print_tsv(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
