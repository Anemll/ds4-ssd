#!/usr/bin/env python3
"""Convert a DS4 Flash-MoE MXFP4 sidecar to native plane-split MXFP4.

The source sidecar stores ggml block_mxfp4 records:

    [scale][16 split-half FP4 bytes] repeated

The native MPP 4.1 path consumes the same bytes rearranged inside each family
region:

    [all 16-byte sequential-pair FP4 blocks][all 1-byte E8M0 scales]

The conversion preserves layer files, family offsets, bytes_per_expert, and
expert_stride. Only bytes inside each MXFP4 family region are rearranged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


N_EXPERT = 256
BLOCK_BYTES = 17
BLOCK_DATA_BYTES = 16
LAYOUT = "mxfp4_plane_split_v1"
SOURCE_LAYOUT = "ggml_block_mxfp4_split_half_v1"


def convert_mxfp4_blocks(block_bytes: bytes) -> tuple[bytes, bytes]:
    """Return (seq-pair FP4 data plane, E8M0 scale plane)."""
    if len(block_bytes) % BLOCK_BYTES:
        raise ValueError(f"MXFP4 byte length {len(block_bytes)} is not divisible by {BLOCK_BYTES}")
    blocks = np.frombuffer(block_bytes, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    qs = blocks[:, 1:17]
    even = qs[:, 0::2]
    odd = qs[:, 1::2]
    data = np.empty((blocks.shape[0], BLOCK_DATA_BYTES), dtype=np.uint8)
    data[:, 0:8] = (even & 0x0F) | ((odd & 0x0F) << 4)
    data[:, 8:16] = (even >> 4) | (odd & 0xF0)
    scales = blocks[:, 0].copy()
    return data.tobytes(order="C"), scales.tobytes(order="C")


def copy_exact(src, dst, n: int, bufsize: int = 16 * 1024 * 1024) -> None:
    remaining = n
    while remaining:
        chunk = src.read(min(bufsize, remaining))
        if not chunk:
            raise EOFError("unexpected EOF while copying sidecar data")
        dst.write(chunk)
        remaining -= len(chunk)


def read_exact(src, n: int) -> bytes:
    data = src.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, read {len(data)}")
    return data


def entry_plane_sizes(entry: dict) -> tuple[int, int]:
    family_bytes = int(entry["bytes_per_expert"])
    if family_bytes % BLOCK_BYTES:
        raise ValueError(f"{entry.get('tensor_name')} bytes_per_expert is not MXFP4 block aligned")
    blocks = family_bytes // BLOCK_BYTES
    return blocks * BLOCK_DATA_BYTES, blocks


def annotate_manifest(manifest: dict, src_dir: Path) -> dict:
    out = json.loads(json.dumps(manifest))
    out["schema_version"] = max(int(out.get("schema_version", 1)), 2)
    out["storage_layout"] = LAYOUT
    out["source_storage_layout"] = SOURCE_LAYOUT
    out["converted_from"] = str(src_dir)
    out["converter"] = Path(__file__).name
    out["converted_at_unix"] = int(time.time())
    for entry in out["entries"]:
        if entry.get("quant_type") != "MXFP4":
            continue
        data_bytes, scale_bytes = entry_plane_sizes(entry)
        family_off = int(entry["repacked_offset"])
        entry["source_quant_type"] = "MXFP4"
        entry["quant_type"] = "MXFP4_NATIVE"
        entry["storage_layout"] = LAYOUT
        entry["source_storage_layout"] = SOURCE_LAYOUT
        entry["plane_data_format"] = "FP4_E2M1_SEQPAIR"
        entry["plane_scale_format"] = "E8M0"
        entry["plane_data_offset"] = family_off
        entry["plane_data_bytes"] = data_bytes
        entry["plane_scale_offset"] = family_off + data_bytes
        entry["plane_scale_bytes"] = scale_bytes
    return out


def load_manifest(src_dir: Path) -> dict:
    manifest_path = src_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest_path} has no entries")
    for entry in entries:
        if entry.get("quant_type") != "MXFP4":
            raise ValueError(f"only MXFP4 entries are supported, got {entry.get('quant_type')}")
        if not entry.get("expert_major", True):
            raise ValueError("only expert_major sidecars are supported")
        if int(entry.get("block_size", 32)) != 32:
            raise ValueError("only MXFP4 block_size=32 is supported")
    return manifest


def grouped_entries(manifest: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest["entries"]:
        grouped[entry["repacked_file"]].append(entry)
    for file_entries in grouped.values():
        file_entries.sort(key=lambda e: int(e["repacked_offset"]))
        strides = {int(e["expert_stride"]) for e in file_entries if int(e.get("expert_stride", 0))}
        if len(strides) != 1:
            raise ValueError("each layer file must have one non-zero expert_stride")
    return dict(grouped)


def convert_layer_file(src_path: Path, dst_path: Path, entries: list[dict], *, resume: bool) -> None:
    src_size = src_path.stat().st_size
    if resume and dst_path.exists() and dst_path.stat().st_size == src_size:
        print(f"skip complete {dst_path.name}", flush=True)
        return

    tmp_path = dst_path.with_name(dst_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    stride = int(entries[0]["expert_stride"])
    min_record = max(int(e["repacked_offset"]) + int(e["bytes_per_expert"]) for e in entries)
    if stride < min_record:
        raise ValueError(f"{src_path.name}: expert_stride is smaller than family regions")
    expected_min = (N_EXPERT - 1) * stride + min_record
    if src_size < expected_min:
        raise ValueError(f"{src_path.name}: source is truncated")

    with src_path.open("rb", buffering=0) as src, tmp_path.open("wb", buffering=0) as dst:
        for expert in range(N_EXPERT):
            record_base = expert * stride
            cursor = record_base
            for entry in entries:
                family_start = record_base + int(entry["repacked_offset"])
                family_bytes = int(entry["bytes_per_expert"])
                if family_start < cursor:
                    raise ValueError(f"{src_path.name}: overlapping family regions")
                if family_start > cursor:
                    src.seek(cursor)
                    copy_exact(src, dst, family_start - cursor)
                else:
                    src.seek(family_start)
                raw = read_exact(src, family_bytes)
                data, scales = convert_mxfp4_blocks(raw)
                dst.write(data)
                dst.write(scales)
                cursor = family_start + family_bytes
            record_end = record_base + stride
            if record_end > cursor:
                src.seek(cursor)
                copy_exact(src, dst, record_end - cursor)

        trailing = src_size - N_EXPERT * stride
        if trailing > 0:
            src.seek(N_EXPERT * stride)
            copy_exact(src, dst, trailing)

    if tmp_path.stat().st_size != src_size:
        raise ValueError(f"{tmp_path}: converted size {tmp_path.stat().st_size} != source {src_size}")
    shutil.copymode(src_path, tmp_path)
    os.replace(tmp_path, dst_path)


def copy_dense(src_dir: Path, dst_dir: Path, *, mode: str, resume: bool) -> None:
    src_dense = src_dir / "dense"
    if not src_dense.exists():
        return
    dst_dense = dst_dir / "dense"
    if mode == "symlink":
        if dst_dense.exists() or dst_dense.is_symlink():
            if resume:
                return
            raise FileExistsError(dst_dense)
        os.symlink(src_dense, dst_dense, target_is_directory=True)
        return
    dst_dense.mkdir(parents=True, exist_ok=True)
    for item in src_dense.iterdir():
        target = dst_dense / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=resume)
            continue
        if resume and target.exists() and target.stat().st_size == item.stat().st_size:
            continue
        tmp = target.with_name(target.name + ".tmp")
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(item, tmp)
        os.replace(tmp, target)


def self_test() -> None:
    root = Path(__file__).resolve().parents[1]
    vec = root / "tests" / "test-vectors" / "mxfp4"
    blocks = (vec / "blocks_ggml.bin").read_bytes()
    want_data = (vec / "plane_data.bin").read_bytes()
    want_scales = (vec / "plane_scales.bin").read_bytes()
    got_data, got_scales = convert_mxfp4_blocks(blocks)
    if got_data != want_data or got_scales != want_scales:
        raise AssertionError("MXFP4 converter does not match golden plane vectors")
    print("self-test: MXFP4 plane conversion matches golden vectors")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir", type=Path, nargs="?")
    ap.add_argument("dst_dir", type=Path, nargs="?")
    ap.add_argument("--dense-mode", choices=("copy", "symlink"), default="copy")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    if args.src_dir is None or args.dst_dir is None:
        ap.error("src_dir and dst_dir are required unless --self-test is used")

    src_dir = args.src_dir.resolve()
    dst_dir = args.dst_dir.resolve()
    if not (src_dir / "manifest.json").exists():
        raise FileNotFoundError(src_dir / "manifest.json")
    if dst_dir.exists() and any(dst_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"{dst_dir} exists and is not empty; use --resume to continue")
    dst_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(src_dir)
    grouped = grouped_entries(manifest)
    annotated = annotate_manifest(manifest, src_dir)

    print(f"source: {src_dir}")
    print(f"dest:   {dst_dir}")
    print(f"layers: {len(grouped)}")
    print(f"layout: {LAYOUT}")

    for i, file_name in enumerate(sorted(grouped), 1):
        src_path = src_dir / file_name
        dst_path = dst_dir / file_name
        t0 = time.monotonic()
        convert_layer_file(src_path, dst_path, grouped[file_name], resume=args.resume)
        dt = time.monotonic() - t0
        gib = src_path.stat().st_size / (1024 ** 3)
        print(f"[{i:02d}/{len(grouped):02d}] {file_name}: {gib:.2f} GiB in {dt:.1f}s ({gib / max(dt, 1e-9):.2f} GiB/s)", flush=True)

    copy_dense(src_dir, dst_dir, mode=args.dense_mode, resume=args.resume)

    tmp_manifest = dst_dir / "manifest.json.tmp"
    with tmp_manifest.open("w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2)
        f.write("\n")
    os.replace(tmp_manifest, dst_dir / "manifest.json")
    print(f"wrote {dst_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
