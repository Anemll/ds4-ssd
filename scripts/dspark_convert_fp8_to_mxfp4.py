#!/usr/bin/env python3
"""Convert DSpark draft package FP8_E4M3 records to MXFP4_NATIVE in-place-safe.

Reads an existing dspark_draft package (no source checkpoint needed), copies
each layer .bin verbatim, appends MXFP4 plane-split data for selected FP8
records at the end of the file, and rewrites the manifest entries to point at
the appended planes. The original package is untouched.

Layout contracts (must match metal/mxfp4_common.h + metal/dense.metal):
  FP8 source  : data[out][in] bytes row-major; scales[ceil(out/128)][ceil(in/128)]
                E8M0 (one scale per 128x128 tile; kernel: scale_row=row>>7, col=k>>7);
                w = e4m3fn(byte) * 2^(scale-127).
  MXFP4 target: data[out][in/2] bytes SEQPAIR (byte b: low nibble = elem 2b,
                high = elem 2b+1); scales[out][in/32] E8M0 per 32-block per row;
                values decode via the E2M1 LUT [0,.5,1,1.5,2,3,4,6, -0..-6];
                e8m0 value = 2^(bits-127), bits==0 decodes to 0.0.

attn_output_a records are skipped by default: they run through the dedicated
grouped-strided FP8 rows5 kernel, which has no MXFP4 variant yet.

Usage:
  python3 scripts/dspark_convert_fp8_to_mxfp4.py \
      --src ~/Models/DSv4-Flash-DSpark-draft \
      --dst ~/Models/DSv4-Flash-DSpark-draft-mxfp4 [--include-output-a] [--force]
"""

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def build_e4m3fn_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 0:
            v = (man / 8.0) * 2.0 ** (-6)
        elif exp == 0xF and man == 0x7:
            v = np.nan  # e4m3fn NaN
        else:
            v = (1.0 + man / 8.0) * 2.0 ** (exp - 7)
        lut[b] = sign * v
    return lut


E4M3FN_LUT = build_e4m3fn_lut()


def e8m0_to_float(bits: np.ndarray) -> np.ndarray:
    # value = 2^(bits-127); bits==0 -> 0.0 (matches ds4mx_e8m0_to_float bit op).
    return np.where(bits == 0, 0.0,
                    np.ldexp(1.0, bits.astype(np.int32) - 127)).astype(np.float32)


def dequant_fp8(data: np.ndarray, scales: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    w = E4M3FN_LUT[data].reshape(out_dim, in_dim)
    s = e8m0_to_float(scales)
    # Expand 128x128 tile scales to full size.
    s_full = np.repeat(np.repeat(s, 128, axis=0)[:out_dim], 128, axis=1)[:, :in_dim]
    return w * s_full


def quant_mxfp4(w: np.ndarray):
    """Per-row per-32-block E2M1 + E8M0 quantization. Returns (packed, scale_bits, wq)."""
    out_dim, in_dim = w.shape
    assert in_dim % 32 == 0
    blocks = w.reshape(out_dim, in_dim // 32, 32)
    amax = np.abs(blocks).max(axis=2)  # [out][in/32]
    # Power-of-two scale so amax/scale <= 6 (E2M1 max).
    with np.errstate(divide="ignore"):
        exp = np.ceil(np.log2(amax / 6.0))
    exp = np.where(amax > 0, exp, -127.0)
    bits = np.clip(exp + 127.0, 0, 254).astype(np.uint8)  # 0 -> zero block
    scale = e8m0_to_float(bits)
    scale_safe = np.where(scale > 0, scale, 1.0)
    v = blocks / scale_safe[:, :, None]
    # Nearest E2M1 value by magnitude.
    mag = np.abs(v).astype(np.float32)
    idx = np.argmin(np.abs(mag[..., None] - E2M1_POS[None, None, None, :]), axis=3)
    neg = (v < 0) | ((v == 0) & (np.signbit(v)))
    nib = (idx + np.where(neg, 8, 0)).astype(np.uint8)
    # Zero blocks encode as all-zero nibbles.
    nib = np.where((scale[:, :, None] > 0), nib, 0).astype(np.uint8)
    flat = nib.reshape(out_dim, in_dim)
    packed = (flat[:, 0::2] | (flat[:, 1::2] << 4)).astype(np.uint8)
    wq = (E2M1_POS[idx] * np.where(neg, -1.0, 1.0) * scale[:, :, None]).reshape(out_dim, in_dim)
    return packed, bits, wq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--include-output-a", action="store_true",
                    help="also convert attn_output_a (needs MXFP4 strided kernel)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.src).expanduser()
    dst = pathlib.Path(args.dst).expanduser()
    if dst.exists():
        if not args.force:
            print(f"error: {dst} exists (use --force)", file=sys.stderr)
            return 2
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    manifest = json.loads((src / "manifest.json").read_text())
    entries = manifest["entries"]

    # Copy every package file verbatim first.
    for f in manifest["files"]:
        shutil.copyfile(src / f["path"], dst / f["path"])
    if (src / "golden").exists():
        shutil.copytree(src / "golden", dst / "golden")

    handles = {f["path"]: open(dst / f["path"], "r+b") for f in manifest["files"]}
    src_handles = {f["path"]: open(src / f["path"], "rb") for f in manifest["files"]}

    converted = 0
    report = []
    for e in entries:
        if e.get("quant_type") != "FP8_E4M3":
            continue
        if "attn_output_a" in e["name"] and not args.include_output_a:
            report.append((e["name"], "skipped (strided kernel path)", 0.0))
            continue
        out_dim, in_dim = int(e["shape"][0]), int(e["shape"][1])
        if in_dim % 32 != 0:
            report.append((e["name"], "skipped (in_dim % 32)", 0.0))
            continue
        sh = src_handles[e["file"]]
        sh.seek(e["plane_data_offset"])
        data = np.frombuffer(sh.read(e["plane_data_bytes"]), dtype=np.uint8)
        sh.seek(e["plane_scale_offset"])
        srows, scols = int(e["scale_shape"][0]), int(e["scale_shape"][1])
        scales = np.frombuffer(sh.read(e["plane_scale_bytes"]), dtype=np.uint8).reshape(srows, scols)

        w = dequant_fp8(data, scales, out_dim, in_dim)
        if not np.isfinite(w).all():
            report.append((e["name"], "skipped (non-finite fp8 values)", 0.0))
            continue
        packed, bits, wq = quant_mxfp4(w)
        denom = float(np.sqrt((w.astype(np.float64) ** 2).mean())) or 1.0
        rel = float(np.sqrt(((wq - w).astype(np.float64) ** 2).mean())) / denom

        out = handles[e["file"]]
        out.seek(0, 2)
        data_off = out.tell()
        out.write(packed.tobytes())
        scale_off = out.tell()
        out.write(bits.tobytes())

        e["quant_type"] = "MXFP4_NATIVE"
        e["storage_layout"] = "mxfp4_plane_split_v1"
        e["plane_data_format"] = "FP4_E2M1_SEQPAIR"
        e["plane_data_offset"] = data_off
        e["plane_data_bytes"] = packed.nbytes
        e["plane_scale_offset"] = scale_off
        e["plane_scale_bytes"] = bits.nbytes
        e["scale_shape"] = [out_dim, in_dim // 32]
        e["converted_from"] = "FP8_E4M3"
        converted += 1
        report.append((e["name"], "converted", rel))

    for h in handles.values():
        h.close()
    for h in src_handles.values():
        h.close()

    manifest["source"] = dict(manifest.get("source") or {},
                              converted="fp8_e4m3->mxfp4 via dspark_convert_fp8_to_mxfp4.py")
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"converted {converted} records -> {dst}")
    for name, status, rel in report:
        extra = f" rel_rms={rel:.4f}" if status == "converted" else ""
        print(f"  {name}: {status}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
