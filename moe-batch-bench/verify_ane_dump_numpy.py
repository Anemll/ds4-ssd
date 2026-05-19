#!/usr/bin/env python3
import argparse
import os

import numpy as np


def read(path, shape):
    arr = np.fromfile(path, dtype=np.float16)
    expect = int(np.prod(shape))
    if arr.size != expect:
        raise SystemExit(f"{path}: got {arr.size} elements, expected {expect}")
    return arr.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    args = ap.parse_args()

    meta = {}
    with open(os.path.join(args.dump_dir, "meta.txt")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k] = int(v)
    b, h, i = meta["B"], meta["H"], meta["I"]

    x = read(os.path.join(args.dump_dir, "input.bin"), (b, h)).astype(np.float32)
    wg = read(os.path.join(args.dump_dir, "W_gate.bin"), (h, i)).astype(np.float32)
    wu = read(os.path.join(args.dump_dir, "W_up.bin"), (h, i)).astype(np.float32)
    wd = read(os.path.join(args.dump_dir, "W_down.bin"), (i, h)).astype(np.float32)
    out = read(os.path.join(args.dump_dir, "output_ane.bin"), (1, b, h)).reshape(b, h).astype(np.float32)

    gate = x @ wg
    up = x @ wu
    mid = (gate / (1.0 + np.exp(-gate))) * up
    ref = mid @ wd

    diff = np.abs(out - ref)
    rel = diff / np.maximum(np.abs(ref), 1e-3)
    print(
        f"B={b} H={h} I={i} "
        f"max_abs={diff.max():.6g} mean_abs={diff.mean():.6g} "
        f"max_rel={rel.max():.6g} mean_rel={rel.mean():.6g}"
    )


if __name__ == "__main__":
    main()
