#!/usr/bin/env python3
"""Numerical validation of the multi-expert CoreML model against a NumPy
reference. Bypasses the int8-input predict() issue in coremltools by passing
float32 inputs; CoreML re-quantizes internally."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import coremltools as ct
import numpy as np


def numpy_reference(Wgq_stack, Wuq_stack, Wdq_stack, Xq, expert_idx,
                    w_scale, x_scale, mid_scale):
    """Reference using the same dequant/swiglu/quant chain as the CoreML
    program, in f16 throughout."""
    K, H, I = Wgq_stack.shape
    B = Xq.shape[0]
    X_dq = (Xq.astype(np.float32) * np.float32(x_scale)).astype(np.float16)
    Y = np.zeros((B, H), dtype=np.float16)
    for r in range(B):
        e = int(expert_idx[r])
        Wg = (Wgq_stack[e].astype(np.float32) * np.float32(w_scale)).astype(np.float16)
        Wu = (Wuq_stack[e].astype(np.float32) * np.float32(w_scale)).astype(np.float16)
        Wd = (Wdq_stack[e].astype(np.float32) * np.float32(w_scale)).astype(np.float16)
        x_row = X_dq[r]
        gate = (x_row.astype(np.float32) @ Wg.astype(np.float32)).astype(np.float16)
        up = (x_row.astype(np.float32) @ Wu.astype(np.float32)).astype(np.float16)
        gate_c = np.clip(gate, -10.0, 10.0).astype(np.float16)
        up_c = np.clip(up, -10.0, 10.0).astype(np.float16)
        # silu in f16
        sig = (1.0 / (1.0 + np.exp(-gate_c.astype(np.float32)))).astype(np.float16)
        act = (gate_c * sig).astype(np.float16)
        hidden = (act * up_c).astype(np.float16)
        hidden_q = np.clip(np.round(hidden.astype(np.float32) / mid_scale), -128, 127).astype(np.int8)
        hidden_dq = (hidden_q.astype(np.float32) * np.float32(mid_scale)).astype(np.float16)
        Y[r] = (hidden_dq.astype(np.float32) @ Wd.astype(np.float32)).astype(np.float16)
    return Y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--I", type=int, required=True)
    p.add_argument("--B", type=int, required=True)
    p.add_argument("--w-qscale", type=float, default=512.0)
    p.add_argument("--x-qscale", type=float, default=32.0)
    p.add_argument("--mid-qscale", type=float, default=32.0)
    p.add_argument("--time-iters", type=int, default=10)
    args = p.parse_args()
    K, H, I, B = args.K, args.H, args.I, args.B
    w_scale = 1.0 / args.w_qscale
    x_scale = 1.0 / args.x_qscale
    mid_scale = 1.0 / args.mid_qscale

    print(f"Loading model: {args.model}")
    m = ct.models.MLModel(str(args.model))

    rng = np.random.default_rng(0)
    Wgq = rng.integers(-8, 8, size=(K, H, I), dtype=np.int8)
    Wuq = rng.integers(-8, 8, size=(K, H, I), dtype=np.int8)
    Wdq = rng.integers(-8, 8, size=(K, I, H), dtype=np.int8)
    Xq  = rng.integers(-32, 32, size=(B, H), dtype=np.int8)
    expert_idx = rng.integers(0, K, size=(B,), dtype=np.int32)
    one_hot = np.zeros((B, K), dtype=np.float16)
    one_hot[np.arange(B), expert_idx] = np.float16(1.0)

    inputs = {
        "Wgq_stack": Wgq.astype(np.float32),
        "Wuq_stack": Wuq.astype(np.float32),
        "Wdq_stack": Wdq.astype(np.float32),
        "Xq": Xq.astype(np.float32),
        "OneHot": one_hot,
    }
    print(f"Predicting K={K} H={H} I={I} B={B} via CoreML…")
    # Warmup
    _ = m.predict(inputs)
    t0 = time.time()
    for _ in range(args.time_iters):
        out = m.predict(inputs)
    t1 = time.time()
    per_call_ms = (t1 - t0) / args.time_iters * 1000.0
    Y_coreml = np.array(list(out.values())[0]).astype(np.float16)
    if Y_coreml.ndim == 3:
        Y_coreml = Y_coreml.squeeze(0)
    print(f"  predict shape: {Y_coreml.shape}  per-call: {per_call_ms:.2f} ms")

    print("Computing NumPy reference…")
    Y_ref = numpy_reference(Wgq, Wuq, Wdq, Xq, expert_idx, w_scale, x_scale, mid_scale)

    diff = Y_coreml.astype(np.float32) - Y_ref.astype(np.float32)
    max_diff = float(np.max(np.abs(diff)))
    rms_diff = float(np.sqrt(np.mean(diff**2)))
    ref_abs_max = float(np.max(np.abs(Y_ref.astype(np.float32))))
    rel_err = max_diff / max(ref_abs_max, 1e-6)
    print(f"max_diff={max_diff:.4f} rms_diff={rms_diff:.4f} ref_abs_max={ref_abs_max:.4f}")
    print(f"relative max diff: {rel_err:.4%}")
    if rel_err > 0.10:
        print("FAIL: relative error >10%")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
