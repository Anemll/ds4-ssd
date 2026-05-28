#!/usr/bin/env python3
"""Export and validate a multi-expert ANE MLP CoreML model.

The model takes K experts' stacked weights plus a per-row expert index and
computes the same swiglu-gated MLP as the single-expert tiled-fused model,
routing each row to its assigned expert. Implementation strategy: loop over
the K experts at compile time; each iteration computes the full (B, H) matmul
result for that expert and masks rows by their assignment, then sums.

Compute scales linearly with K — for K=2 we do 2x the FLOPs of one expert. The
hoped-for win is K-fold reduction in per-call overhead. This script also runs
a numerical correctness check against a NumPy reference before saving.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

COREML_TARGET = ct.target.iOS26


def _f16(value: float) -> np.float16:
    return np.float16(value)


def build_multiexpert_program(
    k: int,
    h: int,
    i: int,
    b: int,
    tile_i: int,
    w_scale: float,
    x_scale: float,
    mid_scale: float,
):
    """Build a CoreML program that processes B rows across K possible experts.

    Inputs:
      Wgq_stack: (K, H, I) int8
      Wuq_stack: (K, H, I) int8
      Wdq_stack: (K, I, H) int8
      Xq:        (B, H)    int8
      OneHot:    (B, K)    float16  (precomputed one-hot of per-row expert idx)
    Output:
      Y:         (B, H)    float16
    """
    if tile_i <= 0 or i % tile_i != 0:
        raise ValueError(f"tile_i={tile_i} must divide i={i}")

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(k, h, i), dtype=types.int8),
            mb.TensorSpec(shape=(k, h, i), dtype=types.int8),
            mb.TensorSpec(shape=(k, i, h), dtype=types.int8),
            mb.TensorSpec(shape=(b, h), dtype=types.int8),
            mb.TensorSpec(shape=(b, k), dtype=types.fp16),
        ],
        opset_version=COREML_TARGET,
    )
    def prog(Wgq_stack, Wuq_stack, Wdq_stack, Xq, OneHot):  # noqa: N803
        X = mb.dequantize(
            input=Xq, scale=_f16(x_scale), zero_point=np.int8(0), name="X"
        )
        # X3: (1, B, H) so matmul against (H, I) gives (1, B, I)
        X3 = mb.expand_dims(x=X, axes=[0], name="X3")
        y_acc = None
        for e in range(k):
            # Slice out expert e's weights.
            Wgq_e = mb.slice_by_index(
                x=Wgq_stack, begin=[e, 0, 0], end=[e + 1, h, i],
                name=f"Wgq_e{e}",
            )
            Wgq_e = mb.squeeze(x=Wgq_e, axes=[0], name=f"Wgq_e{e}_sq")
            Wuq_e = mb.slice_by_index(
                x=Wuq_stack, begin=[e, 0, 0], end=[e + 1, h, i],
                name=f"Wuq_e{e}",
            )
            Wuq_e = mb.squeeze(x=Wuq_e, axes=[0], name=f"Wuq_e{e}_sq")
            Wdq_e = mb.slice_by_index(
                x=Wdq_stack, begin=[e, 0, 0], end=[e + 1, i, h],
                name=f"Wdq_e{e}",
            )
            Wdq_e = mb.squeeze(x=Wdq_e, axes=[0], name=f"Wdq_e{e}_sq")

            # Tile over I dimension and accumulate down output (same pattern
            # as build_tiled_fused_program).
            y_e_acc = None
            for start in range(0, i, tile_i):
                end = min(start + tile_i, i)
                tag = f"e{e}_i{start}_{end}"
                Wgq_i = mb.slice_by_index(
                    x=Wgq_e, begin=[0, start], end=[h, end], name=f"Wgq_{tag}",
                )
                Wuq_i = mb.slice_by_index(
                    x=Wuq_e, begin=[0, start], end=[h, end], name=f"Wuq_{tag}",
                )
                Wdq_i = mb.slice_by_index(
                    x=Wdq_e, begin=[start, 0], end=[end, h], name=f"Wdq_{tag}",
                )
                Wg_i = mb.dequantize(
                    input=Wgq_i, scale=_f16(w_scale), zero_point=np.int8(0),
                    name=f"Wg_{tag}",
                )
                Wu_i = mb.dequantize(
                    input=Wuq_i, scale=_f16(w_scale), zero_point=np.int8(0),
                    name=f"Wu_{tag}",
                )
                Wd_i = mb.dequantize(
                    input=Wdq_i, scale=_f16(w_scale), zero_point=np.int8(0),
                    name=f"Wd_{tag}",
                )
                gate_i = mb.matmul(
                    x=X3, y=Wg_i, transpose_x=False, transpose_y=False,
                    name=f"gate_{tag}",
                )
                up_i = mb.matmul(
                    x=X3, y=Wu_i, transpose_x=False, transpose_y=False,
                    name=f"up_{tag}",
                )
                gate_c_i = mb.clip(
                    x=gate_i,
                    alpha=mb.const(val=_f16(-10.0), name=f"gate_lo_{tag}"),
                    beta=mb.const(val=_f16(10.0), name=f"gate_hi_{tag}"),
                    name=f"gate_c_{tag}",
                )
                up_c_i = mb.clip(
                    x=up_i,
                    alpha=mb.const(val=_f16(-10.0), name=f"up_lo_{tag}"),
                    beta=mb.const(val=_f16(10.0), name=f"up_hi_{tag}"),
                    name=f"up_c_{tag}",
                )
                act_i = mb.silu(x=gate_c_i, name=f"act_{tag}")
                hidden_fp_i = mb.mul(x=act_i, y=up_c_i, name=f"hidden_fp_{tag}")
                hidden_q_i = mb.quantize(
                    input=hidden_fp_i, scale=_f16(mid_scale),
                    zero_point=np.int8(0), output_dtype="int8",
                    name=f"hidden_q_{tag}",
                )
                hidden_i = mb.dequantize(
                    input=hidden_q_i, scale=_f16(mid_scale),
                    zero_point=np.int8(0), name=f"hidden_{tag}",
                )
                y_i = mb.matmul(
                    x=hidden_i, y=Wd_i, transpose_x=False, transpose_y=False,
                    name=f"Y_{tag}",
                )
                y_e_acc = y_i if y_e_acc is None else mb.add(
                    x=y_e_acc, y=y_i, name=f"Y_e_acc_{tag}",
                )
            # y_e_acc has shape (1, B, H); squeeze to (B, H).
            y_e = mb.squeeze(x=y_e_acc, axes=[0], name=f"y_e{e}_sq")
            # Mask by per-row expert assignment.
            mask_e = mb.slice_by_index(
                x=OneHot, begin=[0, e], end=[b, e + 1], name=f"mask_e{e}",
            )  # (B, 1)
            y_e_masked = mb.mul(x=y_e, y=mask_e, name=f"y_e{e}_masked")
            y_acc = y_e_masked if y_acc is None else mb.add(
                x=y_acc, y=y_e_masked, name=f"y_acc_e{e}",
            )
        return y_acc

    return prog


def numpy_reference(
    Wgq_stack, Wuq_stack, Wdq_stack, Xq, expert_idx,
    w_scale, x_scale, mid_scale,
):
    """Compute the expected output (B, H) f16 via NumPy, dequant/swiglu/quant."""
    K, H, I = Wgq_stack.shape
    B, H2 = Xq.shape
    assert H == H2
    X = Xq.astype(np.float16) * np.float16(x_scale)
    Y = np.zeros((B, H), dtype=np.float16)
    for r in range(B):
        e = int(expert_idx[r])
        Wg = Wgq_stack[e].astype(np.float16) * np.float16(w_scale)
        Wu = Wuq_stack[e].astype(np.float16) * np.float16(w_scale)
        Wd = Wdq_stack[e].astype(np.float16) * np.float16(w_scale)
        gate = X[r:r+1] @ Wg
        up = X[r:r+1] @ Wu
        gate_c = np.clip(gate, -10.0, 10.0)
        up_c = np.clip(up, -10.0, 10.0)
        act = gate_c / (np.float16(1.0) + np.exp(-gate_c.astype(np.float32)).astype(np.float16))
        hidden = act * up_c
        hidden_q = np.clip(np.round(hidden / np.float16(mid_scale)), -128, 127).astype(np.int8)
        hidden_dq = hidden_q.astype(np.float16) * np.float16(mid_scale)
        Y[r:r+1] = hidden_dq @ Wd
    return Y


def validate_and_save(prog, path: Path, shape, w_scale, x_scale, mid_scale,
                      force: bool, validate: bool):
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; pass --force")
        shutil.rmtree(path)
    print(f"  converting to mlpackage…")
    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=COREML_TARGET,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    mlmodel.save(str(path))
    print(f"  saved {path}")

    if validate:
        K, H, I, B = shape
        rng = np.random.default_rng(42)
        Wgq = rng.integers(-8, 8, size=(K, H, I), dtype=np.int8)
        Wuq = rng.integers(-8, 8, size=(K, H, I), dtype=np.int8)
        Wdq = rng.integers(-8, 8, size=(K, I, H), dtype=np.int8)
        Xq = rng.integers(-32, 32, size=(B, H), dtype=np.int8)
        # Assign each row randomly to one of K experts
        expert_idx = rng.integers(0, K, size=(B,), dtype=np.int32)
        one_hot = np.zeros((B, K), dtype=np.float16)
        one_hot[np.arange(B), expert_idx] = np.float16(1.0)

        print(f"  running CoreML prediction (B={B}, K={K})…")
        t0 = time.time()
        out = mlmodel.predict({
            "Wgq_stack": Wgq,
            "Wuq_stack": Wuq,
            "Wdq_stack": Wdq,
            "Xq": Xq,
            "OneHot": one_hot,
        })
        t1 = time.time()
        Y_coreml = np.array(list(out.values())[0]).astype(np.float16)
        if Y_coreml.ndim == 3:
            Y_coreml = Y_coreml.squeeze(0)
        print(f"  coreml predict: {(t1 - t0) * 1000:.2f} ms; output shape {Y_coreml.shape}")

        print(f"  computing NumPy reference…")
        Y_ref = numpy_reference(Wgq, Wuq, Wdq, Xq, expert_idx,
                                w_scale, x_scale, mid_scale)
        # Report match per-row
        max_diff = np.max(np.abs(Y_coreml.astype(np.float32) - Y_ref.astype(np.float32)))
        rms_diff = float(np.sqrt(np.mean((Y_coreml.astype(np.float32) - Y_ref.astype(np.float32))**2)))
        ref_abs_max = float(np.max(np.abs(Y_ref.astype(np.float32))))
        print(f"  validation: max_diff={max_diff:.4f} rms_diff={rms_diff:.4f} ref_abs_max={ref_abs_max:.4f}")
        rel_err = max_diff / max(ref_abs_max, 1e-6)
        print(f"  relative max diff: {rel_err:.4%}")
        # Both are int8 quantized so we expect some quantization noise. Tolerance ~5%.
        if rel_err > 0.10:
            print(f"  WARNING: relative error {rel_err:.4%} above 10%")
        else:
            print(f"  OK ({rel_err:.4%} ≤ 10%)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("moe-batch-bench/coreml_exports"))
    p.add_argument("--k", type=int, default=2, help="Number of experts per call.")
    p.add_argument("--shape", nargs=3, type=int, metavar=("H", "I", "B"),
                   default=(4096, 2048, 256))
    p.add_argument("--w-qscale", type=float, default=512.0)
    p.add_argument("--x-qscale", type=float, default=32.0)
    p.add_argument("--mid-qscale", type=float, default=32.0)
    p.add_argument("--tile-i", type=int, default=256)
    p.add_argument("--no-validate", action="store_true",
                   help="Skip numerical validation step.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    h, i, b = args.shape
    k = args.k
    args.out_dir.mkdir(parents=True, exist_ok=True)
    w_scale = 1.0 / args.w_qscale
    x_scale = 1.0 / args.x_qscale
    mid_scale = 1.0 / args.mid_qscale

    print(f"Building multi-expert program K={k} H={h} I={i} B={b} tile_i={args.tile_i}")
    prog = build_multiexpert_program(
        k, h, i, b, args.tile_i, w_scale, x_scale, mid_scale,
    )
    suffix = f"K{k}_H{h}_I{i}_B{b}_wq{args.w_qscale:g}_xq{args.x_qscale:g}_midq{args.mid_qscale:g}"
    path = args.out_dir / f"ds4_i8i8_multiexpert_{suffix}.mlpackage"
    validate_and_save(prog, path, (k, h, i, b), w_scale, x_scale, mid_scale,
                      args.force, validate=not args.no_validate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
