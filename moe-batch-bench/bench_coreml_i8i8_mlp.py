#!/usr/bin/env python3
"""Benchmark exported DS4 W8A8 Core ML MLP packages."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.proto import FeatureTypes_pb2


def random_i8(rng: np.random.Generator, shape: tuple[int, ...], dtype) -> np.ndarray:
    values = rng.integers(-127, 128, size=shape, dtype=np.int16)
    return values.astype(dtype, copy=False)


def model_kind(path: Path) -> str:
    name = path.name
    if "_fused_expert_" in name:
        return "full"
    for kind in ("gateup", "down", "full"):
        if f"_{kind}_" in name:
            return kind
    raise ValueError(f"could not infer model kind from {path.name}")


def make_inputs(
    kind: str,
    h: int,
    i: int,
    b: int,
    rng: np.random.Generator,
    input_dtypes: dict[str, np.dtype],
) -> dict[str, np.ndarray]:
    if kind == "gateup":
        return {
            "Wgq": random_i8(rng, (h, i), input_dtypes["Wgq"]),
            "Wuq": random_i8(rng, (h, i), input_dtypes["Wuq"]),
            "Xq": random_i8(rng, (b, h), input_dtypes["Xq"]),
        }
    if kind == "down":
        return {
            "Wdq": random_i8(rng, (i, h), input_dtypes["Wdq"]),
            "hidden_q": random_i8(rng, (b, i), input_dtypes["hidden_q"]),
        }
    if kind == "full":
        return {
            "Wgq": random_i8(rng, (h, i), input_dtypes["Wgq"]),
            "Wuq": random_i8(rng, (h, i), input_dtypes["Wuq"]),
            "Wdq": random_i8(rng, (i, h), input_dtypes["Wdq"]),
            "Xq": random_i8(rng, (b, h), input_dtypes["Xq"]),
        }
    raise ValueError(kind)


def effective_tflops(kind: str, h: int, i: int, b: int, ms: float) -> float:
    matmuls = {"gateup": 4.0, "down": 2.0, "full": 6.0}[kind]
    flops = matmuls * float(b) * float(h) * float(i)
    return flops / (ms / 1000.0) / 1.0e12


def parse_compute_units(value: str):
    table = {
        "cpu_ane": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
        "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }
    try:
        return table[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unknown compute unit: {value}") from exc


def input_dtypes(model: ct.models.MLModel) -> tuple[dict[str, str], dict[str, np.dtype]]:
    spec_dtypes = {}
    dtypes = {}
    for inp in model.get_spec().description.input:
        dtype = inp.type.multiArrayType.dataType
        if dtype == FeatureTypes_pb2.ArrayFeatureType.INT8:
            spec_dtypes[inp.name] = "INT8"
            # coremltools 9 / macOS 26 exposes native INT8 in the model spec, but
            # the Python predict bridge accepts constrained int8 values through
            # an int32 NumPy carrier. Passing np.int8 raises "value type not
            # convertible" on this local toolchain.
            dtypes[inp.name] = np.int32
        elif dtype == FeatureTypes_pb2.ArrayFeatureType.INT32:
            spec_dtypes[inp.name] = "INT32"
            dtypes[inp.name] = np.int32
        else:
            raise ValueError(f"unsupported input dtype for {inp.name}: {dtype}")
    return spec_dtypes, dtypes


def infer_shape(model: ct.models.MLModel, kind: str) -> tuple[int, int, int]:
    inputs = {
        inp.name: tuple(inp.type.multiArrayType.shape)
        for inp in model.get_spec().description.input
    }
    if kind in ("full", "gateup") and "Wgq" in inputs and "Xq" in inputs:
        h, i = inputs["Wgq"]
        b, xh = inputs["Xq"]
        if xh != h:
            raise ValueError(f"shape mismatch: Wgq H={h}, Xq H={xh}")
        return int(h), int(i), int(b)
    if kind == "down" and "Wdq" in inputs and "hidden_q" in inputs:
        i, h = inputs["Wdq"]
        b, hi = inputs["hidden_q"]
        if hi != i:
            raise ValueError(f"shape mismatch: Wdq I={i}, hidden_q I={hi}")
        return int(h), int(i), int(b)
    raise ValueError(f"could not infer shape from model inputs: {sorted(inputs)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--shape", nargs=3, type=int, metavar=("H", "I", "B"))
    p.add_argument("--kind", choices=("auto", "gateup", "down", "full"), default="auto")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--compute-units", choices=("cpu_ane", "all", "cpu_gpu", "cpu"), default="cpu_ane")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    kind = model_kind(args.model) if args.kind == "auto" else args.kind
    rng = np.random.default_rng(args.seed)
    model = ct.models.MLModel(
        str(args.model), compute_units=parse_compute_units(args.compute_units)
    )
    h, i, b = tuple(args.shape) if args.shape else infer_shape(model, kind)
    spec_dtypes, dtypes = input_dtypes(model)
    inputs = make_inputs(kind, h, i, b, rng, dtypes)

    for _ in range(args.warmup):
        model.predict(inputs)

    times_ms = []
    output_shapes = None
    for _ in range(args.iters):
        t0 = time.perf_counter()
        outputs = model.predict(inputs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)
        if output_shapes is None:
            output_shapes = {k: tuple(v.shape) for k, v in outputs.items()}

    mean_ms = statistics.fmean(times_ms)
    median_ms = statistics.median(times_ms)
    min_ms = min(times_ms)
    tflops = effective_tflops(kind, h, i, b, mean_ms)
    print(
        f"model={args.model} kind={kind} compute_units={args.compute_units} "
        f"H={h} I={i} B={b} warmup={args.warmup} iters={args.iters}"
    )
    print(f"spec_input_dtypes={{{', '.join(f'{k}: {v}' for k, v in spec_dtypes.items())}}}")
    print(f"python_input_carriers={{{', '.join(f'{k}: {v.__name__}' for k, v in dtypes.items())}}}")
    print(f"outputs={output_shapes}")
    print(
        f"mean_ms={mean_ms:.6f} median_ms={median_ms:.6f} min_ms={min_ms:.6f} "
        f"effective_TFLOPs={tflops:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
