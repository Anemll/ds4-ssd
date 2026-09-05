# HY4 routed quantization validation

The native HY4 sidecar path supports these on-disk GGML types. Values and
codebooks were ported from `Anemll/anemll-flash-llama.cpp` at
`34cccef1bc8db4f93048bff4033638024cbd8f34` (`ggml-common.h`,
`ggml-quants.c`, and the STQ1_0 Metal matvec).

| GGML type | Name | Weights/block | Bytes/block | HY4 use |
|---|---|---:|---:|---|
| 43 | STQ1_0 | 256 | 42 | Routed gate/up |
| 16 | IQ2_XXS | 256 | 66 | Routed gate/up |
| 18 | IQ3_XXS | 256 | 98 | Routed down |
| 23 | IQ4_XS | 256 | 136 | Routed down |

STQ1_0's scale is at byte 40, after 32 code bytes and 8 sign bytes. Each
four-value group has three signed nonzero values and one zero; the values
are strided by 16 within each 64-value chunk. Treating these as consecutive
four-value groups produces incorrect weights. The 42-byte layout stores
1.3125 bits per weight including its scale.

`hy4/hy4_quants.c` provides byte-safe CPU dequantization and a double-accumulated
reference matvec. `metal/hy4.metal` consumes the packed weights directly and
accumulates F32 values. `ds4_gpu_hy4_quant_matvec_tensor` accepts an individual
expert tensor view and joins the existing command batch. Existing DS4 fused
kernels are unchanged. The wrapper checks types, block alignment, row stride,
buffer sizes, and overflow before encoding.

## Reproduce focused tests

The model-free test can run without a llama.cpp installation. Supplying a
built source `libggml-base.dylib` additionally compares every dequantized
value against its independent exported CPU dequantizers:

```sh
make tests/test_hy4_quants tests/test_hy4_quants_sanitize
./tests/test_hy4_quants --metal \
  --reference /path/to/anemll-flash-llama.cpp/build/bin/libggml-base.dylib
./tests/test_hy4_quants_sanitize \
  --reference /path/to/anemll-flash-llama.cpp/build/bin/libggml-base.dylib
```

The 32 deterministic cases cover four types, widths 256/512/2048/6144,
17 output rows, plain/padded row strides, unaligned CPU input, nonzero Metal
tensor view offsets, zero/subnormal/positive/negative scales, and malformed
wrapper arguments. A separate invariant check covers all STQ codebook slots
and opposite sign halves. Each synthetic case uses less than 64 KiB of weights.
CPU source-reference values must compare exactly. GPU matvec outputs must be
finite and within `5e-5 * max(1, abs(cpu_reference))` per output.

To test three real packed rows without loading an expert bank or model:

```sh
./tests/test_hy4_quants --metal \
  --reference /path/to/anemll-flash-llama.cpp/build/bin/libggml-base.dylib \
  --sample 43 6144 /path/to/sidecar/layer_002.bin 0
```

## Recorded local results

On Apple M5 Max, all 32 source-reference/Metal cases and the CPU ASan+UBSan
run passed. Three rows from each quant type in
`Hy4-preview-Flash-STQ1_0/sidecar` also passed (13,392 sidecar bytes total):

| File | Offset | Type | Input width | Rows |
|---|---:|---|---:|---:|
| `layer_001.bin` | 0 | IQ2_XXS | 6144 | 3 |
| `layer_001.bin` | 6488064 | IQ3_XXS | 2048 | 3 |
| `layer_002.bin` | 0 | STQ1_0 | 6144 | 3 |
| `layer_075.bin` | 6488064 | IQ4_XS | 2048 | 3 |

Logs are retained locally under `profile_runs/hy4-validation/quants-*.log`.
These are isolated quantization and tensor lifetime checks. They do not
establish full-model logits parity, generation quality, slot-bank correctness,
long-context behavior, or performance. CUDA has no HY4 quant matvec port here.
