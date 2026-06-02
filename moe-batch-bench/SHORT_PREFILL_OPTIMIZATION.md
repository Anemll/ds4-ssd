# Short-Prefill Dedup MoE Optimization - M3 Ultra

Goal: maximize prefill throughput for short and medium prompts while keeping
route-only ANE offload isolated from dense shared-expert and O-proj ANE work.

## Root Cause

Dedup MoE prefill is expert-weight-streaming bound. Even short prompts touch a
large fraction of experts per layer, so the bytes streamed from the expert set
are nearly constant while token count varies. Short prompt throughput is
therefore mostly limited by cold SSD bandwidth and staging overhead, not by the
GPU math itself.

## Route-Only ANE Sweep

Script:

```bash
moe-batch-bench/run_prefill_chunk_minrefs_sweep_m3u.sh
```

The sweep isolates routed ANE:

```bash
DS4_FLASH_MOE_ANE_SHARED_EXPERT=0
DS4_FLASH_MOE_ANE_OUTPUT_PROJ=0
```

Common tuned I/O:

```bash
DS4_FLASH_MOE_PREAD_THREADS=6
DS4_FLASH_MOE_ASYNC_READAHEAD=12
DS4_FLASH_MOE_PREFETCH=3
DS4_FLASH_MOE_ASYNC_PREAD=1
DS4_FLASH_MOE_ASYNC_PREAD_AFTER_STAGE=1
```

ANE settings used for the M3U sweep: dual cluster, 2 threads, output queue 4,
`BATCHES=256`, `MAX_REFS=256`, and `CHUNK_BIG_REFS=1`.

## Key Result

`min_refs=384/512` is too conservative for routed ANE on M3U. For large prefill,
`32/64` usually wins. `128` remains good at 8K/16K but is not consistently best.

### Fixed chunk=16K

| prompt | GPU t/s | best ANE t/s | best min_refs |
|--------|--------:|-------------:|--------------:|
| 128    | 28.90   | no win       | n/a           |
| 256    | 41.00   | no win       | n/a           |
| 1K     | 93.84   | 98.30        | 32            |
| 4K     | 157.90  | 216.43       | 32            |
| 8K     | 206.19  | 284.76       | 128, with 32/64 within noise |
| 16K    | 189.57  | 261.61       | 32            |

### 8K prompt, chunk sweep

| chunk | GPU t/s | best ANE t/s | best min_refs |
|------:|--------:|-------------:|--------------:|
| 1024  | 84.52   | 97.57        | 32            |
| 2048  | 112.76  | 146.54       | 32            |
| 4096  | 145.85  | 201.36       | 64            |
| 8192  | 176.05  | 244.51       | 64            |
| 16384 | 206.19  | 284.76       | 128           |

## Practical Guidance

Do not use routed ANE for sub-512-token prefill on this system. Around 1K tokens
it is only a small win. At 4K+ it is a large win.

Chunk size dominates absolute throughput. ANE improves each chunk size, but it
does not rescue small chunks. For 8K+ prefill on M3U, prefer
`DS4_METAL_PREFILL_CHUNK=16384` unless memory pressure requires less.

## M3U Profile Scripts

The top-level M3U scripts default to Flash-MoE SSD/slot-bank mode:

```bash
./run_ane_prefill_profile_m3u.sh
./run_gpu_prefill_profile_m3u.sh
./run_bench_sweep_m3u.sh
```

All three support:

```bash
--help
--resident
--prefill-size 1k|4k|8k|16k|...
--prompt-file FILE
--chunk 16k
--raw-cap 16896
```

`--resident` switches from sidecar/slot-bank to full-model resident mode:

- `run_ane_prefill_profile_m3u.sh --resident` uses the resident ANE routed-MoE path (`ane_gpu`).
- `run_gpu_prefill_profile_m3u.sh --resident` uses the resident GPU/ALU baseline (`mulmm`).
- `run_bench_sweep_m3u.sh --resident` passes `--resident-ane-prefill` to `ds4-bench`.

The scripts print `prefill_kernel:` before the run so the log makes the
requested routed-MoE path explicit. The profile scripts keep summaries concise by
default; set `DS4_PREFILL_TEST_VERBOSE=1` to include generated per-layer planner
lines. Resident `resident MPP/NAX prefill layer=...` lines still require
`DS4_RESIDENT_MOE_MPP_STATS=1`. The bench harness also defaults
`DS4_RESIDENT_MOE_MPP_STATS=0`; set it to `1` only when per-layer resident stats
are the point of the run.

`--prefill-size` selects the matching `coding_${size}.txt` prompt and, for the
two `ds4` profile scripts, auto-sizes `--ctx` and raw KV cap unless those are
explicitly overridden. For example:

```bash
./run_ane_prefill_profile_m3u.sh --resident --prefill-size 16k --ctx 20k
./run_gpu_prefill_profile_m3u.sh --resident --prefill-size 16k --ctx 20k
./run_bench_sweep_m3u.sh --resident --prefill-size 16k --ctx-alloc 20k --gen-tokens 1
```

The resident 16K default raw cap is `16896` (`16384 + 512`), which avoids the
old 8K raw-cap clamp when testing a 16K chunk.
