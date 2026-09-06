# Profiling the native HY4 path

`DS4_HY4_PROFILE=1` emits one `HY4_PROFILE` JSON line per completed token on
stderr. It adds clock/counter reads at existing boundaries, with no extra GPU
submissions or waits. Unset or 0 disables it. GPU measurements use completed
Metal command-buffer timestamps; CPU measurements use the worker thread's CPU
clock. The phase buckets describe work completed at the existing joins, so
CPU attention/pointwise fallbacks can move work between buckets.

`DS4_FLASH_MOE_PROFILE=1` adds per-layer router synchronization and
remap/install wall time. The latter includes slot bookkeeping, synchronous
SSD/file-cache reads, and installation; it is not pure SSD device latency.

## Reproduce the 48-slot agent measurement

Run from the permanent worktree. Keep the dense/sidecar package explicit and
leave the native top-8 unchanged. This uses the existing isolated system cache.

```sh
HY4_PACKAGE="$HOME/Models/HY4/Hy4-preview-Flash-STQ1_0"
HY4_LOG_DIR="$PWD/profile_runs/slot-protect-validation"
DS4_PROFILE=none DS4_FLASH_MOE_DIRECT_MMAP_AUTO=0 \
DS4_HY4_PROFILE=1 DS4_FLASH_MOE_PROFILE=1 \
DS4_AGENT_CACHE_DIR="$HY4_LOG_DIR/agent-cache" \
./ds4-agent \
  -m "$HY4_PACKAGE/model-dense-f16head.gguf" \
  --moe-sidecar "$HY4_PACKAGE/sidecar" \
  --moe-mode slot-bank --moe-slot-bank 48 --ctx 2048 \
  --tokens 64 --temp 0 --seed 1 --nothink --no-int8 \
  --trace "$HY4_LOG_DIR/hy4-profile-agent.trace" \
  --non-interactive \
  -p 'Explain how binary search works, then show a small Python example. Respond in text only; no tools.' \
  > "$HY4_LOG_DIR/hy4-profile.stdout" 2> "$HY4_LOG_DIR/hy4-profile.stderr"
```

Read the agent trace's `prefill ... transcript=N` and use N as `--decode-start`.
The measured run restored 709 system tokens and added 31 prompt tokens, so N
was 740. This avoids mixing prefill with decode:

```sh
python3 tools/summarize_hy4_profile.py \
  "$HY4_LOG_DIR/hy4-profile.stderr" --decode-start 740 --slots 48
```

## Recorded result on M5 Max

On the `db1cf53` compute path with profiling enabled, the complete run produced
64 tokens at 1.53 t/s. Its 64 decode evaluations covered positions 740-803:

| Measurement | Mean per decode token |
| --- | ---: |
| Whole token wall time | 653.7 ms |
| Completed GPU command-buffer time | 285.1 ms |
| FFN GPU work (routed plus shared) | 221.7 ms |
| Attention GPU work | 55.6 ms |
| Router GPU work | 5.0 ms |
| Worker CPU time | 180.2 ms |
| iHC pre-processing CPU time | 97.1 ms |
| iHC post-processing CPU time | 7.2 ms |
| Remap/install wall time | 194.2 ms |
| Router synchronization wall time | 18.9 ms |
| Expert cache hit rate | 60.8% |
| Expert bytes installed | 2.35 GiB |

GPU phase rows are included in GPU total; iHC CPU rows are included in worker
CPU time. Install wall time includes host work. These different clocks overlap
and must not be added as independent wall-time components. File-cache residency
was not flushed; this is not a cold-SSD measurement.

Every token logged eight selected experts and 48 cache slots, with 616 expert
references across the 77 routed layers. The executed routed path was
**per-expert**, with 1,848 quant matvec, 616 SwiGLU and 77 reduction dispatches:
**2,541 routed dispatches per token**. HY4 does not enter the DS4 six-expert fused
kernel. Increasing cache capacity does not enable fused HY4 kernels.

The source branch's fused `slot8` implementation dispatches two phases per MoE
layer (154 per token): gate/up/SwiGLU followed by down/weighted reduction. It
also has GPU iHC and optional shared-FFN/SSD-read overlap. Those major paths
have not yet been ported. The earlier gate/reduction optimization removes CPU
round trips but keeps the separate expert matvec loop.

Before reusing the fused source shaders, preserve the model's clamp-10
semantics: the source generic HY4 graph clamps routed gate/up, while the source
`kernel_hyv4_fused_phaseA_*` code at `34cccef` computes unclamped SwiGLU and its
fused eligibility check does not exclude clamped layers. This is a source-code
finding, not a claim about its effect on the reported 5 t/s workload. A port
must test clamp activation explicitly instead of copying that omission.

## Metal trace and CPU stack evidence

A native `Metal System Trace` was recorded for 70 seconds from the same
48-slot agent configuration, plus a 10-second `sample` of the launched process.
The trace includes prefill and partial decode; its recording deadline stopped
the launched process, so the complete timing above comes from the separate
profiling run. Instruments materially affects timing and is not the throughput
baseline.

The permanent local artifacts under `profile_runs/slot-protect-validation/` are:

- `hy4-slots48-metal.trace`: open with Instruments; about 10 GiB.
- `hy4-slots48-metal-toc.xml`, `hy4-slots48-gpu-intervals.xml`, and
  `hy4-slots48-metal-encoders.xml`: exported GPU/encoder evidence.
- `hy4-slots48-metal-summary.json`: merged GPU-active intervals, avoiding
  double-counting nested events.
- `hy4-slots48-cpu-sample.txt`: CPU stacks show waits in `hy4_post`,
  reads under `metal_graph_flash_moe_prepare_decode`, and CPU `hy4_pre` work.
- `hy4-slots48-profile.stderr` and `hy4-slots48-profile-summary.json`:
  complete per-token and per-layer accounting.

The exported Metal intervals are command/encoder-level GPU timings, not
hardware timings for individual kernels. Actual HY4 dispatch counts come from
the native runtime counters. No fused execution is claimed from generic startup
messages or from the `gate/reduce=Metal` line.

Enabling profiling passed the eight-slot native lifecycle regression with
bit-identical top-16 logits versus the previous run, including cancellation,
rewind and full-capacity snapshot restore. Profiling does not change math,
cache capacity, expert count, model bytes or compute defaults.
