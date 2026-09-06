# Profiling the native HY4 path

`DS4_HY4_PROFILE=1` emits one `HY4_PROFILE` JSON line per completed token on
stderr. It adds clock/counter reads at existing boundaries, with no extra GPU
submissions or waits. Unset or 0 disables it. GPU measurements use completed
Metal command-buffer timestamps; CPU measurements use the worker thread's CPU
clock. The phase buckets describe work completed at the existing joins, so
CPU attention/pointwise/iHC fallbacks can move work between buckets. With GPU
iHC, the attention bucket includes its iHC pre/post, the router bucket includes
FFN iHC pre, and the FFN bucket includes its iHC post. These are completion
phases rather than isolated shader timings.

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
kernel. This is the pre-fusion baseline; cache capacity and expert count remain fixed.

## Fused top-8 follow-up

The native fused path now adapts source `slot8` to DS4 bank views and resolved
host slot IDs: gate/up/SwiGLU followed by down/ordered weighting. Unlike the
source fused shaders at `34cccef`, it preserves the generic HY4 graph's routed
clamp10. Synthetic cases explicitly activate both gate and up clamps.
`DS4_HY4_UNFUSED=1` retains the reference; unset/0 enables the two-dispatch path
for contiguous banks. CPU pointwise, trace, per-slot and chunked-bank modes
retain separate operations.

The identical command above, actual model, 709-token system cache and 31-token
user suffix produced **2.14 tokens/s** (64 tokens in 29.972 s), versus baseline
1.53 tokens/s. The generated 64-token text was identical. Every decode profile
reports `routed_path=fused_top8`, 154 fused dispatches, zero separate routed
dispatches, top-8 and 48 slots. Native 16-token testing also matched all 272
top-logit IDs with maximum logit delta 0.0000095; lifecycle regressions pass.

| Measurement | Unfused | Fused |
| --- | ---: | ---: |
| Token wall | 653.7 ms | 468.2 ms |
| Completed GPU time | 285.1 ms | 100.5 ms |
| FFN GPU time | 221.7 ms | 33.5 ms |
| Attention GPU time | 55.6 ms | 58.3 ms |
| Worker CPU time | 180.2 ms | 175.8 ms |
| iHC pre CPU time | 97.1 ms | 97.2 ms |
| Remap/install wall | 194.2 ms | 195.9 ms |
| Expert hit rate | 60.8% | 60.8% |

These are bounded serial warm-file-cache runs, not cold-SSD throughput or an
8 tokens/s result. Small routing differences from F32 reduction order can
change cache counters even with identical generated text. The next section
measures the GPU iHC follow-up. Clock overlap definitions above still apply.

Evidence: `hy4-slots48-fused.{stdout,stderr}`, `hy4-slots48-fused-summary.json`,
`hy4-fused-test.log`, `hy4-fused-lifecycle.{jsonl,log}` and
`hy4-fused-greedy16.{jsonl,log}` in the same permanent validation directory.
The following Metal System Trace describes the earlier unfused baseline.
Fused execution is verified by counters on the actual encode path and GPU
command timestamps; no new fused Instruments capture is claimed here.

## GPU iHC follow-up

GPU independent HC keeps the existing residual completion joins, with exact
F32 post multiply/add and parallel mix dots. `DS4_HY4_CPU_IHC=1` selects the
scalar oracle; unset/0 uses GPU. No session cache layout changes are needed.

The identical fixed workload produces **2.72 tokens/s** (64 tokens / 23.503 s),
with the same generated text. All 64 decode rows confirm `ihc_path=metal`,
`routed_path=fused_top8`, top-8 and 48 slots. Mean token wall is 367.1 ms;
worker CPU is 74.9 ms, GPU command time 105.1 ms, iHC pre CPU 2.21 ms, post CPU
1.87 ms and head CPU 0.015 ms. CPU iHC's corresponding total was about 104 ms.
Remap/install remains 194.8 ms and hit rate 60.8%; SSD installation is now the
largest measured remaining phase. Device traffic is not inferred solely from
installed-byte counters because reads can be serviced by the OS file cache.

The focused HC suite passes 129,820 checks; post is bit exact and pre/head
maximum scaled error is 1.28e-5 (bound 4e-5). The eight-slot native lifecycle and
16-token model comparison pass, all 272 top IDs agree, maximum logit delta
0.0001049 (bound 0.001). The CPU opt-out matches the earlier lifecycle exactly.
This remains a bounded warm-cache result, below the 8 tokens/s objective.

Evidence: `hy4-hc-test.log`, `hy4-hc-lifecycle.{jsonl,log}`,
`hy4-hc-greedy16.{jsonl,log}`, `hy4-hc-cpu-lifecycle.{jsonl,log}`,
`hy4-slots48-hc.{stdout,stderr}`, `hy4-slots48-hc-summary.json`. GPU phase clocks
include iHC work at the joins described above; they cannot be treated as pure
attention/router/FFN kernel timings. No new Instruments trace is claimed.

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
