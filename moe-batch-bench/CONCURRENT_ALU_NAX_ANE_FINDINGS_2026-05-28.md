# Concurrent ALU + NAX + ANE Findings

Date: 2026-05-28

## Summary

Concurrent ALU+NAX on the GPU is real and useful. The current production winner
remains the resident GPU Plan A path: NAX-half fused gate/up/SwiGLU in the
compact-bridge routed MoE path.

Concurrent GPU+ANE is also real. A dedicated ANE producer `MTLCommandQueue`
works and avoids draining the main GPU queue. However, K=1 ANE offload does not
save enough GPU work to pay for ANE eval/write-surface overhead. Useful headroom
only appears when many experts are skipped/offloaded, so the promising next
architecture is grouped or batched multi-expert ANE, not one `_eval` per expert.

## Measurements

Same local build and prompt (`moe-batch-bench/coding_187k.txt`) on M5 Max:

| Path | Result |
|---|---:|
| Plan A GPU ALU+NAX baseline | 558.7 t/s |
| Direct ANE queue K=1, output read | 550.0 t/s |
| Direct ANE queue K=1, no output read | 551.4 t/s |
| Free skip K=1 hot expert | 560.3-560.8 t/s |
| Free skip all eligible experts, `min_refs=1` | 589.5 t/s |

Earlier cooldown-separated Plan A peak was about 573 t/s. The local 558.7 t/s
baseline was likely lower due to thermal/run conditions, but the relative result
is still clear.

## Dedicated Producer Queue

The direct ANE producer path now has a separate queue:

```objc
static id<MTLCommandQueue> g_queue;
static id<MTLCommandQueue> g_ane_producer_queue;

g_queue = [g_device newCommandQueue];
g_ane_producer_queue = [g_device newCommandQueue];
```

The important rule is that ANE producer command buffers must not touch
`g_batch_cb`, `ds4_gpu_flush_commands`, `ds4_gpu_synchronize`, or `g_pending_cbs`.
Otherwise the path serializes with the main GPU queue and loses the overlap we
are trying to measure.

## Direct ANE Producer Shape

The producer encodes weight dequant and activation gather on the dedicated ANE
producer queue:

```objc
id<MTLCommandBuffer> cb = [g_ane_producer_queue commandBuffer];

encode_dequant_gate_up_down(cb);
encode_gather_token_f32_i8(
    cb,
    x_f32,
    hids,
    x_i8,
    refs,
    expert_in_dim,
    DS4_N_EXPERT_USED,
    x_qscale);

[cb commit];
pthread_create(&job->thread, NULL, ds4_gpu_ane_direct_eval_thread, job);
```

The worker waits for that producer CB, then calls ANE:

```objc
[job->producer_cb waitUntilCompleted];

ds4_ane_mlp_i8w_i8x_tiled_fused_eval(
    job->ctx,
    gate_i8,
    up_i8,
    down_i8,
    x_i8,
    out_f16);
```

There is also a no-read probe using:

```objc
ds4_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface(...);
```

That removed CPU output readback from the probe, but only improved throughput
from 550.0 to 551.4 t/s, so readback is not the dominant issue.

## Main Scheduling

The outer routed-MoE schedule is:

```c
sync_router_results_to_cpu();
select_hottest_experts_by_ref_count();
build_hids_pair_ids();

start_ane_direct_job_on_dedicated_queue();
ds4_gpu_set_ane_skip_mask(mask, 256);

ds4_gpu_routed_moe_batch_tensor(...);  // main GPU ALU+NAX continues

ds4_gpu_clear_ane_skip_mask();
finish_ane_direct_jobs();
```

The overlap is mechanically real: ANE runs from a pthread while the GPU continues
through the resident routed MoE path.

## Pair-ID Gather Bug

The earlier direct-eval probe had a correctness bug. `hids` are pair IDs:

```c
hids[row] = token * DS4_N_EXPERT_USED + slot;
```

The gather kernel decodes token as:

```metal
token = hids[row] / selected_experts;
```

Therefore `selected_experts` must be `DS4_N_EXPERT_USED` (`6`), not `1`.
The invalid earlier direct-eval result should not be used as evidence.

## Why K=1 Loses

Warm direct producer timing is small:

```text
producer encode ~= 0.03-0.05 ms
producer wait   ~= 0.7 ms
ANE eval        ~= 1.8-2.2 ms typical
post-GPU drain  ~= 2.1-2.4 ms typical
```

The win equation is:

```text
producer_overhead + max(GPU_remaining, ANE_eval) < GPU_all
```

For K=1:

```text
GPU_all - GPU_remaining is too small
ANE_eval is ~2 ms
therefore no win
```

Even free-skipping one hot expert only reached about 560 t/s, barely above the
local Plan A baseline. That means one expert per layer is not enough work to be
a useful ANE relief valve.

## Headroom Check

Free-skipping all eligible experts with `DS4_RESIDENT_MOE_ANE_MIN_REFS=1`
reached 589.5 t/s. This is intentionally incorrect output, but it proves there
is theoretical headroom when a broad set of routed experts is removed from the
GPU path.

That points away from K=1 and toward a grouped ANE architecture:

- batch multiple experts into one ANE submission;
- use persistent/hot expert weight surfaces;
- use `_eval_xonly` where the same expert can reuse already-written weights;
- or build a multi-expert ANE graph so the producer overhead is amortized.

## Conclusion

ALU+NAX on the GPU is the right current production path. ANE sidecar concurrency
is real, and the dedicated producer queue is useful infrastructure, but per-expert
direct eval is too fine-grained. The next viable path is grouped/multi-expert ANE
offload with much lower scheduling overhead per unit of routed MoE work.

