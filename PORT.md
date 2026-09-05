# Flash-MoE slot correctness port

Source: [Anemll/anemll-flash-llama.cpp@34cccef1bc8db4f93048bff4033638024cbd8f34](https://github.com/Anemll/anemll-flash-llama.cpp/commit/34cccef1bc8db4f93048bff4033638024cbd8f34),
“Improve HY4 SSD streaming correctness and enable safe defaults”.
Target base: `Anemll/ds4-ssd` `hy3` at `6aa38025f16435a29e14d5e2d75ed063bd28ffc1`.

## Mapping

| Source change | DS4 equivalent and port |
| --- | --- |
| `flash_moe_protect_request_slots()` before `reserve_expert_slot()` | `ds4_flash_moe_protect_request_slots()` and `ds4_flash_moe_resolve_request_slots()` in `ssd/ssd_flash_moe_slots.h` validate the whole request and reciprocal ownership before hard-reserving every resident hit. `metal_graph_flash_moe_pick_slot()` delegates to the shared selector, retaining soft prefetch hints but never choosing a hard-reserved slot. |
| Request-local duplicate tracking; install only after all slots are resolved | `metal_graph_flash_moe_prepare_decode_prefetch_ids()`, `metal_graph_flash_moe_decode_async_handout()`, `metal_graph_flash_moe_prepare_decode()`, and `metal_graph_flash_moe_run_tiny_batch_slotbank()` use `metal_graph_flash_moe_resolve_request()` / `metal_graph_flash_moe_install_request()` before writes. Matching hits may already be reserved; repeated IDs reuse their chosen slot. Capacity/validation failure cannot partially install a request. |
| Assert expert/slot ownership; invalidate destinations after a failed direct read | DS4 reserve, commit, and invalidate helpers maintain reciprocal mappings. Failed or discarded direct-to-slot jobs invalidate destinations and replay state after joining, before the bank is reused. |
| `finish_shared_io()` at callback stop, graph exit, and runtime destruction | `metal_graph_flash_moe_decode_prefetch_finish()` / `_cleanup()` and `metal_graph_flash_moe_async_load_cleanup()` join outstanding jobs and invalidate abandoned direct writes. `ds4_flash_prefill_async_cancel_and_drain()` is called at failed DeDup layers, `metal_graph_prefill_layer_major()` / `metal_graph_prefill_chunked_range()` boundaries and `metal_graph_reset_prefill_state()`, with existing destructor joins retained. Scratch-read cancellation uses atomic loads/stores. DS4 uses pthread workers rather than futures. |
| Avoid invoking the downstream evaluation callback twice | Audited: DS4 has no llama scheduler `ask`/evaluation callback wrapper. Its progress/display callbacks report distinct events and return void. No duplicate-evaluation equivalent found; no callback API change. |
| Slot reservation regression and HY4 lifecycle tests | Model-free tests exercise production C slot helpers and DS4 reader cleanup. Native DS4 Metal smoke tests exercise the existing SSD runtime when a local model is available. HY4-specific test fixtures are excluded. |

## Prefill and shared-bank semantics

Decode and tiny-batch prefill consume several resident slots together, so their
whole request needs hard protection. Larger expert-major DeDup prefill consumes
one expert at a time and synchronizes queued GPU work before mutating the shared
cache. Its cache-population hints remain soft: pinning the whole prompt's expert
union would incorrectly fail when that union exceeds bank capacity.

Hard request protection is unconditional. `DS4_FLASH_MOE_PREPROTECT_TOPK` remains
an optional soft eviction hint; correctness does not depend on it. No new knobs,
profile defaults, sidecar layout changes, or compute kernels are introduced.

## Excluded from 34cccef

- HY4 shared-FFN overlap eligibility/defaults and associated timing/startup output.
- Exact HY4 iHC post broadcast, GGUF/DSA/HY4 Metal support, `HY4.md`, and HY4 tests.
- llama.cpp callback APIs and futures; only the relevant lifetime invariants map to DS4.
- Unrelated ANE/ALU/fusion work and antirez upstream merges.

## Validation and risk

Validated on Apple M5 Max, macOS 27 / Xcode beta:

- `make -j8 all ds4_test`: all five Metal executables and the existing test binary
  build successfully. Existing signedness, unused-function, and SDK deprecation
  warnings remain; no new warning originates in the changed slot/drain code.
- `make flash-moe-slot-test flash-moe-slot-test-sanitize`: deterministic fixtures
  and 40,000 seeded randomized requests pass, including ASan/UBSan. Tests call the
  production planner/protection/picker; only payload upload/commit is simulated.
  An extracted `hy3` picker reproduces the failure: the first miss selects slot 0,
  which contains the oldest resident required later in the same request.
- `make flash-moe-io-test`: the production implementation passes deterministic
  blocked-worker drain, scratch cancellation, paused split, pool reuse, and
  partial direct-write invalidation tests. The same test passes ASan/UBSan with
  the translation unit instrumented (the existing Metal support objects are not).
- `./ds4_test --server` and `./ds4_test --metal-kernels`: pass on the native host.
- `tests/test_flash_moe_session.c`: a local DS4 IQ2_XXS/Q2_K sidecar passes real
  six-slot prefill (23 tokens), four decode steps, resumed prefill, another decode,
  invalidation, and repeated-prompt argmax (`343`). Direct overlapping reads and
  scratch-prefetch/synchronous fallback both pass. Startup and runtime status
  confirm `slots=6`; full mmap auto-selection is explicitly disabled.
- `git diff --check`: passes.

The native session smoke is opt-in and needs a local **DS4** package. Example
for direct overlapping reads (its model-free counterparts need no model):

```sh
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major \
DS4_PROFILE=none DS4_FLASH_MOE_DIRECT_MMAP_AUTO=0 \
DS4_FLASH_MOE_DIRECT_MMAP_BANK=0 DS4_FLASH_MOE_PREPROTECT_TOPK=0 \
DS4_FLASH_MOE_ANE_PREFILL=0 DS4_FLASH_MOE_ASYNC_PREAD=1 \
DS4_FLASH_MOE_XLAYER_PREFETCH=1 DS4_METAL_PREFILL_CHUNK=32 \
DS4_FLASH_MOE_DECODE_PREFETCH=1 DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=6 \
DS4_FLASH_MOE_DECODE_PREFETCH_SCRATCH_ONLY=0 \
DS4_FLASH_MOE_DECODE_PREFETCH_SHARED_DOWN=1 make flash-moe-session-test
```

The scratch variant uses `DS4_FLASH_MOE_DECODE_PREFETCH_SCRATCH_ONLY=1`,
`DS4_FLASH_MOE_DECODE_PREFETCH_MAX_LOADS=2`, and shared-down overlap unset.
These are test overrides; runtime/profile defaults are unchanged.

Correctness may change which unrequested expert is evicted. Interrupted or failed
direct reads may discard cached destinations and require a later SSD reread.
No throughput improvement is claimed. Full-logit parity, native tiny-batch MTP,
HY3/GLM model inference, all optional cache layouts, CUDA, and device-specific
performance were not validated. Interruption/failure injection is deterministic
and model-free; the native session smoke tests successful lifecycle transitions.
