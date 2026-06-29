# ds4-ssd

`ds4-ssd` is an alpha fork of antirez's DwarfStar 4 (`ds4`) inference engine
for DeepSeek V4 Flash. The fork keeps the narrow, self-contained DS4 runtime and
adds an SSD-streamed routed-MoE sidecar path for Apple Silicon systems where a
fully resident model is not practical.

The main alpha feature is SSD streaming: dense tensors stay in a normal GGUF,
while routed experts live in a sidecar directory and are paged through a
slot-bank cache. Resident full-GGUF mode is still supported for high-memory
machines. Apple Silicon optimizations include NAX, the Apple neural-accelerator
backed `matmul2d` path used by Metal on M5-class hardware, plus Apple Neural
Engine routed-MLP prefill paths where the measured profile says ANE wins.

This branch is intentionally narrower than the research branch. It keeps the
runtime, Metal shaders, GGUF tools, correctness tests, sidecar smoke, and core
docs, while dropping profiling scripts, handoff notes, session exports, and
bench-only ANE probes from the public alpha tree.

## Status

Alpha means:

- SSD sidecar mode is the release headline.
- Resident full-GGUF mode remains available.
- Correctness vectors and an executable 16K sidecar smoke are the current test
  bar.
- Broader mode coverage, CI, and performance regression automation are planned
  for the next stage.

The in-repo `gguf-tools/deepseek4-quantize` tool builds resident GGUFs. It does
not yet emit the sidecar layout. For this alpha, use the prebuilt sidecar
package at
[anemll/dsv4-iq2xxs-expert-major](https://huggingface.co/anemll/dsv4-iq2xxs-expert-major).

## Build

On macOS:

```sh
make
```

This builds:

- `./ds4`: CLI runner.
- `./ds4-server`: OpenAI/Anthropic/Responses-compatible local server.
- `./ds4-bench`: throughput sweeps.
- `./ds4-eval`: evaluation helper.
- `./ds4-agent`: local coding-agent frontend.

`metal/` is a required build input. Do not prune it.

CUDA sources are inherited from upstream DS4 and kept in tree, but the alpha
validation focus is Apple Silicon SSD streaming.

## Run SSD Sidecar Mode

Download the prebuilt sidecar package:

```sh
./download_model.sh sidecar
```

Or the native MXFP4 package (bit-exact MXFP4 routed experts, ~156 GB,
[anemll/DSv4-Flash-MXFP4-native-flash](https://huggingface.co/anemll/DSv4-Flash-MXFP4-native-flash)):

```sh
./download_model.sh mxfp4
./ds4 -m models/DSv4-Flash-MXFP4-native-flash --ssd-cache auto -p "Hello"
```

Sidecar manifests that contain `MXFP4_NATIVE` storage automatically default
`DS4_MXFP4_NATIVE=1` when the variable is unset. Explicitly setting
`DS4_MXFP4_NATIVE=0` keeps the guard enabled and will reject native MXFP4
sidecars.

`--ssd-cache` sizes the resident expert slot bank (`auto`, or an explicit value
like `32GB`). Any size is safe: on RAM-limited machines the bank is clamped so
prefill cannot overflow memory and auto-shrinks after prefill so decode-miss
reads stay served by the OS file cache.

Then set `DS4_SIDECAR_DIR` to the sidecar package root containing
`manifest.json` and `dense/model-dense.gguf`:

```sh
export DS4_SIDECAR_DIR="$PWD/models/dsv4-iq2xxs-expert-major"
```

Run the package root directly. DS4 detects the dense GGUF and sidecar metadata;
no explicit `--moe-sidecar` or `--moe-mode` flag is needed:

```sh
./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --moe-slot-bank 8 \
  --ctx 8192 \
  -p "Hello"
```

`--ctx 8192` is the KV window in this conservative first-run example. Leave the
Metal raw-KV cap automatic so it follows the prefill chunk size and server
checkpoint frontiers stay aligned.

Start with `--moe-slot-bank 8` and raise it once you confirm there is headroom.
The slot bank is the cap on resident routed-expert slots, so a larger value
trades RAM for fewer SSD reads. `--moe-slot-bank 64 --ctx 32768` is a
high-memory setting, not the safest default.

For cache-budget comparisons, especially against upstream SSD-streaming runs,
use `--ssd-cache` instead of manually choosing a slot count. Explicit sizes set
the target routed-expert slot-bank budget, while `auto` sizes the slot bank from
currently available memory after dense weights and context buffers are
estimated:

```sh
./ds4 -m "$DS4_SIDECAR_DIR" --ssd-cache 32G --ctx 32768 -p "Hello"
./ds4 -m "$DS4_SIDECAR_DIR" --ssd-cache 64G --ctx 32768 -p "Hello"
./ds4 -m "$DS4_SIDECAR_DIR" --ssd-cache auto --ctx 32768 -p "Hello"
```

Run the committed sidecar smoke:

```sh
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major make sidecar-smoke
```

To confirm SSD streaming is active, look for these startup lines:

```text
applied sidecar tuning profile
Flash-MoE sidecar loaded
Flash-MoE slot banks allocated
```

If `-m` points at a directory containing `manifest.json` and
`dense/model-dense.gguf`, DS4 auto-detects SSD sidecar mode, rewrites the model
path to the dense GGUF, and enables sidecar slot-bank mode internally. If you
pass only `-m /path/to/full-model.gguf`, DS4 is in resident/full-GGUF mode.

See [docs/SIDECAR.md](docs/SIDECAR.md). For the external expert-sidecar
export wrapper, see [docs/SIDECAR_EXPORT.md](docs/SIDECAR_EXPORT.md); the
prebuilt Hugging Face sidecar remains the turnkey low-RAM package.

Machine-specific defaults for M5, M5 Max, M3 Ultra, and M1 Max are selected
from `ds4_profile.json`. Profiles set defaults only; exported environment
variables still win. Profiles choose ANE only for chunk shapes where it has
measured faster than GPU or NAX on that machine. See
[docs/PROFILES.md](docs/PROFILES.md) and
[docs/STREAMING_KNOBS.md](docs/STREAMING_KNOBS.md).

## Run Resident Sidecar Mode

On high-memory Apple Silicon systems, a sidecar package can also be loaded as a
fully resident all-expert slot bank. This keeps the sidecar package layout
(`manifest.json` plus `dense/model-dense.gguf`) but avoids decode-time SSD
expert misses.

Use `--resident` with the sidecar package directory:

```sh
./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --ctx 8192 \
  -p "Hello"
```

`--resident` autodetects the dense GGUF, enables sidecar slot-bank mode,
defaults the slot bank to all experts, preloads and touches the resident bank,
and disables direct-mmap auto selection. If you explicitly pass
`--moe-slot-bank`, that value is honored.

The same simplified startup is supported by the local server:

```sh
./ds4-server \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --ctx 32768 \
  --host 127.0.0.1 \
  --port 8000
```

The OpenAI-compatible server advertises `deepseek-v4-flash` from
`GET /v1/models`:

```sh
curl http://127.0.0.1:8000/v1/models
```

Use that id in API calls unless you intentionally want a compatibility alias:
`deepseek-chat` disables thinking and `deepseek-reasoner` enables thinking.

## Run MTP With A Sidecar

MTP speculative decoding is optional. It uses the normal sidecar or resident
sidecar model as the target, plus a small support GGUF that drafts candidate
tokens. Download the support model first:

```sh
./download_model.sh mtp
export DS4_MTP_GGUF="$PWD/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
```

For a basic sidecar MTP smoke test, keep the draft length at 2 and use greedy
decoding:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 ./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --mtp "$DS4_MTP_GGUF" \
  --mtp-draft 2 \
  --mtp-margin 0 \
  --temp 0 \
  --nothink \
  -n 128 \
  -p "Write a short Python function that parses a CSV line with quoted fields."
```

Expected startup logs include:

```text
MTP support model loaded
MTP sidecar verifier
```

Expected summary output includes an acceptance line when MTP ran:

```text
ds4: mtp acceptance: 86.3% (1740/2016 draft tokens)
```

If the acceptance line is missing, MTP did not actually draft or verify tokens.
Check that `--mtp "$DS4_MTP_GGUF"` was passed and that `--mtp-draft` is greater
than 1.

For fully resident sidecar runs, enable the sidecar batch verifier. This is the
path to test MTP with the all-expert resident slot bank and without SSD
decode-miss I/O:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 \
DS4_MTP_SIDECAR_BATCH_VERIFY=1 \
./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --mtp "$DS4_MTP_GGUF" \
  --mtp-draft 2 \
  --mtp-margin 0 \
  --temp 0 \
  --nothink \
  -n 256 \
  -p "Hello"
```

For full native MXFP4 sidecars, set `DS4_MTP_SIDECAR_BATCH_VERIFY=1`; otherwise
DS4 skips MTP by default because the exact sidecar verifier is slower than
ordinary banked decode on that layout.

For hybrid resident sidecars with `IQ2_XXS` gate/up and `MXFP4_NATIVE` down,
`DS4_MTP_SIDECAR_BATCH_VERIFY=1` intentionally falls back to exact decode2 by
default. The experimental hybrid batch verifier is available for profiling, but
was measured slower than exact decode2:

```sh
DS4_MTP_HYBRID_BATCH_VERIFY_EXPERIMENT=1 \
DS4_MTP_SIDECAR_BATCH_VERIFY=1 \
./ds4 -m "$DS4_SIDECAR_DIR" --resident --mtp "$DS4_MTP_GGUF" --mtp-draft 2
```

MTP is only expected to help when the target verifier is cheaper than the
accepted target tokens it replaces. A high acceptance rate alone does not
guarantee a speedup; compare the final `generation:` tokens-per-second line
against the same command without `--mtp`.

## Run DSpark With A Sidecar

The primary DSpark path is the Flash sidecar target plus a separate DS4-owned
DSpark draft package. Treat GGUF main-model runs as a compatibility path for
agent demos; the sidecar path is the reference for speed and correctness. The
draft checkpoint must match the target shape, so use the Flash DSpark checkpoint
with Flash sidecars, not the Pro DSpark checkpoint.

Use these paths in the examples below:

```sh
export DS4_SIDECAR_DIR=/Users/anemll/Models/flash/dsv4-iq2xxs-expert-major
export DS4_DSPARK_DRAFT=/Users/anemll/Models/DSv4-Flash-DSpark-draft
export TEST_PROMPT='Make a game of Space Invader in Pygame'
```

Download the pre-exported Flash DSpark draft package:

```sh
DS4_DSPARK_DRAFT_DIR="$DS4_DSPARK_DRAFT" ./download_model.sh dspark
```

This downloads [anemll/DSv4-Flash-DSpark-draft](https://huggingface.co/anemll/DSv4-Flash-DSpark-draft)
directly into the DS4 runtime package layout. To rebuild the package locally
from the original DeepSeek shards instead, download only the Flash DSpark draft
shards:

```sh
mkdir -p /Volumes/TB36/Models/DS/DeepSeek-V4-Flash-DSpark
hf download deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --local-dir /Volumes/TB36/Models/DS/DeepSeek-V4-Flash-DSpark \
  --include config.json \
  --include model.safetensors.index.json \
  --include model-00046-of-00048.safetensors \
  --include model-00047-of-00048.safetensors \
  --include model-00048-of-00048.safetensors
```

Export the DS4-owned draft package:

```sh
scripts/export_dspark_draft.sh \
  --source-dir /Volumes/TB36/Models/DS/DeepSeek-V4-Flash-DSpark \
  --out-dir "$DS4_DSPARK_DRAFT" \
  --variant flash \
  --force
```

Validate the package against the Flash sidecar target:

```sh
./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 5 \
  --inspect
```

The expected package metadata is DSpark-5: block size 5, target layers
`40,41,42`, three draft layers, 256 experts, and Markov rank 256. The default
static verifier budget is 5, matching the DSpark-5 checkpoint. In static mode
the active proposal length is `min(block_size, --draft-verify)`, so the default
proposes and verifies 5 draft tokens and the loader prints `active=5`. Pass
`--draft-verify 2`, `3`, or `4` explicitly for fixed smaller-block A/B tests.

Run a paired sidecar baseline first:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 ./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --temp 0 \
  --nothink \
  -n 1000 \
  -c 4096 \
  -p "$TEST_PROMPT"
```

Then run DSpark on the same sidecar target:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 DS4_DSPARK_PERF=1 ./ds4 \
  -m "$DS4_SIDECAR_DIR" \
  --resident \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 5 \
  --draft-scheduler static \
  --temp 0 \
  --nothink \
  -n 1000 \
  -c 4096 \
  -p "$TEST_PROMPT"
```

The sidecar command is the current clean reference path. DS4 selects the strict
commit-safe hybrid verifier by default for DSpark greedy runs unless
`--quality`, `DS4_DSPARK_EXACT_VERIFY=1`, or `DS4_DSPARK_FAST_VERIFY_DISABLE=1`
is set. A successful run prints:

```text
ds4: DSpark draft package loaded: ... (block=5 verify=5 active=5 ...)
ds4: DSpark draft inference enabled: MPP 4.1 FP8/MXFP4 draft kernels ...
ds4: dspark perf: draft=... verify=... block=... tau=...
ds4: dspark acceptance: ...
ds4: dspark acceptance by position: ...
ds4: dspark avg scheduled: ...
```

Here `tau` means emitted tokens per speculation block:
`1 + accepted_draft_tokens / blocks`. With `--draft-verify 4`, the maximum
`tau` is therefore `5.0`: one ordinary target token plus up to four accepted
draft tokens.

For `ds4-agent`, keep the same sidecar model and draft package:

```sh
DS4_AGENT_ALLOW_BACKEND_STATS=1 DS4_DSPARK_PERF=1 DS4_AGENT_TURN_STATS=1 \
./ds4-agent \
  --model "$DS4_SIDECAR_DIR" \
  --resident \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 4 \
  --temp 0 \
  --nothink \
  --ctx 24096 \
  --debug-status
```

Add `--dspark-attn-force-mma` only for a faster Mode-B/demo run where exact
byte identity with the strict verifier is not the goal.

If a demo must use a GGUF main model, keep DSpark as the same external draft
package and pass `--draft-path` explicitly:

```sh
export DS4_GGUF=/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf

DS4_AGENT_ALLOW_BACKEND_STATS=1 DS4_DSPARK_PERF=1 DS4_AGENT_TURN_STATS=1 \
./ds4-agent \
  --model "$DS4_GGUF" \
  --draft dspark \
  --draft-path "$DS4_DSPARK_DRAFT" \
  --draft-verify 4 \
  --temp 0 \
  --nothink \
  --ctx 24096 \
  --debug-status
```

The sidecar path remains preferred for DSpark. GGUF main-model runs have higher
memory pressure; the runtime keeps the large prefill chunk for prompt ingestion
and shrinks speculative verifier scratch to the active DSpark rows during
generation.

Runtime DSpark inference uses MPP 4.1 FP8/MXFP4 draft kernels with persistent
resident draft experts. Startup refuses DSpark if the native MXFP4
batch-strided routed-MoE kernels are unavailable, so a normal DSpark run does
not silently fall back to an unfused expert path. The routed MoE part is fused
per DSpark draft layer through the native pair-SwiGLU kernel plus the native
down-sum6 kernel; the three draft layers are kept inside one GPU command
sequence with the attention, HC, router, residual, Markov, and confidence work
between them. A literal single kernel for all three DSpark layers is not the
current fusion boundary, because each layer depends on the previous layer's
attention and routing state.
Set `DS4_DSPARK_MARKOV_CHAIN_DISABLE=1` only to compare against the older
per-row Markov path. `--draft dspark` expects the draft package to match the
target sidecar metadata and keeps the 9 DSpark MXFP4 expert records resident by
default; plain `DS4_DSPARK_RESIDENT=0` is ignored so accidental disk-backed
draft runs do not pollute speed measurements. For a deliberate debug-only
nonresident run, set both `DS4_DSPARK_RESIDENT=0` and
`DS4_DSPARK_ALLOW_NONRESIDENT=1`. Accepted-block DSpark main-KV maintenance is
batched for contiguous positions in the 128-token DSpark window; set
`DS4_DSPARK_MAIN_KV_BATCH_DISABLE=1` only for A/B debugging against the older
one-row update loop.
The FP8 draft dense path uses a rows5 kernel by default for N<=5 draft blocks.
It loops the same FP8 row matvec over draft rows inside one kernel while
preserving each token's reduction order. Current local smokes are byte-clean at
n=64/n=160/n=1000/n=4000 and reduce the measured draft block from roughly
16.0 ms to 14.5-14.9 ms on the b4 sweeps. Set
`DS4_DSPARK_DRAFT_FP8_ROWS5_DISABLE=1` or `DS4_DSPARK_DRAFT_NO_FP8_ROWS5=1` to
A/B the older per-token FP8 draft matvecs.

Compare the final `generation:` line against the same resident command without
`--draft dspark`. With backend stats enabled, DS4 prints total DSpark acceptance,
acceptance by draft position, and average scheduled draft length. For lightweight
DSpark timing without VM/backend stats, set `DS4_DSPARK_PERF=1`; for per-block
diagnostic timings, set `DS4_DSPARK_BLOCK_TIMING=1` (`DS4_DSPARK_TIMING=1` is
kept as an older alias). To normalize a DSpark run against a paired no-draft
baseline, also set either `DS4_DSPARK_BASELINE_TPS=<t/s>` or
`DS4_DSPARK_BASELINE_DECODE_MS=<ms>`; the footer will print decode-equivalent
draft, verify, overhead, block, and `tau`.
For routed-MoE verifier planning, `DS4_DSPARK_ROUTE_OVERLAP_LOG=1` prints the
per-layer active-route reuse census. It is a diagnostic with readback/fences,
not a speed path; the production grouped-MoE work must preserve row-exact router
semantics, separate slot-down outputs, and the existing exact ordered FP32 sum.
For the full Phase-0 diagnostic sweep, use:

```bash
N=160 scripts/dspark_phase0_sweep.sh
```

The sweep runs the paired no-draft baseline plus `--draft-verify 1..5`, stores
logs under `bench-results/dspark_phase0_*`, and writes a parseable
`summary.tsv` with generation t/s, acceptance, draft/verify/block timing,
decode-equivalents, dispatch census, routed-overlap fields, and optional
unified target-forward backend/timing fields. These are diagnostic runs; use
clean no-stats paired commands for headline speed. To add
Plan C backend comparison rows to the same sweep, set for example
`PLAN_C_BACKENDS="batch strict_v1"`; this runs `--draft-mode unified` for each
listed `DS4_TARGET_FORWARD_UNIFIED_BACKEND`. Set `TARGET_FORWARD_PROFILE=0` to
suppress the per-call unified wrapper timing in those optional rows.

For Plan C attention work, use the row-shape profiler before writing verifier
kernels:

```bash
N=64 BUDGETS="2 3 4 5" ROUTE_OVERLAP=0 DISPATCH_PROFILE=0 BLOCK_TIMING=0 \
SHARED_PREFIX_PROFILE=1 ATTN_ROWS_SHAPE_PROFILE=1 \
scripts/dspark_phase0_sweep.sh
```

The sweep adds `sp_*` shared-prefix estimates and `ar_*` attention row-shape
columns to `summary.tsv`, including the split-lane alignment fields
`ar_raw_intersection`, `ar_raw_lane_aligned`, `ar_raw_lane`,
`ar_raw_lane_runs`, `ar_raw_lane_avg_run`, `ar_raw_lane_max_run`,
`ar_comp_common`, `ar_comp_lane_aligned`, `ar_comp_lane`,
`ar_comp_lane_runs`, `ar_comp_lane_avg_run`, and `ar_comp_lane_max_run`. If
`ar_raw_same_count` is zero, do not build a same-mask/same-span rows5 attention
shortcut; use the shared-prefix design instead. The intended Plan C kernel is
shared committed-prefix attention for N<=5 plus an exact row-local block tail.
Set
`DS4_DSPARK_ATTN_ROWS_SHAPE_PROFILE_VERBOSE=1` only when a kernel agent needs
per-layer descriptors such as `ratio`, `raw_common`, `raw_reuse`,
`raw_tail=[...]`, `comp_common`, `comp_reuse`, and `comp_tail=[...]`.
The first Plan C implementation target is raw shared-prefix attention only:
batch the committed raw-KV prefix for N<=5 rows, keep the block-local tail
row-exact, and leave compressor/indexer and routed MoE untouched until that
single subsystem passes the one-layer and full-block audits.
`DS4_DSPARK_ATTN_RAW_VEC_ROWS_EXPERIMENT=1` is a narrower raw-only probe for
that work: raw layers with matching raw starts enter the safe vector
flash-attention family as N<=5 rows with per-row masks; layers with compressed
rows or differing raw starts fall back to rows-exact attention. It is not the
final shared-prefix scan-sharing kernel. The named
`DS4_TARGET_FORWARD_UNIFIED_BACKEND=shared_prefix` backend now enables the
byte-clean varmap rows5 attention scaffold while scoping `strict_v1` so
batch-canonical attention shortcuts do not leak into the strict delegate. Set
`DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_ROWS_DISABLE=1` to fall back to the
older raw vec-rows landing point for A/B tests. Setting
`DS4_DSPARK_ATTN_VARMAP_ROWS=1` now also enables the required deferred
row-exact attention gate by itself; older notes that pair it with
`DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS=1` are no longer required for activation.
The direct-resident diagnostic,
`DS4_DSPARK_ATTN_VARMAP_DIRECT_ROWS=1`, now has the same gate fix and actually
enters the direct F32 K/V kernel when set. Current local n=160 tests are
byte-clean but slower than scratch varmap: direct `32.73 t/s`, direct+dynamic
`37.52 t/s`, versus scratch varmap about `38.4 t/s`; keep direct-resident
diagnostic-only.
Current varmap profiling shows CPU encoding is not the limiter. In
`bench-results/dspark_varmap_profile_025345`, 2337 varmap calls cost only
`2.98 ms` total host time while the n=320 DSpark run reached `41.79 t/s`; the
remaining work is GPU/cache traffic. A corrected attention subprofile in
`bench-results/dspark_attn_subprofile_after_profilefix_025610` no longer
double-counts row-output HC as an empty `hc_post` fence and points the next
kernel work at HC-pre/output and attention-half fusion, not pair-row projection
flags.
For a fenced varmap microscope, set `DS4_DSPARK_ATTN_VARMAP_STAGE_PROFILE=1`
or `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_STAGE_PROFILE=1`. It splits the
scratch varmap helper into copy/stage, vec, and reduce fences and prints
`stage_ms` fields in the varmap aggregate footer. It is diagnostic-only and
will slow the run; use it to decide whether the next kernel should remove
scratch staging or attack vec/reduce work.
`DS4_DSPARK_ATTN_VARMAP_COMP_F16_SHADOW=1` is a narrower compressed-KV shadow
diagnostic, with `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_COMP_F16_SHADOW=1`
as the Plan C alias. It is byte-clean, but the current local A/B shows no speed
win: `bench-results/dspark_varmap_comp_shadow_fix_063107` measured n=64 default
DSpark `37.52 t/s` versus shadow `37.42 t/s`, both `cmp=0`. The fenced follow-up
`bench-results/dspark_varmap_comp_shadow_stage_063245` measured copy/stage
`961.712 ms` default versus `967.990 ms` with the shadow, while vec/reduce were
nearly unchanged. Keep this off by default; the copy bucket is dominated by
raw-union scratch materialization, so the next Plan C attention target is a
resident raw plus compressed shared-prefix path, not this compressed-only blit.
`DS4_DSPARK_ATTN_VARMAP_RAW_F16_SHADOW=1` adds the matching raw-KV F16 shadow
for the scratch varmap path, and
`DS4_DSPARK_ATTN_VARMAP_DIRECT_F16_SHADOW=1` uses raw plus compressed shadows in
the direct-resident varmap kernel. Both are byte-clean diagnostics, but current
local smokes reject them for speed: `bench-results/dspark_varmap_raw_shadow_064157`
measured default `37.56 t/s` versus raw-shadow `37.28 t/s`; the fenced follow-up
`bench-results/dspark_varmap_raw_shadow_stage_064338` measured copy/stage
`942.495 ms` default versus `953.118 ms` with raw shadow. Direct-F16 shadows in
`bench-results/dspark_varmap_direct_f16_064826` were also byte-clean but much
slower: default `37.45 t/s`, direct-F32 `36.93 t/s`, direct-F16 `30.65 t/s`.
Do not promote the shadow-to-scratch or direct-F16 paths; the next real Plan C
kernel has to share committed-prefix work inside the attention kernel rather
than just changing where the same row-local reads are staged.
For a finer HC/output microscope, set
`DS4_DSPARK_HC_PRE_SUBPROFILE=1 DS4_DSPARK_OUTPUT_SUBPROFILE=1` together with
`DS4_DSPARK_HYBRID_ATTN_SUBPROFILE=1`. `bench-results/dspark_hc_output_subprofile_030709`
shows the current fenced active-5 split: HC RMS about `40-42 ms/43`, HC function
about `10-11 ms/43`, HC split/norm about `10-11 ms/43`, output inverse-RoPE
about `8-10 ms/43`, output-low Q8 about `19-23 ms/43`, and output-HC expand
about `17-20 ms/43`. These numbers are fence-inflated; use the ordering to pick
kernel work, not as wall time.
`DS4_DSPARK_OUTPUT_LOW_Q8_ROWS5=1` is an opt-in output-low Q8 rows5 diagnostic.
It is layout-correct and byte-clean, but the current local A/B is slower:
`bench-results/dspark_output_low_rows5_034252` measured n=96 enabled
`37.93 t/s` versus disabled `38.76 t/s`, both `cmp=0`. Keep the default off
until this shape is redesigned or a longer-context profile proves otherwise.
`DS4_DSPARK_HC_PRE_SCALED_F16=1` is an opt-in HC-pre diagnostic that computes
per-row RMS scales and feeds the F16 HC projection directly from the HC state
without materializing `batch_flat_hc`. It is byte-clean in the n=64 smoke, but
slower than default (`34.65 t/s` versus `37.91 t/s` in
`bench-results/dspark_hc_scaled_f16_031619`), so keep it diagnostic-only. The
next HC-pre attempt needs a different fusion shape than scale-buffer plus rows5
F16 projection.
`DS4_DSPARK_VERIFY_F16_ROWS5=1` and
`DS4_DSPARK_VERIFY_F16_ROWS5_SEQ=1` are also diagnostic-only for now:
`bench-results/dspark_f16_rows5_ab_025918` was byte-clean, but default active-5
was faster (`40.07 t/s`) than shared F16 rows5 (`39.14 t/s`) or seq F16 rows5
(`39.30 t/s`).
The scratch varmap and direct-resident varmap kernels use dynamic split count by
default. Set `DS4_DSPARK_ATTN_VARMAP_DYNAMIC_NWG_DISABLE=1` or
`DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_DYNAMIC_NWG_DISABLE=1` to A/B the older
fixed-32 split path; use the corresponding
`DS4_DSPARK_ATTN_VARMAP_DIRECT_DYNAMIC_NWG_DISABLE=1` only when debugging the
direct-resident diagnostic. `DS4_DSPARK_ATTN_VARMAP_NWG=<1..32>` and
`DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_NWG=<1..32>` are diagnostic override
knobs for the scratch varmap split count; direct-resident also accepts
`DS4_DSPARK_ATTN_VARMAP_DIRECT_NWG=<1..32>`. Current probe
`bench-results/dspark_varmap_nwg_probe_035519` rejects forced `nwg=1`: it stayed
final-output byte-clean at n=160, but slowed default DSpark from `39.99 t/s` to
`38.28 t/s`, dropped acceptance from `76.4%` to `68.3%`, and the all-layer
compare showed `max_delta=1.9e-6` versus rows-exact while default dynamic split
was exact (`max_delta=0`). Keep dynamic split as default.
Explicit varstream diagnostics now take precedence over the default varmap
scaffold. Set `DS4_DSPARK_ATTN_VARSTREAM_ROWS=1` directly; no separate
`DS4_DSPARK_ATTN_VARMAP_ROWS_DISABLE=1` is required just to enter varstream.
Fresh n=160 A/B in `bench-results/dspark_varstream_recheck_040853` was
byte-clean but not faster: default varmap `40.07 t/s`, row-exact fallback
`38.83`, varstream `38.95`, and varstream dynamic `38.69`. Keep varstream as a
correctness scaffold for shared-prefix work, not a production speed flag.
`DS4_DSPARK_ATTN_VARMAP_COMMON_KSTAGE=1` and
`DS4_DSPARK_ATTN_VARMAP_COMMON_VSTAGE=1` are opt-in varmap common-prefix staging
probes, with matching `DS4_TARGET_FORWARD_SHARED_PREFIX_VARMAP_COMMON_*` aliases.
They stage full 32-key raw-prefix tiles shared by every active verifier row while
leaving the row-local online-softmax order unchanged. Both are byte-clean in
local compare-all tests, but neither is a speed path as implemented:
`bench-results/dspark_varmap_common_kstage_042048` had 738 compare checks all
`max=0` and final `cmp=0`, but no-compare n=96 was only `38.26 t/s`; fresh
`bench-results/dspark_varmap_common_vstage_043114` also had final `cmp=0` and
zero nonzero compare lines, but n=160 slowed default DSpark from `39.86 t/s` to
`39.30 t/s`. Treat these as proof scaffolds only; the production Plan C kernel
must split common-prefix online-softmax from the row-local tail, not just stage
K or V inside the current per-row stream.
The varmap aggregate now reports both strict row-local prefix reuse and the
larger raw-window interval intersection. `bench-results/dspark_varmap_intersection_profile_043922`
stayed `cmp=0` at n=320 and showed why prefix-only staging underwhelms:
`raw_split_reuse=1.22x`, but `raw_intersection_reuse=4.46x`, with
`raw_intersection_full=190240` full-tile keys; compressed reuse was
`comp_split_reuse=4.79x`. The next attention kernel should therefore combine
private-old, shared-intersection, and private-new/tail online-softmax states
rather than assuming the shared segment is a prefix for every row.
`DS4_DSPARK_ATTN_VARMAP_RAW_UNION_KSTAGE=1` is an opt-in proof that row-local
raw tiles can source K from a shared shifted raw-union staging buffer while
preserving each row's exact 32-key tile boundaries. Compare-all
`bench-results/dspark_varmap_raw_union_kstage_044329` had final `cmp=0` and
zero nonzero `varmap-attn compare` lines, but clean n=160
`bench-results/dspark_varmap_raw_union_kstage_clean_044600` slowed default
DSpark from `39.65 t/s` to `38.47 t/s` with identical acceptance. Keep it
diagnostic-only; it proves address/order safety, not a speed win.
`DS4_DSPARK_ATTN_VARMAP_RAW_UNION_VSTAGE=1` adds the matching value-tile probe.
It is also byte-clean: `bench-results/dspark_varmap_raw_union_vstage_050432`
had final `cmp=0` and zero nonzero `varmap-attn compare` lines. Clean n=160
`bench-results/dspark_varmap_raw_union_vstage_clean_050547` slowed default
DSpark from `39.42 t/s` to `38.07 t/s`, with identical acceptance. Keep V-stage
diagnostic-only too; sharing raw K/V inside the current row-local stream adds
more staging/barrier cost than it saves.
`DS4_DSPARK_ATTN_VARMAP_RAW_TILE_INTERSECTION_KSTAGE=1` narrows the K proof
further by staging only the overlapping absolute K rows inside each shifted
row-local tile and loading private tile edges directly. It fixes the previous
kernel/encoder ABI by binding the new tile-intersection cap at index 15. The
probe is exact: `bench-results/dspark_varmap_tile_intersection_kstage_051154`
had final `cmp=0` and zero nonzero `varmap-attn compare` lines. It is still the
wrong fusion for speed: clean n=160
`bench-results/dspark_varmap_tile_intersection_kstage_clean_051307` kept
`cmp=0` but slowed default DSpark from `39.08 t/s` to `37.53 t/s`. Keep it
diagnostic-only. The useful speed kernel must batch the shared intersection as
its own online-softmax phase instead of staging fragments inside the current
per-row tile loop.
`DS4_DSPARK_ATTN_VARMAP_RAW_TILE_INTERSECTION_VSTAGE=1` adds the matching
value-tile proof. It is byte-clean in current local checks:
`bench-results/dspark_raw_tile_vstage_split_084328` kept n=64 V-only `cmp=0`,
and `bench-results/dspark_raw_tile_vstage_vonly_084543` kept n=160 V-only
`cmp=0` with matching acceptance. It is still not a clear speed win
(`36.38 t/s` in that n=160 V-only run), and simultaneous raw tile K+V staging is
guarded back to K-only after `bench-results/dspark_raw_tile_vstage_084149`
showed K+V corrupting acceptance (`1.7%`, `cmp=1`). This reinforces the same
conclusion: changing the per-row staging layout is not the lever; removing
repeated K/V stream materialization with a first-class shared raw/compressed
intersection phase is the lever.
`DS4_DSPARK_ATTN_VARMAP_COMP_UNION_KSTAGE=1` and
`DS4_DSPARK_ATTN_VARMAP_COMP_UNION_VSTAGE=1` are the matching compressed-row
union staging probes. They are guarded so simultaneous K+V requests use K-stage
only; staging both exceeded the safe threadgroup-memory shape and corrupted
later compressed tiles. With the guard, clean n=160 stayed byte-clean, but
slowed default DSpark from `38.77 t/s` to K-stage `37.42 t/s`, V-stage
`37.49 t/s`, and guarded-both `37.29 t/s`
(`bench-results/dspark_varmap_comp_union_guarded160_071734`). Keep these
diagnostic-only. Like the raw-union probes, they prove some address/order
safety inside the row-local stream, but they do not deliver the Plan C speed
kernel; shared compressed scans need their own online-softmax phase.
The current profile footer now reports `comp_common_full`,
`comp_common_partial`, `comp_full_est`, `comp_full_reuse`, and
`comp_full_tile_calls` to size that phase. In
`bench-results/dspark_comp_fulltile_profile_072415`, default varmap active-5
was byte-clean at n=320 (`41.28 t/s`) and reported `comp_common=52890`,
`comp_common_full=34272`, `comp_tail=2979`, `comp_split_reuse=4.79x`, and
`comp_full_tile_calls=777/2337`. That means a useful shared-compressed kernel
must handle both full 32-key common blocks and the remaining common tail; a
full-tile-only shortcut would leave too much compressed reuse on the table.
The fresh footer smoke `bench-results/dspark_comp_tail_profile_073055` stayed
`cmp=0`, measured `41.51 t/s` at n=160, and quantified that gap:
`comp_common=14649`, `comp_common_full=6048`,
`comp_common_partial=8601`, `comp_split_reuse=4.60x`, but
`comp_full_reuse=1.47x`. Do not start the shared compressed scan with a
full-tile-only implementation unless it is explicitly a throwaway bring-up
probe. The matching raw-intersection footer in
`bench-results/dspark_raw_comp_tail_profile_073546` stayed `cmp=0` at n=96 and
showed the same shape: `raw_intersection_reuse=4.24x` versus
`raw_intersection_full_reuse=2.36x`; compressed had
`comp_common_full=0`, `comp_common_partial=6048`, so full-tile-only compressed
sharing would be a no-op for that early block. Plan C should treat partial
common tiles as first-class, not as cleanup after full tiles.
The C-side row-shape profiler now also measures row-local split-lane alignment
for those shared keys. In `bench-results/dspark_phase_alignment_all_074309`,
the active-5 n=160 run stayed `cmp=0`, measured `39.82 t/s`, and across 27
printed blocks had weighted raw intersection lane alignment
`96464/100517 = 96.0%` and compressed common lane alignment
`13137/14019 = 93.7%`. The first Plan C shared-scan kernel should therefore
use a natural-lane fast path with a remap fallback for the minority of
misaligned shared keys.
The follow-up run-shape profile
`bench-results/dspark_phase_runs_profile_074806` stayed `cmp=0`, measured
`39.68 t/s`, and found those natural-lane keys are not tiny crumbs:
raw aligned keys covered `2184` runs with average run `44.2` and max `124`,
while compressed aligned keys covered `908` runs with average run `14.5` and
max `44`. Use compact range descriptors for the Plan C natural-lane phase; do
not start with per-key scatter tables unless the remap fallback proves it needs
them.
The Metal-side varmap host path now has the matching descriptor builder inside
`ds4_gpu_attention_decode_varmap_rows_tensor()`. It constructs compact
natural-lane raw and compressed ranges from `row_raw_base`, `row_n_raw`,
`row_n_comp`, and `nwg`; the future phase kernel can consume the same
descriptor shape directly. `bench-results/dspark_varmap_phase_desc_profile_075259`
stayed `cmp=0` at n=96 and printed
`phase_raw_keys=48257`, `phase_raw_ranges=1845`, `phase_raw_avg_run=26.2`,
`phase_raw_max_run=32`, `phase_comp_keys=5397`, `phase_comp_ranges=504`,
`phase_comp_avg_run=10.7`, `phase_comp_max_run=23`, with zero descriptor
overflow.
`DS4_DSPARK_ATTN_VARMAP_PHASE_PROBE=1` now uploads those descriptors to Metal
and verifies the GPU-side ABI with `kernel_dspark_phase_range_probe`; add
`DS4_DSPARK_ATTN_VARMAP_PHASE_PROBE_ALL=1` to check every varmap call. Fresh
smokes `bench-results/dspark_varmap_phase_probe_080002` and
`bench-results/dspark_varmap_phase_probe_all_080115` both stayed `cmp=0`; the
all-call probe checked 123 varmap calls with no descriptor mismatch. This
clears the descriptor-buffer plumbing for the next shared-intersection
FlashAttention phase.
`DS4_DSPARK_ATTN_VARMAP_PHASE_KV_PROBE=1` also reads the descriptor-addressed
F16 `raw-union | compressed` K/V stream and validates the expected half4 read
count/checksum shape. Fresh smokes
`bench-results/dspark_varmap_phase_kv_probe_080742` and
`bench-results/dspark_varmap_phase_kv_probe_all_080914` both stayed `cmp=0`; the
all-call K/V probe checked 123 varmap calls. This is still proof plumbing, not
the final `dspark_attn_shared_intersection_mixed_exact_n5` kernel.
`DS4_DSPARK_ATTN_VARMAP_PHASE_SOFTMAX_PROBE=1` computes per-row/per-head/
per-split-lane `S/M` online-softmax state over the descriptor-covered shared
ranges. Fresh smokes `bench-results/dspark_varmap_phase_softmax_probe_081521`
and `bench-results/dspark_varmap_phase_softmax_probe_all_081634` both stayed
`cmp=0`; the all-call softmax probe checked 123 varmap calls. It still does not
write candidate `so4` or heads.
`DS4_DSPARK_ATTN_VARMAP_PHASE_HEAD_PROBE=1` is the next scaffold: it consumes
the same descriptor-addressed F16 K/V stream, writes shared-range candidate
`so4 + S/M` into a scratch split-state buffer, then reduces through the existing
FlashAttention reducer into candidate heads. Fresh smokes
`bench-results/dspark_varmap_phase_head_probe_082347` and
`bench-results/dspark_varmap_phase_head_probe_all_082459` both stayed `cmp=0`;
the first-call probe logged `heads=320 floats=163840 nonzero=163840
rms=0.349892`, and the all-call probe exercised every varmap call. This is still
not a strict-head compare because private-old/private-new/block-tail phases are
not included yet. The next exact step is to add those phases and gate on
candidate-vs-strict attention-head max delta `0`.
`DS4_DSPARK_ATTN_VARMAP_PHASE_HEAD_SHARED_PROBE=1` is a rows-together variant of
that scaffold. It dispatches one threadgroup per head/split lane, stages each
descriptor K/V range once in threadgroup memory, and lets all N<=5 row simdgroups
consume it. Current smoke `bench-results/dspark_phase_head_shared_probe_085449`
stayed `cmp=0` and preserved acceptance, but it is slower as an add-on probe:
default n=64 `36.69 t/s`, old per-row head probe `36.66 t/s`, shared head probe
`33.53 t/s`. Treat it as implementation scaffolding for
`dspark_attn_shared_intersection_mixed_exact_n5`, not as a speed flag. The speed
version must replace repeated F16 K/V stream materialization and include
private-old/private-new/block-tail phases, not run beside strict varmap.
Latest refreshed stage profile
`bench-results/dspark_varmap_stage_refresh_085143` confirms why: over 738 varmap
attention calls, copy/materialization was `1402.552 ms` while vec was
`274.759 ms` and reduce was `187.447 ms`. The main-track attention target is
therefore `dspark_attn_shared_prefix_mixed_exact_n5` (also called
`shared_intersection_mixed_exact_n5` in older notes): full mixed/compressed
private/shared/tail descriptors, F16 consumed K/V, F32 per-row online-softmax
state, candidate heads to scratch, and strict heads authoritative until
candidate-vs-strict attention-head `max=0`. Do not spend main-track time on
raw-only, same-shape, direct-resident, shadow-blit, raw-union mask, plain mixed
unsafe, or grouped-Q2 detours unless they directly feed this candidate.

Latest mixed-prefix checkpoint: `DS4_DSPARK_ATTN_SHARED_PREFIX_MIXED_DESC_PROBE=1`
validates the full descriptor split and stayed `cmp=0` in
`bench-results/dspark_mixed_desc_probe_090253`; the first active-5 block reduced
`raw_row=115` to `raw_desc=31` and `comp_row=27` to `comp_desc=7`.
`DS4_DSPARK_ATTN_SHARED_PREFIX_MIXED_COMPARE=1` now runs a compare-only GPU
candidate while leaving strict varmap heads authoritative. In
`bench-results/dspark_mixed_prefix_compare_real_091336` the GPU range probe was
correct and final output still matched default (`cmp=0`), but candidate heads
had large deltas (`max` about `2.7..7.6`). The phase split is wrong for strict
exactness: strict varmap processes each row-local 32-key FlashAttention chunk as
one online-softmax update, and early chunks can contain both raw and compressed
keys. The next candidate must preserve those strict chunk boundaries while
sharing K/V loads inside the chunk.
`DS4_DSPARK_ATTN_VARMAP_MIXED_CHUNK_UNION_KSTAGE=1` and
`DS4_DSPARK_ATTN_VARMAP_MIXED_CHUNK_UNION_VSTAGE=1` are the first
chunk-boundary-aware staging probes inside the strict varmap kernel. They stage
the exact row-local 32-key mixed raw/compressed chunk only when every active
verifier row participates in that chunk, avoiding the barrier/state corruption
seen in the first K-stage attempt. Evidence:
`bench-results/dspark_mixed_chunk_guard_092621` had K-only `cmp=0` at n=96
after the all-row guard and V-only `cmp=0`; simultaneous K+V is still unsafe and
is guarded back to K-only. Longer smoke
`bench-results/dspark_mixed_chunk_kstage_n160_092915` stayed `cmp=0`, but did
not improve speed: default `36.96 t/s`, verifier `76.92 ms`; K-stage
`36.79 t/s`, verifier `79.89 ms`. Keep these as exactness diagnostics, not the
main speed path. The main target remains replacing repeated consumed-format K/V
materialization with a real mixed/shared-prefix verifier attention kernel.
For the corresponding unified-forward contract, compare against the local MLX
reference at `/tmp/dspark-research/mlx-vlm`: `mlx_vlm/speculative/mtp.py`
`_mtp_verify_target()` runs the target verify block, `_mtp_rounds()` and
`_mtp_rounds_batch()` build `[bonus, draft_tokens]`, walk target-vs-draft tokens,
and call rollback on rejection. `mlx_vlm/models/deepseek_v4/language.py`
`_speculative_verify()` calls the same DeepSeek V4 model forward for the verify
block, while `rollback_speculative_cache()` restores/replays or trims/zeros
rejected cache tails. Keep those symbols beside Plan C implementation work.
`DS4_DSPARK_ATTN_RAW_VEC_ROWS_FUSED=1` tests an experimental rows5 fused
threadgroup version of the same raw vec-rows path. Current local smoke found it
byte-clean but slower, so keep it diagnostic-only.
`DS4_DSPARK_ATTN_MIXED_SHARED_COMPARE_GEOM=1` adds row geometry beside the
existing mixed-shared candidate-vs-strict attention compare. It is useful only
with `DS4_DSPARK_ATTN_MIXED_SHARED_COMPARE=1`; disable varmap rows if you need
to force that diagnostic path. Fresh forced n=96
`bench-results/dspark_mixed_shared_geom_forced_052550` stayed `cmp=0` because
compare mode restored strict heads, but logged the important failure shape:
even with `raw_start=0`, raw intersection as a real prefix, and shared
compressed prefix, the plain mixed shared-row candidate differs from strict by
about `1e-6`. Example: layer 2 at `pos=20` had raw intersection `[0,21)`,
`raw_shared_est=31`, `comp_common=5`, and max head delta `9.54e-7`. This
rules out raw-ring visibility as the cause for this probe. The candidate uses
the separate plain online attention kernel, while strict row-exact uses the
FlashAttention vec/reduce realization, so keep mixed-shared diagnostic-only.
The next strict-compatible attention path must preserve the vec/reduce contract
or explicitly switch to a unified canonical N=1/N<=5 target-forward contract.
Speed-first recheck confirms this is not just a correctness problem:
`bench-results/dspark_mixed_speed_ceiling_083126` measured n=320 strict/default
varmap DSpark `39.01 t/s`, unsafe plain mixed-shared `40.15 t/s`, and unsafe
heads8 mixed-shared `37.88 t/s`; `bench-results/dspark_mixed_speed_ceiling_n1000_083357`
measured n=1000 strict/default varmap `36.01 t/s` versus unsafe plain
mixed-shared `37.87 t/s`. Both unsafe outputs were `cmp=1`. Treat the old plain
mixed-shared kernel as a small speed-ceiling probe, not the production
`dspark_attn_shared_intersection_mixed_exact_n5` target.
Current go/no-go rule for more attention exactness work: only promote a new
candidate family if it saves repeated consumed-format K/V materialization and
shows at least an `8%` n=1000 speed ceiling without tau/acceptance collapse.
The current plain mixed-shared family and row-local tile staging probes do not
clear that bar.
The committed/no-compare variant is guarded now: setting the old
`DS4_DSPARK_ATTN_MIXED_ROWS_SHARED_EXACT=1` flag without compare falls back to
rows-exact heads; add `DS4_DSPARK_ATTN_MIXED_SHARED_UNSAFE=1` only when
reproducing the known-divergent committed diagnostic.
`DS4_DSPARK_ATTN_RAW_VEC_ROWS_KSTAGE=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_ROWS5_KSTAGE=1`) selects the opt-in
rows5 K-stage variant. It stages each committed-prefix K tile once per
threadgroup for N<=5 rows when `nsg=1`, then preserves per-row online-softmax
state and exact tail masking. Current n=160 smoke is byte-clean but not faster
than strict/raw vec-rows, so keep it diagnostic until a larger context or revised
K/V staging shape proves useful.
Set `DS4_DSPARK_ATTN_RAW_VEC_ROWS_PROFILE=1` (or
`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_VEC_ROWS_PROFILE=1`) to print the first
raw vec-rows resource line: active rows, max raw span, `nsg/nwg`, mask/KV/tmp
scratch, threadgroup memory, dispatch shape, and estimated K-only/KV tile-staging
memory for the future shared-prefix kernel. Add `_ALL=1` to either env name only
when you need one line per raw vec-rows call. The same env also prints one
aggregate line at exit with call count, average/max raw span, `nsg` histogram,
and total scratch/staging estimates. `DS4_DSPARK_ATTN_RAW_VEC_ROWS_KVSTAGE_UNSAFE=1`
selects a failed K+V staging diagnostic that is not byte-clean in local smokes;
keep it off except when debugging that exact experiment.
`DS4_DSPARK_ATTN_RAW_VEC_ROWS_DYNAMIC_NWG=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_RAW_VEC_ROWS_DYNAMIC_NWG=1`) shrinks the
FlashAttention split count for small raw spans and skips the reduce pass when
`nwg=1`. Current n=320 smoke is byte-clean but slower despite reducing tmp
traffic sharply, so keep it diagnostic until the kernel shape changes.
`DS4_DSPARK_ATTN_MIXED_VEC_ROWS_EXPERIMENT=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_VEC_ROWS_EXPERIMENT=1`) is a
pack+mask scaffold for mixed/compressed rows. It packs the per-block raw union
plus compressed rows into the vector FlashAttention backend and preserves final
greedy bytes in short local smokes, but n=160 changed DSpark acceptance/blocking
statistics while still `cmp=0`. Keep it diagnostic; it is not strict-v1
bit-identical verifier evidence.
`DS4_DSPARK_ATTN_MIXED_ROWS_SHARED_EXACT=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_ROWS_SHARED_EXACT=1`) is a stricter
shared-row mixed attention probe. It uses the row-exact attention update helper
and skips invisible keys instead of masking them, but groups up to five rows and
two heads in one threadgroup. It matched the deferred row-exact control at n=64,
then diverged by n=160 (`cmp=1`, acceptance changed from `126/165` to
`125/171`). Keep it diagnostic until a stage audit proves the remaining
row/head grouping delta. This flag now auto-activates the deferred row-exact
head path, but the committed candidate is additionally guarded: without
`DS4_DSPARK_ATTN_MIXED_SHARED_COMPARE=1` it falls back to rows-exact unless
`DS4_DSPARK_ATTN_MIXED_SHARED_UNSAFE=1` is also set. Use compare mode for
fallback checks; `bench-results/dspark_mixed_shared_gatefix_020444` entered the
branch, logged `~1e-6` head deltas, fell back to rows-exact, and stayed `cmp=0`.
`DS4_DSPARK_ATTN_MIXED_ROWS_SHARED_HEADS8=1`
(`DS4_TARGET_FORWARD_SHARED_PREFIX_MIXED_ROWS_SHARED_HEADS8=1`) selects a
follow-up diagnostic that keeps the strict path's eight-head simdgroup layout
and loops the N<=5 rows inside that layout. It still is not byte-clean: compare
mode shows `~1e-6` head deltas and falls back to rows-exact, while no-compare
n=160 with the fixed gate diverged for both heads2 and heads8
(`bench-results/dspark_mixed_shared_defer_nocompare_020252`). Heads8 measured
`37.33 t/s`, acceptance `70.5%`. This closes the
"heads8 grouping only" hypothesis; true strict shared-prefix attention needs a
stage-level exactness fix or a unified/canonical target-forward contract.
`DS4_DSPARK_ATTN_ROWS_EXACT_PROFILE=1`
(`DS4_TARGET_FORWARD_ROWS_EXACT_PROFILE=1`) prints an aggregate for the remaining
rows-exact attention path. Use it with the raw vec-rows profile to separate
raw-only landing-point work from the larger mixed/compressed verifier cost. The
aggregate includes intersection counters such as `same_counts`, `same_shape`,
and `same_shape_mixed`; only `same_shape*` means a same raw-start/raw-count/
comp-count rows5 shortcut would have coverage. Current local n=320 Plan C
profiling has `same_counts=720` but `same_shape=0`, so do not build a
same-shape mixed shortcut for that trace. Target true shared-prefix
mixed/compressed attention instead. The same line also prints conservative
common-to-all scan estimates (`raw_shared_est`, `raw_reuse_est`,
`comp_shared_est`, `comp_reuse_est`) for sizing that kernel; current n=320 shows
about `4.46x` raw and `4.79x` compressed scan reuse on the remaining rows-exact
mixed/compressed path. The profiler also reports host-side encode, finish, and
total milliseconds for the rows-exact and varmap rows helpers. Inside the
shared decodeN command sequence this is not full GPU verifier wall time; use it
to separate host encode overhead from the broader decodeN verifier bucket.

`DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_EXACT=1` currently enables the GPU route
group descriptor scaffold and then falls back to row-exact math; it is an
opt-in correctness/structure probe for the next grouped exact MoE kernels, not
a production speed flag yet. `DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_IQ2=1`
additionally tries the first grouped IQ2 gate/up+SwiGLU consumer; it is
byte-clean in short smokes but currently slower than the default pair kernel, so
leave it off for speed runs.
The default greedy verifier is the strict decode-order hybrid path: it keeps
attention/cache mutation, indexer scoring, router, routed MoE, shared expert,
and HC accumulation row-exact, while batching only audited row-independent
subpaths for N<=5. Set
`DS4_DSPARK_FAST_VERIFY_DISABLE=1`, `DS4_DSPARK_EXACT_VERIFY=1`, or `--quality`
to force exact verification. Budget 2 now also uses this strict decodeN hybrid
path by default; set `DS4_DSPARK_DECODEN_VERIFY=0` or
`DS4_DSPARK_DECODEN_DISABLE=1` only to A/B the older specialized exact decode2
verifier. Exact decodeN captures prefix frontiers for accepted lengths 1..4, so
partial accepts from DSpark-5 can commit without replay; set
`DS4_DSPARK_DECODEN_PREFIXN_DISABLE=1` or the older
`DS4_DSPARK_DECODEN_PREFIX1_DISABLE=1` to A/B the replay path. Set
`DS4_DSPARK_SEQUENTIAL_VERIFY=1` only to compare against the older one-token
replay loop, or `DS4_DSPARK_DECODE2_CAPTURE_PREFIX1_DISABLE=1` to force the
slower decode2 exact-replay path for one-token accepts. Experimental target
suffix verification is opt-in with `DS4_DSPARK_BATCH_VERIFY=1`;
`DS4_DSPARK_BATCH_APPROX_STATE=1` may be faster, but it can diverge from exact
greedy output because the target state does not yet match the exact decode path.
Set `DS4_DSPARK_DECODEN_DISABLE=1`/`DS4_DSPARK_DECODE_N_DISABLE=1` and
`DS4_DSPARK_DECODE2_DISABLE=1` only when deliberately routing N=2..5 blocks into
that experimental batch verifier for diagnosis. Pair it with
`DS4_DSPARK_BATCH_MARGIN_GUARD=<float>` for diagnostic runs; the guard now checks
all accepted verifier rows and falls back to exact replay when the batch logits
are too close. Use `DS4_DSPARK_BATCH_STATE_AUDIT=1` to compare the batch verifier
frontier against exact replay when chasing that state drift; add
`DS4_DSPARK_BATCH_STATE_AUDIT_DSPARK_KV=1` to also compare the phase-correct
DSpark target-hidden and draft KV cache rows, and
`DS4_DSPARK_BATCH_STATE_AUDIT_DECODE_HC=1` to compare the batch verifier's final
hidden row against exact replay's current decode hidden row.
Use `--draft-mode batch` for the existing batch-canonical diagnostic path. Use
`--draft-mode unified` or `DS4_DSPARK_VERIFY_CANONICAL=unified` for the
MLX-inspired unified-greedy diagnostic label; the CLI also sets
`DS4_TARGET_FORWARD_UNIFIED=1`. Today that target-forward API routes no-draft
N=1 through a wrapper around the current decode path and routes DSpark N<=5
through a selectable backend. `DS4_TARGET_FORWARD_UNIFIED_BACKEND=batch` is the
current default and uses the existing batch target-forward verifier;
`DS4_TARGET_FORWARD_UNIFIED_BACKEND=strict_v1` routes N<=5 through the current
byte-clean strict verifier from the same unified wrapper, useful for comparing
the Plan C API against the shipping-safe verifier;
`DS4_TARGET_FORWARD_UNIFIED_BACKEND=shared_prefix` is the named Plan-C backend
entry point. It currently uses the strict-v1 core plus the byte-clean varmap
rows5 attention scaffold; current paired n=1000 tests keep `cmp=0` but do not
beat default strict, so treat it as Plan-C scaffolding until the exact resident
shared-prefix attention/cache/indexer kernel lands. Set
`DS4_TARGET_FORWARD_UNIFIED_SHARED_PREFIX_BATCH_FALLBACK=1` only to reproduce
the older diagnostic batch fallback, or set
`DS4_TARGET_FORWARD_UNIFIED_STRICT_BACKEND=1` to make the unified wrapper reject
the pending shared-prefix backend instead of using the strict-v1 core. Use
`DS4_TARGET_FORWARD_UNIFIED_PROFILE=1` to print per-call unified wrapper timing.
The next implementation step is replacing that backend with a real shared-prefix
rows path before treating unified-greedy as a production correctness contract.
For the strict verifier's dense Q8 row projections, DS4 now enables the
shared-weight rows5 kernel by default for N<=5. Set
`DS4_DSPARK_VERIFY_Q8_ROWS5_DISABLE=1` or `DS4_DSPARK_VERIFY_NO_Q8_ROWS5=1` to
A/B the older per-row Q8 exact matvecs. `DS4_DSPARK_VERIFY_Q8_ROWS5_SEQ=1`
keeps the diagnostic sequential rows5 kernel. The F16 rows5 path remains
opt-in with `DS4_DSPARK_VERIFY_F16_ROWS5=1`; current local smokes found it
byte-clean but slower than the default.
For the hybrid decodeN verifier, `DS4_DSPARK_HYBRID_STATE_AUDIT=1` and
`DS4_DSPARK_HYBRID_LAYER_HC_AUDIT=1` compare the hybrid path against exact
decodeN without committing unsafe rows. Add
`DS4_DSPARK_HYBRID_STATE_AUDIT_DSPARK_KV=1` to also compare the accepted-prefix
DSpark draft KV rows after both hybrid and exact paths rebuild them from their
captured target-hidden state. Add
`DS4_DSPARK_HYBRID_ATTN_STAGE_AUDIT=1` to split the attention half into
Q-rope, KV/cache row, attention heads, and attention output. Add
`DS4_DSPARK_HYBRID_FFN_STAGE_AUDIT=1` to also split that audit into FFN-pre,
FFN-norm, routed-output, and shared-output stages, and use
`DS4_DSPARK_HYBRID_LAYER_HC_AUDIT_EPS=<float>` to set the mismatch threshold.
The audit prints a `dspark hybrid first divergence` line with the earliest
material stage/layer/row/index in chronological layer order; use around
`1e-5` for practical drift and `1e-8` for byte-clean checks.
`DS4_DSPARK_STATE_AUDIT_EPS=<float>` controls the companion frontier-state
audit threshold; its summary reports the first state mismatch across compressor
frontiers, raw KV rows, final HC, DSpark target-hidden, and any captured DSpark
KV rows.
For a detailed reviewer handoff with the current benchmark table, safe/unsafe
verifier splits, and the next N<=5 exact-microbatch target, see
`docs/DSPARK_VERIFIER_HANDOFF.md`.
`DS4_DSPARK_HYBRID_STAGE_PROFILE=1` is a blocking diagnostic profiler for the
hybrid decodeN verifier; it prints one summary line splitting time across
attention, row-FFN-pre, row-router, row-routed MoE, row-shared expert,
batch-tail, DSpark target-hidden capture, output head, and readback.
`DS4_DSPARK_ATTN_SUBPROFILE=1` adds a fenced attention-half profile. Add
`DS4_DSPARK_ATTN_FINE_SUBPROFILE=1` only for microscope runs: it further splits
KV store, compressor projection/update/quant/capture, indexer-compressor
projection/update/capture, and indexer query/score/top-k. This is intentionally
slow and should not be used for headline t/s. The first active-5 smoke with the
fine split is saved under `bench-results/dspark_fine_subprofile_055320`; it
kept `cmp=0`, and the normal non-profile companion run reached 37.41 t/s
against a 33.33 t/s no-draft baseline.
`DS4_DSPARK_HYBRID_BATCH_ATTN_DECODE_ORDER=1` is an experimental path that uses
the batched attention kernel in decode-order mode while keeping FFN/router/MoE
row-exact. Current audits show the plain batch-attention composition is not
commit-safe, so use it only to reproduce attention-path drift.
`DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS_VARLEN_UNSAFE=1` enables the newer
N<=5 varlen plain-head diagnostic with explicit per-row `n_raw`, `raw_start`,
and `n_comp`; n=64 still diverged, so it is intentionally marked unsafe.
`DS4_DSPARK_HYBRID_DEFER_BATCH_HEADS_ROW_EXACT=1` instead defers only plain
heads and replays them through the existing row-exact flash-attention encoders in
one helper call. It is byte-clean in local n=64/n=160 smokes, but it is not a
speed win; use it as a measurement scaffold, not the final shared-prefix kernel.
`DS4_DSPARK_SHARED_PREFIX_PROFILE=1` (or
`DS4_TARGET_FORWARD_SHARED_PREFIX_PROFILE=1`) prints a non-fenced first-block
estimate of how many raw/compressed attention keys the current row-exact verifier
scans versus how many a shared-prefix N<=5 attention kernel would scan. Use it
to size the `shared_prefix` backend; it does not change verifier math or timing.
Add `DS4_DSPARK_SHARED_PREFIX_PROFILE_ALL=1` only when you deliberately want one
line per verifier block. `scripts/dspark_phase0_sweep.sh` enables the first-block
profile by default and writes `sp_*` columns in `summary.tsv`; set
`SHARED_PREFIX_PROFILE=0` to omit them.
`DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_QKV=1` keeps that path but computes HC-pre,
Q/KV, RoPE, and FP8 KV row preparation with single-row kernels before resuming
the batch attention helper. HC-pre F16 projection and Q8 Q/KV projections use
exact multirow helpers by default to reduce dispatch overhead while preserving
the single-row reduction kernels. Set
`DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_HC_EXACT_F16_ROWS_DISABLE=1` or
`DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_HC_EXACT_QKV_ROWS_DISABLE=1` only to restore
the older per-row projection calls for diagnosis. Attention-compressor
projections for all compressed layers also use exact multirow F16 helpers in
decode-order mode; set `DS4_DSPARK_HYBRID_ATTN_COMP_ROWS_DISABLE=1` only to
restore the older singleton pair-projection path. The default also enables
`DS4_DSPARK_HYBRID_INDEX_COMP_ROWS=1`, which gives the ratio-4 indexer
compressor its own exact multirow projection buffer while preserving row-local
state mutation; set `DS4_DSPARK_HYBRID_INDEX_COMP_ROWS_DISABLE=1` only for A/B
timing. In decode-order mode raw-KV visibility is deferred to the row attention
point.
`DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_OUTPUT=1` also makes the attention output
projection and HC update row-exact; current DSpark-KV audits show it is required
for the state-clean reference split even though the faster byte-match smoke can
omit it. Its exact Q8 output-HC tail is batched by default for N<=5; set
`DS4_DSPARK_HYBRID_ROW_OUTPUT_BATCH_HC_DISABLE=1` only to restore the older
per-row tail for diagnosis. The inverse RoPE and low projection subpaths also
use exact multirow helpers by default; set
`DS4_DSPARK_HYBRID_ROW_OUTPUT_BATCH_ROPE_DISABLE=1` or
`DS4_DSPARK_HYBRID_ROW_OUTPUT_LOW_ROWS_DISABLE=1` only for A/B diagnosis.
With row-QKV, row-output, row-router, row-routed, and row-shared enabled, DS4
captures prefix frontiers for accepted lengths 1..4
so partial accepts can commit with `decodeN-prefix` instead of exact replay;
disable with `DS4_DSPARK_DECODEN_PREFIXN_DISABLE=1` or
`DS4_DSPARK_HYBRID_BATCH_ATTN_PREFIXN_DISABLE=1` only while benchmarking. Current
local A/B shows that disabling prefix-N stays byte-clean but slows active-5
because partial-accept commit falls back to exact replay. This strict
decode-order hybrid keeps compressor/indexer state mutation row-local and is
audit-clean through N=5 by default. Use
`DS4_DSPARK_HYBRID_BATCH_ATTN_MAX_CLEAN=<2..5>` only as a diagnostic clamp, or
`DS4_DSPARK_HYBRID_BATCH_ATTN_ALLOW_UNSAFE_N_GT3=1` only for diagnostics.
`DS4_DSPARK_HYBRID_ROW_FFN_PRE=1` is an experimental verifier split that makes
FFN HC-pre/norm row-exact before resuming the batched FFN tail; it is useful for
diagnosis but is not yet a complete speedup path. `DS4_DSPARK_HYBRID_ROW_ROUTER=1`
also makes router selection row-exact. The row-router helper writes exact
single-row router outputs directly into the batch verifier rows by default; set
`DS4_DSPARK_HYBRID_ROW_ROUTER_DIRECT_DISABLE=1` only to restore the older
scratch-and-copy path for A/B timing. The F16 router logits and exact selector
subpaths use N<=5 row wrappers by default; set
`DS4_DSPARK_HYBRID_ROW_ROUTER_LOGITS_ROWS_DISABLE=1` or
`DS4_DSPARK_HYBRID_ROW_ROUTER_SELECT_ROWS_DISABLE=1` to restore the older
per-row calls.
`DS4_DSPARK_HYBRID_ROW_SHARED=1` makes the shared-expert gate/up path row-exact,
and `DS4_DSPARK_HYBRID_ROW_ROUTED=1` makes routed MoE row-exact. With row-QKV
enabled, both routed MoE and the shared expert need the row-exact guards for
long-run byte identity. The default also enables
`DS4_DSPARK_HYBRID_ROW_ROUTED_DECODE2_ROWS=1`, which keeps routed MoE in exact
decode order while encoding the N<=5 row loop through the banked native helper;
set it to `0` only for A/B timing against the older per-row path.
The default DSpark path also enables `DS4_DSPARK_ORDERED_MOE_SUM=1`, an exact
ordered expert-sum kernel that preserves the old routed-down plus FP32
ordered-add contract while reducing expert-add dispatches; set
`DS4_DSPARK_ORDERED_MOE_SUM_DISABLE=1` for A/B timing against the older add
chain.
The local Flash IQ2/Q2 strict verifier path uses direct Q2 down+ordered-sum by
default. It keeps independent per-slot accumulators inside one kernel and uses
the same `slot0 + slot1 + ... + slot5` add order, then writes the summed row
directly. Set `DS4_DSPARK_HYBRID_ROW_ROUTED_FUSED_ORDERED_Q2_DISABLE=1` to
restore the older separate Q2 down plus ordered FP32 expert-sum path for A/B.
Current clean A/B evidence: n=1000 conservative active-5 `39.12 t/s`; direct
ordered-Q2 `39.66 t/s`, also `cmp=0`; n=4000 conservative `36.47 t/s`, direct
ordered-Q2 `36.73 t/s`, with identical acceptance. This proves the direct
Q2-down/sum boundary for this local Q2 path; it does not prove grouped
shared-weight MoE fusion is safe, and it is not the large DSpark speed lever.
Fresh n=160 A/B in `bench-results/dspark_hc_output_subprofile_030709/clean`
measured no-direct-Q2 `39.56 t/s` versus direct ordered-Q2 `39.99 t/s`
(`cmp=0` for both). Treat this as a narrow micro-win; do not extend it into
unordered `sum6` or grouped shared-weight MoE without a new exactness proof.
For a focused MoE-boundary regression proof,
set
`DS4_DSPARK_HYBRID_ROW_ROUTED_DIRECT_Q2_COMPARE=1`. The hook is symmetric: if
the separate path is primary it computes direct Q2 into scratch, and if the
default fused direct-Q2 path is primary it computes the separate down plus
ordered-sum fallback into scratch. It prints exact/max/rms deltas with
`mode=primary-separate` or `mode=primary-direct`.
`bench-results/dspark_direct_q2_compare_buckets_022458` showed exact matches
(`max=0`) for `tokens=1`, active-5 verifier calls, and the active-3 tail while
keeping final output `cmp=0`. Post-promotion default validation in
`bench-results/dspark_direct_q2_promoted_default_023137` stayed clean: n=160
`cmp=0`, and n=1000 matched the paired no-draft baseline output while reaching
`39.71 t/s`. The symmetric default-primary guard
`bench-results/dspark_direct_q2_primary_compare_051916` also stayed `cmp=0`;
it logged 1720 direct-Q2 boundary compares, including active-5
`mode=primary-direct exact=yes`, with no `exact=no` lines. Fresh clean
active-size sweep
`bench-results/dspark_direct_q2_clean_sweep_023454` kept active-5 as best:
baseline `33.01 t/s`, b2 `33.83`, b3 `36.70`, b4 `39.10`, b5 `39.64 t/s`.
Current refresh `bench-results/dspark_budget_sweep_refresh_045720` gives the
same operating point at n=320: all budgets 2..5 are `cmp=0`, baseline
`33.24 t/s`, b2 `33.98`, b3 `36.94`, b4 `39.24`, and b5 `41.27 t/s`
(`84.6%`, `258/305`, tau `5.23`, block `97.68 ms`, draft `18.05 ms`, verify
`77.53 ms`). Treat active-5 as the current headline DSpark-5 path.
With direct-Q2 enabled, `DS4_DSPARK_VERIFY_DISPATCH_PROFILE=1` now accounts for
the removed ordered-sum dispatch correctly. `bench-results/dspark_direct_q2_dispatchfix_024232`
reported active-5 `ordered_sum=0`, `routed_moe=43`, and total dispatch estimate
`893` instead of the stale `936`; the largest remaining row-scaled buckets are
attention heads `215`, compressor `205`, and indexer `105`.
`DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN=1` is a diagnostic grouped-down
consumer. It now defaults to the exact wrapper because the first shared-weight
grouped Q2 kernel is not correct (`cmp=1`, n=160 `33.17 t/s`, verifier
`77.89 ms`). The companion rejected path can still be forced with
`DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_DOWN_UNSAFE=1` for focused debugging.
The safe wrapper uses the same GPU route descriptor but calls the known exact Q2
row kernel per pair; it is byte-clean
(`cmp=0`, n=160 `38.15 t/s`). That isolates the bug to the shared-weight
grouped Q2 math, not route grouping or output layout. Keep both flags
diagnostic only. The focused compare hook
`DS4_DSPARK_HYBRID_ROW_ROUTED_GROUPED_Q2_COMPARE=1` runs the safe wrapper and
the unsafe shared-weight kernel in the same verifier call and compares the
slot-down tensors before the ordered expert sum. Current audit
`bench-results/dspark_default_after_fusion_guard_034504` keeps the default
strict path clean at n=160 (`cmp=0`, DSpark `39.94 t/s`, baseline `33.28 t/s`)
while the unsafe grouped-Q2 path is both wrong and slower (`cmp=1`, `33.83 t/s`,
acceptance `68.0%`). The compare hook in the same directory shows the unsafe
shared-weight path is not slot-bit-exact before the ordered sum. Fresh recheck
`bench-results/dspark_wrong_fusion_recheck_040257` gives the same conclusion:
direct ordered-Q2 compare is exact, while unsafe grouped Q2 has active-5
slot-down deltas up to `4.768e-7` and diverges at n=160 (`cmp=1`, first diff
swaps `RED`/`GREEN`). Do not treat grouped shared-weight Q2 as a template for
DSpark verifier fusion; only row/slot-exact fusion with the same accumulator and
ordered-add contract can be promoted.
Fresh no-compare clean A/B
`bench-results/dspark_grouped_q2_clean_042538` makes the boundary sharper:
paired baseline `33.37 t/s`, default direct ordered-Q2 DSpark `39.87 t/s`
(`cmp=0`), grouped-safe descriptor wrapper `38.34 t/s` (`cmp=0`), and unsafe
shared-weight grouped Q2 `33.74 t/s` (`cmp=1`). The route grouping and output
layout are safe; the current shared-weight grouped dot is the wrong FP
realization and is slower anyway. Focused compare
`bench-results/dspark_grouped_q2_compare_044810` makes that more precise:
single-token calls are exact, while active-5 calls differ from the safe wrapper
by about `4.47e-8` to `4.77e-7` in the slot-down tensor. That is FP-order drift,
not a descriptor/stride bug, and it is enough to corrupt the long greedy state.
Current-tree recheck `bench-results/dspark_wrong_fusion_current_n160_062305`
kept default active-5 clean (`cmp=0`, DSpark `39.23 t/s`) while the forced
unsafe grouped-Q2 path diverged (`cmp=1`, first diff swaps `RED`/`GREEN`), so
the rejected fusion remains grouped shared-weight Q2, not direct ordered-Q2.
The row-shared gate/up/SwiGLU subpath uses the existing fused single-row kernel
by default; set `DS4_DSPARK_HYBRID_ROW_SHARED_FUSED_DISABLE=1` only to restore
the older separate gate/up/SwiGLU row path for A/B timing.
The row-shared tail now also uses an exact N<=5 shared-down+HC rows kernel by
default. It keeps the same per-token Q8 reduction and the same HC add order, but
removes the old CPU-side loop over verifier rows. Set
`DS4_DSPARK_HYBRID_ROW_SHARED_DOWN_HC_ROWS_DISABLE=1` to restore the older
per-row shared-down+HC calls. Current A/B:
`bench-results/dspark_shared_down_rows_n1000_054326` stayed `cmp=0` for both
paths with identical acceptance `77.1% (794/1030)`; rows default measured
`38.29 t/s`, verifier `77.23 ms`, versus disabled `38.00 t/s`, verifier
`78.24 ms`.
`DS4_DSPARK_HYBRID_ROW_ROUTED_SLOTWISE=1` is an opt-in diagnostic that makes the
row-routed verifier helper use the slotwise resident MoE kernel when the Flash
sidecar is in identity slot mode. It exists to test whether that single-row
kernel family can reduce verifier cost without changing state.
`DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_ROW_EXACT=1` is enabled by default. It calls
the tiny-batch row-exact encoder from the strict row-routed helper. For local
Flash Q2 down experts, the backend may use the direct ordered-Q2 down+sum
kernel; this is not the older unsafe `sum6` path. It keeps six independent slot
accumulators and writes the exact slot0..slot5 FP32 ordered sum. Set
`DS4_DSPARK_HYBRID_ROW_ROUTED_FUSED_ORDERED_Q2_DISABLE=1` to force the older
separate-down plus ordered-add path, or
`DS4_DSPARK_HYBRID_ROW_ROUTED_DIRECT_Q2_COMPARE=1` to compare both paths while
keeping the conservative output authoritative. Set
`DS4_DSPARK_HYBRID_ROW_ROUTED_BATCH_ROW_EXACT=0` only for A/B timing against the
older per-row row-loop helper.
`DS4_DSPARK_HYBRID_BATCH_ROUTER_ROW_ROUTED=1` is a diagnostic that keeps routed
MoE row-exact while batching router selection. Current audits still show
layer-1 `attn-heads-raw` drift, and it is not faster than the exact row-router
path, so leave it off outside A/B experiments.
`DS4_DSPARK_HYBRID_BATCH_ATTN_ROW_HC_BATCH_QKV=1` is a fast but unsafe
diagnostic: it reached about 36.1 t/s on the local n=1000 Flash prompt, but
changed greedy output. `DS4_DSPARK_HYBRID_MARGIN_GUARD=<float>` can force
exact replay for low-margin accepted rows and `DS4_DSPARK_HYBRID_READ_ALL_LOGITS=1`
forces all verifier row logits to be read back, but a 0.5..4.0 guard sweep did
not restore byte-identical output for that batch-QKV path.
`DS4_DSPARK_TINY_BATCH_ROW_KERNEL=1` (alias:
`DS4_MTP_SIDECAR_BATCH_SLOTBANK_ROW_EXACT=1`) is a diagnostic tiny-batch routed
MoE path that encodes the single-row native kernels in one command buffer; it did
not by itself remove the hybrid drift in current Flash resident tests.
`DS4_DSPARK_HYBRID_EXACT_PREFIX_LAYERS=N` is a diagnostic only; it can make
early target layers exact, but current tests show that enough exact-prefix depth
to preserve n=1000 output is slower than the normal exact verifier.

On the local M5 Max Flash resident prompt, the strict path is now above baseline:
the default DSpark-5 run measured 39.04 t/s on n=1000 versus a refreshed no-draft
baseline of 35.34 t/s, and 36.48 t/s on the long n=4000 run versus 33.90 t/s.
Active 2, 3, 4, and 5 all use the same DSpark draft fusion (`dspark-strided`
MXFP4 routed MoE) and the same strict decodeN hybrid verifier helpers. Current
timing shows the draft module is not the limiting term: active 5 proposes about
265 draft tokens/s, while the target verifier checks only about 58 proposed
tokens/s. The remaining production work is a cheaper exact N<=5 verifier,
especially attention/cache mutation and routed MoE, not more acceptance-rate
tuning.

### Experimental Pro Support

DeepSeek V4 Pro sidecar support is experimental. For Pro agent runs, use
`--nothink`, keep the slot bank at or below 32 slots while tuning, and keep
shared-down decode prefetch enabled:

```sh
DS4_FLASH_MOE_DECODE_PREFETCH_SHARED_DOWN=1 ./ds4-agent \
  -m ~/Models/DSv4Pro-flash/ \
  --moe-slot-bank 32 \
  --ctx 32768 \
  --nothink
```

Larger Pro slot banks can consume enough memory bandwidth and residency budget
to collapse decode throughput, so only raise `--moe-slot-bank` after measuring
reuse and decode stalls on your machine.

For diagnostic fanout tests, add `--moe-expert-topk 4`. This is different from
`--moe-prefetch-topk`: it changes the actual routed expert count for both
prefill and decode, so quality and logits are expected to change.

`--no-int8` is optional. Normal runs use the fastest measured profile path.
For quality-preserving runs, pass `--no-int8`; it disables current int8 dense,
NAX, Flash-MoE, and ANE accelerator paths, using NAX-half where safe and GPU
fallbacks otherwise. `--quality` implies `--no-int8`.

## Run Resident GGUF Mode

Download a resident GGUF:

```sh
./download_model.sh q2-imatrix
```

An alternate resident GGUF,
[Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF/resolve/main/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf),
is available from
[huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF):

```sh
./download_model.sh huihui-iq2xxs
```

Then run:

```sh
./ds4 -p "Hello"
```

Resident mode loads the full model file and is meant for high-memory machines.
It is still useful for baseline comparison, server use, and systems with enough
RAM to hold the selected quantization. The Huihui IQ2_XXS resident GGUF can be
used on a 96 GB M3 Ultra, but memory headroom is tight; run with little to
nothing else active. Reaped/pruned resident models are still under investigation.

See [docs/RESIDENT.md](docs/RESIDENT.md) and
[docs/MODEL_SETUP.md](docs/MODEL_SETUP.md).

## Validate

The alpha validation gate is:

```sh
make clean
make
./ds4_test --server --metal-kernels
make ane-smoke
DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major make sidecar-smoke
```

The sidecar smoke uses `tests/test-vectors/prompts/long_code_audit.txt`, a
shorter 4K-class prompt, and generates 64 deterministic tokens with a 4K
prefill chunk cap.

## Docs

- [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md): model files, downloads, and
  sidecar package expectations.
- [docs/SIDECAR.md](docs/SIDECAR.md): SSD streaming mode and smoke test.
- [docs/SIDECAR_EXPORT.md](docs/SIDECAR_EXPORT.md): external expert-sidecar
  export wrapper and its dense-GGUF caveats.
- [docs/STREAMING_KNOBS.md](docs/STREAMING_KNOBS.md): SSD sidecar slot-bank,
  prefill, I/O, ANE, and profile knobs.
- [docs/RESIDENT.md](docs/RESIDENT.md): full-GGUF resident mode.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): runtime layout and accelerator
  paths.
- [docs/PROFILES.md](docs/PROFILES.md): machine-specific tuning defaults and
  override rules.
- [docs/ANE_KERNELS.md](docs/ANE_KERNELS.md): experimental Apple Neural Engine
  kernel families and private API notes.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md): current benchmark stance.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): common first-run failures.
- [docs/DWARFSTAR4_REFERENCE.md](docs/DWARFSTAR4_REFERENCE.md): original DS4
  README retained for reference.

## Attribution

`ds4-ssd` is derived from antirez's DwarfStar 4 / `ds4` work and keeps the DS4
model-specific design: GGUF loading, prompt rendering, KV handling, server API,
and DeepSeek V4 Flash validation. The project also depends conceptually on the
GGUF, quantization, and kernel work pioneered by `llama.cpp` and GGML.

The SSD-streaming direction is also indebted to Apple's
[LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://machinelearning.apple.com/research/efficient-large-language)
paper and to the original [danveloper/flash-moe](https://github.com/danveloper/flash-moe)
work by Claude Opus 4.6 and Daniel Woods. Read the
[original Flash-MoE paper](https://github.com/danveloper/flash-moe/blob/main/paper/flash_moe.pdf)
for the full story of how they built that engine in 24 hours.

The Apple Neural Engine path uses GPU-side int8 dequantization/packing together
with ANE MLP execution through private Apple APIs and additional scheduling
optimizations. GPU int8 dequantization for this class of local inference was
pioneered by Liu Liu (Draw Things, @liuliu), and the private ANE API path was
first documented publicly by @maderix.

Keep the repository `LICENSE` with redistributions and preserve attribution to
antirez, llama.cpp, GGML, and their contributors.
