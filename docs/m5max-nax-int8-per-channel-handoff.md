# M5 Max NAX INT8 per-channel quality and speed handoff

## Status and objective

This handoff has now been executed on `M5M.local` (`Apple M5 Max`, 128 GiB).
It remains a reproducibility guide; the completed quality and speed results are
summarized below.

The experiment answers two questions:

1. Does NAX INT8 recover GPU/no-int8 quality when it consumes the same
   expert-specific scale vectors already qualified for ANE?
2. What is the cold-prefill speed cost, if any, versus the original scalar
   NAX INT8 path?

Use branch `ANE-INT8-IMPROVE`. Keep this run isolated from the other active
profiling task. Coordinate ownership of the M5 Max and one shared profiling
lock before starting. Do not kill its processes, reuse or clean its run
directory, or change any of its environment or model files.

Completed resident production speed means (2048 / 4096 / 8192 tokens):

- scalar `h-i8`: 218.34 / 281.43 / 348.00 t/s;
- plain NAX-half: 206.60 / 267.70 / 341.50 t/s;
- NAX-half+ALU: 214.00 / 275.03 / 344.95 t/s;
- per-channel `h-i8-pc`: 213.04 / 281.21 / 348.04 t/s.

Thus PC is -2.43% / -0.08% / +0.01% versus scalar and +3.12% /
+5.05% / +1.92% versus plain NAX-half. All 12 balanced runs used the fully
preloaded 256-slot identity bank with zero transient staging.

The strict 100-case resident quality arm is exactly identical to strict
streaming PC: average NLL 0.404266357 in both, 100/100 score ties, zero logit
KL/RMS/max-absolute difference, correlation 1.0, and 100% top-1 agreement.
See `bench-results/m5max-resident-prefill-speed-20260717-v2/report.md` and
`gguf-tools/quality-testing/runs/m5max-nax-pc-20260717/report.md`.

## Scale contract and implementation scope

The ANE scale data is reusable by NAX because it describes the quantized
weights, not the execution backend. Use the existing v2 package on the SN8100
volume; do not generate another sidecar or add metadata.

For every one of 256 experts in every routed layer, the expert record contains
one packed little-endian FP16 dequantization tensor:

```text
[ gate: 2048 | up: 2048 | down: 4096 ]
```

That is 8,192 FP16 values, or 16 KiB per expert. Each expert has its own gate,
up, and down values. Across 43 layers and 256 experts, the scale payload is
180,355,072 bytes (172 MiB, 0.168 GiB). The stored contract remains FP16 even
if NAX creates a temporary FP32 reciprocal view for weight requantization.

The per-channel executor supports both the original streaming sidecar path and
a fully preloaded 256-slot identity bank. Resident execution uses zero-copy
per-expert slot views for weights and reads each expert's FP16 scale tail from
the same resident record. It deliberately does not use the separate full-GGUF
resident grouped/dedup executor. Therefore:

- always pass the v2 package as a sidecar;
- explicitly disable resident grouped/dedup routing;
- use `--resident --moe-slot-bank 256` plus full slot-bank preload for the
  zero-copy resident path;
- require the identity-bank, `h-i8-pc`, FP16-scale, and zero-stage markers;
- do not interpret a standalone full-GGUF resident result as per-channel.

`REQUIRE=1` is also an execution contract. The h-i8 NAX kernel has a 64-row
M-tile correctness floor, while official-corpus expert groups are often much
smaller. The sidecar caller therefore zero-pads every strict logical group
to a 64-row dispatch (or the next 64-row boundary), runs `h-i8-pc`, and
scatters only the logical rows. Padded activation and route-weight rows are
zero, so they cannot contribute to model output. A strict call that reaches
the lower helper without this padding is rejected rather than split or
silently reported as fully NAX-backed.

The scale vectors exist only in the v2 expert-major sidecar. A standalone
full-GGUF or non-identity resident-grouped run with per-channel `REQUIRE=1`
must fail explicitly; it is not allowed to fall back to scalar NAX. For
quality, always pass `SIDECAR` as the positional package argument, never the
standalone dense `MODEL` path.

The new switches are:

- `DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=1`: consume the packed scale tensor;
- `DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=1`: fail closed on missing or
  invalid scales and require padded per-channel NAX execution for every
  logical expert group.

With the isolated settings below, the selected lower path must report
`mode=h-i8-pc`. The per-channel implementation demotes scalar-only fused W8A8
paths, but those paths are also forced off here so the A/B is explicit.

A forced-non-M5 one-case smoke on M3 first passed this contract. It reported
`strict NAX INT8 per-channel M-tile padding active: logical_refs=2 dispatch_refs=64 tile=64`,
the packed count `8192 (2048/2048/4096)`, and `mode=h-i8-pc`. The mandatory M5
streaming and resident smokes have since passed as well.

## 1. Coordinate the box and acquire the shared lock

Run this in a dedicated terminal. Ask the other profiling task to use the same
`PROFILE_LOCK` path. An existing lock or active uncoordinated DS4 process is a
stop condition: contact its owner. Never delete an unfamiliar lock and never
use `pkill` to make the box available.

```zsh
set -euo pipefail

export PROFILE_LOCK="/tmp/ds4-m5max-exclusive-profile.lock"

pgrep -fl 'ds4|score_official' || true

if ! mkdir "$PROFILE_LOCK"; then
  print -u2 "M5 Max profiling lock is already held: $PROFILE_LOCK"
  exit 75
fi
trap 'rmdir "$PROFILE_LOCK"' EXIT INT TERM
```

Do not proceed until the process list is understood and the other profiling
task has acknowledged this run window.

## 2. Set target paths

Replace every `/SET/ME/...` value. The paths are intentionally not guessed for
the M5 Max. `SIDECAR` must be the existing SN8100 v2 package;
`MODEL` must be its dense GGUF. The 100-case official Flash corpus is saved
data and may live outside this checkout, so set `MANIFEST` to its actual target
path.

Choose a new, experiment-owned `RUNROOT`. The preflight refuses to reuse an
existing path and therefore cannot overwrite another task's results.

```zsh
export REPO="/SET/ME/path/to/ds4-ssd"
export SIDECAR="/SET/ME/SN8100/dsv4-iq2xxs-expert-major-ane-i8-v2"
export MODEL="/SET/ME/SN8100/dsv4-iq2xxs-expert-major-ane-i8-v2/dense/model-dense.gguf"
export MANIFEST="/SET/ME/gguf-tools/quality-testing/data/flash/manifest.tsv"
export RUNROOT="/SET/ME/results/m5max-nax-int8-pc-YYYYMMDD-HHMMSS"

test -d "$REPO/.git"
test -d "$SIDECAR"
test -f "$SIDECAR/manifest.json"
test -f "$MODEL"
test -f "$MANIFEST"
test ! -e "$RUNROOT"

mkdir -p "$RUNROOT/quality" "$RUNROOT/smoke" "$RUNROOT/speed"
```

Record provenance before any run:

```zsh
git -C "$REPO" rev-parse HEAD > "$RUNROOT/git-commit.txt"
git -C "$REPO" status --short > "$RUNROOT/git-status.txt"
sw_vers > "$RUNROOT/macos.txt"
system_profiler SPHardwareDataType > "$RUNROOT/hardware.txt"
shasum -a 256 "$SIDECAR/manifest.json" > "$RUNROOT/sidecar-manifest.sha256.before"
```

Validate that this is the locked v2 scale layout. This reads the manifest only;
it does not modify the package.

```zsh
python3 - "$SIDECAR/manifest.json" <<'PY' | tee "$RUNROOT/scale-contract.txt"
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text())
s = d["ane_i8_scale_scheme"]
assert d["runtime_loadable"] is True
assert d["storage_layout"] == "expert_major_weights_plus_ane_i8_output_scales_v1"
assert s["family_order"] == ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
assert s["dtype"] == "F16"
assert s["semantics"] == "dequant_multiplier"
assert s["axis"] == 1 and s["group_size"] == 1
assert len(d["layer_files"]) == 43

expected = {
    "ffn_gate_exps": 2048,
    "ffn_up_exps": 2048,
    "ffn_down_exps": 4096,
}
seen = {(int(e["layer"]), e["tensor_family"]): int(e["ane_i8_scale_count"])
        for e in d["entries"]}
assert len(seen) == 43 * 3
for layer in range(43):
    for family, count in expected.items():
        assert seen[(layer, family)] == count
for layer in d["layer_files"]:
    assert int(layer["expert_count"]) == 256
    assert int(layer["scale_region_bytes"]) == 16384

total = sum(int(x["expert_count"]) * int(x["scale_region_bytes"])
            for x in d["layer_files"])
assert total == 180355072
print("contract=valid")
print("layers=43 experts_per_layer=256")
print("packed_counts=2048/2048/4096 dtype=F16")
print(f"total_scale_bytes={total}")
PY
```

## 3. Check out and build the branch

Use an idle worktree that belongs to this run. Do not switch branches in the
other profiler's live worktree.

```zsh
test "$(git -C "$REPO" branch --show-current)" = "ANE-INT8-IMPROVE"
git -C "$REPO" diff --check

make -C "$REPO" -j"$(sysctl -n hw.logicalcpu)" ds4 ds4-bench
make -C "$REPO/gguf-tools" quality-score

test -x "$REPO/ds4-bench"
test -x "$REPO/gguf-tools/quality-testing/score_official"
```

Save the actual command interfaces with the run:

```zsh
"$REPO/ds4-bench" --help > "$RUNROOT/ds4-bench-help.txt"
"$REPO/gguf-tools/quality-testing/score_official" --help \
  > "$RUNROOT/score-official-help.txt"
python3 "$REPO/scripts/compare_logits.py" --help \
  > "$RUNROOT/compare-logits-help.txt"
```

## 4. Isolated backend environment

Keep this array in the same zsh session for all commands below. Explicit zeroes
prevent the machine profile or inherited shell state from selecting ANE,
ANE+NAX hybrid, NAX-half, resident grouped/dedup, or scalar-only fused routes.
`DS4_FLASH_MOE_KERNEL_LOG=1` is the implementation's kernel/scale diagnostic
gate; scheduler statistics provide the surrounding routing evidence.

```zsh
COMMON_ENV=(
  DS4_ANE=0
  DS4_FLASH_MOE_ANE_PREFILL=0
  DS4_FLASH_MOE_ANE_PIPELINE_PREFILL=0
  DS4_FLASH_MOE_ANE_PER_CHANNEL=0
  DS4_FLASH_MOE_ANE_REQUIRE=0
  DS4_FLASH_MOE_ANE_I8I8_PREFILL=0
  DS4_FLASH_MOE_ANE_I8I8_FUSED_PREFILL=0
  DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=0
  DS4_FLASH_MOE_ANE_I8I8_FULL_FUSED_PREFILL=0
  DS4_FLASH_MOE_HYBRID_PREFILL=0
  DS4_FLASH_MOE_HYBRID_CONCURRENT_PREFILL=0
  DS4_FLASH_MOE_CONCURRENT_PREFILL=0
  DS4_FLASH_MOE_OVERLAP_PREFILL=0
  DS4_RESIDENT_MOE_ANE_NAX_HYBRID=0
  DS4_RESIDENT_MOE_NAX_HALF=0
  DS4_RESIDENT_MOE_NAX_HALF_MAX_TOKENS=0
  DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP=0
  DS4_FLASH_MOE_RESIDENT_GROUPED_PREFILL=0
  DS4_RESIDENT_MOE_MPP_DEDUP_PREFILL=0
  DS4_RESIDENT_MOE_NAX_DEDUP_PREFILL=0
  DS4_RESIDENT_MOE_MPP_INT8_PREFILL=0
  DS4_RESIDENT_MPP_INT8_PREFILL=0
  DS4_FLASH_MOE_MPP_INT8_ACT=0
  DS4_FLASH_MOE_MPP_I8I8_PREFILL=0
  DS4_FLASH_MOE_MPP_I8I8_FUSED_PREFILL=0
  DS4_FLASH_MOE_MPP_I8I8_TILED_FUSED_PREFILL=0
  DS4_FLASH_MOE_MPP_I8I8_FULL_FUSED_PREFILL=0
  DS4_FLASH_MOE_MPP_INT8_QSCALE=512
  DS4_FLASH_MOE_KERNEL_LOG=1
  DS4_FLASH_MOE_SCHED_STATS=1
)
```

The three quality modes differ only as follows:

| Mode | `DS4_NO_INT8` | MPP/NAX INT8 | Per-channel | Require |
| --- | ---: | ---: | ---: | ---: |
| GPU/no-int8 reference | 1 | 0 | 0 | 0 |
| Original scalar NAX INT8 | 0 | 1 | 0 | 0 |
| New per-channel NAX INT8 | 0 | 1 | 1 | 1 |

## 5. Fail-fast one-case smoke

Run the strict candidate first. It must exit zero and prove that a short
logical group was padded to a legal 64-row dispatch, executed by `h-i8-pc`,
and not sent to the classic GPU fallback.

```zsh
mkdir -p "$RUNROOT/smoke/pc-logits"

env "${COMMON_ENV[@]}" \
  DS4_NO_INT8=0 \
  DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
  DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=1 \
  DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=1 \
  "$REPO/gguf-tools/quality-testing/score_official" \
    --ctx 4096 --limit 1 --first-token-only \
    --dump-logits-dir "$RUNROOT/smoke/pc-logits" \
    "$SIDECAR" "$MANIFEST" "$RUNROOT/smoke/pc.tsv" \
    > "$RUNROOT/smoke/pc.stdout.log" \
    2> "$RUNROOT/smoke/pc.log"

rg -F \
  'Flash-MoE NAX INT8 per-channel active: packed gate/up/down count=8192 (2048/2048/4096) mode=h-i8-pc' \
  "$RUNROOT/smoke/pc.log"
rg -n \
  'strict NAX INT8 per-channel M-tile padding active: logical_refs=[1-9][0-9]* dispatch_refs=64 tile=64' \
  "$RUNROOT/smoke/pc.log"
rg -F \
  'Flash-MoE MPP prefill engaged: tokens=64 mode=h-i8-pc' \
  "$RUNROOT/smoke/pc.log"

if rg -n \
  'NAX INT8 per-channel (scales )?unavailable|using scalar NAX INT8 weight scale|ERROR:|unpadded partial M tile' \
  "$RUNROOT/smoke/pc.log"; then
  print -u2 'strict per-channel smoke reported an error or scalar fallback'
  exit 1
fi
```

If any positive marker is absent, stop. Do not run the full corpus under an
unproven backend. The padding marker may show a logical count other than the
M3 smoke's `2`; `dispatch_refs=64` and `mode=h-i8-pc` are the required facts.

## 6. Run the 100-case official-continuation quality matrix

`score_official` accepts the package directory as its model argument and
autodetects `dense/model-dense.gguf` plus the slot-bank sidecar. The first
matrix below establishes the streaming baseline. Repeat its strict PC arm with
`--resident --moe-slot-bank 256` and the preload environment to validate the
zero-copy path against it. Each run dumps one 129,280-value FP32 distribution
immediately after prefill for every case.

```zsh
Q="$RUNROOT/quality"
mkdir -p "$Q/gpu-logits" "$Q/scalar-nax-logits" "$Q/pc-nax-logits"

env "${COMMON_ENV[@]}" \
  DS4_NO_INT8=1 \
  DS4_FLASH_MOE_MPP_INT8_PREFILL=0 \
  DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=0 \
  DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=0 \
  "$REPO/gguf-tools/quality-testing/score_official" \
    --ctx 4096 --no-int8 \
    --dump-logits-dir "$Q/gpu-logits" \
    "$SIDECAR" "$MANIFEST" "$Q/gpu.tsv" \
    > "$Q/gpu.stdout.log" 2> "$Q/gpu.log"

env "${COMMON_ENV[@]}" \
  DS4_NO_INT8=0 \
  DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
  DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=0 \
  DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=0 \
  "$REPO/gguf-tools/quality-testing/score_official" \
    --ctx 4096 \
    --dump-logits-dir "$Q/scalar-nax-logits" \
    "$SIDECAR" "$MANIFEST" "$Q/scalar-nax.tsv" \
    > "$Q/scalar-nax.stdout.log" 2> "$Q/scalar-nax.log"

env "${COMMON_ENV[@]}" \
  DS4_NO_INT8=0 \
  DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
  DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=1 \
  DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=1 \
  "$REPO/gguf-tools/quality-testing/score_official" \
    --ctx 4096 \
    --dump-logits-dir "$Q/pc-nax-logits" \
    "$SIDECAR" "$MANIFEST" "$Q/pc-nax.tsv" \
    > "$Q/pc-nax.stdout.log" 2> "$Q/pc-nax.log"
```

The first run is the GPU/no-int8 quality reference. NAX-half is forced off, so
verify the routed-expert diagnostic names a GPU path rather than NAX-half.
The scalar and strict per-channel runs use byte-identical model and sidecar
inputs. The strict candidate intentionally also sets `REQUIRE=1`: its 64-row
zero-padding makes every short official-corpus expert group exercise
per-channel NAX. The production `REQUIRE=0` A/B is reserved for the speed
matrix, where partial tails retain the existing classic-GPU behavior.

## 7. Compare official scores and dense logits

Run every pair so the result set captures GPU versus scalar NAX, GPU versus
per-channel NAX, and the direct scalar-to-per-channel improvement.

```zsh
python3 "$REPO/gguf-tools/quality-testing/compare_scores.py" \
  "$Q/gpu.tsv" "$Q/scalar-nax.tsv" \
  > "$Q/gpu-vs-scalar-nax-scores.txt"
python3 "$REPO/gguf-tools/quality-testing/compare_scores.py" \
  "$Q/gpu.tsv" "$Q/pc-nax.tsv" \
  > "$Q/gpu-vs-pc-nax-scores.txt"
python3 "$REPO/gguf-tools/quality-testing/compare_scores.py" \
  "$Q/scalar-nax.tsv" "$Q/pc-nax.tsv" \
  > "$Q/scalar-vs-pc-nax-scores.txt"

python3 "$REPO/scripts/compare_logits.py" \
  "$Q/gpu-logits" "$Q/scalar-nax-logits" \
  --per-case "$Q/gpu-vs-scalar-nax-logits.tsv" \
  > "$Q/gpu-vs-scalar-nax-logits.txt"
python3 "$REPO/scripts/compare_logits.py" \
  "$Q/gpu-logits" "$Q/pc-nax-logits" \
  --per-case "$Q/gpu-vs-pc-nax-logits.tsv" \
  > "$Q/gpu-vs-pc-nax-logits.txt"
python3 "$REPO/scripts/compare_logits.py" \
  "$Q/scalar-nax-logits" "$Q/pc-nax-logits" \
  --per-case "$Q/scalar-vs-pc-nax-logits.tsv" \
  > "$Q/scalar-vs-pc-nax-logits.txt"
```

The standard comparator reports KL, JS, RMS, maximum absolute error,
correlation, top-1 agreement, and top-5 overlap. The following companion audit
adds the promotion-specific top-3 recall and high-confidence flip test:

```zsh
python3 - "$Q/gpu-logits" "$Q/pc-nax-logits" <<'PY' \
  > "$Q/gpu-vs-pc-nax-gates.tsv"
import math
import sys
from pathlib import Path

import numpy as np

ref_dir = Path(sys.argv[1])
cand_dir = Path(sys.argv[2])
names = sorted(p.name for p in ref_dir.glob("*.f32")
               if (cand_dir / p.name).is_file())
assert len(names) == 100, len(names)

kls = []
corrs = []
top1 = 0
top5 = []
ref_winner_in_cand_top3 = 0
flips = 0
both_margin_gt_1 = 0

for name in names:
    ref = np.fromfile(ref_dir / name, dtype=np.float32)
    cand = np.fromfile(cand_dir / name, dtype=np.float32)
    assert ref.shape == cand.shape == (129280,)
    assert np.isfinite(ref).all() and np.isfinite(cand).all()

    ref64 = ref.astype(np.float64)
    cand64 = cand.astype(np.float64)
    lr = ref64 - (ref64.max() + math.log(np.exp(ref64 - ref64.max()).sum()))
    lc = cand64 - (cand64.max() + math.log(np.exp(cand64 - cand64.max()).sum()))
    p = np.exp(lr)
    kls.append(float(np.sum(p * (lr - lc))))
    corrs.append(float(np.corrcoef(ref64, cand64)[0, 1]))

    rt = int(np.argmax(ref))
    ct = int(np.argmax(cand))
    same = rt == ct
    top1 += int(same)
    flips += int(not same)
    cand_top3 = set(np.argpartition(cand, -3)[-3:].tolist())
    ref_winner_in_cand_top3 += int(rt in cand_top3)
    ref_top5 = set(np.argpartition(ref, -5)[-5:].tolist())
    cand_top5 = set(np.argpartition(cand, -5)[-5:].tolist())
    top5.append(len(ref_top5 & cand_top5))

    if not same:
        ref_two = np.partition(ref, -2)[-2:]
        cand_two = np.partition(cand, -2)[-2:]
        ref_margin = float(ref_two.max() - ref_two.min())
        cand_margin = float(cand_two.max() - cand_two.min())
        both_margin_gt_1 += int(ref_margin > 1.0 and cand_margin > 1.0)

print(f"cases\t{len(names)}")
print(f"top1_match_rate\t{top1 / len(names):.9g}")
print(f"top1_flips\t{flips}")
print(f"ref_winner_recall_at_candidate_top3\t{ref_winner_in_cand_top3 / len(names):.9g}")
print(f"both_margin_gt_1_flips\t{both_margin_gt_1}")
print(f"kl_mean\t{np.mean(kls):.9g}")
print(f"kl_p95\t{np.percentile(kls, 95):.9g}")
print(f"kl_max\t{np.max(kls):.9g}")
print(f"corr_mean\t{np.mean(corrs):.9g}")
print(f"corr_min\t{np.min(corrs):.9g}")
print(f"top5_overlap_mean\t{np.mean(top5):.9g}")
PY
```

## 8. Prove backend engagement and artifact integrity

These checks are part of the result, not optional log browsing. The strict
per-channel process exiting zero establishes that no requested call failed
closed. The padding, packed-scale, and lower-kernel markers prove that short
groups ran through legal 64-row `h-i8-pc` dispatches; the negative checks prove
the run did not split into the production GPU-tail path or silently use scalar
NAX scales.

```zsh
rg -F \
  'Flash-MoE MPP prefill gate: requested=1 allowed=1' \
  "$Q/scalar-nax.log" "$Q/pc-nax.log"
rg -F \
  'Flash-MoE NAX INT8 per-channel active: packed gate/up/down count=8192 (2048/2048/4096) mode=h-i8-pc' \
  "$Q/pc-nax.log"
rg -n \
  'strict NAX INT8 per-channel M-tile padding active: logical_refs=[1-9][0-9]* dispatch_refs=64 tile=64' \
  "$Q/pc-nax.log"
rg -F \
  'Flash-MoE MPP prefill engaged: tokens=64 mode=h-i8-pc' \
  "$Q/pc-nax.log"

if rg -n \
  'NAX INT8 per-channel (scales )?unavailable|using scalar NAX INT8 weight scale|ERROR:|unpadded partial M tile|MPP/NAX int8 partial-tile workaround active|evaluation failed|write failed|reopen failed|reject(ed)?=[1-9][0-9]*|fail(ure)?s?=[1-9][0-9]*' \
  "$Q/pc-nax.log"; then
  print -u2 'per-channel log contains a failure, reject, or scalar fallback'
  exit 1
fi

if rg -F \
  'Flash-MoE NAX INT8 per-channel active:' \
  "$Q/scalar-nax.log"; then
  print -u2 'scalar control unexpectedly enabled per-channel scales'
  exit 1
fi

rg -n 'prefill compute: routed experts = .*GPU' "$Q/gpu.log"
if rg -n 'routed experts = .*ANE|routed experts = .*NAX-half|ANE\+NAX|hybrid.*active' \
  "$Q/gpu.log" "$Q/scalar-nax.log" "$Q/pc-nax.log"; then
  print -u2 'an excluded ANE, NAX-half, or hybrid route was selected'
  exit 1
fi
```

Check the corpus and raw distributions exactly. Each logits file must contain
129,280 little-endian FP32 values, or 517,120 bytes.

```zsh
python3 - "$Q" <<'PY'
import csv
import sys
from pathlib import Path

q = Path(sys.argv[1])
for mode in ("gpu", "scalar-nax", "pc-nax"):
    with (q / f"{mode}.tsv").open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 100, (mode, len(rows))
    files = sorted((q / f"{mode}-logits").glob("*.f32"))
    assert len(files) == 100, (mode, len(files))
    bad = [(p.name, p.stat().st_size) for p in files if p.stat().st_size != 517120]
    assert not bad, (mode, bad)
    print(f"{mode}: scores=100 logits=100 bytes_per_logits=517120")
PY

shasum -a 256 "$SIDECAR/manifest.json" > "$RUNROOT/sidecar-manifest.sha256.after"
cmp "$RUNROOT/sidecar-manifest.sha256.before" \
    "$RUNROOT/sidecar-manifest.sha256.after"
```

## 9. Cold-prefill speed matrix

Measure scalar versus per-channel NAX at 128, 512, 2,048, 4,000, and 8,192
tokens on both tracked long prompts. The added 8,192 frontier makes NAX-eligible
64-row expert groups more likely and is supported by these long prompt files;
`ds4-bench` will fail before timing if a target copy tokenizes too short.

This is a production-policy A/B, not a strict-padding benchmark. Both speed
modes set `REQUIRE=0`. The per-channel candidate sets only
`DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL=1`: eligible 64-row prefixes use
`h-i8-pc`, while partial tails use the same classic GPU path as scalar NAX.
No strict zero-padding overhead is charged to the production candidate.

Use five fresh-process repeats and alternate mode order by repeat. Every
process has one frontier, zero decode tokens, no weight warmup, and
`--full-prefill-each-frontier`, so the CSV reports a cold full prefill. A
30-second cooldown reduces thermal order bias; increase it if the M5 Max is
not thermally stable.

Do not run this matrix concurrently with any other model, benchmark, indexing,
or profiling workload.

```zsh
REPEATS=5
COOLDOWN=30

run_cold_prefill() {
  local mode="$1"
  local prompt_name="$2"
  local prompt_file="$3"
  local ctx="$4"
  local rep="$5"
  local pc

  if [[ "$mode" == "pc" ]]; then
    pc=1
  else
    pc=0
  fi

  env "${COMMON_ENV[@]}" \
    DS4_NO_INT8=0 \
    DS4_FLASH_MOE_MPP_INT8_PREFILL=1 \
    DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL="$pc" \
    DS4_FLASH_MOE_NAX_INT8_PER_CHANNEL_REQUIRE=0 \
    "$REPO/ds4-bench" \
      --model "$MODEL" \
      --moe-sidecar "$SIDECAR" \
      --moe-mode slot-bank \
      --moe-slot-bank 256 \
      --prompt-file "$prompt_file" \
      --ctx-start "$ctx" \
      --ctx-max "$ctx" \
      --full-prefill-each-frontier \
      --gen-tokens 0 \
      --csv "$RUNROOT/speed/${mode}-${prompt_name}-ctx${ctx}-r${rep}.csv" \
      > "$RUNROOT/speed/${mode}-${prompt_name}-ctx${ctx}-r${rep}.stdout.log" \
      2> "$RUNROOT/speed/${mode}-${prompt_name}-ctx${ctx}-r${rep}.log"
}

for rep in $(seq 1 "$REPEATS"); do
  if (( rep % 2 )); then
    order=(scalar pc)
  else
    order=(pc scalar)
  fi

  for prompt_name in story security; do
    if [[ "$prompt_name" == "story" ]]; then
      prompt_file="$REPO/tests/long_context_story_prompt.txt"
    else
      prompt_file="$REPO/tests/long_context_security_prompt.txt"
    fi

    for ctx in 128 512 2048 4000 8192; do
      for mode in "${order[@]}"; do
        run_cold_prefill "$mode" "$prompt_name" "$prompt_file" "$ctx" "$rep"
        sleep "$COOLDOWN"
      done
    done
  done
done
```

Reject actual scale-contract or runtime errors, but do not require a
per-channel marker in every speed log. At 128 and 512 tokens, all routed groups
may be smaller than one legal NAX tile, in which case production correctly uses
the classic GPU path. Record engagement per run instead:

```zsh
for log in "$RUNROOT"/speed/pc-*.log; do
  if rg -q \
    'NAX INT8 per-channel (scales )?unavailable|using scalar NAX INT8 weight scale|ERROR:' \
    "$log"; then
    print -u2 "invalid production per-channel speed log: $log"
    exit 1
  fi
done

python3 - "$RUNROOT/speed" <<'PY' > "$RUNROOT/speed-engagement.tsv"
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
pat = re.compile(r"pc-(story|security)-ctx(\d+)-r(\d+)\.log$")
active = (
    "Flash-MoE NAX INT8 per-channel active: packed gate/up/down "
    "count=8192 (2048/2048/4096) mode=h-i8-pc"
)
print("prompt\tctx\trepeat\tpc_active\th_i8_pc_engaged\tpartial_gpu_tail_marker")
for path in sorted(root.glob("pc-*.log")):
    match = pat.fullmatch(path.name)
    assert match, path
    prompt, ctx, repeat = match.groups()
    text = path.read_text(errors="replace")
    print(
        f"{prompt}\t{ctx}\t{repeat}\t{int(active in text)}\t"
        f"{int('Flash-MoE MPP prefill engaged:' in text and 'mode=h-i8-pc' in text)}\t"
        f"{int('MPP/NAX int8 partial-tile workaround active' in text)}"
    )
PY
```

Aggregate medians without discarding the per-run CSVs:

```zsh
python3 - "$RUNROOT/speed" <<'PY' > "$RUNROOT/speed-summary.tsv"
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
pat = re.compile(r"(scalar|pc)-(story|security)-ctx(\d+)-r(\d+)\.csv$")
values = defaultdict(list)
engaged = defaultdict(list)
partial_tail = defaultdict(list)
for path in root.glob("*.csv"):
    m = pat.fullmatch(path.name)
    if not m:
        continue
    mode, prompt, ctx, _rep = m.groups()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, (path, len(rows))
    values[(mode, prompt, int(ctx))].append(float(rows[0]["prefill_tps"]))

active_marker = (
    "Flash-MoE NAX INT8 per-channel active: packed gate/up/down "
    "count=8192 (2048/2048/4096) mode=h-i8-pc"
)
for path in root.glob("pc-*.log"):
    m = re.fullmatch(r"pc-(story|security)-ctx(\d+)-r(\d+)\.log", path.name)
    assert m, path
    prompt, ctx, _rep = m.groups()
    text = path.read_text(errors="replace")
    key = (prompt, int(ctx))
    engaged[key].append(int(active_marker in text))
    partial_tail[key].append(int("MPP/NAX int8 partial-tile workaround active" in text))

print("prompt\tctx\trepeats\tscalar_median_tps\tpc_median_tps\tpc_delta_percent\tpc_engaged_repeats\tpartial_gpu_tail_marker_repeats")
for prompt in ("story", "security"):
    for ctx in (128, 512, 2048, 4000, 8192):
        scalar = values[("scalar", prompt, ctx)]
        pc = values[("pc", prompt, ctx)]
        assert len(scalar) == len(pc) >= 3, (prompt, ctx, len(scalar), len(pc))
        assert len(engaged[(prompt, ctx)]) == len(pc)
        sm = statistics.median(scalar)
        pm = statistics.median(pc)
        delta = (pm / sm - 1.0) * 100.0
        print(
            f"{prompt}\t{ctx}\t{len(pc)}\t{sm:.6f}\t{pm:.6f}\t{delta:.3f}\t"
            f"{sum(engaged[(prompt, ctx)])}\t{sum(partial_tail[(prompt, ctx)])}"
        )
PY
```

The speed objective is minimal degradation: zero is ideal. Report every length
and both prompts; do not hide a short- or long-context regression in one grand
average. Treat any repeatable negative delta as an optimization item. A loss
larger than 2% should be investigated before promotion even if quality passes;
the project owner may set a stricter final speed gate after seeing M5 Max noise.
Zero PC engagement at 128 or 512 is valid production behavior. If every 8,192
run also reports zero engagement, treat the speed experiment as inconclusive
and inspect routing rather than claiming zero per-channel overhead.

## 10. Acceptance gates

The per-channel NAX candidate is quality-acceptable only if all of these hold
against the GPU/no-int8 dense-logit reference:

- prompt-final top-1 agreement is at least 90%;
- the GPU winner is in the NAX top 3 for 100% of cases;
- zero top-1 flips have both paths' winning margins above 1.0;
- mean KL is at most 0.10, P95 KL at most 0.35, and maximum KL at most 1.5;
- mean correlation is at least 0.990 and minimum correlation at least 0.960;
- mean top-5 overlap is at least 4.4/5;
- official-continuation NLL regression is no greater than 1%;
- strict execution reports 64-row M-tile padding plus `mode=h-i8-pc`, with
  zero scale fallbacks, classic-GPU partial-tail splits, rejects, or evaluation
  failures;
- all 100 score rows and all 100 correctly sized logits files exist per mode.

Speed is reported separately for scalar and production per-channel NAX
(`REQUIRE=0`) at every matrix point, including how many repeats actually
engaged `h-i8-pc`. Quality passing does not excuse a material speed regression,
and a no-engagement speed result does not measure per-channel overhead.

## 11. Generate the Markdown result handback

The following command creates a self-contained result table from the saved
TSVs and CSVs. It also records the strict activation/fallback evidence. Review
the generated file and add any thermal, process, or outlier notes before
handing it back.

```zsh
python3 - "$RUNROOT" <<'PY' > "$RUNROOT/M5MAX-RESULTS.md"
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
q = root / "quality"

def score_summary(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    tokens = sum(int(r["target_tokens"]) for r in rows)
    avg_nll = sum(float(r["nll"]) for r in rows) / tokens
    return {
        "cases": len(rows),
        "tokens": tokens,
        "nll": avg_nll,
        "ppl": math.exp(avg_nll),
        "bits": avg_nll / math.log(2.0),
        "first": sum(int(r["first_match"]) for r in rows),
        "lcp": sum(int(r["greedy_lcp"]) for r in rows) / len(rows),
    }

scores = {
    "GPU/no-int8": score_summary(q / "gpu.tsv"),
    "Scalar NAX INT8": score_summary(q / "scalar-nax.tsv"),
    "Per-channel NAX INT8": score_summary(q / "pc-nax.tsv"),
}
gpu_nll = scores["GPU/no-int8"]["nll"]
pc_nll = scores["Per-channel NAX INT8"]["nll"]
nll_delta = (pc_nll / gpu_nll - 1.0) * 100.0

def key_values(path):
    result = {}
    for line in path.read_text().splitlines():
        if "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        try:
            result[key] = float(value)
        except ValueError:
            pass
    return result

pairwise = {
    "GPU -> scalar NAX": key_values(q / "gpu-vs-scalar-nax-logits.txt"),
    "GPU -> per-channel NAX": key_values(q / "gpu-vs-pc-nax-logits.txt"),
    "Scalar -> per-channel NAX": key_values(q / "scalar-vs-pc-nax-logits.txt"),
}

gates = {}
for line in (q / "gpu-vs-pc-nax-gates.tsv").read_text().splitlines():
    key, value = line.split("\t", 1)
    gates[key] = float(value)

pc_log = (q / "pc-nax.log").read_text(errors="replace")
active_marker = (
    "Flash-MoE NAX INT8 per-channel active: packed gate/up/down "
    "count=8192 (2048/2048/4096) mode=h-i8-pc"
)
padding_marker = "strict NAX INT8 per-channel M-tile padding active:"
forbidden = re.compile(
    r"NAX INT8 per-channel (?:scales )?unavailable|"
    r"using scalar NAX INT8 weight scale|ERROR:|unpadded partial M tile|"
    r"MPP/NAX int8 partial-tile workaround active|evaluation failed|"
    r"write failed|reopen failed|reject(?:ed)?=[1-9][0-9]*|"
    r"fail(?:ure)?s?=[1-9][0-9]*"
)
active_count = pc_log.count(active_marker)
padding_count = pc_log.count(padding_marker)
mode_count = pc_log.count("mode=h-i8-pc")
forbidden_count = len(forbidden.findall(pc_log))

pat = re.compile(r"(scalar|pc)-(story|security)-ctx(\d+)-r(\d+)\.csv$")
speed = defaultdict(list)
speed_engaged = defaultdict(list)
speed_partial_tail = defaultdict(list)
for path in (root / "speed").glob("*.csv"):
    m = pat.fullmatch(path.name)
    if not m:
        continue
    mode, prompt, ctx, _rep = m.groups()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 1:
        speed[(mode, prompt, int(ctx))].append(float(rows[0]["prefill_tps"]))
for path in (root / "speed").glob("pc-*.log"):
    m = re.fullmatch(r"pc-(story|security)-ctx(\d+)-r(\d+)\.log", path.name)
    if not m:
        continue
    prompt, ctx, _rep = m.groups()
    text = path.read_text(errors="replace")
    key = (prompt, int(ctx))
    speed_engaged[key].append(int(active_marker in text))
    speed_partial_tail[key].append(
        int("MPP/NAX int8 partial-tile workaround active" in text)
    )

checks = [
    ("Top-1 >= 90%", gates["top1_match_rate"] >= 0.90),
    ("GPU winner recall@NAX top-3 = 100%",
     gates["ref_winner_recall_at_candidate_top3"] == 1.0),
    ("Both-margin>1 flips = 0", gates["both_margin_gt_1_flips"] == 0),
    ("Mean KL <= 0.10", gates["kl_mean"] <= 0.10),
    ("P95 KL <= 0.35", gates["kl_p95"] <= 0.35),
    ("Max KL <= 1.5", gates["kl_max"] <= 1.5),
    ("Mean corr >= 0.990", gates["corr_mean"] >= 0.990),
    ("Min corr >= 0.960", gates["corr_min"] >= 0.960),
    ("Top-5 overlap >= 4.4", gates["top5_overlap_mean"] >= 4.4),
    ("NLL regression <= 1%", nll_delta <= 1.0),
    ("Strict padding and PC path, no forbidden marker",
     padding_count > 0 and active_count > 0 and mode_count > 0 and
     forbidden_count == 0),
]

commit = (root / "git-commit.txt").read_text().strip()
print("# M5 Max NAX INT8 per-channel results")
print()
print(f"- Commit: `{commit}`")
print("- Model/sidecar paths: see execution environment and archived logs")
print("- Local authoring run: none; these are M5 Max target results")
print()
print("## Official-continuation quality")
print()
print("| Mode | Cases | Tokens | Avg NLL | PPL | Bits/token | First match | Avg LCP |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for name, s in scores.items():
    print(f"| {name} | {s['cases']} | {s['tokens']} | {s['nll']:.9f} | "
          f"{s['ppl']:.6f} | {s['bits']:.6f} | {s['first']}/100 | {s['lcp']:.3f} |")
print()
print(f"Per-channel NAX NLL delta versus GPU/no-int8: `{nll_delta:+.3f}%`.")
print()
print("## Pairwise dense-logit metrics")
print()
print("| Pair (reference -> candidate) | Mean KL | P95 KL | Mean JS | "
      "Mean RMS | Max abs | Mean corr | Min corr | Top-1 | Top-5 |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for name, m in pairwise.items():
    print(f"| {name} | {m['kl_mean']:.6f} | {m['kl_p95']:.6f} | "
          f"{m['js_mean']:.6f} | {m['rms_mean']:.6f} | {m['max_abs_max']:.6f} | "
          f"{m['corr_mean']:.6f} | {m['corr_min']:.6f} | "
          f"{m['top1_match_rate']:.3%} | {m['top5_overlap_mean']:.3f}/5 |")
print()
print("## Dense-logit quality gate")
print()
print("| Metric | Result | Gate |")
print("| --- | ---: | ---: |")
print(f"| Top-1 agreement | {gates['top1_match_rate']:.3%} | >= 90% |")
print(f"| GPU winner recall@NAX top-3 | {gates['ref_winner_recall_at_candidate_top3']:.3%} | 100% |")
print(f"| Both-margin>1 flips | {int(gates['both_margin_gt_1_flips'])} | 0 |")
print(f"| Mean KL | {gates['kl_mean']:.6f} | <= 0.10 |")
print(f"| P95 KL | {gates['kl_p95']:.6f} | <= 0.35 |")
print(f"| Max KL | {gates['kl_max']:.6f} | <= 1.5 |")
print(f"| Mean correlation | {gates['corr_mean']:.6f} | >= 0.990 |")
print(f"| Minimum correlation | {gates['corr_min']:.6f} | >= 0.960 |")
print(f"| Mean top-5 overlap | {gates['top5_overlap_mean']:.3f}/5 | >= 4.4/5 |")
print()
print("## Strict execution evidence")
print()
print("| Evidence | Count |")
print("| --- | ---: |")
print(f"| Strict 64-row padding marker | {padding_count} |")
print(f"| Packed-scale active marker | {active_count} |")
print(f"| `mode=h-i8-pc` marker | {mode_count} |")
print(f"| Forbidden failure/fallback markers | {forbidden_count} |")
print()
print("## Cold full-prefill speed medians")
print()
print("Production A/B uses per-channel enabled with `REQUIRE=0`; missing PC "
      "engagement at a short frontier is valid.")
print()
print("| Prompt | Tokens | Repeats | Scalar NAX t/s | PC NAX t/s | PC delta | "
      "PC engaged | GPU-tail marker |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for prompt in ("story", "security"):
    for ctx in (128, 512, 2048, 4000, 8192):
        scalar = speed[("scalar", prompt, ctx)]
        pc = speed[("pc", prompt, ctx)]
        if not scalar or not pc:
            print(f"| {prompt} | {ctx} | 0 | MISSING | MISSING | MISSING | "
                  "MISSING | MISSING |")
            continue
        sm = statistics.median(scalar)
        pm = statistics.median(pc)
        delta = (pm / sm - 1.0) * 100.0
        engaged = sum(speed_engaged[(prompt, ctx)])
        partial = sum(speed_partial_tail[(prompt, ctx)])
        print(f"| {prompt} | {ctx} | {min(len(scalar), len(pc))} | "
              f"{sm:.3f} | {pm:.3f} | {delta:+.2f}% | "
              f"{engaged}/{len(pc)} | {partial}/{len(pc)} |")
print()
print("## Gate verdicts")
print()
for label, passed in checks:
    print(f"- [{'x' if passed else ' '}] {label}")
print()
print("## Operator notes")
print()
print("- Thermal stability / cooldown observations: TODO")
print("- Competing processes or interruptions: TODO")
print("- Outliers and reruns (never delete original run files): TODO")
PY
```

Review the generated report, then package the entire immutable run directory
beside it. This keeps raw logits, per-case TSVs, diagnostics, help text, and
provenance together.

```zsh
ARCHIVE="${RUNROOT%/}.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$RUNROOT")" "$(basename "$RUNROOT")"
shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"

print "Result report: $RUNROOT/M5MAX-RESULTS.md"
print "Archive:       $ARCHIVE"
print "Checksum:      $ARCHIVE.sha256"
```

After the result paths are recorded and the other profiling task is notified,
release only the lock created by this session:

```zsh
rmdir "$PROFILE_LOCK"
trap - EXIT INT TERM
```
