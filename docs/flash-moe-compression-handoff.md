# Flash-MoE IOAccelerator compression cliff handoff

Date: 2026-06-18

This handoff is **only** for the M5 Max native MXFP4 `ds4-agent`
IOAccelerator/shared-buffer compression cliff. Do not use it as a general
Flash-MoE performance tuning note, L2-cache note, or page-cache cliff note.

The older page-cache cliff is already documented in `docs/mxfp4-handoff.md`
and `docs/STREAMING_KNOBS.md`. That older issue was about true SSD reads after
a wired bank evicted the filesystem cache. The issue here is different:
macOS appears to compress cold/inactive Metal/IOAccelerator-backed pages, and
decode later stalls while those pages are decompressed.

## Problem

On Apple M5 Max 128 GB, native MXFP4 sidecar decode can run near 15 t/s with a
large mixed slot bank, then suddenly fall to 0.1-0.5 t/s. During the bad state,
SSD read can be low and the slot bank can still report high residency, so this
is not just the old "wired bank evicted OS file cache, every miss hits SSD"
failure mode.

The current evidence points at macOS compressing cold/inactive IOAccelerator /
shared Metal buffer pages. When decode later touches those pages,
decompression bursts dominate and token rate collapses.

## Compression signature

A run is in the compression cliff when some combination of these appears:

- `task-compressed` rises by GiBs.
- `gpu-compressed` rises above zero, even briefly.
- `decompressions` jumps by hundreds of thousands or millions.
- t/s falls to 0.1-0.5 or enters long sync/tool-result stalls.
- SSD read is not necessarily saturated.
- Slot residency may still look good or may be refilling after a grow.
- macOS Activity Monitor can show memory pressure and compressed memory even
  while total app RSS/physical memory looks below the machine limit.

The strongest current marker is the decompression counter. In the bad logs,
the t/s collapse lines coincide with `decompressions` jumping from 0 to
millions.

## Compression-specific working theory

Large slot banks are not automatically fatal. The fatal pattern is:

1. Allocate or grow a large slot bank.
2. Many pages are resident/allocated from the VM perspective but cold from
   recent-access perspective.
3. Memory pressure rises.
4. macOS compresses cold IOAccelerator/shared buffer pages.
5. Decode touches those pages and triggers huge decompression work.
6. t/s collapses, sometimes to 0.1.

This fits the latest logs better than "total RAM is too high" or "112 slots is
intrinsically too large." A 96-slot segment reached 15.8-16.5 t/s cleanly; the
cliff happened immediately after a cold grow to 112 that reset resident slots
to zero.

`DS4_FLASH_MOE_SLOT_BANK_TOUCH_PAGES=1` only touches pages at allocation time.
It does not keep pages hot during decode and is not a keepalive.

## Do not conflate with page-cache cliff

Old cliff:

- Cause: huge wired slot bank crowds out OS file cache.
- Symptom: decode misses become true SSD reads.
- Mitigation: shrink decode bank after prefill.

This cliff:

- Cause hypothesis: cold shared/IOAccelerator pages are compressed under memory
  pressure, then decompressed on use.
- Symptom: decompression counters explode, `task-compressed` or
  `gpu-compressed` rises, t/s collapses even without obvious SSD saturation.
- Mitigation under test: avoid cold bank realloc/grow, preserve hot pages,
  delay growth, recover to a useful bank instead of permanently falling to 64,
  maybe add a real keepalive/touch strategy.

For this handoff, do not chase SSD I/O first unless the log shows high sustained
SSD reads during the actual slow tokens. The compression evidence is stronger
than the page-cache evidence for the current M5 Max failure.

## Compression-specific code map

- `ds4.c`
  - `metal_graph_flash_moe_slot_snapshot_*`
  - `metal_graph_flash_moe_compression_recovery_*`
  - VM stats fields in slot snapshots:
    `gpu-footprint`, `gpu-compressed`, `task-compressed`, `phys`,
    `decompressions`, `mem-pressure`, `mem-free`, `swap-used`
- `ssd/ssd_flash_moe_slot_cache.c`
  - decode shrink/grow logic
  - `DS4_FLASH_MOE_GROW_SLOT_BANK_CARRY`
  - compression recovery shrink
  - grow suppression after recovery unless
    `DS4_FLASH_MOE_GROW_AFTER_COMPRESSION_RECOVERY=1`
- `ssd/ssd_flash_moe_allocation.c`
  - slot-bank allocation and startup page touch
- `ds4_metal.m`
  - `ds4_gpu_tensor_touch_pages`
  - Metal VM/stat plumbing

## Compression instrumentation

Use token-based snapshots so the log captures grow/recovery events:

```sh
mkdir -p /tmp/ds4-snapshots
[ -f /tmp/ds4-slot-snapshot.log ] && \
  mv /tmp/ds4-slot-snapshot.log \
     "/tmp/ds4-snapshots/ds4-slot-snapshot-$(date +%Y%m%d-%H%M%S).log"
```

Useful knobs:

```sh
DS4_FLASH_MOE_SLOT_SNAPSHOT_TOKENS=16
DS4_FLASH_MOE_SLOT_SNAPSHOT_FILE=/tmp/ds4-slot-snapshot.log
DS4_FLASH_MOE_SLOT_SNAPSHOT_HOT_TOKENS=128
DS4_FLASH_MOE_RESIDENCY_STATS_START=1
DS4_FLASH_MOE_RESIDENCY_STATS=0
```

Snapshot checks are cheap enough for this diagnosis. Observed one-shot checks
were typically sub-ms to a few ms; the visible long pauses in bad runs are not
from printing one line, they coincide with decompression/refill events.

Fields that matter most:

- `tps`: symptom.
- `slots`: current bank size.
- `resident`: whether a grow/shrink reset the cache.
- `gpu-compressed`: graphics compressed pages.
- `task-compressed`: process compressed pages.
- `decompressions`: best cliff trigger signal seen so far.
- `mem-pressure`, `mem-free`, `swap-used`: pressure context.

## Key evidence from logs

### `/tmp/ds4-slot-grow-64-134.log`

This run grew by cold free/reallocate steps. Summary:

- `64 -> 80` at token 512: refill completed, no compression.
- `80 -> 96` at token 1024: refill completed, no compression.
- `96 -> 112` at token 1536: refill completed, no compression in that run.
- Full 112 segment token 1712-2032:
  - average about 14.17 t/s
  - max about 15.60 t/s
  - no compression
- `112 -> 128` at token 2048 reset residency to `0/5504`.
- Token 2064:
  - `tps=0.27`
  - `gpu-compressed=1.84 GiB`
  - `task-compressed=21.46 GiB`
  - `decompressions=19,904,202`
- Recovery shrank to 64. After refill, GPU compression cleared and t/s
  recovered, but the run stayed at 64.

Conclusion: 128 cold grow is clearly unsafe. 112 can be good once hot.

### `/tmp/ds4-slot-grow-64-112.log`

This was the 112-capped run. It did not stay at 112; recovery fired almost
immediately after the cold grow.

Important lines:

```text
tok=1408 tps=15.82 slots=96  resident=4128/4128 gc=0.00 tc=0.00
tok=1424 tps=15.56 slots=96  resident=4128/4128 gc=0.00 tc=0.00
tok=1440 tps=16.27 slots=96  resident=4128/4128 gc=0.00 tc=0.00
tok=1456 tps=16.50 slots=96  resident=4128/4128 gc=0.00 tc=0.00
tok=1536 tps=8.28  slots=112 resident=0/4816    gc=0.00 tc=0.00
tok=1552 tps=0.45  slots=112 resident=2021/4816 gc=0.06 tc=3.52 decompressions=1181953
tok=1568 tps=0.51  slots=64  resident=0/2752    gc=0.00 tc=6.00 decompressions=1998926
```

Conclusion: this does not prove 112 steady-state is bad. It proves the current
grow method is bad because it creates a cold, partially-filled 112-slot bank
under memory pressure.

## Compression false leads

- `DS4_METAL_RESIDENT_DENSE=all` plus `DS4_METAL_RESIDENT_DENSE_FORCE=1`
  did not prevent the cliff. Dense residency is not the primary trigger.
- `sudo sysctl iogpu.wired_limit_mb=120000` did not prevent compression. It
  raises the allowed GPU working set but does not guarantee cold pages remain
  uncompressed.
- `DS4_FLASH_MOE_RESIDENCY_STATS=256` only logs; it does not clean up or
  recover.
- Compression recovery can shrink the bank, but by default it marks recovery
  done and prevents later growth. This avoids oscillation, but it can
  permanently leave the run at 64 slots and lower throughput.
- `--moe-slot-bank 64` avoids compression more often, but wastes memory and
  gives up the 96/112-slot speed potential.
- Shared L2 cache experiments were not clearly helpful for this native FP4
  decode path. Treat L2 as unproven for this specific cliff unless a log shows
  L2 hits replacing sidecar/refill pressure.

## Next compression experiment

The next best test is not another cold 64->112 grow. Start at 96, let it get
hot, then carry-grow to 112 much later. Recovery should go back to 96, not 64.
`DS4_FLASH_MOE_GROW_AFTER_COMPRESSION_RECOVERY=1` is mandatory in this test:
without it, a decode-start compression sample can mark recovery done at the
96-slot recovery target and suppress the later 96->112 grow entirely.
Add the KV touch flags for the next pass. They are diagnostic knobs: if the
decode-start cliff improves or the snapshot immediately after touching reports
less compressed GPU/task memory, then cold shared KV/cache pages are part of the
trigger. If they do not change the cliff, the problem is more likely slot-bank
or other graph state.

Also enable the slow-decode dump guard for this experiment. It watches a real
wall-clock decode window and, if TPS stays below the threshold for the requested
seconds, frees and reallocates the Flash-MoE slot bank while preserving KV/cache
context. This is intentionally an emergency recovery path, not a root-cause fix.
Omitting `DS4_FLASH_MOE_SLOW_DECODE_SLOT_BANK` means "free/reallocate the live
slot count"; setting it to `96` or `64` turns the same guard into a forced
emergency shrink.

```sh
/usr/bin/time -lp env DS4_LOCK_FILE=/tmp/ds4-cli-grow-96-112-carry.lock \
  DS4_METAL_RESUME_PREFILL_MIN=1 \
  DS4_FLASH_MOE_PREFILL_SLOT_BANK=96 \
  DS4_FLASH_MOE_RESUME_PREFILL_SLOT_BANK=96 \
  DS4_FLASH_MOE_FULL_PREFILL_SLOT_BANK=96 \
  DS4_FLASH_MOE_FORCE_PREFILL_SLOT_BANK=1 \
  DS4_FLASH_MOE_RESTORE_SLOT_BANK_AFTER_PREFILL=0 \
  DS4_FLASH_MOE_STEADY_SLOT_BANK=112 \
  DS4_FLASH_MOE_GROW_SLOT_BANK_DURING_DECODE=1 \
  DS4_FLASH_MOE_GROW_AFTER_COMPRESSION_RECOVERY=1 \
  DS4_FLASH_MOE_GROW_SLOT_BANK_CARRY=1 \
  DS4_FLASH_MOE_GROW_SLOT_BANK_STEP=16 \
  DS4_FLASH_MOE_GROW_SLOT_BANK_WARMUP_TOKENS=512 \
  DS4_FLASH_MOE_GROW_SLOT_BANK_INTERVAL_TOKENS=256 \
  DS4_FLASH_MOE_GROW_SLOT_BANK_MIN_RESIDENT_PCT=99 \
  DS4_FLASH_MOE_GROW_SLOT_BANK_MIN_HIT_PCT=80 \
  DS4_FLASH_MOE_GROW_SLOT_BANK_MAX_COMPRESSED_MB=256 \
  DS4_FLASH_MOE_GROW_SLOT_BANK_MAX_TASK_COMPRESSED_MB=1024 \
  DS4_FLASH_MOE_COMPRESSION_RECOVERY=1 \
  DS4_FLASH_MOE_COMPRESSION_RECOVERY_INTERVAL=16 \
  DS4_FLASH_MOE_COMPRESSION_RECOVERY_MB=1024 \
  DS4_FLASH_MOE_COMPRESSION_RECOVERY_TASK=1 \
  DS4_FLASH_MOE_COMPRESSION_RECOVERY_TASK_MB=2048 \
  DS4_FLASH_MOE_COMPRESSION_RECOVERY_SAMPLES=1 \
  DS4_FLASH_MOE_COMPRESSION_RECOVERY_SLOT_BANK=96 \
  DS4_FLASH_MOE_SLOT_SNAPSHOT_TOKENS=16 \
  DS4_FLASH_MOE_SLOT_SNAPSHOT_FILE=/tmp/ds4-slot-grow-96-112-carry.log \
  DS4_FLASH_MOE_SLOT_SNAPSHOT_HOT_TOKENS=128 \
  DS4_METAL_KV_TOUCH_ON_PREFILL_START=1 \
  DS4_METAL_KV_TOUCH_ON_DECODE_START=1 \
  DS4_FLASH_MOE_SLOW_DECODE_DUMP_CACHE=1 \
  DS4_FLASH_MOE_SLOW_DECODE_SECONDS=3 \
  DS4_FLASH_MOE_SLOW_DECODE_TPS=1 \
  DS4_FLASH_MOE_SLOW_DECODE_MAX_DUMPS=1 \
  DS4_FLASH_MOE_SLOW_DECODE_COOLDOWN_TOKENS=128 \
  DS4_FLASH_MOE_RESIDENCY_STATS=0 \
  DS4_FLASH_MOE_RESIDENCY_STATS_START=1 \
  DS4_MXFP4_NATIVE=1 \
  DS4_FLASH_MOE_ANE_PREFILL=1 \
  DS4_METAL_PREFILL_CHUNK=4096 \
  DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS=1 \
  DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO=1 \
  DS4_FLASH_MOE_SLOTWISE_DECODE=1 \
  ./ds4-agent \
    -m /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash \
    --ctx 32768 \
    -n 20000 \
    --temp 0 \
    --nothink \
    --moe-slot-bank 134
```

Success criteria:

- 96-slot steady segment is still around 15 t/s.
- Carry-grow log says resident records were preserved.
- No `resident=0/...` reset at the 112 transition.
- `gpu-compressed` stays near 0.
- `task-compressed` stays below 1-2 GiB and decompressions do not jump.
- If recovery fires, it shrinks to 96, not 64.

## Compression control runs

No-grow 96-slot baseline:

```sh
/usr/bin/time -lp env DS4_LOCK_FILE=/tmp/ds4-cli-96-baseline.lock \
  DS4_MXFP4_NATIVE=1 \
  DS4_FLASH_MOE_ANE_PREFILL=1 \
  DS4_METAL_PREFILL_CHUNK=4096 \
  DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS=1 \
  DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO=1 \
  DS4_FLASH_MOE_SLOTWISE_DECODE=1 \
  DS4_FLASH_MOE_SLOT_SNAPSHOT_TOKENS=16 \
  DS4_FLASH_MOE_SLOT_SNAPSHOT_FILE=/tmp/ds4-slot-96-baseline.log \
  DS4_FLASH_MOE_SLOT_SNAPSHOT_HOT_TOKENS=128 \
  DS4_FLASH_MOE_RESIDENCY_STATS=0 \
  DS4_FLASH_MOE_RESIDENCY_STATS_START=1 \
  ./ds4-agent \
    -m /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash \
    --ctx 32768 \
    -n 20000 \
    --temp 0 \
    --nothink \
    --moe-slot-bank 96
```

No-grow 112-slot baseline:

```sh
/usr/bin/time -lp env DS4_LOCK_FILE=/tmp/ds4-cli-112-baseline.lock \
  DS4_MXFP4_NATIVE=1 \
  DS4_FLASH_MOE_ANE_PREFILL=1 \
  DS4_METAL_PREFILL_CHUNK=4096 \
  DS4_FLASH_MOE_DISABLE_AUTO_PER_SLOT_BUFFERS=1 \
  DS4_FLASH_MOE_DISABLE_DIRECT_MMAP_AUTO=1 \
  DS4_FLASH_MOE_SLOTWISE_DECODE=1 \
  DS4_FLASH_MOE_SLOT_SNAPSHOT_TOKENS=16 \
  DS4_FLASH_MOE_SLOT_SNAPSHOT_FILE=/tmp/ds4-slot-112-baseline.log \
  DS4_FLASH_MOE_SLOT_SNAPSHOT_HOT_TOKENS=128 \
  DS4_FLASH_MOE_RESIDENCY_STATS=0 \
  DS4_FLASH_MOE_RESIDENCY_STATS_START=1 \
  ./ds4-agent \
    -m /Users/anemll/Models/DSv4-Flash-MXFP4-unpacked-flash \
    --ctx 32768 \
    -n 20000 \
    --temp 0 \
    --nothink \
    --moe-slot-bank 112
```

If 112 baseline is fast once hot and no-grow 96 is clean, the fix should focus
on compression around grow/recovery mechanics, not lowering the steady target.

## Compression-specific code directions

1. Make grow-carry the default for mixed expert-major banks when growing
   during decode. Cold free/reallocate grow is too dangerous under pressure.
2. Add a post-recovery grow policy that can return to a higher bank after a
   long clean window. It must not immediately oscillate.
3. Add a real keepalive experiment:
   - CPU touch is easy but may synchronize shared buffers and cost too much.
   - GPU-side touch/no-op kernel or blit may better mark pages recently used.
   - The keepalive should be token-periodic and capped so it cannot dominate.
4. Add memory-pressure gating before grow, not just compression gating after
   grow. Current grow checks compression thresholds, but the bad 112 grow had
   no compression before the grow; compression appeared after the cold bank was
   created/refilled.
5. Consider growing by fewer slots or per-layer batches so the newly cold
   allocation is smaller.

Do not spend the next pass on generic decode kernel speed, L2 policy, or
page-cache prefetch unless the compression counters are flat. The immediate bug
is avoiding or recovering from compressed Metal/IOAccelerator pages.

## Quick parser

Use this to summarize a snapshot log:

```sh
python3 - <<'PY'
import re, statistics as st, sys
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ds4-slot-snapshot.log"
pat = re.compile(
    r'slot-snapshot\s+(\S+)\s+tok=(\d+)\s+t=([0-9.]+)s\s+'
    r'tps=([0-9.]+)\s+slots=(\d+)\s+resident=(\d+)/(\d+).*?'
    r'hit=([0-9.]+)%\s+gpu-footprint=([0-9.]+)GiB\s+'
    r'gpu-compressed=([0-9.]+)GiB\s+task-compressed=([0-9.]+)GiB\s+'
    r'phys=([0-9.]+)GiB\s+decompressions=(\d+)\s+mem-pressure=(\d+)%')
rows = []
for i, line in enumerate(open(path, errors="replace"), 1):
    m = pat.search(line)
    if not m:
        continue
    phase,tok,t,tps,slots,res,tot,hit,gf,gc,tc,phys,decomp,mp = m.groups()
    rows.append(dict(line=i, phase=phase, tok=int(tok), t=float(t),
                     tps=float(tps), slots=int(slots), res=int(res),
                     tot=int(tot), hit=float(hit), gf=float(gf),
                     gc=float(gc), tc=float(tc), phys=float(phys),
                     decomp=int(decomp), mp=int(mp)))

last = None
print("transitions:")
for r in rows:
    if r["slots"] != last or r["phase"] != "decode":
        print(f'{r["line"]:4d} {r["phase"]:22s} tok={r["tok"]:5d} '
              f't={r["t"]:7.1f}s slots={r["slots"]:3d} '
              f'res={r["res"]:4d}/{r["tot"]:<4d} tps={r["tps"]:5.2f} '
              f'gc={r["gc"]:4.2f} tc={r["tc"]:4.2f} '
              f'decomp={r["decomp"]} mp={r["mp"]}%')
        last = r["slots"]

print("\nslot summaries:")
for s in sorted({r["slots"] for r in rows}):
    xs = [r for r in rows if r["phase"] == "decode" and r["slots"] == s and r["tps"] > 0]
    if not xs:
        continue
    vals = [r["tps"] for r in xs]
    full = [r for r in xs if r["res"] >= r["tot"]]
    print(f'slots {s:3d}: n={len(xs):3d} tok {xs[0]["tok"]:5d}-{xs[-1]["tok"]:5d} '
          f'avg={sum(vals)/len(vals):5.2f} med={st.median(vals):5.2f} '
          f'maxComp={max(r["gc"] + r["tc"] for r in xs):.2f}GiB '
          f'decompDelta={xs[-1]["decomp"] - xs[0]["decomp"]}')
    if full:
        fvals = [r["tps"] for r in full]
        print(f'  full: n={len(full):3d} avg={sum(fvals)/len(fvals):5.2f} '
              f'med={st.median(fvals):5.2f}')
PY /tmp/ds4-slot-grow-96-112-carry.log
```
