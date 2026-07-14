# Overnight YOLO DSpark campaign — final as of ~03:05 PDT 2026-07-09

## Champion (opt-in, default-OFF)

```bash
export DS4_DSPARK_FORCE_TARGET_FIRST=1
export DS4_DSPARK_CONF_SCALE=0.85
export DS4_DSPARK_CONF_THRESHOLD=0.50
```

### n=4000 noise-confirmed (arcade HTML, resident IQ2XXS, -c 20000)

| run | gen t/s |
|-----|--------:|
| champion rep0 | 38.04 |
| champion rep1 | 38.08 |
| champion rep2 | 38.14 |
| **champion mean** | **~38.09** |
| nodraft | 37.71 |
| nodraft rep2 | 37.73 |
| **nodraft mean** | **~37.72** |
| force only | 37.46 |
| control (earlier) | 36.69 |

**Champion beats nodraft by ~+0.37 t/s at n=4000** (stable across 3 reps).
**Champion beats force-only by ~+0.6 t/s** and control by ~+1.4 t/s.

### n=2000

| config | t/s |
|--------|----:|
| control | 38.35 |
| force | 39.33 |
| champion (s0.85/t0.5) | **39.93–40.03** |
| s0.83 | 40.12 (short-ctx only; n4000 38.02 ≈ champion) |
| verify-4 | 39.36 (worse) |

## Code landed tonight (all default-OFF)

1. **FORCE_TARGET_FIRST** — teacher-force drafts[0], Markov rechain
2. **Skip row0 Markov matmul** under force (unless KEEP_ROW0_MM)
3. **Conf schedule remap** — CONF_SCALE / CONF_BIAS / CONF_CALIB_APPLY
4. **CONF_THRESHOLD** env override
5. **Adaptive draft_cap** (neutral in tests)
6. Force conf polish (sched≥1) + prefetch live-redraft
7. Margin gate / margin+rate (earlier; don't stack with force)
8. Lazy main-KV always-on **REJECTED** (draft 62ms)

## Rejected / do not stack

- force + margin gate
- margin + rate gate  
- MARKOV_SCALE ≠ 1
- LAZY_MAIN_KV always rebuild
- CONF_SCALE ≤ 0.80 at long ctx (over-trims)
- draft-verify 4 with champion

## Why champion works

Force-first kills first-miss (22%→0) and lifts tau, but conf over-schedules (~4.5).  
Scale 0.85 + thresh 0.50 trims avg scheduled to ~3.6–3.9, cuts verify ~10ms/block, full-accept ~81–82%, net win even at long ctx.

## Command

```bash
PROMPT='aders arcade game in a single self-contained HTML file with inline CSS and JS. Replicate the original 1978 Taito arcade cabinet experience with full fidelity to the original rules, sprite designs, and mechanics.'

env DS4_DSPARK_PERF=1 \
  DS4_DSPARK_FORCE_TARGET_FIRST=1 \
  DS4_DSPARK_CONF_SCALE=0.85 \
  DS4_DSPARK_CONF_THRESHOLD=0.50 \
  ./ds4 -m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major \
  --draft dspark --draft-path /Users/anemll/Models/DSv4-Flash-DSpark-draft \
  --draft-verify 5 --nothink --temp 0 --resident -c 20000 -n 4000 \
  -p "$PROMPT"
```

## Logs
- `bench-results/yolo-20260709/` — first YOLO matrix + markov scale
- `bench-results/yolo-20260709-round2/` — conf scale sweeps + n4000 champion
- `bench-results/force-first-20260709/` — force discovery
- `bench-results/margin-rate-gate-20260709/` — combined gate reject

## Residual path to +10 t/s
Still needs ANE draft and/or draft retrain. Host scheduling gains are largely harvested for this workload.

## n=5000 long-horizon

| config | t/s | delta vs nodraft |
|--------|----:|-----------------:|
| **champion** | **37.21** | **+0.99** |
| nodraft | 36.22 | 0 |

Champion holds and **widens** the lead at n=5000 (first-miss 0%, full-accept 80.6%, avg sched 3.92).

## Noise summary n=4000 champion

38.04 / 38.08 / 38.14 vs nodraft 37.71 / 37.73 → **stable +0.35–0.4 t/s**.
