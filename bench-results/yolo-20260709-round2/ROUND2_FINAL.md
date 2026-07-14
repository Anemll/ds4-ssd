# YOLO Round 2 — force-first + conf scheduling (2026-07-09 ~02:20)

## Code shipped (default-OFF)

| knob | effect |
|------|--------|
| `DS4_DSPARK_FORCE_TARGET_FIRST=1` | drafts[0]=target argmax + Markov rechain; first-miss→0 |
| force row0 Markov skip | under force-first, skip row0 gather/matmul (conf[0] forced 1.0) |
| force conf polish | scheduled≥1 when force-first |
| force prefetch polish | mismatch → live redraft |
| `DS4_DSPARK_CONF_SCALE` / `CONF_BIAS` | scale conf probs before hard/cost schedule |
| `DS4_DSPARK_CONF_CALIB_APPLY=1` | map via online reliability bins |
| `DS4_DSPARK_CONF_THRESHOLD` | env override of conf threshold |
| `DS4_DSPARK_ADAPTIVE_DRAFT_CAP=1` | draft_cap from committed EMA (neutral in tests) |

## n=1200 (arcade HTML, resident)

| config | t/s | tau | avg sched | notes |
|--------|----:|----:|----------:|-------|
| force | 40.43 | 3.55 | 4.34 | baseline force |
| force+calib apply | 40.41 | 3.55 | 4.34 | neutral (ECE 0.07) |
| **force+scale0.85** | **40.65** | 3.41 | 4.06 | slight win |
| force+adaptive | 40.49 | 3.56 | 4.34 | neutral |
| nodraft | 38.46 | - | - | |

## n=2000

| config | t/s | tau | avg sched | verify ms | full-acc |
|--------|----:|----:|----------:|----------:|---------:|
| control | 38.35 | 2.83 | 4.45 | 60.9 | 52% |
| force | 39.33 | 3.58 | 4.41 | 77.4 | 67% |
| force+s0.85 | 39.69 | 3.47 | 4.14 | 73.6 | 73% |
| force+s0.75 | 39.44 | 3.31 | 3.91 | 70.3 | 76% |
| force+t0.55 | 39.59 | 3.27 | 3.82 | 69.0 | 78% |
| **force+s0.85+t0.5** | **39.93** | **3.21** | **3.64** | **66.6** | **82%** |

## n=4000 (critical long-ctx)

| config | t/s | vs nodraft | tau | avg sched | verify ms |
|--------|----:|-----------:|----:|----------:|----------:|
| force alone | 37.46 | −0.25 | 3.77 | 4.50 | 87.7 |
| nodraft | 37.71 | 0 | - | - | - |
| **force+s0.85+t0.5** | **38.04** | **+0.33** | **3.47** | **3.91** | **78.3** |

**First time DSpark clear-wins nodraft on n=4000 arcade HTML in this campaign.**

## Best recommended env stack

```bash
export DS4_DSPARK_FORCE_TARGET_FIRST=1
export DS4_DSPARK_CONF_SCALE=0.85
export DS4_DSPARK_CONF_THRESHOLD=0.50   # same as default 0.4 CLI? wait — CLI default is 0.4
```

Note: CLI default conf is 0.4; THRESHOLD=0.5 is stricter than default. Combined with SCALE=0.85 it trims over-long verify under force-first.

Do NOT stack: margin gate, lazy main-KV always, MARKOV_SCALE≠1, adaptive draft alone.

## Why it works

Force-first removes first-miss waste and raises tau, but then conf scheduler over-schedules (avg~4.5) and pays heavy verify at long ctx. Scaling conf + slightly higher threshold shortens scheduled prefix (3.6–3.9), cuts verify ~10ms/block, full-accept rises to ~81–82%, net t/s up.

## n=2500 fine sweep (all FORCE_TARGET_FIRST)

| scale / thresh | t/s | avg sched | full-acc |
|----------------|----:|----------:|---------:|
| 0.85 / 0.50 | 38.95 | 3.69 | 80.6% |
| **0.80 / 0.50** | **39.25** | 3.52 | 85.0% |
| 0.85 / 0.55 | 39.20 | 3.48 | 85.1% |
| 0.90 / 0.50 | 38.78 | 3.83 | 78.6% |

At n=2500 s0.80 looks best; at n=4000 it **regressed** to 37.67 (under-schedules). **Champion remains s0.85/t0.50**.

## n=4000 full compare

| config | t/s |
|--------|----:|
| force only | 37.46 |
| s0.80/t0.50 | 37.67 |
| s0.85/t0.55 | 37.70 |
| nodraft | 37.71 |
| **s0.85/t0.50 (champion)** | **38.04** |

