# YOLO campaign results — 2026-07-09

Parallel agents: long bench matrix, MARKOV_SCALE A/B, draft-lever explore,
worktree adaptive-cap implement, 4AM handoff synthesize.

## Shipped tonight (all default-OFF)

| env | purpose | status |
|-----|---------|--------|
| `DS4_DSPARK_FORCE_TARGET_FIRST=1` | force drafts[0]=target argmax + Markov rechain | **WIN** |
| `DS4_DSPARK_ADAPTIVE_DRAFT_CAP=1` | draft_cap from committed EMA after 16 blocks | shipped, unproven vs force alone |
| force-first conf polish | scheduled≥1 when force-first active | shipped |
| force-first prefetch polish | mismatch → live redraft not host patch | shipped |
| `DS4_DSPARK_LAZY_MAIN_KV` | was always-skip; **REJECTED** as full lazy (draft 10→62ms); now alias of rate-dormant main-KV skip only | reject full form |
| `DS4_DSPARK_EVAL_MARGIN_GATE_THRESHOLD=4.5` | pre-draft margin skip | modest alone; don't stack with force |
| `DS4_DSPARK_EVAL_MARGIN_RATE_GATE=1` | combined margin+rate | **reject** (prior session) |
| `DS4_DSPARK_MARKOV_SCALE` | draft markov mix | **leave 1.0** |

## Headline benches

### n=1000 (earlier force-first session)
| config | t/s | tau | first-miss |
|--------|----:|----:|-----------:|
| control | 39.76 | 2.91 | 22.3% |
| force | **41.01** | **3.67** | **0%** |
| margin 4.5 | 40.02 | 3.25 | 13.6% |

### n=2000 matrix (`BENCH_MATRIX.md`)
| config | t/s | tau | first-miss |
|--------|----:|----:|-----------:|
| **force** | **39.23** | **3.58** | **0%** |
| margin 4.5 | 38.81 | 3.25 | 12.4% |
| force+margin | 38.54 | 3.60 | 0% |
| control | 38.44 | 2.83 | 22.4% |
| nodraft | 38.12 | - | - |

### n=4000 confirm
| config | t/s | tau | first-miss |
|--------|----:|----:|-----------:|
| nodraft | **37.65** | - | - |
| force | 37.35 | 3.77 | 0% |
| control | 36.69 | 3.02 | 20.4% |

Force still beats control (+0.66) at n=4000; plain decode slightly ahead (verify slope).

### MARKOV_SCALE + force (n=800)
Best = **1.0** (40.45 t/s). 0.75/1.25/1.5 all worse.

### force+adaptive+lazy stack (n=1000) — REJECT stack
| config | t/s | draft ms | notes |
|--------|----:|---------:|-------|
| force only | **40.98** | 10.52 | good |
| force+adaptive+lazy | **28.30** | **62.14** | lazy rebuild kills draft |

## Recommendation

```bash
export DS4_DSPARK_FORCE_TARGET_FIRST=1
# optional alone: DS4_DSPARK_EVAL_MARGIN_GATE_THRESHOLD=4.5
# do NOT stack force+margin, force+lazy-always, or MARKOV_SCALE≠1
```

Structural +10 still needs ANE draft or retrain. Force-first is the best host-side win tonight.

## Artifacts
- `bench-results/yolo-20260709/BENCH_MATRIX.md`
- `bench-results/yolo-20260709/MARKOV_SCALE.md`
- `bench-results/force-first-20260709/SUMMARY.txt`
- `bench-results/margin-rate-gate-20260709/SUMMARY.txt`
