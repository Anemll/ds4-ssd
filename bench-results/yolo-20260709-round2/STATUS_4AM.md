# 4AM DSpark YOLO status (2026-07-09 ~03:30 PDT)

## Champion env (opt-in)

```bash
export DS4_DSPARK_FORCE_TARGET_FIRST=1
export DS4_DSPARK_CONF_SCALE=0.85
export DS4_DSPARK_CONF_THRESHOLD=0.50
```

## Headline results (arcade HTML, resident, -c 20000)

| n | control | force | **champion** | nodraft | champ−nodraft |
|--:|--------:|------:|-------------:|--------:|--------------:|
| 2000 | 38.35 | 39.33 | **39.93–40.03** | — | — |
| 4000 | 36.69 | 37.46 | **38.04–38.14** (3 reps) | 37.71–37.73 | **+0.35–0.4** |
| 5000 | 35.63 | 36.82 | **37.21** | 36.22 | **+0.99** |

Strict DSpark **beats no-draft** at long context. First-miss **0%**. Full-accept **~81%**.

## What shipped (default-OFF)

- Force-target-first + row0 Markov skip + conf scale/bias/threshold/calib-apply
- Adaptive draft_cap (neutral)
- Prefetch force live-redraft; conf sched≥1 under force
- Margin/rate gates earlier (do not stack with force)

## Rejected

lazy main-KV always, force+margin, margin+rate, MARKOV_SCALE≠1, verify-4, scale≤0.80 long-ctx

## Still for +10 t/s

ANE draft / retrain. Host scheduling largely harvested for this prompt.

## Logs

`bench-results/yolo-20260709-round2/` (OVERNIGHT.md, CHAMPION.md, this file)
