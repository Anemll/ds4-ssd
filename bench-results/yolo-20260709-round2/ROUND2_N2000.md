# YOLO Round2 n=2000 — force-first vs force+CONF_CALIB_APPLY vs control

Date: 2026-07-09  
Prompt: `aders arcade game in a single self-contained HTML file with inline CSS and JS. Replicate the original 1978 Taito arcade cabinet experience with full fidelity to the original rules, sprite designs, and mechanics.`

Model / common flags:
```
-m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major \
  --draft dspark --draft-path /Users/anemll/Models/DSv4-Flash-DSpark-draft \
  --draft-verify 5 --nothink --temp 0 --resident -c 20000 -n 2000
```

Env base: `DS4_DSPARK_PERF=1`  
Logs: `bench-results/yolo-20260709-round2/n2000-*.{out,err}`

## Results

| config | env extras | gen t/s | decode s | tau | first-miss | full-accept | accept % | avg sched | blocks | skip-pre | skip-verify | block ms | draft ms | verify ms | logs |
|--------|------------|--------:|---------:|----:|-----------:|------------:|---------:|----------:|-------:|---------:|------------:|---------:|---------:|----------:|------|
| control | (none) | **38.35** | 52.145 | 2.83 | 22.4% | 52.3% | 63.6% | 4.45 | 522 | 0 | 23 | 73.03 | 10.63 | 60.89 | `n2000-control.{out,err}` |
| force | `FORCE_TARGET_FIRST=1` | **39.30** | 50.887 | 3.58 | 0.0% | 66.5% | 81.4% | 4.41 | 436 | 0 | 0 | 89.80 | 10.39 | 77.48 | `n2000-force.{out,err}` |
| force+calib | `FORCE_TARGET_FIRST=1` + `CONF_CALIB=1` + `CONF_CALIB_APPLY=1` | **39.29** | 50.898 | 3.58 | 0.0% | 66.5% | 81.4% | 4.41 | 436 | 0 | 0 | 89.80 | 10.40 | 77.46 | `n2000-force-calib.{out,err}` |

Full env names:
- `DS4_DSPARK_FORCE_TARGET_FIRST=1`
- `DS4_DSPARK_CONF_CALIB=1`
- `DS4_DSPARK_CONF_CALIB_APPLY=1`

## Ranking (gen t/s)

1. **force**: 39.30 t/s (tau=3.58, first-miss=0.0%)
2. **force+calib**: 39.29 t/s (tau=3.58, first-miss=0.0%) — within noise of force
3. **control**: 38.35 t/s (tau=2.83, first-miss=22.4%)

Δ force vs control: **+0.95 t/s** (+2.5%)  
Δ force+calib vs force: **−0.01 t/s** (neutral)

## Acceptance structure

| config | pos1 | pos2\|1 | pos3\|1-2 | pos4\|1-3 | pos5\|1-4 |
|--------|-----:|--------:|----------:|----------:|----------:|
| control | 77.6% | 87.7% | 88.3% | 91.3% | 89.8% |
| force | 100.0% | 88.3% | 89.3% | 88.0% | 88.3% |
| force+calib | 100.0% | 88.3% | 89.3% | 88.0% | 88.3% |

Force zeros first-miss (22.4% → 0%) and lifts tau (2.83 → 3.58). Block wall rises (73.0 → 89.8 ms) because more draft tokens are verified (no first-miss early-out / skip-verify). Net still wins on gen t/s via fewer blocks (522 → 436) and higher tokens/block.

## CONF_CALIB_APPLY detail (force+calib only)

```
dspark conf-calib apply enabled (lookup bins after 32 samples/bin; scale=1.000 bias=0.000)
conf-calib reliability (bin pred-mean realized n):
  [0.4-0.5) pred=0.454 realized=0.464 n=84
  [0.5-0.6) pred=0.552 realized=0.558 n=129
  [0.6-0.7) pred=0.650 realized=0.531 n=128
  [0.7-0.8) pred=0.752 realized=0.673 n=162
  [0.8-0.9) pred=0.852 realized=0.709 n=227
  [0.9-1.0) pred=0.984 realized=0.935 n=1191
conf-calib ECE=0.0625 samples=1921
```

Apply is live but with scale=1.000 bias=0.000 and online bin lookup after 32 samples/bin; at n=2000 under force-first it did **not** change scheduling vs force alone (identical tau, blocks, accept, first-miss; gen t/s within 0.01).

## Notes / recommendation

- **force-first alone is the win** at n=2000: +0.95 t/s vs control, first-miss→0, tau 2.83→3.58.
- **CONF_CALIB_APPLY on top of force is neutral** here (39.29 vs 39.30). No harm, no gain at this horizon with default scale/bias.
- Matches prior yolo-20260709 n=2000 force result (39.23 t/s) within run-to-run noise.
- Prefer `DS4_DSPARK_FORCE_TARGET_FIRST=1` as the opt-in; leave `CONF_CALIB_APPLY` for diagnostics / longer-horizon experiments unless a non-default scale/bias is tuned.
