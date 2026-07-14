# YOLO DSpark Bench Matrix — 2026-07-09

Prompt: `aders arcade game in a single self-contained HTML file with inline CSS and JS. Replicate the original 1978 Taito arcade cabinet experience with full fidelity to the original rules, sprite designs, and mechanics.`

Model: `-m /Users/anemll/Models/flash/dsv4-iq2xxs-expert-major`

Common flags: `--draft dspark --draft-path /Users/anemll/Models/DSv4-Flash-DSpark-draft --draft-verify 5 --nothink --temp 0 --resident -c 20000` (omitted for nodraft).

Env base: `DS4_DSPARK_PERF=1` (ANE not used).

Logs: `bench-results/yolo-20260709/yolo-n{N}-{tag}.{out,err}`

## Results

| n | config | env extras | gen t/s | decode s | tau | first-miss | avg sched | skip-pre | skip-verify | block ms | draft ms | verify ms | logs |
|--:|--------|------------|--------:|---------:|----:|-----------:|----------:|---------:|------------:|---------:|---------:|----------:|------|
| 2000 | control | (none) | 38.44 | 52.030 | 2.83 | 22.4% | 4.45 | 0 | 23 | 72.88 | 10.59 | 60.77 | `yolo-n2000-ctrl.{out,err}` |
| 2000 | force | FORCE_TARGET_FIRST=1 | 39.23 | 50.980 | 3.58 | 0.0% | 4.41 | 0 | 0 | 89.96 | 10.52 | 77.50 | `yolo-n2000-force.{out,err}` |
| 2000 | margin45 | EVAL_MARGIN_GATE_THRESHOLD=4.5 | 38.81 | 51.538 | 3.25 | 12.4% | 4.52 | 180 | 4 | 82.02 | 10.60 | 69.69 | `yolo-n2000-margin45.{out,err}` |
| 2000 | force+margin | FORCE + MARGIN 4.5 | 38.54 | 51.901 | 3.60 | 0.0% | 4.51 | 170 | 0 | 91.70 | 10.54 | 79.21 | `yolo-n2000-force-margin.{out,err}` |
| 2000 | nodraft | (none) | 38.12 | 52.466 | - | - | - | - | - | - | - | - | `yolo-n2000-nodraft.{out,err}` |
| 4000 | nodraft | (none) | 37.65 | 106.252 | - | - | - | - | - | - | - | - | `yolo-n4000-nodraft.{out,err}` |
| 4000 | force | FORCE_TARGET_FIRST=1 | 37.35 | 107.106 | 3.77 | 0.0% | 4.50 | 0 | 0 | 100.46 | 10.53 | 87.98 | `yolo-n4000-force.{out,err}` |
| 4000 | control | (none) | 36.69 | 109.028 | 3.02 | 20.4% | 4.45 | 0 | 33 | 82.28 | 10.58 | 70.11 | `yolo-n4000-ctrl.{out,err}` |

## n=2000 ranking (by gen t/s)

1. **force**: 39.23 t/s (tau=3.58, first-miss=0.0%, skip-pre=0)
2. **margin45**: 38.81 t/s (tau=3.25, first-miss=12.4%, skip-pre=180)
3. **force+margin**: 38.54 t/s (tau=3.60, first-miss=0.0%, skip-pre=170)
4. **control**: 38.44 t/s (tau=2.83, first-miss=22.4%, skip-pre=0)
5. **nodraft**: 38.12 t/s (tau=-, first-miss=-, skip-pre=-)

## n=4000 ranking (force led n=2000 → confirm nodraft / force / control)

1. **nodraft**: 37.65 t/s (tau=-, first-miss=-, skip-pre=-)
2. **force**: 37.35 t/s (tau=3.77, first-miss=0.0%, skip-pre=0)
3. **control**: 36.69 t/s (tau=3.02, first-miss=20.4%, skip-pre=0)

## Notes

- At n=2000, **force** leads (39.23 t/s) over margin45 (38.81), force+margin (38.54), control (38.44), nodraft (38.12).
- Force zeros first-miss (22.4% → 0%) and lifts tau (2.83 → 3.58) but block time rises (72.9 → 90.0 ms) because more drafts are verified (no first-miss early-outs / skip-verify).
- Stacking force+margin is **worse** than force alone at n=2000 (38.54 vs 39.23): margin skip-pre=170 over-skips after force already eliminated first-miss.
- At n=4000 confirm: force (37.35) still beats control (36.69, +0.66 t/s) with first-miss 0% and tau 3.77 vs 3.02; **nodraft** is slightly highest (37.65), so draft win is small / context-sensitive at long horizon.
- Control shows skip-verify=23 (n=2000) / 33 (n=4000) first-miss early exits that force cannot take (by design first-miss=0).

## Recommendation

**Best opt-in for DSpark (byte-exact / strict verify): `DS4_DSPARK_FORCE_TARGET_FIRST=1` alone.**

- Do **not** default-stack with `DS4_DSPARK_EVAL_MARGIN_GATE_THRESHOLD=4.5` (force alone wins).
- Margin 4.5 alone remains a modest safe gate vs control when force is unavailable.
- Long-horizon (n=4000) force still beats control but gap to plain decode is small; prioritize force for acceptance structure (tau↑, first-miss→0), not large absolute t/s.

