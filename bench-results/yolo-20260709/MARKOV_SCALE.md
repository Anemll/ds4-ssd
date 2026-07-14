# DS4_DSPARK_MARKOV_SCALE A/B (FORCE_TARGET_FIRST)

Date: 2026-07-09  
Prompt: arcade HTML (`aders arcade game...` 1978 Taito fidelity)  
Base: `./ds4 -m .../dsv4-iq2xxs-expert-major --draft dspark --draft-path .../DSv4-Flash-DSpark-draft --draft-verify 5 --nothink --temp 0 --resident -c 20000 -n 800`  
Env (all runs): `DS4_DSPARK_PERF=1 DS4_DSPARK_FORCE_TARGET_FIRST=1 DS4_DSPARK_MARKOV_SCALE=<v>`

Logs: `bench-results/yolo-20260709/markov-scale-*.{out,err}`

## Results

| scale | t/s   | tau  | first-miss | full-accept | avg scheduled | blocks |
|------:|------:|-----:|-----------:|--------------:|--------------:|-------:|
| 0.75  | 39.50 | 3.47 | 0.0%       | 61.5%         | 4.45          | 179    |
| **1.0** | **40.45** | **3.57** | **0.0%** | **64.6%** | **4.46** | **175** |
| 1.25  | 38.82 | 3.44 | 0.0%       | 59.4%         | 4.52          | 180    |
| 1.5   | 38.46 | 3.35 | 0.0%       | 59.2%         | 4.46          | 184    |

Notes:
- `first-miss` is 0% for all rows by construction of `FORCE_TARGET_FIRST` (drafts[0] = target argmax).
- `avg scheduled` stays ~4.45–4.52 (confidence scheduler + verify=5); scale mainly moves accept quality, not draft length.

## Best vs 1.0 (control)

**Best scale = 1.0 (default).**

| metric        | best (1.0) | vs 0.75     | vs 1.25     | vs 1.5      |
|---------------|-----------:|------------:|------------:|------------:|
| gen t/s       | 40.45      | +0.95       | +1.63       | +1.99       |
| tau           | 3.57       | +0.10       | +0.13       | +0.22       |
| full-accept   | 64.6%      | +3.1 pp     | +5.2 pp     | +5.4 pp     |
| avg scheduled | 4.46       | ~flat       | −0.06       | ~flat       |

### Deltas vs 1.0

| scale | Δ t/s  | Δ tau  | Δ full-accept |
|------:|-------:|-------:|--------------:|
| 0.75  | −0.95  | −0.10  | −3.1 pp       |
| 1.25  | −1.63  | −0.13  | −5.2 pp       |
| 1.5   | −1.99  | −0.22  | −5.4 pp       |

## Interpretation

`DS4_DSPARK_MARKOV_SCALE` rescales the Markov head contribution (`logits = base + F·markov`). Under force-target-first:

- **F &lt; 1 (0.75):** slightly weaker Markov → lower tau / full-accept → fewer blocks fully accepted → more blocks / lower t/s.
- **F = 1.0:** peak tau (3.57), full-accept (64.6%), and gen speed (40.45 t/s).
- **F &gt; 1 (1.25, 1.5):** over-weighting Markov hurts draft quality after the forced first token; tau and full-accept fall monotonically; t/s follows.

## Recommendation

Keep **`DS4_DSPARK_MARKOV_SCALE=1.0`** (or unset). Do not promote 0.75 / 1.25 / 1.5 for this force-first + arcade-HTML workload.

Stack remains: `DS4_DSPARK_FORCE_TARGET_FIRST=1` alone; no markov-scale change.
