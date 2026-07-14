# DSpark Generated-Output Quality And Speed Report Card

Run root:
`bench-results/dspark_quality_20260630_090516`

This report compares the saved generated outputs for the same 100 prompt set.
`generated_first_match_vs_official` and `generated_avg_lcp_vs_official` compare
the generated tokens to the official continuation tokens. `generated_prefix_nll`
is the model NLL of each saved generated prefix after the prompt, not the
teacher-forced official-continuation NLL from `score_official`.

| mode | cases | official tokens | scored generated tokens | generated_prefix_nll | generated_first_match_vs_official | generated_avg_lcp_vs_official | generated tokens | generation t/s | speedup vs no draft | DSpark acceptance | byte equal vs no draft |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no draft | 100 | 2290 | 2281 | 0.170559803 | 64 | 7.000 | 11754 | 31.769544 | 1.000000x | n/a | baseline |
| strict DSpark | 100 | 2290 | 2281 | 0.170559803 | 64 | 7.000 | 11754 | 28.279020 | 0.890130x | 42.966600% | 100/100 |
| fast relaxed DSpark | 100 | 2290 | 2289 | 0.444579396 | 64 | 5.500 | 11825 | 32.942353 | 1.036916x | 67.128933% | 2/100 |

Strict DSpark copies the no-draft quality row because the saved strict outputs
are byte-equal to no-draft for all 100 cases. It still has its own speed and
acceptance row.

## First-Match Audit

The `64` count was checked two ways against the current sidecar no-draft model:

- Saved generated-output comparison:
  `bench-results/dspark_quality_20260630_090516/no_draft/generated_vs_official.tsv`
  has `sum(first_match)=64`.
- Independent live greedy scorer in first-token-only mode:
  `bench-results/dspark_quality_20260630_090516/official_first_token_current.tsv`
  reports `first_match=64` and has the same no-draft mismatch case IDs.

Strict DSpark is also `64` because it is byte-equal to no-draft for all 100
saved cases. Fast relaxed DSpark also lands at `64`, but with a different
mismatch set and only `2/100` byte-equal cases versus no-draft.

The fast-relaxed `64` is not the same 64 cases. It fixes five no-draft
first-token misses (`case_029`, `case_039`, `case_044`, `case_066`,
`case_072`) and introduces five new first-token misses (`case_040`,
`case_042`, `case_049`, `case_062`, `case_086`). The equal total is a net
tie, not evidence that the fast run preserves no-draft first-token behavior.

## Raw Artifacts

- Overall speed: `bench-results/dspark_quality_20260630_090516/quality_overall.tsv`
- No-draft per-case quality: `bench-results/dspark_quality_20260630_090516/no_draft/generated_vs_official.tsv`
- Fast relaxed per-case quality: `bench-results/dspark_quality_20260630_090516/fast_relaxed/generated_vs_official.tsv`
- No-draft official first-token cross-check: `bench-results/dspark_quality_20260630_090516/official_first_token_current.tsv`
- Machine-readable report: `gguf-tools/quality-testing/dspark_generated_quality_report.tsv`

## Historical Official Model-Variant Scores

Source: `gguf-tools/quality-testing/HYBRID_QUALITY_REPRO.md`.

| mode | cases | target tokens | avg_nll | first_match | avg_lcp |
|---|---:|---:|---:|---:|---:|
| HYBRID resident `--no-int8` | 100 | 2290 | 0.379792583 | 70 | 7.570 |
| HYBRID normal | 100 | 2290 | 0.382228881 | 72 | 7.540 |
| IQ2 | 100 | 2290 | 0.412994818 | 66 | 6.470 |
