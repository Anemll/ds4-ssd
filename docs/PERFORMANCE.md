# Performance

This alpha does not claim a complete benchmark matrix. The release priority is:

1. sidecar mode builds and runs;
2. correctness gates still pass;
3. the 16K sidecar smoke proves the headline path is not only a manual script.

Use `ds4-bench` for local throughput sweeps:

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 32768 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/ds4-speed.csv
```

For sidecar runs, prefer comparing the same prompt, slot-bank count, context
window, power state, and `DS4_METAL_PREFILL_CHUNK=16384` setting. Leave
`DS4_METAL_GRAPH_RAW_CAP` unset unless you are explicitly debugging raw-KV
allocation.

`ds4_profile.json` contains measured Apple Silicon defaults and comments for
M5, M5 Max, M3 Ultra, and M1 Max classes. Treat it as the current tuning source
of truth. Broader CI and performance regression automation are planned for a
post-alpha stage.
