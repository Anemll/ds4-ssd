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

That command is resident/full-GGUF mode. To benchmark SSD streaming, pass the
dense sidecar GGUF and `--moe-sidecar`:

```sh
export DS4_SIDECAR_DIR=/path/to/dsv4-iq2xxs-expert-major

DS4_METAL_PREFILL_CHUNK=16384 ./ds4-bench \
  -m "$DS4_SIDECAR_DIR/dense/model-dense.gguf" \
  --moe-sidecar "$DS4_SIDECAR_DIR" \
  --moe-slot-bank 64 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 32768 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/ds4-sidecar-speed.csv
```

For sidecar runs, prefer comparing the same prompt, slot-bank count, context
window, power state, and `DS4_METAL_PREFILL_CHUNK=16384` setting. Confirm the
log says `Flash-MoE sidecar loaded`; if it only says `applied tuning profile`,
you are benchmarking resident mode. Leave `DS4_METAL_GRAPH_RAW_CAP` unset unless
you are explicitly debugging raw-KV allocation.

`ds4_profile.json` contains measured Apple Silicon defaults and comments for
M5, M5 Max, M3 Ultra, and M1 Max classes. Treat it as the current tuning source
of truth; see [PROFILES.md](PROFILES.md). Broader CI and performance regression
automation are planned for a post-alpha stage.
