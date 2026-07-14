# DS4 Live Monitor

This is the macOS port of the DSpark live dashboard. It reads the native
`/metrics` endpoint in `ds4-server`; it does not parse prompts, trace files, or
generated text.

## Start it

Rebuild the server first, then restart your existing DS4 server process so it
uses the new binary:

```bash
cd ~/SourceRelease/GITHUB/ML_playground/ds4-ssd
make ds4-server

# Start ds4-server with your normal model and runtime arguments.
./dashboard/run-dashboard.sh
```

Open `http://127.0.0.1:11002`.

The dashboard assumes DS4 is listening at `http://127.0.0.1:8000`. It binds to
loopback by default. Override the metrics endpoint or dashboard port when
needed:

```bash
DS4_METRICS_URL=http://127.0.0.1:8100/metrics \
DASHBOARD_PORT=11003 \
./dashboard/run-dashboard.sh
```

For a dashboard opened from another machine, set `DASHBOARD_BIND` deliberately
and set `DS4_AGENT_API_BASE_URL` to the address that Pi and Droid can actually
reach:

```bash
DASHBOARD_BIND=0.0.0.0 \
DS4_AGENT_API_BASE_URL=http://m5m.local:8000/v1 \
./dashboard/run-dashboard.sh
```

## What it measures

- Green/right graph: decode output tokens per second.
- Red/left graph: non-cached prompt tokens evaluated per second.
- Dashed averages exclude zero-rate samples, so idle time does not pull the
  active throughput average down.
- Queue, TTFT, inter-token latency, end-to-end latency, completion totals, and
  error totals come directly from `ds4-server`.
- DSpark mirrors the `--debug-status` draft summary: proposed and accepted
  draft tokens, acceptance, tau, verified blocks, pre-draft skips, post-draft
  verify skips, and dynamic verification budget when active. These counters
  work without enabling `DS4_DSPARK_PERF`.
- The runtime panel shows the last worker-side memory snapshot: DS4 process
  footprint/resident memory, unified GPU footprint, GPU/task/system compression,
  system memory pressure, swap, and Flash-MoE slots.

DS4 has one inference worker, so the dashboard reports that scheduler directly;
it does not aggregate DGX tensor-parallel workers. On Apple Silicon, GPU and
CPU share unified memory, so the panel labels GPU footprint separately from the
process and system-memory views rather than treating them as independent RAM.

## Server start, stop, and restart

The lifecycle controls only manage a child `ds4-server` process launched by the
dashboard. They intentionally never discover or kill an independently started
server on the metrics port.

Configure its fixed startup argv and a control token before starting the
dashboard. `DASHBOARD_SERVER_COMMAND` must be a JSON string array whose first
entry is an executable absolute path. It is executed without a shell.

```bash
export DASHBOARD_CONTROL_TOKEN='replace-with-a-long-random-secret'
export DASHBOARD_SERVER_CWD="$HOME/SourceRelease/GITHUB/ML_playground/ds4-ssd"
export DASHBOARD_SERVER_COMMAND='["/Users/you/SourceRelease/GITHUB/ML_playground/ds4-ssd/ds4-server","-m","/absolute/path/to/model.gguf","--port","8000"]'
export DASHBOARD_SERVER_AUTOSTART=1  # optional; otherwise press Start

./dashboard/run-dashboard.sh
```

The dashboard writes the owned child’s stdout/stderr to
`/tmp/ds4-dashboard-server.log` by default; override it with
`DASHBOARD_SERVER_LOG`. Set `DASHBOARD_SERVER_STOP_GRACE_SECONDS` to change the
grace period before a forced stop.

Controls require the token in the browser for every action and are not exposed
through CORS. Keep the dashboard on loopback, or use an SSH/VPN/TLS-protected
route when you need remote access; a plain LAN bind is not appropriate for
server control.

## Pi and Droid setup

The dashboard has copy buttons for mergeable Pi and Droid model entries. They
use the API base derived from `DS4_METRICS_URL` by default, or the explicit
`DS4_AGENT_API_BASE_URL` value when set. Merge the Pi entry into
`~/.pi/agent/models.json` and the Droid entry into
`~/.factory/settings.json`; do not replace unrelated providers or models.

## Apple Silicon power estimates

The dashboard detects the local Apple GPU without privilege. Estimated CPU/GPU/
ANE rail power comes from `powermetrics`, which needs elevated privileges, so it
is disabled by default and never prompts for a password. If you deliberately
configure a non-interactive `sudo` rule for `/usr/bin/powermetrics`, opt in with:

```bash
DS4_DASHBOARD_POWERMETRICS=1 ./dashboard/run-dashboard.sh
```

Those are OS estimates, not wall-power measurements.
