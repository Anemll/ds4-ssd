#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

make ds4-agent
./ds4-agent --self-test-tools

if [[ "${DS4_AGENT_LIVE:-0}" != "1" ]]; then
  exit 0
fi

: "${DS4_AGENT_MODEL:=/Users/anemll/Models/GLM-5.2-IQ1-Dense/model-dense.gguf}"
: "${DS4_AGENT_SIDECAR:=/Users/anemll/Models/GLM-5.2-sidecar}"
: "${DS4_AGENT_TRACE:=/tmp/ds4-agent-tool-live.trace}"
: "${DS4_AGENT_TOKENS:=2048}"
: "${DS4_LOCK_FILE:=/tmp/ds4-agent-tool-live.lock}"

rm -f "$DS4_AGENT_TRACE"
DS4_AGENT_TRACE_TOKENS=0 DS4_LOCK_FILE="$DS4_LOCK_FILE" ./ds4-agent \
  -m "$DS4_AGENT_MODEL" \
  --moe-mode slot-bank \
  --moe-sidecar "$DS4_AGENT_SIDECAR" \
  --moe-slot-bank 32 \
  --non-interactive \
  --debug-status \
  --nothink \
  --temp 0 \
  --tokens "$DS4_AGENT_TOKENS" \
  --trace "$DS4_AGENT_TRACE" \
  -p "${DS4_AGENT_PROMPT:-Use local tools only, one tool call at a time, with no narration before tool calls: list the current directory, search for tokens per second, read the matching source lines, create /tmp/ds4_agent_live_tool_test.txt, edit it, search it, run a short bash command, then stop.}"

if grep -E "dsml error|glm_tool error|foreign tool syntax|unsupported tool-call syntax|invalid DSML|invalid native GLM" "$DS4_AGENT_TRACE" >/dev/null; then
  echo "FAIL live: malformed tool syntax detected in $DS4_AGENT_TRACE" >&2
  grep -E "dsml error|glm_tool error|foreign tool syntax|unsupported tool-call syntax|invalid DSML|invalid native GLM|tool_call|tool_result" "$DS4_AGENT_TRACE" >&2 || true
  exit 1
fi

if [[ -z "${DS4_AGENT_EXPECT_TOOLS+x}" ]]; then
  DS4_AGENT_EXPECT_TOOLS="list search read write edit bash"
fi
for tool in $DS4_AGENT_EXPECT_TOOLS; do
  if ! grep -E "tool_call .*name=\"$tool\"" "$DS4_AGENT_TRACE" >/dev/null; then
    echo "FAIL live: missing tool_call for $tool in $DS4_AGENT_TRACE" >&2
    grep -E "tool_call|tool_result|dsml" "$DS4_AGENT_TRACE" >&2 || true
    exit 1
  fi
done

if [[ -n "${DS4_AGENT_EXPECT_ANY_TOOL:-}" ]]; then
  found_any_tool=0
  for tool in $DS4_AGENT_EXPECT_ANY_TOOL; do
    if grep -E "tool_call .*name=\"$tool\"" "$DS4_AGENT_TRACE" >/dev/null; then
      found_any_tool=1
      break
    fi
  done
  if [[ "$found_any_tool" != "1" ]]; then
    echo "FAIL live: missing any expected tool_call from: $DS4_AGENT_EXPECT_ANY_TOOL in $DS4_AGENT_TRACE" >&2
    grep -E "tool_call|tool_result|dsml|glm_tool" "$DS4_AGENT_TRACE" >&2 || true
    exit 1
  fi
fi

echo "PASS live: $DS4_AGENT_TRACE"
