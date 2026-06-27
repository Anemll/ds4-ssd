#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll-flash-llama.cpp}"
MODEL="${MODEL:-/Users/anemll/Models/GLM-5.2-IQ1-Dense/model-dense.gguf}"
SIDECAR="${SIDECAR:-/Users/anemll/Models/GLM-5.2-sidecar}"
OUT_DIR="${OUT_DIR:-/tmp/glm52-longprompt-compare}"
PROMPT_FILE="${PROMPT_FILE:-/Users/anemll/SourceRelease/GITHUB/ML_playground/long_prompt.txt}"
ROWS="${ROWS:-900}"
TOKENS="${TOKENS:-1536}"
CTX="${CTX:-32768}"
SLOT_BANK="${SLOT_BANK:-32}"
SEED="${SEED:-1}"
RUN_DS4="${RUN_DS4:-1}"
RUN_LLAMA="${RUN_LLAMA:-1}"
RUN_AGENT="${RUN_AGENT:-1}"

PROMPT="$OUT_DIR/prompt-${ROWS}.txt"
DS4_OUT="$OUT_DIR/ds4-${ROWS}.out"
LLAMA_OUT="$OUT_DIR/llama-${ROWS}.out"
AGENT_OUT="$OUT_DIR/ds4-agent-${ROWS}.out"
LLAMA_PROMPT="$OUT_DIR/llama-rendered-${ROWS}.txt"

mkdir -p "$OUT_DIR"
if [ -f "$PROMPT_FILE" ]; then
  PROMPT="$PROMPT_FILE"
else
  {
    echo "You are doing a long-context retrieval test."
    echo "The reference table below contains many distractor rows and two sentinel rows."
    echo "Do not write code. Do not explain. After the table, answer exactly in this format:"
    echo "A=<sentinel A value>"
    echo "B=<sentinel B value>"
    echo
    echo "REFERENCE TABLE START"
    i=1
    while [ "$i" -le "$ROWS" ]; do
      case "$i" in
        471)
          printf 'ROW %04d | SENTINEL_A | VALUE=DS4CHECK-ALPHA-0471-PINEAPPLE | filler=%08d\n' "$i" "$((i * 73 + 41))"
          ;;
        888)
          printf 'ROW %04d | SENTINEL_B | VALUE=DS4CHECK-OMEGA-0888-QUARTZ | filler=%08d\n' "$i" "$((i * 73 + 41))"
          ;;
        *)
          printf 'ROW %04d | ordinary distractor | VALUE=ignore-%04d-%08d | words=metal attention cache router prefill\n' "$i" "$i" "$((i * 73 + 41))"
          ;;
      esac
      i=$((i + 1))
    done
    echo "REFERENCE TABLE END"
    echo
    echo "Now answer by copying only the two sentinel VALUE strings, exactly:"
    echo "A="
    echo "B="
  } > "$PROMPT"
fi

echo "prompt=$PROMPT"
echo "bytes=$(wc -c < "$PROMPT") lines=$(wc -l < "$PROMPT")"
{
  printf '[gMASK]<sop><|user|>\n'
  cat "$PROMPT"
  printf '<|assistant|>\n<think></think>'
} > "$LLAMA_PROMPT"

if [ "$RUN_DS4" = "1" ]; then
  echo "running ds4..."
  (
    cd "$ROOT"
    ./ds4 \
      -m "$MODEL" \
      --moe-mode slot-bank \
      --moe-sidecar "$SIDECAR" \
      --moe-slot-bank "$SLOT_BANK" \
      --ctx "$CTX" \
      --prompt-file "$PROMPT" \
      -sys "" \
      --nothink \
      --temp 0 \
      --seed "$SEED" \
      --tokens "$TOKENS"
  ) > "$DS4_OUT" 2>&1
  echo "ds4_out=$DS4_OUT"
else
  echo "skipping ds4; ds4_out=$DS4_OUT"
fi

if [ "$RUN_LLAMA" = "1" ]; then
  echo "running llama-cli..."
  (
    cd "$LLAMA_ROOT"
    build/bin/llama-cli \
      -m "$MODEL" \
      --moe-mode slot-bank \
      --moe-sidecar "$SIDECAR" \
      --moe-slot-bank "$SLOT_BANK" \
      --slot8 \
      -ngl 99 \
      -c "$CTX" \
      -f "$LLAMA_PROMPT" \
      -no-cnv \
      -sp \
      --simple-io \
      --no-display-prompt \
      --reasoning off \
      --temp 0 \
      --seed "$SEED" \
      -n "$TOKENS" \
      --no-warmup
  ) > "$LLAMA_OUT" 2>&1
  echo "llama_out=$LLAMA_OUT"
else
  echo "skipping llama-cli; llama_out=$LLAMA_OUT"
fi

if [ "$RUN_AGENT" = "1" ]; then
  echo "running ds4-agent..."
  (
    cd "$ROOT"
    ./ds4-agent \
      -m "$MODEL" \
      --moe-mode slot-bank \
      --moe-sidecar "$SIDECAR" \
      --moe-slot-bank "$SLOT_BANK" \
      --ctx "$CTX" \
      --non-interactive \
      -p "$(cat "$PROMPT")" \
      --nothink \
      --temp 0 \
      --seed "$SEED" \
      --tokens "$TOKENS"
  ) > "$AGENT_OUT" 2>&1
  echo "agent_out=$AGENT_OUT"
else
  echo "skipping ds4-agent; agent_out=$AGENT_OUT"
fi

python3 - "$DS4_OUT" "$LLAMA_OUT" "$AGENT_OUT" <<'PY'
from pathlib import Path
import re
import sys

def collapse_score(text):
    body = re.sub(r"^ds4: .*$", "", text, flags=re.M)
    issues = []
    if re.search(r"\b\d+\.\d+\.\d+\b", body):
        issues.append("multi-dot-number")
    if body.count("command>") >= 3:
        issues.append("repeated-command-tag-fragment")
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    counts = {}
    for ln in lines:
        if re.fullmatch(r'"[. 0-9A-Fa-f]+",?', ln):
            continue
        if re.fullmatch(r"[-/*=\\ ]{12,}", ln):
            continue
        if len(ln) >= 12:
            counts[ln] = counts.get(ln, 0) + 1
    if counts and max(counts.values()) >= 6:
        issues.append(f"repeated-line-x{max(counts.values())}")
    consts = re.findall(r"\bconst\s+([A-Z_][A-Z0-9_]*)\s*=", body)
    for name in sorted(set(consts)):
        if consts.count(name) >= 5:
            issues.append(f"repeated-const-{name}-x{consts.count(name)}")
            break
    if re.search(r"</content><!DOCTYPE|path=.*?</content>", body, re.S):
        issues.append("malformed-tool-content-boundary")
    return issues

for path in map(Path, sys.argv[1:]):
    if not path.exists():
        print(f"{path.name}: skipped bytes=0 issues=-")
        continue
    text = path.read_text(errors="replace")
    issues = collapse_score(text)
    status = "COLLAPSE" if issues else "CAP" if "Tool call exceeded --tokens" in text else "ok"
    print(f"{path.name}: {status} bytes={len(text)} issues={','.join(issues) if issues else '-'}")
PY

echo
echo "=== ds4 tail ==="
if [ -f "$DS4_OUT" ]; then
  tail -80 "$DS4_OUT"
else
  echo "(missing: $DS4_OUT)"
fi
echo
echo "=== llama tail ==="
if [ -f "$LLAMA_OUT" ]; then
  tail -80 "$LLAMA_OUT"
else
  echo "(missing: $LLAMA_OUT)"
fi
echo
echo "=== ds4-agent tail ==="
if [ -f "$AGENT_OUT" ]; then
  tail -80 "$AGENT_OUT"
else
  echo "(missing: $AGENT_OUT)"
fi
