#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SIDECAR_DIR=${DS4_SIDECAR_DIR:-"$ROOT/models/dsv4-iq2xxs-expert-major"}
DENSE_GGUF=${DS4_DENSE_GGUF:-"$SIDECAR_DIR/dense/model-dense.gguf"}
PROMPT_FILE=${DS4_SIDECAR_PROMPT:-"$ROOT/tests/test-vectors/prompts/sidecar_16k.txt"}
SLOT_BANK=${DS4_SLOT_BANK:-64}
CTX=${DS4_CTX:-32768}
GEN_TOKENS=${DS4_GEN_TOKENS:-1}

if [ ! -x "$ROOT/ds4" ]; then
    echo "tests/sidecar_smoke.sh: build ./ds4 first" >&2
    exit 2
fi
if [ ! -f "$DENSE_GGUF" ]; then
    echo "tests/sidecar_smoke.sh: missing dense GGUF: $DENSE_GGUF" >&2
    echo "set DS4_SIDECAR_DIR or DS4_DENSE_GGUF" >&2
    exit 2
fi
if [ ! -f "$SIDECAR_DIR/manifest.json" ]; then
    echo "tests/sidecar_smoke.sh: missing sidecar manifest: $SIDECAR_DIR/manifest.json" >&2
    echo "set DS4_SIDECAR_DIR to a sidecar directory" >&2
    exit 2
fi

export DS4_METAL_PREFILL_CHUNK=${DS4_METAL_PREFILL_CHUNK:-16384}
export DS4_METAL_GRAPH_RAW_CAP=${DS4_METAL_GRAPH_RAW_CAP:-16640}

"$ROOT/ds4" \
    -m "$DENSE_GGUF" \
    --moe-sidecar "$SIDECAR_DIR" \
    --moe-mode slot-bank \
    --moe-slot-bank "$SLOT_BANK" \
    --ctx "$CTX" \
    -n "$GEN_TOKENS" \
    --temp 0 \
    --prompt-file "$PROMPT_FILE"
