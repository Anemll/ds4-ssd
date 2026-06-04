#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SIDECAR_DIR=${DS4_SIDECAR_DIR:-"$ROOT/models/dsv4-iq2xxs-expert-major"}
DENSE_GGUF=${DS4_DENSE_GGUF:-}
PROMPT_FILE=${DS4_SIDECAR_PROMPT:-"$ROOT/tests/test-vectors/prompts/long_code_audit.txt"}
SLOT_BANK=${DS4_SLOT_BANK:-8}
CTX=${DS4_CTX:-8192}
GEN_TOKENS=${DS4_GEN_TOKENS:-64}

if [ ! -x "$ROOT/ds4" ]; then
    echo "tests/sidecar_smoke.sh: build ./ds4 first" >&2
    exit 2
fi
if [ ! -f "$SIDECAR_DIR/manifest.json" ]; then
    echo "tests/sidecar_smoke.sh: missing sidecar manifest: $SIDECAR_DIR/manifest.json" >&2
    echo "set DS4_SIDECAR_DIR to a sidecar directory" >&2
    exit 2
fi
if [ -z "$DENSE_GGUF" ]; then
    if [ ! -f "$SIDECAR_DIR/dense/model-dense.gguf" ]; then
        echo "tests/sidecar_smoke.sh: missing package dense GGUF: $SIDECAR_DIR/dense/model-dense.gguf" >&2
        echo "set DS4_SIDECAR_DIR to a package root or DS4_DENSE_GGUF to a compatible dense GGUF" >&2
        exit 2
    fi
    MODEL_ARG=$SIDECAR_DIR
else
    if [ ! -f "$DENSE_GGUF" ]; then
        echo "tests/sidecar_smoke.sh: missing dense GGUF: $DENSE_GGUF" >&2
        echo "set DS4_DENSE_GGUF to a compatible dense GGUF" >&2
        exit 2
    fi
    MODEL_ARG=$DENSE_GGUF
fi

export DS4_METAL_PREFILL_CHUNK=${DS4_METAL_PREFILL_CHUNK:-4096}

if [ -z "$DENSE_GGUF" ]; then
    "$ROOT/ds4" \
        -m "$MODEL_ARG" \
        --moe-slot-bank "$SLOT_BANK" \
        --ctx "$CTX" \
        -n "$GEN_TOKENS" \
        --temp 0 \
        --prompt-file "$PROMPT_FILE"
else
    "$ROOT/ds4" \
        -m "$MODEL_ARG" \
        --moe-sidecar "$SIDECAR_DIR" \
        --moe-mode slot-bank \
        --moe-slot-bank "$SLOT_BANK" \
        --ctx "$CTX" \
        -n "$GEN_TOKENS" \
        --temp 0 \
        --prompt-file "$PROMPT_FILE"
fi
