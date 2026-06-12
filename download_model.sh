#!/bin/sh
set -e

REPO="antirez/deepseek-v4-gguf"
SIDECAR_REPO="anemll/dsv4-iq2xxs-expert-major"
SIDECAR_DIR_NAME="dsv4-iq2xxs-expert-major"
MXFP4_REPO="anemll/DSv4-Flash-MXFP4-native-flash"
MXFP4_DIR_NAME="DSv4-Flash-MXFP4-native-flash"
HUIHUI_REPO="huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF"
Q2_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
Q2_IMATRIX_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
Q4_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf"
Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
MTP_FILE="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
HUIHUI_IQ2XXS_FILE="Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-IQ2_XXS.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
SIDECAR_OUT_DIR=${DS4_SIDECAR_DIR:-"$ROOT/models/$SIDECAR_DIR_NAME"}
case "$SIDECAR_OUT_DIR" in
    /*) ;;
    *) SIDECAR_OUT_DIR="$ROOT/$SIDECAR_OUT_DIR" ;;
esac
MXFP4_OUT_DIR=${DS4_MXFP4_DIR:-"$ROOT/models/$MXFP4_DIR_NAME"}
case "$MXFP4_OUT_DIR" in
    /*) ;;
    *) MXFP4_OUT_DIR="$ROOT/$MXFP4_OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}
MODEL_REPO=$REPO

usage() {
    cat <<EOF
DeepSeek V4 Flash GGUF downloader

Usage:
  ./download_model.sh q2-imatrix [--token TOKEN]
  ./download_model.sh q4-imatrix [--token TOKEN]
  ./download_model.sh q2 [--token TOKEN]
  ./download_model.sh q4 [--token TOKEN]
  ./download_model.sh mtp [--token TOKEN]
  ./download_model.sh huihui-iq2xxs [--token TOKEN]
  ./download_model.sh sidecar [--token TOKEN]
  ./download_model.sh mxfp4 [--token TOKEN]

Targets:
  *** PREFERRED GGUF FILES: USE THE IMATRIX VERSIONS BELOW ***

  q2-imatrix
       2-bit routed experts, about 81 GB on disk.
       Recommended model for 96 and 128 GB RAM machines.

  q4-imatrix
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  Legacy GGUF files:

  q2   2-bit routed experts, about 81 GB on disk.
       Older non-imatrix model for 96 and 128 GB RAM machines. Prefer
       q2-imatrix unless you specifically need the legacy quant.

  q4   4-bit routed experts, about 153 GB on disk.
       Older non-imatrix model for machines with 256 GB RAM or more. Prefer
       q4-imatrix unless you specifically need the legacy quant.

  mtp  Optional speculative decoding component, about 3.5 GB on disk.
       It is useful with q2-imatrix, q4-imatrix, q2, and q4, but must be
       enabled explicitly with --mtp when running ds4 or ds4-server.

  huihui-iq2xxs
       Resident GGUF from $HUIHUI_REPO:
       $HUIHUI_IQ2XXS_FILE
       About 80 GB on disk. Can run resident on a 96 GB M3 Ultra if almost
       nothing else is using memory; SSD sidecar remains the safer 96 GB path.

  sidecar
       SSD-streaming sidecar package from $SIDECAR_REPO.
       Downloads dense/model-dense.gguf plus routed-expert sidecar files.

  mxfp4
       Native MXFP4 SSD-streaming package from $MXFP4_REPO.
       About 156 GB on disk: Q8_0/F16 dense GGUF plus bit-exact native
       MXFP4 routed-expert sidecar (manifest.json + layer_*.bin).
       Recommended SSD-streaming package; runs on 96 GB+ machines with
       --ssd-cache (the slot bank auto-shrinks after prefill on
       RAM-limited machines, so any --ssd-cache size is safe).

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache is used if present.

Environment:
  DS4_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

  DS4_SIDECAR_DIR
                 Directory used for the downloaded SSD sidecar package.
                 Default: ./models/$SIDECAR_DIR_NAME

  DS4_MXFP4_DIR  Directory used for the downloaded MXFP4 package.
                 Default: ./models/$MXFP4_DIR_NAME

After q2-imatrix/q4-imatrix/q2/q4/huihui-iq2xxs downloads the script updates:
  ./ds4flash.gguf -> <download directory>/<selected model>

Then the default commands work:
  ./ds4 -p "Hello"
  ./ds4-server --ctx 100000

After downloading mtp, enable it explicitly, for example:
  ./ds4 --mtp <download directory>/$MTP_FILE --mtp-draft 2

After downloading sidecar, run:
  DS4_SIDECAR_DIR=<sidecar directory> make sidecar-smoke
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift

case "$MODEL" in
    q2-imatrix) MODEL_FILE=$Q2_IMATRIX_FILE ;;
    q4-imatrix) MODEL_FILE=$Q4_IMATRIX_FILE ;;
    q2) MODEL_FILE=$Q2_FILE ;;
    q4) MODEL_FILE=$Q4_FILE ;;
    mtp) MODEL_FILE=$MTP_FILE ;;
    huihui-iq2xxs)
        MODEL_REPO=$HUIHUI_REPO
        MODEL_FILE=$HUIHUI_IQ2XXS_FILE
        ;;
    sidecar) MODEL_FILE= ;;
    mxfp4) MODEL_FILE= ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

download_one() {
    file=$1
    out="$OUT_DIR/$file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$MODEL_REPO/resolve/main/$file"

    mkdir -p "$OUT_DIR"

    if [ -e "$aria2_part" ]; then
        echo "Found incomplete aria2 download sidecar: $aria2_part" >&2
        echo "Finish or remove that partial download before using this curl downloader." >&2
        exit 1
    fi

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$MODEL_REPO"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

download_sidecar() {
    if ! command -v hf >/dev/null 2>&1; then
        echo "The sidecar package is a multi-file Hugging Face repo." >&2
        echo "Install the Hugging Face CLI first: https://huggingface.co/docs/huggingface_hub/guides/cli" >&2
        exit 1
    fi

    mkdir -p "$SIDECAR_OUT_DIR"

    echo "Downloading SSD sidecar package"
    echo "from https://huggingface.co/$SIDECAR_REPO"
    echo "to $SIDECAR_OUT_DIR"

    if [ -n "$TOKEN" ]; then
        HF_TOKEN=$TOKEN hf download "$SIDECAR_REPO" --local-dir "$SIDECAR_OUT_DIR"
    else
        hf download "$SIDECAR_REPO" --local-dir "$SIDECAR_OUT_DIR"
    fi

    echo
    echo "Set:"
    echo "  export DS4_SIDECAR_DIR=$SIDECAR_OUT_DIR"
    echo
    echo "Then run:"
    echo "  ./ds4 -m \"\$DS4_SIDECAR_DIR\" --moe-slot-bank 8 --ctx 8192 -p 'Hello'"
    echo
    echo "After checking memory pressure, raise --moe-slot-bank and --ctx as needed."
    echo "  make sidecar-smoke"
}

download_mxfp4() {
    if ! command -v hf >/dev/null 2>&1; then
        echo "The MXFP4 package is a multi-file Hugging Face repo." >&2
        echo "Install the Hugging Face CLI first: https://huggingface.co/docs/huggingface_hub/guides/cli" >&2
        exit 1
    fi

    mkdir -p "$MXFP4_OUT_DIR"

    echo "Downloading native MXFP4 SSD-streaming package (about 156 GB)"
    echo "from https://huggingface.co/$MXFP4_REPO"
    echo "to $MXFP4_OUT_DIR"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        HF_TOKEN=$TOKEN hf download "$MXFP4_REPO" --local-dir "$MXFP4_OUT_DIR"
    else
        hf download "$MXFP4_REPO" --local-dir "$MXFP4_OUT_DIR"
    fi

    echo
    echo "Run it with SSD streaming:"
    echo "  ./ds4 -m \"$MXFP4_OUT_DIR\" --ssd-cache auto -p 'Hello'"
    echo
    echo "Or with an explicit slot-bank budget (any size is safe; on RAM-limited"
    echo "machines the bank auto-shrinks after prefill for decode):"
    echo "  ./ds4 -m \"$MXFP4_OUT_DIR\" --ssd-cache 32GB -p 'Hello'"
}

if [ "$MODEL" = "sidecar" ]; then
    download_sidecar
    echo
    echo "Done."
    exit 0
fi

if [ "$MODEL" = "mxfp4" ]; then
    download_mxfp4
    echo
    echo "Done."
    exit 0
fi

download_one "$MODEL_FILE"

if [ "$MODEL" = "mtp" ]; then
    echo
    echo "MTP is an optional component for q2-imatrix, q4-imatrix, q2, and q4."
    echo "Enable it explicitly, for example:"
    echo "  ./ds4 --mtp $OUT_DIR/$MTP_FILE --mtp-draft 2"
else
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
    echo "Linked ./ds4flash.gguf -> $OUT_DIR/$MODEL_FILE"
fi

echo
echo "Done."
