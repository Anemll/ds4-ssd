#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${DS4_TUI_CONFIG:-./ds4-tui.conf}"

quote_sh() {
    printf "'%s'" "$(printf "%s" "$1" | sed "s/'/'\\\\''/g")"
}

create_config() {
    local project_dir
    local hf_cache_dir
    project_dir="$(quote_sh "$PROJECT_DIR_DEFAULT")"
    if [ -n "${HOME:-}" ] && [ -d "$HOME/.cache/huggingface" ]; then
        hf_cache_dir="$(quote_sh "$HOME/.cache/huggingface")"
    else
        hf_cache_dir="''"
    fi
    cat >"$CONFIG_FILE" <<EOF
# ds4 local runner config.
#
# This file is created by ./ds4-tui.sh on first run and sourced as a shell
# script on later runs. Keep values quoted, avoid spaces around "=", and edit
# this file whenever you want to change where the model lives or how ds4 runs.
#
# Goal for this wrapper: maximize usable tokens/second for evaluating whether a
# huge, highly-compressed, SSD-streamed DeepSeek V4 Flash package can beat the
# usefulness of a smaller fully-resident local model on a 32 GB Apple Silicon
# system. Defaults therefore bias toward performance exploration, not maximum
# verifier conservatism. If a fast setting fails to start or produces a bad A/B
# result, lower it explicitly.

# Absolute path to this ds4 checkout. The wrapper uses this to find ./ds4,
# ./ds4-server, and Makefile targets even when you start the TUI from another
# current directory. Change this only if you move the whole source checkout.
PROJECT_DIR=$project_dir

# Model package type. The default "sidecar" is the SSD-streaming package used
# by this wrapper. "mxfp4" is another sidecar-style package. "gguf" can be used
# for a single resident GGUF file, but the download menu is aimed at Hugging
# Face directory-style packages.
MODEL_KIND='sidecar'

# Hugging Face repository passed to:
#   hf download "\$MODEL_REPO" --local-dir "\$MODEL_DIR"
# Change this if you want the wrapper to download a different compatible model
# package.
MODEL_REPO='anemll/dsv4-iq2xxs-expert-major'

# Local model location. For sidecar packages this must be the directory that
# contains manifest.json and dense/model-dense.gguf. The default keeps the
# large files under ./models/, which is ignored by git. If you later move the
# model to another disk, update this path to the new package directory.
MODEL_DIR='./models/dsv4-iq2xxs-expert-major'

# Optional SSD-backed cache/scratch root for the wrapper. This does not move
# the model files; MODEL_DIR controls the actual sidecar/GGUF location that ds4
# reads during inference. If this is set, the wrapper uses it as the default
# parent for Hugging Face download cache and ds4-server KV disk cache unless
# HF_CACHE_DIR or SERVER_KV_DISK_DIR below are set more specifically.
SSD_CACHE_DIR=''

# Optional Hugging Face revision, branch, tag, or commit. Leave empty to use
# the repository default revision.
HF_REVISION=''

# Hugging Face cache/staging root used through HF_HOME when running
# "hf download". This is separate from MODEL_DIR: MODEL_DIR is the final local
# model package path used by ds4, while HF_CACHE_DIR is where the Hugging Face
# client keeps its download cache and metadata. The wrapper intentionally uses
# HF_HOME instead of "hf download --cache-dir" because this Hugging Face CLI
# does not allow --cache-dir and --local-dir together. First-run config
# generation sets this to "\$HOME/.cache/huggingface" when that directory
# already exists. Leave it empty to fall back to "\$SSD_CACHE_DIR/hf-cache" when
# SSD_CACHE_DIR is set, or otherwise let the Hugging Face CLI choose its normal
# default.
HF_CACHE_DIR=$hf_cache_dir

# Optional Hugging Face token for gated/private downloads. Prefer leaving this
# empty and using "hf auth login" or the HF_TOKEN environment variable so the
# token is not stored in this local config file.
HF_TOKEN=''

# DSpark speculative decoding. DSpark is an optional draft model that can
# improve greedy decode throughput when paired with a compatible Flash sidecar.
# Download it from the TUI before setting DSPARK_ENABLED='1'. DSpark only helps
# greedy requests: use TEMP='0' for CLI runs, and send API requests with
# "temperature": 0 when using ds4-server.
DSPARK_ENABLED='0'
DSPARK_REPO='anemll/DSv4-Flash-DSpark-draft'
DSPARK_DIR='./models/DSv4-Flash-DSpark-draft'
DSPARK_REVISION=''

# DSpark verification controls passed to ds4/ds4-server. For this wrapper's
# performance-first goal, verify=4 is the default target: it can accept up to
# four draft tokens after the ordinary target token, while avoiding the slowest
# fifth draft position. Try 2 or 3 only if verify=4 loses speed on this machine;
# try 5 only for explicit benchmarks.
#
# DSPARK_MODE chooses how accepted draft tokens are verified:
#   batch   Performance-first default. Reduces verification overhead by batching
#           more DSpark work. This is the right first setting for this local
#           usability experiment, where tokens/second is the constraint and the
#           whole model path is already a high-compromise SSD-streamed setup.
#   strict  Conservative reference path. Keeps the verifier on the compatibility
#           path intended to match ordinary greedy decoding. Use it to get an
#           A/B correctness baseline or if batch behaves poorly.
#
# Server mode accepts only "strict" and "batch"; CLI mode may support extra
# experimental modes, but this shared config stays server-safe.
#
# DSPARK_SCHEDULER controls how many drafted tokens DSpark asks the verifier to
# check each block:
#   static      Always schedule the fixed DSPARK_VERIFY budget, subject to how
#               many tokens the draft model actually produced. This is the
#               simple, repeatable baseline and the README's reference path.
#   confidence  Let the draft model emit confidence probabilities for each
#               proposed token, then schedule only the leading prefix whose
#               probabilities are at least DSPARK_CONF_THRESHOLD. This can avoid
#               verifier work when the draft model looks uncertain, but setting
#               the threshold too high can erase most or all speculative speedup.
#
# DSPARK_CONF_THRESHOLD only matters when DSPARK_SCHEDULER='confidence'. It is
# a probability cutoff in the 0..1 range. 0 accepts the whole drafted prefix
# like static scheduling; 0.5 requires each scheduled draft token to be at least
# roughly even-confidence; 1 is maximally conservative and will usually schedule
# no draft tokens because probabilities rarely equal exactly 1.
DSPARK_VERIFY='4'
DSPARK_MODE='batch'
DSPARK_SCHEDULER='static'
DSPARK_CONF_THRESHOLD='0'

# Diagnostic/performance environment toggles. Set DSPARK_PERF='1' to print
# DSpark timing/acceptance summaries. BACKEND_STATS='1' enables backend stat
# output used by the examples in the README.
DSPARK_PERF='1'
BACKEND_STATS='1'

# Context window passed to ds4 as --ctx. Larger values allow longer prompts and
# conversations but allocate more KV/context memory.
CTX='8192'

# Sidecar cache budget passed to ds4 as --ssd-cache. Leave empty to use the
# explicit MOE_SLOT_BANK value below. Set to "auto" or a size such as "32GB" if
# you want ds4 to choose the slot bank from a memory budget. On low-free-memory
# systems "auto" can choose a budget below the minimum required to start.
SSD_CACHE=''

# Sidecar expert slots per layer passed as --moe-slot-bank when SSD_CACHE is
# empty. Higher values reduce SSD reads and can improve generation speed, but
# use more memory. This wrapper defaults to 12 as a performance-first attempt
# on a 32 GB M5. If startup fails or memory pressure is ugly, lower to 8 or the
# minimum-start value 6. If it runs comfortably, benchmark 16+.
MOE_SLOT_BANK='12'

# Maximum generated tokens for one-shot and interactive ds4 runs. Server mode
# uses SERVER_TOKENS below instead.
TOKENS='512'

# Sampling temperature for ds4 CLI runs. DSpark only helps greedy decoding, so
# the performance-evaluation default is 0. Server clients must also send
# "temperature": 0 for DSpark to engage.
TEMP='0'

# Thinking mode sent to ds4. Supported values:
#   nothink   direct non-thinking responses
#   think     normal thinking mode
#   think-max gated long-context Think Max mode
#   empty     do not pass a thinking flag
THINK_MODE='nothink'

# Optional system prompt passed with -sys. Leave empty to use ds4's default
# system prompt.
SYSTEM_PROMPT=''

# Host address for ds4-server. 127.0.0.1 keeps the server local to this
# machine. Use 0.0.0.0 only if you intentionally want LAN-accessible serving.
SERVER_HOST='127.0.0.1'

# TCP port for ds4-server.
SERVER_PORT='8000'

# Default max output tokens for ds4-server requests when the client does not
# provide its own limit.
SERVER_TOKENS='4096'

# Optional disk-backed KV cache directory for ds4-server, passed as
# --kv-disk-dir. This is not the model SSD sidecar cache; it is server request
# state used to reuse prompt/KV work across compatible API calls. Leave empty
# to use "\$SSD_CACHE_DIR/server-kv-cache" when SSD_CACHE_DIR is set, otherwise
# let ds4-server use its own default behavior.
SERVER_KV_DISK_DIR=''
EOF
    echo "Created $CONFIG_FILE"
}

load_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        create_config
    fi

    PROJECT_DIR="$PROJECT_DIR_DEFAULT"
    MODEL_KIND="sidecar"
    MODEL_REPO="anemll/dsv4-iq2xxs-expert-major"
    MODEL_DIR="./models/dsv4-iq2xxs-expert-major"
    SSD_CACHE_DIR=""
    HF_REVISION=""
    HF_CACHE_DIR=""
    HF_TOKEN=""
    DSPARK_ENABLED="0"
    DSPARK_REPO="anemll/DSv4-Flash-DSpark-draft"
    DSPARK_DIR="./models/DSv4-Flash-DSpark-draft"
    DSPARK_REVISION=""
    DSPARK_VERIFY="4"
    DSPARK_MODE="strict"
    DSPARK_SCHEDULER="static"
    DSPARK_CONF_THRESHOLD="0"
    DSPARK_PERF="1"
    BACKEND_STATS="1"
    CTX="8192"
    SSD_CACHE=""
    MOE_SLOT_BANK="6"
    TOKENS="512"
    TEMP="0.7"
    THINK_MODE="nothink"
    SYSTEM_PROMPT=""
    SERVER_HOST="127.0.0.1"
    SERVER_PORT="8000"
    SERVER_TOKENS="4096"
    SERVER_KV_DISK_DIR=""

    # shellcheck disable=SC1090
    . "$CONFIG_FILE"
}

effective_hf_cache_dir() {
    local home_dir
    home_dir="${HOME:-}"

    if [ -n "${HF_CACHE_DIR:-}" ]; then
        printf "%s" "$HF_CACHE_DIR"
    elif [ -n "$home_dir" ] && [ -d "$home_dir/.cache/huggingface" ]; then
        printf "%s" "$home_dir/.cache/huggingface"
    elif [ -n "${SSD_CACHE_DIR:-}" ]; then
        printf "%s/hf-cache" "$SSD_CACHE_DIR"
    fi
    return 0
}

effective_server_kv_disk_dir() {
    if [ -n "${SERVER_KV_DISK_DIR:-}" ]; then
        printf "%s" "$SERVER_KV_DISK_DIR"
    elif [ -n "${SSD_CACHE_DIR:-}" ]; then
        printf "%s/server-kv-cache" "$SSD_CACHE_DIR"
    fi
    return 0
}

require_binary() {
    local bin="$1"
    local name="$2"
    if [ ! -x "$bin" ]; then
        echo "Missing $name at $bin"
        echo "Run: make"
        exit 1
    fi
}

require_hf() {
    if ! command -v hf >/dev/null 2>&1; then
        echo "Missing Hugging Face CLI: hf"
        echo "Install it, then run: hf auth login"
        exit 1
    fi
}

ds4_bin() {
    printf "%s/ds4" "$PROJECT_DIR"
}

ds4_server_bin() {
    printf "%s/ds4-server" "$PROJECT_DIR"
}

model_ready() {
    case "$MODEL_KIND" in
        sidecar|mxfp4)
            [ -f "$MODEL_DIR/manifest.json" ]
            ;;
        gguf)
            [ -f "$MODEL_DIR" ]
            ;;
        *)
            [ -e "$MODEL_DIR" ]
            ;;
    esac
}

dspark_enabled() {
    case "${DSPARK_ENABLED:-0}" in
        1|yes|true|on) return 0 ;;
        *) return 1 ;;
    esac
}

dspark_ready() {
    [ -e "$DSPARK_DIR" ]
}

print_status() {
    local hf_cache_dir
    local server_kv_disk_dir
    hf_cache_dir="$(effective_hf_cache_dir)"
    server_kv_disk_dir="$(effective_server_kv_disk_dir)"

    echo
    echo "Config:       $CONFIG_FILE"
    echo "Project:      $PROJECT_DIR"
    echo "Model kind:   $MODEL_KIND"
    echo "Model repo:   $MODEL_REPO"
    echo "Model path:   $MODEL_DIR"
    echo "SSD cache dir:${SSD_CACHE_DIR:-disabled}"
    echo "HF cache:     ${hf_cache_dir:-default}"
    echo "Context:      $CTX"
    echo "SSD cache:    ${SSD_CACHE:-disabled}"
    echo "Slot bank:    ${MOE_SLOT_BANK:-default}"
    echo "DSpark:       $(dspark_enabled && echo enabled || echo disabled)"
    if dspark_enabled; then
        echo "DSpark path:  $DSPARK_DIR"
        echo "DSpark verify:$DSPARK_VERIFY"
    fi
    echo "Server:       http://$SERVER_HOST:$SERVER_PORT"
    echo "Server KV:    ${server_kv_disk_dir:-default}"
    if model_ready; then
        echo "Model state:  present"
    else
        echo "Model state:  not downloaded"
    fi
    echo
}

download_model() {
    require_hf
    mkdir -p "$MODEL_DIR"
    echo "Downloading $MODEL_REPO"
    echo "Into $MODEL_DIR"
    echo "Re-run this action if the download is interrupted; hf will resume."

    local cmd=(hf download "$MODEL_REPO" --local-dir "$MODEL_DIR")
    if [ -n "$HF_REVISION" ]; then
        cmd+=(--revision "$HF_REVISION")
    fi
    local hf_cache_dir
    local hf_env=()
    hf_cache_dir="$(effective_hf_cache_dir)"
    if [ -n "$hf_cache_dir" ]; then
        mkdir -p "$hf_cache_dir"
        hf_env+=(HF_HOME="$hf_cache_dir")
    fi
    if [ -n "$HF_TOKEN" ]; then
        hf_env+=(HF_TOKEN="$HF_TOKEN")
    fi

    if ! env "${hf_env[@]}" "${cmd[@]}"; then
        echo "Download failed."
        return 1
    fi
}

download_dspark() {
    require_hf
    mkdir -p "$DSPARK_DIR"
    echo "Downloading $DSPARK_REPO"
    echo "Into $DSPARK_DIR"
    echo "Re-run this action if the download is interrupted; hf will resume."

    local cmd=(hf download "$DSPARK_REPO" --local-dir "$DSPARK_DIR")
    if [ -n "$DSPARK_REVISION" ]; then
        cmd+=(--revision "$DSPARK_REVISION")
    fi

    local hf_cache_dir
    local hf_env=()
    hf_cache_dir="$(effective_hf_cache_dir)"
    if [ -n "$hf_cache_dir" ]; then
        mkdir -p "$hf_cache_dir"
        hf_env+=(HF_HOME="$hf_cache_dir")
    fi
    if [ -n "$HF_TOKEN" ]; then
        hf_env+=(HF_TOKEN="$HF_TOKEN")
    fi

    if ! env "${hf_env[@]}" "${cmd[@]}"; then
        echo "DSpark download failed."
        return 1
    fi
}

append_env_args() {
    DS4_ENV=()
    case "${DSPARK_PERF:-0}" in
        1|yes|true|on) DS4_ENV+=(DS4_DSPARK_PERF=1) ;;
    esac
    case "${BACKEND_STATS:-0}" in
        1|yes|true|on) DS4_ENV+=(DS4_AGENT_ALLOW_BACKEND_STATS=1) ;;
    esac
}

append_runtime_args() {
    DS4_ARGS=(-m "$MODEL_DIR" --ctx "$CTX")

    if [ -n "$SSD_CACHE" ]; then
        DS4_ARGS+=(--ssd-cache "$SSD_CACHE")
    elif [ -n "$MOE_SLOT_BANK" ]; then
        DS4_ARGS+=(--moe-slot-bank "$MOE_SLOT_BANK")
    fi

    if dspark_enabled; then
        DS4_ARGS+=(--draft dspark --draft-path "$DSPARK_DIR")
        if [ -n "$DSPARK_VERIFY" ]; then
            DS4_ARGS+=(--draft-verify "$DSPARK_VERIFY")
        fi
        if [ -n "$DSPARK_MODE" ]; then
            DS4_ARGS+=(--draft-mode "$DSPARK_MODE")
        fi
        if [ -n "$DSPARK_SCHEDULER" ]; then
            DS4_ARGS+=(--draft-scheduler "$DSPARK_SCHEDULER")
        fi
        if [ -n "$DSPARK_CONF_THRESHOLD" ]; then
            DS4_ARGS+=(--draft-conf-threshold "$DSPARK_CONF_THRESHOLD")
        fi
    fi
}

append_cli_args() {
    append_runtime_args

    case "$THINK_MODE" in
        think) DS4_ARGS+=(--think) ;;
        think-max) DS4_ARGS+=(--think-max) ;;
        nothink) DS4_ARGS+=(--nothink) ;;
        "") ;;
        *)
            echo "Unsupported THINK_MODE='$THINK_MODE' in $CONFIG_FILE"
            exit 1
            ;;
    esac

    if [ -n "$SYSTEM_PROMPT" ]; then
        DS4_ARGS+=(-sys "$SYSTEM_PROMPT")
    fi
}

ensure_model_or_hint() {
    if ! model_ready; then
        echo "Model is not present at: $MODEL_DIR"
        echo "Choose Download model from the menu first."
        return 1
    fi
}

ensure_dspark_or_hint() {
    if dspark_enabled && ! dspark_ready; then
        echo "DSpark is enabled but the draft package is not present at: $DSPARK_DIR"
        echo "Choose Download DSpark draft from the menu first, or set DSPARK_ENABLED='0'."
        return 1
    fi
}

run_prompt() {
    local prompt="$1"
    local bin
    bin="$(ds4_bin)"
    require_binary "$bin" "ds4"
    ensure_model_or_hint || return 1
    ensure_dspark_or_hint || return 1
    append_cli_args
    append_env_args
    env "${DS4_ENV[@]}" "$bin" "${DS4_ARGS[@]}" --temp "$TEMP" -n "$TOKENS" -p "$prompt"
}

run_interactive() {
    local bin
    bin="$(ds4_bin)"
    require_binary "$bin" "ds4"
    ensure_model_or_hint || return 1
    ensure_dspark_or_hint || return 1
    append_cli_args
    append_env_args
    env "${DS4_ENV[@]}" "$bin" "${DS4_ARGS[@]}" --temp "$TEMP" -n "$TOKENS"
}

run_server() {
    local bin
    bin="$(ds4_server_bin)"
    require_binary "$bin" "ds4-server"
    ensure_model_or_hint || return 1
    ensure_dspark_or_hint || return 1
    append_runtime_args
    append_env_args
    SERVER_ARGS=("${DS4_ARGS[@]}" --tokens "$SERVER_TOKENS" --host "$SERVER_HOST" --port "$SERVER_PORT")
    local server_kv_disk_dir
    server_kv_disk_dir="$(effective_server_kv_disk_dir)"
    if [ -n "$server_kv_disk_dir" ]; then
        mkdir -p "$server_kv_disk_dir"
        SERVER_ARGS+=(--kv-disk-dir "$server_kv_disk_dir")
    fi
    env "${DS4_ENV[@]}" "$bin" "${SERVER_ARGS[@]}"
}

run_sidecar_smoke() {
    ensure_model_or_hint || return 1
    DS4_SIDECAR_DIR="$MODEL_DIR" make -C "$PROJECT_DIR" sidecar-smoke
}

edit_config() {
    local editor="${EDITOR:-}"
    if [ -z "$editor" ]; then
        if command -v nano >/dev/null 2>&1; then
            editor="nano"
        else
            editor="vi"
        fi
    fi
    "$editor" "$CONFIG_FILE"
    load_config
}

pause() {
    printf "\nPress return to continue..."
    read -r _
}

menu() {
    while true; do
        clear 2>/dev/null || true
        echo "ds4 local runner"
        echo "================"
        print_status
        echo "1) Download model with hf"
        echo "2) Download DSpark draft with hf"
        echo "3) Run one prompt"
        echo "4) Start interactive ds4"
        echo "5) Start ds4-server"
        echo "6) Run sidecar smoke"
        echo "7) Show config"
        echo "8) Edit config"
        echo "q) Quit"
        printf "\nChoice: "
        read -r choice

        case "$choice" in
            1) download_model || true; pause ;;
            2) download_dspark || true; pause ;;
            3)
                printf "Prompt: "
                read -r prompt
                [ -n "$prompt" ] || prompt="Hello"
                run_prompt "$prompt" || true
                pause
                ;;
            4) run_interactive || true; pause ;;
            5) run_server || true; pause ;;
            6) run_sidecar_smoke || true; pause ;;
            7) sed -n '1,260p' "$CONFIG_FILE"; pause ;;
            8) edit_config ;;
            q|Q) exit 0 ;;
            *) echo "Unknown choice: $choice"; pause ;;
        esac
    done
}

usage() {
    cat <<EOF
Usage: $0 [command]

Without a command, starts the TUI menu. The config file is created in the
current directory as $CONFIG_FILE on first run.

Commands:
  --init             Create config if missing, then exit.
  --show-config      Print the current config.
  --download         Download MODEL_REPO into MODEL_DIR with hf.
  --download-dspark  Download DSPARK_REPO into DSPARK_DIR with hf.
  --prompt TEXT      Run one prompt.
  --interactive      Start interactive ds4.
  --server           Start ds4-server.
  --smoke            Run sidecar smoke with MODEL_DIR.
  -h, --help         Show this help.
EOF
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

load_config

case "${1:-}" in
    "")
        menu
        ;;
    --init)
        exit 0
        ;;
    --show-config)
        sed -n '1,220p' "$CONFIG_FILE"
        ;;
    --download)
        download_model
        ;;
    --download-dspark)
        download_dspark
        ;;
    --prompt)
        shift
        if [ $# -eq 0 ]; then
            echo "--prompt requires text"
            exit 1
        fi
        run_prompt "$*"
        ;;
    --interactive)
        run_interactive
        ;;
    --server)
        run_server
        ;;
    --smoke)
        run_sidecar_smoke
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
