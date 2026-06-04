#!/usr/bin/env bash
set -euo pipefail

repo_url="https://github.com/anemll/anemll-flash-llama.cpp.git"
branch="DeepSeek-V4-SSD"
model=""
out_dir=""
llama_dir="${FLASH_MOE_LLAMA_DIR:-}"
python_bin="${PYTHON:-python3}"
force=0
verify=0
metadata_only=0
include_shared=0
dry_run=0

usage() {
  cat <<'EOF'
Usage:
  scripts/export_flash_moe_sidecar.sh --model MODEL.gguf --out-dir OUT_DIR [options]

Exports a DS4 Flash-MoE expert-major sidecar using the public
anemll/anemll-flash-llama.cpp converter branch.

Required:
  --model FILE         Source DS4 GGUF, single file or first split shard.
  --out-dir DIR       Output sidecar directory.

Options:
  --llama-dir DIR     Existing anemll-flash-llama.cpp checkout.
                      Defaults to FLASH_MOE_LLAMA_DIR, then ./.external/anemll-flash-llama.cpp.
  --repo-url URL      Repo to clone when --llama-dir is missing.
                      Default: https://github.com/anemll/anemll-flash-llama.cpp.git
  --branch NAME       Branch to clone when --llama-dir is missing.
                      Default: DeepSeek-V4-SSD
  --python PATH       Python interpreter. Default: python3 or $PYTHON.
  --force             Pass --force to the converter.
  --verify            Run converter verify after extraction.
  --metadata-only     With --verify, validate metadata only.
  --include-shared    Also export shared expert tensors. DS4 sidecar runtime normally does not need this.
  --dry-run           Print commands without running extraction.
  -h, --help          Show this help.

The converter command used is:
  flashmoe_sidecar.py extract --layout expert-major

Important:
  origin/master of anemll-flash-llama.cpp may not expose --layout.
  Use branch DeepSeek-V4-SSD for the expert-major export path.
  This exports expert sidecar records only; it does not create
  dense/model-dense.gguf.
EOF
}

die() {
  echo "export_flash_moe_sidecar.sh: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || die "--model requires a value"
      model="$2"
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || die "--out-dir requires a value"
      out_dir="$2"
      shift 2
      ;;
    --llama-dir)
      [[ $# -ge 2 ]] || die "--llama-dir requires a value"
      llama_dir="$2"
      shift 2
      ;;
    --repo-url)
      [[ $# -ge 2 ]] || die "--repo-url requires a value"
      repo_url="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || die "--branch requires a value"
      branch="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      python_bin="$2"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    --verify)
      verify=1
      shift
      ;;
    --metadata-only)
      metadata_only=1
      shift
      ;;
    --include-shared)
      include_shared=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "$model" ]] || die "--model is required"
[[ -n "$out_dir" ]] || die "--out-dir is required"
[[ -f "$model" ]] || die "model file not found: $model"
command -v git >/dev/null 2>&1 || die "git is required"
command -v "$python_bin" >/dev/null 2>&1 || die "python not found: $python_bin"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
if [[ -z "$llama_dir" ]]; then
  llama_dir="$repo_root/.external/anemll-flash-llama.cpp"
fi

if [[ ! -d "$llama_dir" ]]; then
  mkdir -p "$(dirname "$llama_dir")"
  echo "Cloning $repo_url branch $branch into $llama_dir" >&2
  git clone --branch "$branch" --single-branch "$repo_url" "$llama_dir"
fi

tool="$llama_dir/tools/flashmoe-sidecar/flashmoe_sidecar.py"
[[ -f "$tool" ]] || die "converter not found: $tool"

if ! "$python_bin" "$tool" extract --help 2>&1 | grep -q -- "--layout"; then
  cat >&2 <<EOF
export_flash_moe_sidecar.sh: converter does not expose --layout.

Use the public branch that contains the expert-major path:
  cd "$llama_dir"
  git fetch origin "$branch"
  git checkout "$branch"

Or pass --llama-dir to a checkout of:
  $repo_url
  branch: $branch
EOF
  exit 2
fi

extract_cmd=(
  "$python_bin" "$tool"
  extract
  --model "$model"
  --out-dir "$out_dir"
  --layout expert-major
)
if [[ "$force" -eq 1 ]]; then
  extract_cmd+=(--force)
fi
if [[ "$include_shared" -eq 1 ]]; then
  extract_cmd+=(--include-shared)
fi

printf 'Running:'
printf ' %q' "${extract_cmd[@]}"
printf '\n'
if [[ "$dry_run" -eq 0 ]]; then
  "${extract_cmd[@]}"
fi

if [[ "$verify" -eq 1 ]]; then
  verify_cmd=(
    "$python_bin" "$tool"
    verify
    --model "$model"
    --sidecar "$out_dir"
  )
  if [[ "$metadata_only" -eq 1 ]]; then
    verify_cmd+=(--metadata-only)
  fi
  printf 'Running:'
  printf ' %q' "${verify_cmd[@]}"
  printf '\n'
  if [[ "$dry_run" -eq 0 ]]; then
    "${verify_cmd[@]}"
  fi
fi

cat <<EOF

Sidecar export command completed.

Note: this wrapper exports the routed expert sidecar records. For the public
alpha low-RAM package layout, pair the sidecar with a compatible dense-only GGUF
at dense/model-dense.gguf. Running with a full resident GGUF as -m is useful for
export validation and experiments, but it is not the same low-RAM package.

If you add a compatible dense-only GGUF at "$out_dir/dense/model-dense.gguf",
use the package root:
  ./ds4 \\
    -m "$out_dir" \\
    --moe-slot-bank 8 \\
    --ctx 8192

For expert-only export validation, use an explicit GGUF plus sidecar:
  ./ds4 \\
    -m "$model" \\
    --moe-sidecar "$out_dir" \\
    --moe-mode slot-bank \\
    --moe-slot-bank 8 \\
    --ctx 8192
EOF
