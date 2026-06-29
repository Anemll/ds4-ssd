#!/usr/bin/env bash
set -euo pipefail

source_dir="/Volumes/TB36/Models/DS/DeepSeek-V4-Flash-DSpark"
out_dir="/Users/anemll/Models/DSv4-Flash-DSpark-draft"
variant="flash"
repo_id="deepseek-ai/DeepSeek-V4-Flash-DSpark"
python_bin="${PYTHON:-python3}"
force=0
metadata_only=0

usage() {
  cat <<'EOF'
Usage:
  scripts/export_dspark_draft.sh [options]

Exports a DeepSeek DSpark draft checkpoint into the DS4 dspark_draft package.

Defaults:
  --source-dir /Volumes/TB36/Models/DS/DeepSeek-V4-Flash-DSpark
  --out-dir    /Users/anemll/Models/DSv4-Flash-DSpark-draft
  --variant    flash

Options:
  --source-dir DIR    Directory with config.json, index, and DSpark draft shards.
  --out-dir DIR       Output package directory.
  --variant NAME      flash, pro, or unknown. Default: flash.
  --repo-id ID        Source HF repo id recorded in the manifest.
  --python PATH       Python interpreter. Default: python3 or $PYTHON.
  --metadata-only     Validate source metadata without writing package files.
  --force             Replace existing output directory.
  -h, --help          Show this help.
EOF
}

die() {
  echo "export_dspark_draft.sh: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      [[ $# -ge 2 ]] || die "--source-dir requires a value"
      source_dir="$2"
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || die "--out-dir requires a value"
      out_dir="$2"
      shift 2
      ;;
    --variant)
      [[ $# -ge 2 ]] || die "--variant requires a value"
      variant="$2"
      shift 2
      ;;
    --repo-id)
      [[ $# -ge 2 ]] || die "--repo-id requires a value"
      repo_id="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      python_bin="$2"
      shift 2
      ;;
    --metadata-only)
      metadata_only=1
      shift
      ;;
    --force)
      force=1
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

[[ "$variant" == "flash" || "$variant" == "pro" || "$variant" == "unknown" ]] ||
  die "--variant must be flash, pro, or unknown"
[[ -d "$source_dir" ]] || die "source directory not found: $source_dir"
command -v "$python_bin" >/dev/null 2>&1 || die "python not found: $python_bin"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

cmd=(
  "$python_bin" "$repo_root/scripts/dspark_export.py"
  --source-dir "$source_dir"
  --out-dir "$out_dir"
  --variant "$variant"
  --repo-id "$repo_id"
)
if [[ "$metadata_only" -eq 1 ]]; then
  cmd+=(--metadata-only)
fi
if [[ "$force" -eq 1 ]]; then
  cmd+=(--force)
fi

"${cmd[@]}"
