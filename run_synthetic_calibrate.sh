#!/usr/bin/env bash
# run_synthetic_calibrate.sh
# Kernel-gate calibration via SYNTHETIC microbenchmarks — NO model load, runs on
# any RAM (incl. 32 GB M5 that can't hold the 81 GB resident model). Isolates the
# routed-MoE matmul kernels per per-expert batch size M, so the crossover points
# directly set the gates (DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS etc.) for THIS GPU.
#
# Probes (built from moe-batch-bench/*.m, no deps on the model/sidecar):
#   fused_kernel_probe  - NAX-half (h_h_f) fused gate+up+swiglu vs separate, per M,
#                         with a tile/SG sweep (SG=1 ~ NAX-half, SG=4 ~ Path C/ALU).
#   nax_fused_probe     - NAX-int8 fused-dequant matmul2d (the int8 baseline kernel).
#
# Output: per-M timing table per probe -> read the crossover (see notes at end).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"

MS="${MS:-32 64 128 256 512 1024}"
N="${N:-2048}"; K="${K:-4096}"; ITERS="${ITERS:-30}"
COOLDOWN="${COOLDOWN:-5}"
TMPDIR_BUILD="${TMPDIR_BUILD:-/tmp}"

FW="-framework Foundation -framework Metal -framework IOSurface"
echo "building synthetic probes..."
cc -O2 -fobjc-arc moe-batch-bench/fused_kernel_probe.m -o "$TMPDIR_BUILD/fused_probe" $FW 2>/dev/null \
  && echo "  fused_kernel_probe OK" || { echo "  fused_kernel_probe FAILED"; }
cc -O2 -fobjc-arc moe-batch-bench/nax_fused_probe.m -o "$TMPDIR_BUILD/nax_fused_probe" $FW 2>/dev/null \
  && echo "  nax_fused_probe OK" || echo "  nax_fused_probe FAILED (optional)"
cc -O2 -fobjc-arc moe-batch-bench/nax_multiexpert_probe.m -o "$TMPDIR_BUILD/nax_multi" $FW 2>/dev/null \
  && echo "  nax_multiexpert_probe OK" || echo "  nax_multiexpert_probe FAILED (optional)"

echo
echo "==================================================================="
echo " NAX-half (h_h_f) fused gate+up+swiglu vs separate   N=$N K=$K"
echo "==================================================================="
echo "M | separate_ms | fused_best_ms | speedup | best_tile/SG"
SUMMARY_ROWS=""   # "M speedup tile" per line, for the auto-recommendation parse
for M in $MS; do
  sleep "$COOLDOWN"
  out=$("$TMPDIR_BUILD/fused_probe" "$M" "$N" "$K" "$ITERS" 2>&1 || true)
  sep=$(echo "$out"  | grep "^timing"    | grep -oE "separate=[0-9.]+ms" | grep -oE "[0-9.]+")
  best=$(echo "$out" | grep "^--- best:" | grep -oE "[0-9.]+ms" | head -1 | sed 's/ms//')
  spd=$(echo "$out"  | grep "^--- best:" | grep -oE "speedup vs separate = [0-9.]+" | grep -oE "[0-9.]+$")
  tile=$(echo "$out" | grep "^--- best:" | grep -oE "fused_v[0-9_]+" | head -1)
  echo "$M | ${sep:-?} | ${best:-?} | ${spd:-?} | ${tile:-?}"
  SUMMARY_ROWS="${SUMMARY_ROWS}${M} ${spd:-0} ${tile:-?}
"
done

if [ -x "$TMPDIR_BUILD/nax_fused_probe" ]; then
  echo
  echo "==================================================================="
  echo " NAX-int8 fused-dequant matmul2d (baseline kernel)"
  echo "==================================================================="
  "$TMPDIR_BUILD/nax_fused_probe" 2>&1 | grep -iE "GF/s|ms|max_abs|verify|OK|FAIL" | head -20
fi

if [ -x "$TMPDIR_BUILD/nax_multi" ]; then
  echo
  echo "==================================================================="
  echo " ALU / multi-expert single-dispatch matmul2d"
  echo "==================================================================="
  "$TMPDIR_BUILD/nax_multi" 2>&1 | grep -iE "GF/s|ms|verify|OK|FAIL|TOTAL|single_dispatch" | head -10
fi

cat <<'NOTES'

-------------------------------------------------------------------------------
READING THE TABLE -> GATES (per THIS GPU; slow GPU differs from M5 Max)
-------------------------------------------------------------------------------
* speedup > 1 at an M  => the FUSED gate+up+swiglu kernel beats separate there.
  If speedup > 1 across ALL M (typical), set MIN_REFS=0 (fuse every expert):
      DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=0
  If fused only wins above some M*, set MIN_REFS=M*.
* best_tile/SG: v32_32_1 (1 simdgroup) is the NAX-half winner; if v*_*_4 (SG=4)
  ever wins, that's the Path C / ALU regime — note it but it rarely beats SG=1.
* On a SLOW GPU the absolute ms are larger but the SPEEDUP ratio is what sets
  the gate; expect the fused win to hold or grow.
NOTES

# ---- recommended gate for THIS GPU (auto-parsed from the NAX-half table) ----
# Find the lowest M from which speedup stays > 1 for every larger M (an unbroken
# winning suffix). A low-M dip (e.g. M=64 on M5 Max) thus raises MIN_REFS instead
# of false-positiving to 0.
first_win_M=""; broke=0; saw_sg4=0
# iterate DESCENDING so we extend the winning suffix until it breaks
rows_desc=$(printf '%s' "$SUMMARY_ROWS" | awk 'NF' | sort -rn)
while IFS=' ' read -r rM rSpd rTile; do
  [ -z "$rM" ] && continue
  case "$rTile" in *_4) saw_sg4=1 ;; esac
  win=$(awk -v s="$rSpd" 'BEGIN{print (s>1.0)?1:0}')
  if [ "$broke" -eq 0 ] && [ "$win" -eq 1 ]; then first_win_M="$rM"
  elif [ "$win" -eq 0 ]; then broke=1
  fi
done <<< "$rows_desc"

echo
echo "==================================================================="
echo " RECOMMENDED GATE FOR THIS GPU"
echo "==================================================================="
if [ -z "$first_win_M" ]; then
  echo " fused never beats separate -> keep it OFF:"
  echo "     export DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=999999"
else
  lowest_M=$(printf '%s' "$SUMMARY_ROWS" | awk 'NF{print $1}' | sort -n | head -1)
  if [ "$first_win_M" = "$lowest_M" ]; then
    echo " fused wins at EVERY tested M -> fuse every expert:"
    echo "     export DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=0"
  else
    echo " fused wins from M=$first_win_M upward (lower M dips) -> gate there:"
    echo "     export DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=$first_win_M"
  fi
fi
[ "$saw_sg4" -eq 1 ] && echo " note: an SG=4 (Path C / ALU) tile won at some M — ALU regime present at small M."