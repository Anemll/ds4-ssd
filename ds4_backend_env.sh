#!/usr/bin/env bash
# ds4_backend_env.sh — single source of truth for routed-MoE prefill *compute
# backend* env blocks. Sourced by run_resident_variant_sweep.sh and
# tune_profile.sh so both scripts force the exact same backends.
#
# RESIDENT mode (full GGUF in RAM, --moe-mode off). The "common" block enables
# the resident MPP/NAX dedup prefill path; each backend then layers its toggles.
#
# Canonical backend names (and the env that forces each):
#   mulmm        upstream mul_mm_id GPU routed — pure ALU, NO resident envs.
#                This is the "ALU only / classic GPU baseline" (no NAX/matmul2d).
#   nax_int8     NAX int8 fused-dequant matmul2d (compact bridge). [common only]
#   nax_half     NAX half x half matmul2d (no fused gate+up).
#   nax_half_alu NAX-half "Plan A": fused gate+up+swiglu (NAX matmul2d + ALU).
#   pathc        Path C i8 in-kernel NAX∥ALU full-fused.
#   ane_gpu      ANE i8i8 + GPU(MPP-int8) scatter-gather hybrid.
#   ane_nax      ANE i8i8 + GPU(NAX-half) scatter-gather hybrid (concurrent).
#                Requires the DS4_RESIDENT_MOE_ANE_NAX_HYBRID build (Stage 2).
#
# Legacy aliases (run_resident_variant_sweep.sh history): int8->nax_int8,
# nax->nax_half_alu, alu->pathc.
#
# Functions:
#   ds4_backend_common_env            -> echoes the shared resident-enable block
#   ds4_backend_variant_env <name>    -> echoes the per-backend extra env, OR the
#                                        sentinel "MULMM" (run with NO resident
#                                        envs), OR "" for the common-only backend.
#   ds4_backend_is_ane <name>         -> exit 0 if the backend uses ANE
#   ds4_backend_canonical <name>      -> echoes the canonical name for an alias
#   ds4_backend_default_list          -> echoes the default M5 backend sweep list

# Shared resident MPP/NAX enablement (everything except per-backend toggles).
# Kept identical to run_resident_variant_sweep.sh's validated COMMON block.
ds4_backend_common_env() {
  cat <<'ENV'
DS4_RESIDENT_MOE_MPP_INT8_PREFILL=1
DS4_RESIDENT_MOE_MPP_FORCE=1
DS4_RESIDENT_MOE_MPP_MIN_TOKENS=64
DS4_RESIDENT_MOE_MPP_COMPACT_MIN_TOKENS=64
DS4_RESIDENT_MOE_MPP_COMPACT_BRIDGE=1
DS4_RESIDENT_MOE_MPP_FUSED_DEQUANT=1
ENV
}

# ANE knobs shared by the ANE backends. MIN_REFS is the swept knob (the caller
# may override via DS4_RESIDENT_MOE_ANE_MIN_REFS in the environment it passes).
# CRUCIAL: the resident ANE hybrid lives in metal_graph_resident_moe_run_mpp_prefill_dedup,
# which only runs when DS4_RESIDENT_MOE_MPP_DEDUP_PREFILL=1 (resident_moe_mpp_dedup_prefill_enabled,
# ds4.c). Without it the prefill falls through to the grouped MPP/NAX path and ANE never engages
# (the non-ANE backends above use that grouped path by design). This is NOT the SSD slot-bank
# flash ANE (DS4_FLASH_MOE_ANE_*) — that's a separate --moe-mode slot-bank path.
# DS4_FLASH_MOE_ANE_I8I8_PREFILL + ..._TILED_FUSED_PREFILL are the mode gate that the
# ANE start (ds4_gpu_routed_moe_expert_banked_batch_ane_start_tensor) checks; without both,
# every ANE start is rejected ("mode_gate") and the hybrid silently runs GPU-only.
ds4_backend_ane_common_env() {
  echo "DS4_RESIDENT_MOE_MPP_DEDUP_PREFILL=1 DS4_RESIDENT_MOE_ANE_HYBRID=1 DS4_RESIDENT_MOE_ANE_MAX_REFS=1024 DS4_RESIDENT_MOE_ANE_QUEUE=8 DS4_FLASH_MOE_ANE_I8I8_PREFILL=1 DS4_FLASH_MOE_ANE_I8I8_TILED_FUSED_PREFILL=1"
}

ds4_backend_canonical() {
  case "$1" in
    int8) echo "nax_int8" ;;
    nax)  echo "nax_half_alu" ;;
    alu)  echo "pathc" ;;
    *)    echo "$1" ;;
  esac
}

ds4_backend_is_ane() {
  case "$(ds4_backend_canonical "$1")" in
    ane_gpu|ane_alu|ane_nax) return 0 ;;
    *) return 1 ;;
  esac
}

# Echoes the per-backend extra env (beyond common). "MULMM" = run with NO
# resident envs at all (pure upstream GPU reference).
ds4_backend_variant_env() {
  case "$(ds4_backend_canonical "$1")" in
    mulmm)        echo "MULMM" ;;
    nax_int8)     echo "" ;;
    nax_half)     echo "DS4_RESIDENT_MOE_NAX_HALF=1" ;;
    nax_half_alu) echo "DS4_RESIDENT_MOE_NAX_HALF=1 DS4_RESIDENT_MOE_NAX_FUSED_GATE_UP=1 DS4_RESIDENT_MOE_NAX_FUSED_MIN_REFS=0" ;;
    pathc)        echo "DS4_RESIDENT_MOE_NAX_FULL_FUSED=1" ;;
    ane_gpu)      echo "$(ds4_backend_ane_common_env)" ;;
    ane_alu)      echo "$(ds4_backend_ane_common_env) DS4_RESIDENT_MOE_ANE_ALU_HYBRID=1" ;;
    ane_nax)      echo "$(ds4_backend_ane_common_env) DS4_RESIDENT_MOE_ANE_NAX_HYBRID=1 DS4_RESIDENT_MOE_NAX_HALF=1" ;;
    *)            echo "__UNKNOWN__" ;;
  esac
}

# Default backend sweep for NAX-capable (M5+) chips. ane_nax is intentionally
# omitted until the Stage-2 build lands; add it once DS4_RESIDENT_MOE_ANE_NAX_HYBRID
# is available (the binary silently no-ops the flag otherwise → duplicate of ane_gpu).
ds4_backend_default_list() {
  echo "mulmm nax_int8 nax_half nax_half_alu ane_gpu"
}
