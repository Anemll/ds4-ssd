/* =========================================================================
 * ssd_flash_moe_streaming.c - SSD Flash-MoE implementation aggregator.
 * =========================================================================
 *
 * This file is included by ds4.c so the refactor can keep ds4_gpu_graph,
 * ds4_model, and ds4_weights private to one translation unit while the actual
 * Flash-MoE implementation lives in focused include files.
 */

#include "ssd_flash_moe_allocation.c"
#include "ssd_flash_moe_runtime.c"
#include "ssd_flash_moe_resident_prefill.c"
#include "ssd_flash_moe_slot_cache.c"
#include "ssd_flash_moe_decode.c"
#include "ssd_flash_moe_prefill.c"
