#ifndef DS4_FLASH_MOE_SLOTS_H
#define DS4_FLASH_MOE_SLOTS_H

#include <stdbool.h>
#include <stdint.h>

/* Validate the complete request before touching reservations. Resident inputs
 * are hard reservations, independent of the optional soft prefetch policy. */
static inline bool ds4_flash_moe_protect_request_slots(
        const int32_t *experts, uint32_t n_ids, uint32_t n_experts,
        const int32_t *expert_to_slot, const int32_t *slot_to_expert,
        uint32_t n_slots, bool *reserved_slots) {
    if (!experts || !expert_to_slot || !slot_to_expert || !reserved_slots ||
        n_slots == 0) return false;
    for (uint32_t i = 0; i < n_ids; i++) {
        const int32_t expert = experts[i];
        if (expert < 0 || (uint32_t)expert >= n_experts) return false;
        const int32_t slot = expert_to_slot[expert];
        if (slot < -1 || (slot >= 0 &&
            ((uint32_t)slot >= n_slots || slot_to_expert[slot] != expert))) {
            return false;
        }
    }
    for (uint32_t i = 0; i < n_ids; i++) {
        const int32_t slot = expert_to_slot[experts[i]];
        if (slot >= 0) reserved_slots[slot] = true;
    }
    return true;
}

/* Every pass skips hard reservations. Only prefetch hints may be ignored when
 * the bank has no unprotected victim. */
static inline uint32_t ds4_flash_moe_select_slot(
        uint32_t n_slots, uint32_t n_experts,
        const int32_t *slot_to_expert, const uint64_t *slot_age,
        const bool *protected_experts, const bool *reserved_slots) {
    for (uint32_t slot = 0; slot < n_slots; slot++) {
        if ((!reserved_slots || !reserved_slots[slot]) && slot_to_expert[slot] < 0) {
            return slot;
        }
    }
    for (uint32_t pass = 0; pass < 2; pass++) {
        uint32_t victim = UINT32_MAX;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            if (reserved_slots && reserved_slots[slot]) continue;
            const int32_t expert = slot_to_expert[slot];
            if (pass == 0 && protected_experts && expert >= 0 &&
                (uint32_t)expert < n_experts && protected_experts[expert]) continue;
            if (victim == UINT32_MAX || slot_age[slot] < oldest) {
                victim = slot;
                oldest = slot_age[slot];
            }
        }
        if (victim != UINT32_MAX) return victim;
    }
    return UINT32_MAX;
}

/* Resolve the entire request without changing mappings or slot bytes. The
 * caller installs only after success. Duplicate references share a slot and
 * only their first occurrence can be a miss. reserved_slots is caller-owned
 * scratch, initially clear for this request. */
static inline bool ds4_flash_moe_resolve_request_slots(
        const int32_t *experts, uint32_t n_ids, uint32_t n_experts,
        const int32_t *expert_to_slot, const int32_t *slot_to_expert,
        const uint64_t *slot_age, uint32_t n_slots,
        const bool *protected_experts, bool *reserved_slots,
        int32_t *slot_ids, int32_t *evicted, bool *misses) {
    if (!slot_age || !slot_ids || !evicted || !misses ||
        !ds4_flash_moe_protect_request_slots(experts, n_ids, n_experts,
                expert_to_slot, slot_to_expert, n_slots, reserved_slots)) return false;
    for (uint32_t i = 0; i < n_ids; i++) {
        slot_ids[i] = -1;
        evicted[i] = -1;
        misses[i] = false;
        for (uint32_t prev = 0; prev < i; prev++) {
            if (experts[prev] == experts[i]) {
                slot_ids[i] = slot_ids[prev];
                break;
            }
        }
        if (slot_ids[i] >= 0) continue;
        const int32_t resident = expert_to_slot[experts[i]];
        if (resident >= 0) {
            slot_ids[i] = resident;
            continue;
        }
        const uint32_t slot = ds4_flash_moe_select_slot(n_slots, n_experts,
                slot_to_expert, slot_age, protected_experts, reserved_slots);
        if (slot == UINT32_MAX) return false;
        reserved_slots[slot] = true;
        slot_ids[i] = (int32_t)slot;
        evicted[i] = slot_to_expert[slot];
        misses[i] = true;
    }
    return true;
}

#endif
