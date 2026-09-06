/* Model-free regression coverage for the production slot request planner.
 * The bank models expert bytes separately from both residency mappings; only
 * uploads/commits are simulated here, never protection or victim selection. */
#include "../ssd/ssd_flash_moe_slots.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TEST_EXPERTS = 384, TEST_MAX_PAIRS = 2 * TEST_EXPERTS };

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__); \
        exit(1); \
    } \
} while (0)

typedef struct {
    uint32_t capacity;
    int32_t bytes[TEST_EXPERTS];
    int32_t slot_to_expert[TEST_EXPERTS];
    int32_t expert_to_slot[TEST_EXPERTS];
    uint64_t ages[TEST_EXPERTS];
    uint64_t age;
    uint64_t reads;
} test_bank;

typedef struct {
    int32_t slots[TEST_MAX_PAIRS];
    int32_t evicted[TEST_MAX_PAIRS];
    bool misses[TEST_MAX_PAIRS];
    bool reserved[TEST_EXPERTS];
} test_plan;

static void bank_init(test_bank *bank, uint32_t capacity) {
    CHECK(capacity > 0 && capacity <= TEST_EXPERTS, "invalid test capacity");
    memset(bank, 0, sizeof(*bank));
    bank->capacity = capacity;
    for (uint32_t i = 0; i < TEST_EXPERTS; i++) {
        bank->bytes[i] = -1;
        bank->slot_to_expert[i] = -1;
        bank->expert_to_slot[i] = -1;
    }
}

static bool bank_plan(const test_bank *bank, const int32_t *experts,
                      uint32_t count, const bool *hints, test_plan *plan) {
    CHECK(count <= TEST_MAX_PAIRS, "test request too large");
    return ds4_flash_moe_resolve_request_slots(experts, count, TEST_EXPERTS,
            bank->expert_to_slot, bank->slot_to_expert, bank->ages,
            bank->capacity, hints, plan->reserved, plan->slots,
            plan->evicted, plan->misses);
}

static void bank_verify(const test_bank *bank, const int32_t *experts,
                        uint32_t count, const test_plan *plan) {
    for (uint32_t i = 0; i < count; i++) {
        const int32_t slot = plan->slots[i];
        CHECK(slot >= 0 && (uint32_t)slot < bank->capacity, "slot outside bank");
        CHECK(bank->bytes[slot] == experts[i], "selected expert bytes overwritten");
        CHECK(bank->slot_to_expert[slot] == experts[i], "wrong slot owner");
        CHECK(bank->expert_to_slot[experts[i]] == slot, "wrong forward mapping");
        for (uint32_t prev = 0; prev < i; prev++) {
            CHECK(plan->slots[prev] != slot || experts[prev] == experts[i],
                  "distinct requested experts alias one slot");
        }
    }
    for (uint32_t slot = 0; slot < bank->capacity; slot++) {
        const int32_t expert = bank->slot_to_expert[slot];
        CHECK(bank->bytes[slot] == expert, "slot metadata does not match bytes");
        if (expert >= 0) {
            CHECK(bank->expert_to_slot[expert] == (int32_t)slot,
                  "reverse mapping inconsistent");
        }
    }
    for (uint32_t expert = 0; expert < TEST_EXPERTS; expert++) {
        const int32_t slot = bank->expert_to_slot[expert];
        CHECK(slot == -1 || (slot >= 0 && (uint32_t)slot < bank->capacity &&
              bank->slot_to_expert[slot] == (int32_t)expert),
              "forward mapping points at an evicted expert");
    }
}

static bool bank_run(test_bank *bank, const int32_t *experts, uint32_t count,
                     const bool *hints, test_plan *plan) {
    const test_bank before = *bank;
    memset(plan, 0, sizeof(*plan));
    const bool ok = bank_plan(bank, experts, count, hints, plan);
    CHECK(memcmp(bank, &before, sizeof(*bank)) == 0,
          "planning changed bank before full request resolution");
    if (!ok) return false;
    for (uint32_t i = 0; i < count; i++) {
        const int32_t slot = plan->slots[i];
        CHECK(slot >= 0 && (uint32_t)slot < bank->capacity, "bad planned slot");
        CHECK(plan->reserved[slot], "selected slot is not hard reserved");
        const int32_t resident = before.expert_to_slot[experts[i]];
        CHECK(resident < 0 || resident == slot,
              "same-request resident hit selected as a miss victim");
        for (uint32_t prev = 0; prev < i; prev++) {
            CHECK(plan->slots[prev] != slot || experts[prev] == experts[i],
                  "request contains a hit-after-miss slot alias");
            CHECK(experts[prev] != experts[i] || !plan->misses[i],
                  "duplicate expert scheduled another read");
        }
        if (plan->misses[i]) {
            CHECK(plan->evicted[i] == before.slot_to_expert[slot],
                  "victim does not match original slot owner");
        }
    }
    /* Simulate completed uploads only after every hit and miss has a slot.
     * Remove all old forward mappings before publishing new residents. */
    for (uint32_t i = 0; i < count; i++) {
        if (!plan->misses[i]) continue;
        const int32_t evicted = plan->evicted[i];
        if (evicted >= 0) bank->expert_to_slot[evicted] = -1;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!plan->misses[i]) continue;
        const int32_t slot = plan->slots[i];
        bank->bytes[slot] = experts[i];
        bank->slot_to_expert[slot] = experts[i];
        bank->expert_to_slot[experts[i]] = slot;
        bank->reads++;
    }
    for (uint32_t i = 0; i < count; i++) {
        bank->ages[plan->slots[i]] = ++bank->age;
    }
    bank_verify(bank, experts, count, plan);
    return true;
}

static void fill_bank(test_bank *bank, test_plan *plan) {
    int32_t initial[TEST_EXPERTS];
    for (uint32_t i = 0; i < bank->capacity; i++) initial[i] = (int32_t)i;
    CHECK(bank_run(bank, initial, bank->capacity, NULL, plan), "initial fill failed");
}

static void hit_after_miss_regression(void) {
    const uint32_t capacities[] = {8, 16, 96};
    for (uint32_t c = 0; c < sizeof(capacities) / sizeof(capacities[0]); c++) {
        test_bank bank;
        test_plan plan;
        bank_init(&bank, capacities[c]);
        fill_bank(&bank, &plan);
        bool hints[TEST_EXPERTS];
        memset(hints, 1, sizeof(hints));
        const int32_t request[] = {383, 2, 3, 4, 5, 6, 7, 0};
        /* All soft hints are set to force fallback victim selection. Hard
         * protection must still retain the later hit in the oldest slot. */
        CHECK(bank_run(&bank, request, 8, hints, &plan), "hit-after-miss failed");
        CHECK(bank.expert_to_slot[0] == 0, "oldest required hit was evicted");
        CHECK(bank.expert_to_slot[383] == 1, "wrong unrequested victim");
        CHECK(bank.reads == bank.capacity + 1u, "resident hit caused redundant read");
        const uint64_t reads = bank.reads;
        const int32_t duplicates[] = {0, 0, 383, 383, 2, 2, 3, 3};
        CHECK(bank_run(&bank, duplicates, 8, NULL, &plan), "duplicate hits failed");
        CHECK(bank.reads == reads, "duplicate hit caused a read");
        const int32_t duplicate_misses[] = {382, 0, 382, 383, 381, 381, 0, 383};
        CHECK(bank_run(&bank, duplicate_misses, 8, hints, &plan), "duplicate misses failed");
        CHECK(bank.reads == reads + 2u, "duplicate miss caused extra reads");
    }
}

static void capacity_and_resume(void) {
    test_bank bank;
    test_plan plan;
    bank_init(&bank, 8);
    fill_bank(&bank, &plan);
    const test_bank before = bank;
    const int32_t overflow[] = {383, 0, 1, 2, 3, 4, 5, 6, 7};
    CHECK(!bank_run(&bank, overflow, 9, NULL, &plan), "overflow was accepted");
    CHECK(memcmp(&bank, &before, sizeof(bank)) == 0, "overflow installed partial request");
    const int32_t recovery[] = {383, 2, 3, 4, 5, 6, 7, 0};
    CHECK(bank_run(&bank, recovery, 8, NULL, &plan), "request after overflow failed");
    const uint64_t reads = bank.reads;
    const int32_t full_hits[] = {0, 7, 6, 5, 4, 3, 2, 383, 0, 383};
    CHECK(bank_run(&bank, full_hits, 10, NULL, &plan), "full-bank hits failed");
    CHECK(bank.reads == reads, "full-bank duplicate hits need extra capacity");
    CHECK(bank_run(&bank, full_hits, 0, NULL, &plan), "empty request failed");

    /* Sequential prefill can stream more experts than the decode bank holds:
     * each separately completed request releases the previous reservations. */
    for (int32_t expert = 0; expert < TEST_EXPERTS; expert++) {
        CHECK(bank_run(&bank, &expert, 1, NULL, &plan), "sequential prefill failed");
    }
    const int32_t decode_after_prefill[] = {0, 376, 377, 378, 379, 380, 381, 383};
    CHECK(bank_run(&bank, decode_after_prefill, 8, NULL, &plan),
          "decode after sequential prefill failed");
}

static void invalid_requests_and_mappings(void) {
    test_bank bank;
    test_plan plan;
    bank_init(&bank, 8);
    fill_bank(&bank, &plan);
    const int32_t invalid[][2] = {{0, -1}, {0, TEST_EXPERTS}};
    for (uint32_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        memset(&plan, 0, sizeof(plan));
        plan.reserved[7] = true;
        const test_plan before_plan = plan;
        const test_bank before_bank = bank;
        CHECK(!bank_plan(&bank, invalid[i], 2, NULL, &plan), "invalid ID was accepted");
        CHECK(memcmp(&plan, &before_plan, sizeof(plan)) == 0,
              "invalid ID changed reservations or outputs");
        CHECK(memcmp(&bank, &before_bank, sizeof(bank)) == 0,
              "invalid ID changed bank");
    }
    const int32_t request[] = {0, 383};
    const int32_t stale_slots[] = {-2, 8, 0};
    for (uint32_t i = 0; i < sizeof(stale_slots) / sizeof(stale_slots[0]); i++) {
        bank.expert_to_slot[383] = stale_slots[i];
        memset(&plan, 0, sizeof(plan));
        const test_plan before_plan = plan;
        const test_bank before_bank = bank;
        CHECK(!bank_plan(&bank, request, 2, NULL, &plan), "stale mapping was accepted");
        CHECK(memcmp(&plan, &before_plan, sizeof(plan)) == 0,
              "stale mapping changed reservations or outputs");
        CHECK(memcmp(&bank, &before_bank, sizeof(bank)) == 0,
              "stale mapping changed bank");
    }
    bank.expert_to_slot[383] = -1;
    CHECK(bank_run(&bank, request, 2, NULL, &plan), "repaired mapping failed");
}

static void hard_reservations_and_soft_hints(void) {
    const int32_t owners[] = {-1, -1, 2, 3};
    const uint64_t ages[] = {0, 0, 1, 2};
    bool reserved[] = {true, false, false, false};
    CHECK(ds4_flash_moe_select_slot(4, TEST_EXPERTS, owners, ages, NULL, reserved) == 1,
          "empty-slot selection ignored hard reservation");
    memset(reserved, 1, sizeof(reserved));
    CHECK(ds4_flash_moe_select_slot(4, TEST_EXPERTS, owners, ages, NULL, reserved) == UINT32_MAX,
          "picker evicted a hard reservation");

    const int32_t full[] = {0, 1, 2, 3};
    const uint64_t old_ages[] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    bool hints[TEST_EXPERTS];
    memset(hints, 1, sizeof(hints));
    memset(reserved, 0, sizeof(reserved));
    reserved[0] = true;
    CHECK(ds4_flash_moe_select_slot(4, TEST_EXPERTS, full, old_ages, hints, reserved) == 1,
          "soft fallback ignored hard reservation or maximum age");
    hints[3] = false;
    CHECK(ds4_flash_moe_select_slot(4, TEST_EXPERTS, full, ages, hints, reserved) == 3,
          "soft prefetch hint stopped influencing victim choice");
    CHECK(ds4_flash_moe_select_slot(4, TEST_EXPERTS, full, ages, NULL, NULL) == 0,
          "single-expert prefill picker cannot release request reservations");
    CHECK(ds4_flash_moe_select_slot(0, TEST_EXPERTS, full, ages, NULL, NULL) == UINT32_MAX,
          "zero-capacity picker returned a slot");
}

static uint32_t random_state = 20260904u;

static uint32_t next_random(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

static void shuffle(int32_t *values, uint32_t count) {
    for (uint32_t i = count; i > 1; i--) {
        const uint32_t other = next_random() % i;
        const int32_t tmp = values[i - 1];
        values[i - 1] = values[other];
        values[other] = tmp;
    }
}

static void randomized_requests(void) {
    const uint32_t capacities[] = {6, 8, 96, TEST_EXPERTS};
    for (uint32_t c = 0; c < sizeof(capacities) / sizeof(capacities[0]); c++) {
        test_bank bank;
        test_plan plan;
        bank_init(&bank, capacities[c]);
        int32_t ids[TEST_EXPERTS];
        for (uint32_t i = 0; i < TEST_EXPERTS; i++) ids[i] = (int32_t)i;
        for (uint32_t step = 0; step < 10000; step++) {
            shuffle(ids, TEST_EXPERTS);
            const uint32_t unique = step % 3 == 0 ? bank.capacity :
                (bank.capacity < 8 ? bank.capacity : 8);
            int32_t request[TEST_MAX_PAIRS];
            memcpy(request, ids, unique * sizeof(ids[0]));
            uint32_t count = unique;
            for (uint32_t i = 0; i < unique; i++) {
                if ((next_random() & 3u) == 0) request[count++] = request[i];
            }
            shuffle(request, count);
            bool hints[TEST_EXPERTS];
            for (uint32_t i = 0; i < TEST_EXPERTS; i++) {
                hints[i] = step % 5 == 0 || (next_random() & 1u) != 0;
            }
            CHECK(bank_run(&bank, request, count, step % 2 ? hints : NULL, &plan),
                  "random valid request failed");
        }
    }
}

int main(void) {
    hit_after_miss_regression();
    capacity_and_resume();
    invalid_requests_and_mappings();
    hard_reservations_and_soft_hints();
    randomized_requests();
    puts("PASS: Flash-MoE hard hit protection, deferred installs, duplicates, overflow, "
         "stale mappings, prefill/decode reuse, and 40000 randomized requests");
    return 0;
}
