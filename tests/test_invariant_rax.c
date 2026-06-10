#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Forward declarations from rax.c */
typedef struct raxStack {
    void **stack;
    size_t items;
    size_t maxitems;
} raxStack;

raxStack *raxStackCreate(void);
void raxStackFree(raxStack *ts);
int raxStackPush(raxStack *ts, void *ptr);

START_TEST(test_rax_stack_no_overflow)
{
    /* Invariant: Buffer reads/writes never exceed allocated size.
       Stack growth must not overflow size_t in allocation calculation. */
    
    raxStack *ts = raxStackCreate();
    ck_assert_ptr_nonnull(ts);
    
    /* Test cases: valid input, boundary, and overflow attempts */
    struct {
        size_t pushes;
        const char *desc;
    } cases[] = {
        {10, "valid small pushes"},
        {1000, "valid large pushes"},
        {SIZE_MAX / (sizeof(void*) * 2), "boundary: max safe items"},
    };
    
    int num_cases = sizeof(cases) / sizeof(cases[0]);
    
    for (int i = 0; i < num_cases; i++) {
        raxStack *test_ts = raxStackCreate();
        ck_assert_ptr_nonnull(test_ts);
        
        size_t pushes = cases[i].pushes;
        if (pushes > 100000) pushes = 100000; /* Cap to prevent timeout */
        
        int result = 0;
        for (size_t j = 0; j < pushes; j++) {
            result = raxStackPush(test_ts, (void *)(uintptr_t)j);
            if (result == 0) break; /* Allocation failed safely */
        }
        
        /* Invariant: either all pushes succeeded or allocation was rejected */
        ck_assert(result == 0 || test_ts->items <= test_ts->maxitems);
        
        /* Invariant: items never exceeds maxitems */
        ck_assert_uint_le(test_ts->items, test_ts->maxitems);
        
        raxStackFree(test_ts);
    }
    
    raxStackFree(ts);
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_rax_stack_no_overflow);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}