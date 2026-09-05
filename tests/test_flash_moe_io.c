/* Model-free lifecycle regression using the production Flash-MoE implementation.
 * Link with the normal GPU support objects; these tests do not create a device.
 * The pread shim holds an actual worker at a deterministic I/O boundary. */
#include <assert.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

static pthread_mutex_t io_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t io_cv = PTHREAD_COND_INITIALIZER;
static int io_entered;
static int io_release;
#define TEST_IO_FD (-42)

static ssize_t test_pread(int fd, void *buf, size_t count, off_t offset) {
    if (fd != TEST_IO_FD) return pread(fd, buf, count, offset);
    pthread_mutex_lock(&io_mu);
    io_entered++;
    pthread_cond_broadcast(&io_cv);
    while (!io_release) pthread_cond_wait(&io_cv, &io_mu);
    pthread_mutex_unlock(&io_mu);
    memset(buf, 0xa5, count);
    return (ssize_t)count;
}

#define pread test_pread
#include "../ds4.c"
#undef pread

static void wait_for_io(void) {
    pthread_mutex_lock(&io_mu);
    while (!io_entered) pthread_cond_wait(&io_cv, &io_mu);
    pthread_mutex_unlock(&io_mu);
}

static void release_io(void) {
    pthread_mutex_lock(&io_mu);
    io_release = 1;
    pthread_cond_broadcast(&io_cv);
    pthread_mutex_unlock(&io_mu);
}

typedef struct {
    ds4_flash_prefill_async_reader *reader;
    int returned;
} drain_job;

static void *drain_reader(void *arg) {
    drain_job *job = arg;
    ds4_flash_prefill_async_cancel_and_drain(job->reader);
    __atomic_store_n(&job->returned, 1, __ATOMIC_RELEASE);
    return NULL;
}

static void test_prefill_drain(void) {
    ds4_flash_prefill_async_reader *reader = calloc(1, sizeof(*reader));
    assert(reader && ds4_flash_prefill_async_init(reader, 64));
    assert(ds4_flash_prefill_async_submit(reader, 0, 1, 0, 1,
                                         TEST_IO_FD, 0, 64, NULL, false));
    wait_for_io();
    /* A second speculative request remains queued while its reader is paused. */
    ds4_flash_prefill_async_set_paused(reader, 1);
    assert(ds4_flash_prefill_async_submit(reader, 1, 2, 0, 1,
                                         TEST_IO_FD, 0, 64, NULL, true));
    drain_job job = { .reader = reader };
    pthread_t drain;
    assert(pthread_create(&drain, NULL, drain_reader, &job) == 0);
    pthread_mutex_lock(&reader->mu);
    while (!reader->slots[0].canceled) pthread_cond_wait(&reader->cv, &reader->mu);
    /* Cancellation is visible, but the blocked worker still owns its buffer. */
    assert(!__atomic_load_n(&job.returned, __ATOMIC_ACQUIRE));
    pthread_mutex_unlock(&reader->mu);
    release_io();
    assert(pthread_join(drain, NULL) == 0);
    assert(job.returned && !reader->xlayer_paused);
    for (int i = 0; i < DS4_FLASH_PREFILL_ASYNC_SLOTS; i++) {
        assert(reader->slots[i].state == DS4_FLASH_ASYNC_EMPTY);
    }
    /* A split read paused between chunks has no active worker to signal its
     * cancellation; draining must reclaim it without waiting for that signal. */
    pthread_mutex_lock(&reader->mu);
    reader->xlayer_paused = 1;
    reader->slots[0].state = DS4_FLASH_ASYNC_READING;
    reader->slots[0].speculative = true;
    reader->slots[0].nsplit = 4;
    reader->slots[0].chunks_claimed = reader->slots[0].chunks_done = 1;
    pthread_mutex_unlock(&reader->mu);
    ds4_flash_prefill_async_cancel_and_drain(reader);
    assert(reader->slots[0].state == DS4_FLASH_ASYNC_EMPTY);
    /* The same pool must accept another request after interruption. */
    assert(ds4_flash_prefill_async_submit(reader, 0, 3, 0, 1,
                                         TEST_IO_FD, 0, 64, NULL, false));
    pthread_mutex_lock(&reader->mu);
    while (reader->slots[0].state != DS4_FLASH_ASYNC_READY)
        pthread_cond_wait(&reader->cv, &reader->mu);
    pthread_mutex_unlock(&reader->mu);
    ds4_flash_prefill_async_cancel_and_drain(reader);
    assert(reader->slots[0].state == DS4_FLASH_ASYNC_EMPTY);
    ds4_flash_prefill_async_destroy(reader);
    free(reader);
}

typedef struct {
    ds4_flash_decode_prefetch *pf;
    int returned;
} cleanup_job;

static void *cleanup_prefetch(void *arg) {
    cleanup_job *job = arg;
    metal_graph_flash_moe_decode_prefetch_cleanup(NULL, 0, job->pf);
    __atomic_store_n(&job->returned, 1, __ATOMIC_RELEASE);
    return NULL;
}

static void test_scratch_stop(void) {
    pthread_mutex_lock(&io_mu);
    io_entered = io_release = 0;
    pthread_mutex_unlock(&io_mu);
    const size_t bytes = 2u << 20;
    uint8_t *buf = calloc(1, bytes);
    assert(buf);
    ds4_flash_decode_prefetch pf = {
        .active = true, .sequential_scratch = true, .n_loads = 2
    };
    for (uint32_t i = 0; i < pf.n_loads; i++) {
        pf.job[i].fd = TEST_IO_FD;
        pf.job[i].bytes = bytes;
        pf.job[i].buf = buf;
    }
    assert(pthread_create(&pf.thread[0], NULL,
                          ds4_flash_decode_scratch_prefetch_worker, &pf) == 0);
    pf.thread_started[0] = true;
    wait_for_io();
    cleanup_job job = { .pf = &pf };
    pthread_t cleanup;
    assert(pthread_create(&cleanup, NULL, cleanup_prefetch, &job) == 0);
    while (!__atomic_load_n(&pf.stop_requested, __ATOMIC_ACQUIRE)) sched_yield();
    assert(!__atomic_load_n(&job.returned, __ATOMIC_ACQUIRE));
    release_io();
    assert(pthread_join(cleanup, NULL) == 0);
    assert(job.returned && !pf.active && !pf.thread_started[0]);
    assert(pf.job[0].canceled && pf.job[1].canceled);
    assert(buf[0] == 0xa5 && buf[1u << 20] == 0); /* no next chunk/job read */
    free(buf);
}

static void init_bank(ds4_gpu_graph *g, ds4_flash_moe_sidecar *sidecar,
                      int32_t *s2e, int32_t *e2s, uint64_t *age) {
    memset(g, 0, sizeof(*g));
    g->flash_moe = sidecar;
    g->flash_slot_bank = 2;
    g->flash_slot_to_expert = s2e;
    g->flash_expert_to_slot = e2s;
    g->flash_slot_age = age;
    for (uint32_t e = 0; e < DS4_N_EXPERT; e++) e2s[e] = -1;
    s2e[0] = 5; s2e[1] = 6;
    e2s[5] = 0; e2s[6] = 1;
    age[0] = 1; age[1] = 2;
    g->flash_decode_ids_valid[0] = 1;
}

static void test_direct_read_failures(void) {
    ds4_gpu_graph *g = calloc(1, sizeof(*g));
    ds4_flash_moe_sidecar *sidecar = calloc(1, sizeof(*sidecar));
    int32_t s2e[2], e2s[DS4_MAX_EXPERT];
    uint64_t age[2];
    uint8_t destinations[2][8] = {{0}};
    assert(g && sidecar);
    char path[] = "/tmp/ds4-flash-io-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0 && unlink(path) == 0);
    assert(write(fd, "abcd", 4) == 4);
    init_bank(g, sidecar, s2e, e2s, age);
    ds4_flash_decode_prefetch pf = { .active = true, .n_loads = 2 };
    for (uint32_t i = 0; i < pf.n_loads; i++) {
        pf.load_slot[i] = (int32_t)i;
        pf.load_expert[i] = (int32_t)(10 + i);
        pf.job[i].direct_record = true;
        pf.job[i].record_dst = destinations[i];
        pf.job[i].fd = fd;
        pf.job[i].bytes = i == 0 ? 8 : 4;  /* partial read / success */
        pf.job[i].io_split = 1;
        assert(pthread_create(&pf.thread[i], NULL, ds4_flash_decode_read_thread,
                              &pf.job[i]) == 0);
        pf.thread_started[i] = true;
    }
    assert(!metal_graph_flash_moe_decode_prefetch_finish(g, 0, &pf));
    assert(memcmp(destinations[0], "abcd", 4) == 0); /* victim was overwritten */
    assert(!pf.active && !pf.thread_started[0] && !pf.thread_started[1]);
    assert(s2e[0] == -1 && s2e[1] == -1 && e2s[5] == -1 && e2s[6] == -1);
    assert(!g->flash_decode_ids_valid[0]);
    /* Abandoning a successful direct read also cannot retain the old mapping. */
    init_bank(g, sidecar, s2e, e2s, age);
    memset(&pf, 0, sizeof(pf));
    pf.active = true;
    pf.n_loads = 1;
    pf.load_slot[0] = 0;
    pf.job[0].direct_record = true;
    pf.job[0].record_dst = destinations[0];
    pf.job[0].fd = fd;
    pf.job[0].bytes = 4;
    assert(pthread_create(&pf.thread[0], NULL, ds4_flash_decode_read_thread,
                          &pf.job[0]) == 0);
    pf.thread_started[0] = true;
    metal_graph_flash_moe_decode_prefetch_cleanup(g, 0, &pf);
    assert(!pf.active && !pf.thread_started[0]);
    assert(s2e[0] == -1 && e2s[5] == -1 && s2e[1] == 6);
    assert(!g->flash_decode_ids_valid[0]);
    /* Async handout cleanup preserves committed slots and invalidates direct
     * destinations whose loads were never committed. */
    init_bank(g, sidecar, s2e, e2s, age);
    ds4_flash_decode_async_load loads[2] = {{0}};
    loads[0].slot = 0;
    loads[0].uploaded = true;
    loads[0].job.direct_record = true;
    loads[1].slot = 1;
    loads[1].job.direct_record = true;
    loads[1].job.record_dst = destinations[1];
    loads[1].job.fd = fd;
    loads[1].job.bytes = 8;
    assert(pthread_create(&loads[1].thread, NULL, ds4_flash_decode_read_thread,
                          &loads[1].job) == 0);
    loads[1].thread_started = true;
    metal_graph_flash_moe_async_load_cleanup(g, 0, loads, 2);
    assert(loads[1].joined && s2e[0] == 5 && s2e[1] == -1 && e2s[6] == -1);
    assert(!g->flash_decode_ids_valid[0]);
    close(fd);
    free(sidecar);
    free(g);
}

/* HY4 consumes host slot IDs without requesting the DS4 replay/grouped
 * kernels. Previously the recorder silently skipped this native path,
 * leaving every selected expert bound to zero-initialized slot zero. */
static void test_hy4_native_decode_slot_ids(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_HY4;
    ds4_gpu_graph *g = calloc(1, sizeof(*g));
    ds4_flash_moe_sidecar *sidecar = calloc(1, sizeof(*sidecar));
    assert(g && sidecar);
    g->flash_moe = sidecar;
    g->flash_slot_bank = 8;
    g->flash_mixed_slot_bank = true;
    const uint32_t layer = 1;
    const int32_t ids[8] = {250, 65, 132, 66, 241, 199, 90, 137};
    const int32_t slots[8] = {3, 0, 7, 4, 1, 6, 5, 2};
    assert(!flash_moe_replay_plan_enabled());
    assert(!flash_moe_mixed_slots6_grouped_enabled());
    for (unsigned pass = 0; pass < 2; ++pass) {
        int32_t requested[8], mapped[8];
        for (unsigned k = 0; k < 8; ++k) {
            requested[k] = ids[(k + pass) % 8];
            mapped[k] = slots[(k + pass) % 8];
        }
        g->flash_decode_ids_valid[layer] = 0;
        metal_graph_flash_moe_record_decode_slots(g, layer, requested, mapped, 8);
        assert(g->flash_decode_ids_valid[layer]);
        for (unsigned k = 0; k < 8; ++k) {
            assert(g->flash_decode_true_ids[layer][k] == requested[k]);
            assert(g->flash_decode_slot_ids[layer][k] == mapped[k]);
            for (unsigned j = 0; j < k; ++j)
                assert(g->flash_decode_slot_ids[layer][k] != g->flash_decode_slot_ids[layer][j]);
        }
    }
    free(sidecar);
    free(g);
    g_ds4_shape = saved_shape;
}

int main(void) {
    alarm(20); /* A lifecycle regression should fail instead of hanging CI. */
    setenv("DS4_FLASH_MOE_BAKED_SLOT_DECODE", "0", 1);
    setenv("DS4_FLASH_MOE_STABLE_REPLAY", "0", 1);
    setenv("DS4_FLASH_MOE_MIXED_SLOTS6_GROUPED", "0", 1);
    setenv("DS4_FLASH_MOE_MIXED_SLOTS6", "0", 1);
    setenv("DS4_FLASH_MOE_PREAD_THREADS", "1", 1);
    setenv("DS4_FLASH_MOE_PREFILL_IO_SPLIT", "1", 1);
    test_hy4_native_decode_slot_ids();
    test_prefill_drain();
    test_scratch_stop();
    test_direct_read_failures();
    puts("PASS Flash-MoE I/O: blocked reader drain, scratch stop, paused split, reuse, direct-read invalidation, HY4 native eight-slot IDs");
    return 0;
}
