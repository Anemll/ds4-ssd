/* Opt-in native DS4 sidecar smoke: no downloads and no default model.
 * Disable direct mmap auto-selection to exercise real six-slot eviction.
 * Checks successful transitions and repeat-prefill argmax, not full numerical
 * equivalence or injected interruption (covered by the I/O test). */
#include "../ds4.h"
#include <stdio.h>
#include <stdlib.h>

static void check(int ok, const char *what, const char *err) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s: %s\n", what, err);
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s /path/to/DS4-sidecar-package\n", argv[0]);
        return 2;
    }
    ds4_engine_options opt = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .moe_mode = DS4_MOE_MODE_SLOT_BANK, .moe_slot_bank = 6,
        .moe_slot_bank_explicit = true, .ctx_size = 512,
        .no_int8 = true,
    };
    check(ds4_engine_options_autodetect_sidecar_package(&opt, argv[0]),
          "sidecar package", argv[1]);
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    char err[512] = {0};
    check(ds4_engine_open(&engine, &opt) == 0, "engine open", err);
    check(ds4_session_create(&session, engine, 512) == 0, "session create", err);
    ds4_tokens prompt = {0}, resumed = {0};
    ds4_tokenize_text(engine,
        "A slot cache stores expert weights. Every requested resident must remain "
        "available until the routed calculation finishes. Explain this rule.", &prompt);
    check(prompt.len > 5, "prompt length", err);
    check(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0, "initial prefill", err);
    ds4_runtime_status status = {0};
    check(ds4_session_runtime_status(session, &status) == 1 &&
          status.available && status.moe_slot_bank == 6,
          "six-slot bank required (disable direct mmap auto)", err);
    const int expected = ds4_session_argmax(session);
    for (int i = 0; i < 4; i++) {
        int token = ds4_session_argmax(session);
        check(ds4_session_eval(session, token, err, sizeof(err)) == 0, "decode", err);
    }
    ds4_tokens_copy(&resumed, ds4_session_tokens(session));
    ds4_tokenize_text(engine, " Continue with a concrete example of a cache miss followed by a hit.", &resumed);
    check(ds4_session_sync(session, &resumed, err, sizeof(err)) == 0,
          "decode to resumed prefill", err);
    int token = ds4_session_argmax(session);
    check(ds4_session_eval(session, token, err, sizeof(err)) == 0, "resumed decode", err);
    ds4_session_invalidate(session);
    check(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0,
          "reuse after invalidate", err);
    check(ds4_session_argmax(session) == expected, "repeat prefill argmax", err);
    printf("PASS: six-slot DS4 sidecar prefill (%d tokens), decode, resumed prefill, "
           "invalidate and repeat argmax=%d\n", prompt.len, expected);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&resumed);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}
