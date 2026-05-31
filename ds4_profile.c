/*
 * Machine tuning profile loader. See ds4_profile.h for the schema and search
 * order. Self-contained: a tiny JSON parser (no external dependency) plus
 * sysctl-based device identification, applied via setenv(overwrite=0).
 */
#include "ds4_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#include <mach-o/dyld.h>
#endif

/* ------------------------------------------------------------------ */
/* Minimal JSON parser (objects, arrays, strings, numbers, bool, null) */
/* ------------------------------------------------------------------ */

static char *pstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

typedef enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ } jtype;

typedef struct jval jval;
struct jval {
    jtype t;
    double num;          /* JNUM, JBOOL (0/1) */
    char  *str;          /* JSTR (owned, NUL-terminated) */
    jval **items; int nitems;            /* JARR */
    char **keys; jval **vals; int nkeys; /* JOBJ */
};

typedef struct {
    const char *p;
    const char *end;
    bool ok;
} jcur;

static jval *jparse_value(jcur *c);

static void jskip_ws(jcur *c) {
    while (c->p < c->end && isspace((unsigned char)*c->p)) c->p++;
}

static void jfree(jval *v) {
    if (!v) return;
    if (v->t == JSTR) free(v->str);
    else if (v->t == JARR) {
        for (int i = 0; i < v->nitems; i++) jfree(v->items[i]);
        free(v->items);
    } else if (v->t == JOBJ) {
        for (int i = 0; i < v->nkeys; i++) { free(v->keys[i]); jfree(v->vals[i]); }
        free(v->keys); free(v->vals);
    }
    free(v);
}

static char *jparse_raw_string(jcur *c) {
    if (c->p >= c->end || *c->p != '"') { c->ok = false; return NULL; }
    c->p++;
    size_t cap = 16, len = 0;
    char *s = malloc(cap);
    if (!s) { c->ok = false; return NULL; }
    while (c->p < c->end && *c->p != '"') {
        char ch = *c->p++;
        if (ch == '\\' && c->p < c->end) {
            char e = *c->p++;
            switch (e) {
                case 'n': ch = '\n'; break;
                case 't': ch = '\t'; break;
                case 'r': ch = '\r'; break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case '/': ch = '/';  break;
                case '\\': ch = '\\'; break;
                case '"': ch = '"';  break;
                case 'u': /* skip \uXXXX -> '?' (env values never need it) */
                    for (int i = 0; i < 4 && c->p < c->end; i++) c->p++;
                    ch = '?';
                    break;
                default: ch = e; break;
            }
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char *ns = realloc(s, cap);
            if (!ns) { free(s); c->ok = false; return NULL; }
            s = ns;
        }
        s[len++] = ch;
    }
    if (c->p >= c->end || *c->p != '"') { free(s); c->ok = false; return NULL; }
    c->p++; /* closing quote */
    s[len] = '\0';
    return s;
}

static jval *jnew(jtype t) {
    jval *v = calloc(1, sizeof(*v));
    if (v) v->t = t;
    return v;
}

static jval *jparse_object(jcur *c) {
    c->p++; /* { */
    jval *v = jnew(JOBJ);
    if (!v) { c->ok = false; return NULL; }
    jskip_ws(c);
    if (c->p < c->end && *c->p == '}') { c->p++; return v; }
    int cap = 0;
    for (;;) {
        jskip_ws(c);
        char *key = jparse_raw_string(c);
        if (!c->ok) { jfree(v); return NULL; }
        jskip_ws(c);
        if (c->p >= c->end || *c->p != ':') { free(key); jfree(v); c->ok = false; return NULL; }
        c->p++;
        jval *val = jparse_value(c);
        if (!c->ok) { free(key); jfree(val); jfree(v); return NULL; }
        if (v->nkeys >= cap) {
            cap = cap ? cap * 2 : 8;
            char **nk = realloc(v->keys, (size_t)cap * sizeof(char *));
            jval **nv = realloc(v->vals, (size_t)cap * sizeof(jval *));
            if (!nk || !nv) { free(nk ? nk : v->keys); free(key); jfree(val); jfree(v); c->ok = false; return NULL; }
            v->keys = nk; v->vals = nv;
        }
        v->keys[v->nkeys] = key;
        v->vals[v->nkeys] = val;
        v->nkeys++;
        jskip_ws(c);
        if (c->p < c->end && *c->p == ',') { c->p++; continue; }
        if (c->p < c->end && *c->p == '}') { c->p++; break; }
        jfree(v); c->ok = false; return NULL;
    }
    return v;
}

static jval *jparse_array(jcur *c) {
    c->p++; /* [ */
    jval *v = jnew(JARR);
    if (!v) { c->ok = false; return NULL; }
    jskip_ws(c);
    if (c->p < c->end && *c->p == ']') { c->p++; return v; }
    int cap = 0;
    for (;;) {
        jval *item = jparse_value(c);
        if (!c->ok) { jfree(item); jfree(v); return NULL; }
        if (v->nitems >= cap) {
            cap = cap ? cap * 2 : 8;
            jval **ni = realloc(v->items, (size_t)cap * sizeof(jval *));
            if (!ni) { jfree(item); jfree(v); c->ok = false; return NULL; }
            v->items = ni;
        }
        v->items[v->nitems++] = item;
        jskip_ws(c);
        if (c->p < c->end && *c->p == ',') { c->p++; continue; }
        if (c->p < c->end && *c->p == ']') { c->p++; break; }
        jfree(v); c->ok = false; return NULL;
    }
    return v;
}

static jval *jparse_value(jcur *c) {
    jskip_ws(c);
    if (c->p >= c->end) { c->ok = false; return NULL; }
    char ch = *c->p;
    if (ch == '{') return jparse_object(c);
    if (ch == '[') return jparse_array(c);
    if (ch == '"') {
        char *s = jparse_raw_string(c);
        if (!c->ok) return NULL;
        jval *v = jnew(JSTR);
        if (!v) { free(s); c->ok = false; return NULL; }
        v->str = s;
        return v;
    }
    if (ch == 't' || ch == 'f') {
        if ((size_t)(c->end - c->p) >= 4 && strncmp(c->p, "true", 4) == 0) { c->p += 4; jval *v = jnew(JBOOL); if (v) v->num = 1; else c->ok = false; return v; }
        if ((size_t)(c->end - c->p) >= 5 && strncmp(c->p, "false", 5) == 0) { c->p += 5; jval *v = jnew(JBOOL); if (v) v->num = 0; else c->ok = false; return v; }
        c->ok = false; return NULL;
    }
    if (ch == 'n') {
        if ((size_t)(c->end - c->p) >= 4 && strncmp(c->p, "null", 4) == 0) { c->p += 4; return jnew(JNULL); }
        c->ok = false; return NULL;
    }
    if (ch == '-' || (ch >= '0' && ch <= '9')) {
        char *e = NULL;
        double d = strtod(c->p, &e);
        if (e == c->p) { c->ok = false; return NULL; }
        c->p = e;
        jval *v = jnew(JNUM);
        if (!v) { c->ok = false; return NULL; }
        v->num = d;
        return v;
    }
    c->ok = false;
    return NULL;
}

static jval *jobj_get(const jval *o, const char *key) {
    if (!o || o->t != JOBJ) return NULL;
    for (int i = 0; i < o->nkeys; i++) {
        if (strcmp(o->keys[i], key) == 0) return o->vals[i];
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Device identity + file IO                                          */
/* ------------------------------------------------------------------ */

static void device_chip(char *buf, size_t bufsz) {
    buf[0] = '\0';
#if defined(__APPLE__)
    size_t len = bufsz;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, NULL, 0) != 0) buf[0] = '\0';
    buf[bufsz - 1] = '\0';
#endif
}

static uint64_t device_ram_bytes(void) {
#if defined(__APPLE__)
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) == 0 && len == sizeof(bytes)) return bytes;
#endif
    return 0;
}

static char *read_file(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long sz = ftell(fp);
    if (sz < 0 || sz > (16 * 1024 * 1024)) { fclose(fp); return NULL; }
    rewind(fp);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[got] = '\0';
    if (out_len) *out_len = got;
    return buf;
}

/* Resolve the profile file path. Returns a malloc'd string or NULL. */
static char *resolve_profile_path(void) {
    const char *env = getenv("DS4_PROFILE");
    if (env && env[0]) {
        if (strcmp(env, "none") == 0 || strcmp(env, "0") == 0) return NULL; /* explicitly disabled */
        return pstrdup(env);
    }
    /* ./ds4_profile.json */
    {
        FILE *fp = fopen("ds4_profile.json", "rb");
        if (fp) { fclose(fp); return pstrdup("ds4_profile.json"); }
    }
    /* next to the executable */
#if defined(__APPLE__)
    {
        char exe[4096];
        uint32_t sz = sizeof(exe);
        if (_NSGetExecutablePath(exe, &sz) == 0) {
            const char *slash = strrchr(exe, '/');
            if (slash) {
                size_t dirlen = (size_t)(slash - exe) + 1;
                char path[4096];
                if (dirlen + strlen("ds4_profile.json") < sizeof(path)) {
                    memcpy(path, exe, dirlen);
                    strcpy(path + dirlen, "ds4_profile.json");
                    FILE *fp = fopen(path, "rb");
                    if (fp) { fclose(fp); return pstrdup(path); }
                }
            }
        }
    }
#endif
    /* ~/.config/ds4/ds4_profile.json */
    {
        const char *home = getenv("HOME");
        if (home && home[0]) {
            char path[4096];
            int n = snprintf(path, sizeof(path), "%s/.config/ds4/ds4_profile.json", home);
            if (n > 0 && (size_t)n < sizeof(path)) {
                FILE *fp = fopen(path, "rb");
                if (fp) { fclose(fp); return pstrdup(path); }
            }
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Match + apply                                                      */
/* ------------------------------------------------------------------ */

static bool profile_matches(const jval *match, const char *chip, uint64_t ram_bytes) {
    if (!match || match->t != JOBJ) return true; /* empty match = wildcard */
    const jval *jchip = jobj_get(match, "chip");
    if (jchip && jchip->t == JSTR && jchip->str[0]) {
        if (!chip || !strstr(chip, jchip->str)) return false; /* substring */
    }
    const jval *jram = jobj_get(match, "min_ram_gib");
    if (jram && jram->t == JNUM) {
        double have_gib = (double)ram_bytes / (1024.0 * 1024.0 * 1024.0);
        if (have_gib + 0.5 < jram->num) return false; /* small slack */
    }
    return true;
}

void ds4_profile_load_and_apply(void) {
    static bool done = false;
    if (done) return;
    done = true;

    char *path = resolve_profile_path();
    if (!path) return;

    size_t len = 0;
    char *text = read_file(path, &len);
    if (!text) { free(path); return; }

    jcur c = { .p = text, .end = text + len, .ok = true };
    jval *root = jparse_value(&c);
    if (!c.ok || !root || root->t != JOBJ) {
        fprintf(stderr, "ds4: profile %s: parse error (ignored)\n", path);
        jfree(root); free(text); free(path);
        return;
    }

    const jval *profiles = jobj_get(root, "profiles");
    if (!profiles || profiles->t != JARR) {
        jfree(root); free(text); free(path);
        return;
    }

    char chip[256];
    device_chip(chip, sizeof(chip));
    uint64_t ram = device_ram_bytes();

    for (int i = 0; i < profiles->nitems; i++) {
        const jval *prof = profiles->items[i];
        if (!prof || prof->t != JOBJ) continue;
        if (!profile_matches(jobj_get(prof, "match"), chip, ram)) continue;

        const jval *env = jobj_get(prof, "env");
        int applied = 0, skipped = 0;
        if (env && env->t == JOBJ) {
            for (int k = 0; k < env->nkeys; k++) {
                const jval *val = env->vals[k];
                if (!val) continue;
                char numbuf[64];
                const char *sval = NULL;
                if (val->t == JSTR) sval = val->str;
                else if (val->t == JBOOL) sval = val->num != 0 ? "1" : "0";
                else if (val->t == JNUM) {
                    /* integer-print when whole, else %g */
                    if (val->num == (double)(long long)val->num)
                        snprintf(numbuf, sizeof(numbuf), "%lld", (long long)val->num);
                    else
                        snprintf(numbuf, sizeof(numbuf), "%g", val->num);
                    sval = numbuf;
                } else continue;
                if (getenv(env->keys[k]) != NULL) { skipped++; continue; } /* user env wins */
                if (setenv(env->keys[k], sval, 0) == 0) applied++;
            }
        }
        fprintf(stderr,
                "ds4: applied tuning profile [%s] from %s (%d env defaults set, %d kept from environment)\n",
                chip[0] ? chip : "unknown-device", path, applied, skipped);
        break; /* first matching profile wins */
    }

    jfree(root);
    free(text);
    free(path);
}
