#ifndef DS4_PROFILE_H
#define DS4_PROFILE_H

#include <stdbool.h>

/*
 * Machine tuning profiles.
 *
 * A profile is a per-device map of environment-variable defaults. At startup
 * ds4_profile_load_and_apply() reads a JSON profile file, finds the entry whose
 * `match` (chip substring + minimum RAM) fits this machine, and applies its
 * `env` map with setenv(overwrite=0) -- so the profile sets defaults that flow
 * through every existing getenv-based knob, while any env var the user already
 * exported still wins. This is a no-op if no file/entry matches.
 *
 * File search order (first found wins):
 *   1. $DS4_PROFILE                       (explicit path; set to "none"/"0" to disable)
 *   2. ./ds4_profile.json                 (current working directory)
 *   3. <dir of executable>/ds4_profile.json (next to the binary)
 *   4. ~/.config/ds4/ds4_profile.json
 *
 * Schema:
 *   {
 *     "version": 1,
 *     "profiles": [
 *       { "match": { "chip": "Apple M3 Ultra", "min_ram_gib": 64 },
 *         "env":   { "DS4_CTX_GROW": "1", "DS4_RESIDENT_MOE_BACKEND": "mulmm", ... },
 *         "sidecar_env": { "DS4_FLASH_MOE_ANE_PREFILL": "1", ... } }
 *     ]
 *   }
 *
 * `env` applies to resident/full-model runs. `sidecar_env` applies only when
 * opening Flash-MoE slot-bank sidecar mode. They intentionally do not mix.
 */

/* Apply the matching profile's env map. Safe to call once at startup before any
 * tuning knob is read; a no-op if no profile file/entry matches. */
void ds4_profile_load_and_apply(void);

/* Profile scope is set by frontends before load. Resident/full-model profiles
 * and Flash-MoE sidecar profiles use different knobs, so the loader must not
 * blindly apply one to the other. */
void ds4_profile_set_sidecar_mode(bool enabled);

/* Disable implicit profile discovery for this process unless the user set
 * DS4_PROFILE explicitly. */
void ds4_profile_disable_auto_default(void);

#endif /* DS4_PROFILE_H */
