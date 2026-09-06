// Adapted from Anemll/anemll-flash-llama.cpp 34cccef (MIT), HY4 slot8.
// Four rows share activation loads. Two dispatches compute all eight experts.
// Native integration supplies validated host slot IDs after the full bank
// reservation/install pass. Unlike the source fused path, preserve clamp10.
struct ds4_metal_args_hy4_fused {
    ulong n_embd, n_ff, n_used;
    ulong gate_nb1, gate_nb2, up_nb1, up_nb2, down_nb1, down_nb2;
    int slots[8];
};
// STQ1_0: 64 codebook groups per 256 block. Group g covers y[chunk*64 + g%16 + p*16],
// p = 0..3, chunk = g/16. Lane t owns groups t and t+32 (kernel_mul_mv_stq1_0_f32_impl).
// Computes gate and up dots for rows j0..j0+nrows-1 (row stride nb1) against y[ne00].
template <int NR>
static inline void hyv4_stq1_0_dot_rows(
        device const char  * gate_base,
        device const char  * up_base,
        uint64_t             gate_nb1,
        uint64_t             up_nb1,
        int                  nrows,
        device const float * y,
        int                  ne00,
        ushort               tiisg,
        thread float2      * sumf) {
    const int nb = ne00 / 256;

    for (int r = 0; r < NR; ++r) {
        sumf[r] = 0.0f;
    }

    for (int ib = 0; ib < nb; ++ib) {
        device const float * yb = y + ib*256;

        // activation taps for groups tiisg and tiisg + 32, shared by all rows
        float yv[2][4];
        for (int gi = 0; gi < 2; ++gi) {
            const int group = tiisg + 32*gi;
            device const float * yi = yb + (group/16)*64 + (group%16);
            for (int p = 0; p < 4; ++p) {
                yv[gi][p] = yi[p*16];
            }
        }

        for (int r = 0; r < NR; ++r) {
            if (r >= nrows) {
                break;
            }

            device const hy4_block_stq1_0 & gb = ((device const hy4_block_stq1_0 *)(gate_base + r*gate_nb1))[ib];
            device const hy4_block_stq1_0 & ub = ((device const hy4_block_stq1_0 *)(up_base   + r*up_nb1))[ib];

            float2 sum = 0.0f;

            for (int gi = 0; gi < 2; ++gi) {
                const int group = tiisg + 32*gi;
                const uint8_t gcode = (gb.qs[group/2] >> (4*(group & 1))) & 0x0f;
                const uint8_t ucode = (ub.qs[group/2] >> (4*(group & 1))) & 0x0f;
                const uint8_t gsign = (gb.sign[group/8] >> (group & 7)) & 0x01;
                const uint8_t usign = (ub.sign[group/8] >> (group & 7)) & 0x01;
                const uint8_t gq = hy4_stq1_codebook[(gsign << 4) | gcode];
                const uint8_t uq = hy4_stq1_codebook[(usign << 4) | ucode];

                for (int p = 0; p < 4; ++p) {
                    sum[0] += yv[gi][p] * float(int((gq >> (2*p)) & 0x03) - 1);
                    sum[1] += yv[gi][p] * float(int((uq >> (2*p)) & 0x03) - 1);
                }
            }

            sumf[r][0] += (float) gb.d * sum[0];
            sumf[r][1] += (float) ub.d * sum[1];
        }
    }
}

// IQ2_XXS: one 32-value sub-block per lane per iteration (kernel_mul_mv_iq2_xxs_f32_impl).
// The default build reads the immutable grid/sign tables directly from Metal constant memory,
// avoiding a 2,176-byte copy and barrier per Phase-A threadgroup. The compile-time fallback
// retains the original threadgroup LUT path for controlled A/B testing.
static inline float hyv4_iq2_xxs_dot_subblock(
        device const block_iq2_xxs * xr,
        int                          ib,
        thread const float         * yl,
        constant const uint64_t    * svalues,
        constant const uint8_t     * ssigns) {
    device const uint16_t * q2   = xr->qs + 4*ib;
    device const uint8_t  * aux8 = (device const uint8_t *) q2;
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) xr->d * (0.5f + (aux32 >> 28));

    float sum = 0.f;
    for (short l = 0; l < 4; ++l) {
        constant const uint8_t * grid = (constant const uint8_t *)(svalues + aux8[l]);
        const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
        for (short j = 0; j < 8; ++j) {
            sum += yl[8*l + j] * grid[j] * (signs & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
        }
    }

    return d * sum;
}

template <int NR>
static inline void hyv4_iq2_xxs_dot_rows(
        device const char          * gate_base,
        device const char          * up_base,
        uint64_t                     gate_nb1,
        uint64_t                     up_nb1,
        int                          nrows,
        device const float         * y,
        int                          ne00,
        constant const uint64_t    * svalues,
        constant const uint8_t     * ssigns,
        ushort                       tiisg,
        thread float2              * sumf) {
    const int nb   = ne00 / 256;
    const int nb32 = nb * (256 / 32);

    float yl[32];

    for (int r = 0; r < NR; ++r) {
        sumf[r] = 0.0f;
    }

    device const float * y4 = y + 32*tiisg;

    for (int ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (256 / 32);
        const int ib  = ib32 % (256 / 32);

        for (int r = 0; r < NR; ++r) {
            if (r >= nrows) {
                break;
            }
            sumf[r][0] += hyv4_iq2_xxs_dot_subblock((device const block_iq2_xxs *)(gate_base + r*gate_nb1) + ibl, ib, yl, svalues, ssigns);
            sumf[r][1] += hyv4_iq2_xxs_dot_subblock((device const block_iq2_xxs *)(up_base   + r*up_nb1)   + ibl, ib, yl, svalues, ssigns);
        }

        y4 += 32*32;
    }
}

// IQ3_XXS: one 32-value sub-block per lane per iteration (kernel_mul_mv_iq3_xxs_f32_impl).
// svalues/ssigns are threadgroup copies of ds4_metal_iq3xxs_grid / ds4_metal_ksigns_iq2xs.
static inline float hyv4_iq3_xxs_dot_subblock(
        device const block_iq3_xxs * xr,
        int                          ib,
        thread const float         * yl,
        threadgroup const uint32_t * svalues,
        threadgroup const uint8_t  * ssigns) {
    device const uint8_t  * q3  = xr->qs + 8*ib;
    device const uint16_t * gas = (device const uint16_t *)(xr->qs + 256/4) + 2*ib;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) xr->d * (0.5f + (aux32 >> 28));

    float2 sum = 0.f;
    for (short l = 0; l < 4; ++l) {
        threadgroup const uint8_t * grid1 = (threadgroup const uint8_t *)(svalues + q3[2*l+0]);
        threadgroup const uint8_t * grid2 = (threadgroup const uint8_t *)(svalues + q3[2*l+1]);
        const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
        for (short j = 0; j < 4; ++j) {
            sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & ds4_metal_kmask_iq2xs[j+0] ? -1.f : 1.f);
            sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & ds4_metal_kmask_iq2xs[j+4] ? -1.f : 1.f);
        }
    }

    return d * (sum[0] + sum[1]);
}

// down rows r0..r0+nrows-1 (row stride nb1) against y[ne00]; returns the unscaled per-lane
// partial sums (caller applies simd_sum and the 0.5 factor per row).
template <int NR>
static inline void hyv4_iq3_xxs_dot_rows(
        device const char          * base,
        uint64_t                     nb1,
        int                          nrows,
        device const float         * y,
        int                          ne00,
        threadgroup const uint32_t * svalues,
        threadgroup const uint8_t  * ssigns,
        ushort                       tiisg,
        thread float               * sumf) {
    const int nb   = ne00 / 256;
    const int nb32 = nb * (256 / 32);

    float yl[32];

    for (int r = 0; r < NR; ++r) {
        sumf[r] = 0.f;
    }

    device const float * y4 = y + 32*tiisg;

    for (int ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (256 / 32);
        const int ib  = ib32 % (256 / 32);

        for (int r = 0; r < NR; ++r) {
            if (r >= nrows) {
                break;
            }
            sumf[r] += hyv4_iq3_xxs_dot_subblock((device const block_iq3_xxs *)(base + r*nb1) + ibl, ib, yl, svalues, ssigns);
        }

        y4 += 32*32;
    }
}

// IQ4_XS: 16 lanes per 256 block, two blocks per iteration; each lane owns one
// 8-value low/high nibble half of a 32-value sub-block (kernel_mul_mv_iq4_xs_f32_impl).
// shmem_f32 holds ds4_metal_kvalues_iq4nl_f[t % 16] for t in 0..31.
template <int NR>
static inline void hyv4_iq4_xs_dot_rows(
        device const char        * base,
        uint64_t                   nb1,
        int                        nrows,
        device const float       * y,
        int                        ne00,
        threadgroup const float  * shmem_f32,
        ushort                     tiisg,
        thread float             * sumf) {
    const int nb = ne00 / 256;

    const short ix = tiisg/16;  // 0 or 1
    const short it = tiisg%16;  // 0...15
    const short ib = it/2;
    const short il = it%2;

    float4 yl[4];

    for (int r = 0; r < NR; ++r) {
        sumf[r] = 0.f;
    }

    device const float * yb = y + ix*256 + ib*32 + il*8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *) aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {
        device const float4 * y4 = (device const float4 *) yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (int r = 0; r < NR; ++r) {
            if (r >= nrows) {
                break;
            }

            device const block_iq4_xs & xb = ((device const block_iq4_xs *)(base + r*nb1))[ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[r] += (float) xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2*256;
    }
}

// Phase A, STQ1_0 gate/up. Grid = (ceil(n_ff/NR), 1, n_used), one simdgroup per NR rows of one expert.
template <int NR>
kernel void kernel_hyv4_fused_phaseA_stq1_0_t(
        constant ds4_metal_args_hy4_fused & args,
        device const char * x,        // [n_embd] f32
        device const char * gate,     // STQ1_0 [n_embd, n_ff, n_slots]
        device const char * up,       // STQ1_0 [n_embd, n_ff, n_slots]
        device       char * h,        // f32 [n_ff, n_used]
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int j0 = tgpig.x * NR;
    const int e  = tgpig.z;
    if (j0 >= args.n_ff || e >= args.n_used) {
        return;
    }
    const int nrows = min(NR, (int) (args.n_ff - j0));

    const int slot = args.slots[e];

    device const float * y         = (device const float *) x;
    device const char  * gate_base = gate + slot*args.gate_nb2 + (uint64_t) j0*args.gate_nb1;
    device const char  * up_base   = up   + slot*args.up_nb2   + (uint64_t) j0*args.up_nb1;

    float2 sumf[NR];
    hyv4_stq1_0_dot_rows<NR>(gate_base, up_base, args.gate_nb1, args.up_nb1, nrows, y, args.n_embd, tiisg, sumf);

    device float * hout = (device float *) h + (uint64_t) e*args.n_ff + j0;
    for (int r = 0; r < NR; ++r) {
        if (r < nrows) {
            const float g = min(simd_sum(sumf[r][0]), 10.0f);
            const float u = clamp(simd_sum(sumf[r][1]), -10.0f, 10.0f);
            if (tiisg == 0) {
                hout[r] = (g / (1.0f + exp(-g))) * u;
            }
        }
    }
}

// Phase A, IQ2_XXS gate/up. Grid = (ceil(n_ff/NR), 1, n_used), one simdgroup per NR rows of one expert.
template <int NR>
kernel void kernel_hyv4_fused_phaseA_iq2_xxs_t(
        constant ds4_metal_args_hy4_fused & args,
        device const char * x,        // [n_embd] f32
        device const char * gate,     // IQ2_XXS [n_embd, n_ff, n_slots]
        device const char * up,       // IQ2_XXS [n_embd, n_ff, n_slots]
        device       char * h,        // f32 [n_ff, n_used]
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int j0 = tgpig.x * NR;
    const int e  = tgpig.z;
    if (j0 >= args.n_ff || e >= args.n_used) {
        return;
    }
    const int nrows = min(NR, (int) (args.n_ff - j0));

    const int slot = args.slots[e];

    device const float * y         = (device const float *) x;
    device const char  * gate_base = gate + slot*args.gate_nb2 + (uint64_t) j0*args.gate_nb1;
    device const char  * up_base   = up   + slot*args.up_nb2   + (uint64_t) j0*args.up_nb1;

    float2 sumf[NR];
    hyv4_iq2_xxs_dot_rows<NR>(gate_base, up_base, args.gate_nb1, args.up_nb1, nrows, y, args.n_embd, ds4_metal_iq2xxs_grid, ds4_metal_ksigns_iq2xs, tiisg, sumf);

    device float * hout = (device float *) h + (uint64_t) e*args.n_ff + j0;
    for (int r = 0; r < NR; ++r) {
        if (r < nrows) {
            const float g = min(simd_sum(sumf[r][0]) * 0.25f, 10.0f);
            const float u = clamp(simd_sum(sumf[r][1]) * 0.25f, -10.0f, 10.0f);
            if (tiisg == 0) {
                hout[r] = (g / (1.0f + exp(-g))) * u;
            }
        }
    }
}

// Phase B, IQ3_XXS down. Grid = (ceil(n_embd/NR), 1, 1), one simdgroup per NR output rows.
template <int NR>
kernel void kernel_hyv4_fused_phaseB_iq3_xxs_t(
        constant ds4_metal_args_hy4_fused & args,
        device const char * h,        // f32 [n_ff, n_used]
        device const char * down,     // IQ3_XXS [n_ff, n_embd, n_slots]
        device const char * weights,  // f32 [1, n_used, 1]
        device       char * dst,      // f32 [n_embd]
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    for (int i = tiisg; i < 256; i += 32) {
        svalues[i] = ds4_metal_iq3xxs_grid[i];
    }
    for (int i = tiisg; i < 128; i += 32) {
        ssigns[i] = ds4_metal_ksigns_iq2xs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int r0 = tgpig.x * NR;
    if (r0 >= args.n_embd) {
        return;
    }
    const int nrows = min(NR, (int) (args.n_embd - r0));

    float acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = 0.f;
    }

    for (int e = 0; e < args.n_used; ++e) {
        const int slot = args.slots[e];

        device const char  * dbase = down + slot*args.down_nb2 + (uint64_t) r0*args.down_nb1;
        device const float * he    = (device const float *) h + (uint64_t) e*args.n_ff;

        float sumf[NR];
        hyv4_iq3_xxs_dot_rows<NR>(dbase, args.down_nb1, nrows, he, args.n_ff, svalues, ssigns, tiisg, sumf);

        const float w = *(device const float *)(weights + (uint64_t) e*sizeof(float));
        for (int r = 0; r < NR; ++r) {
            if (r < nrows) {
                const float d = simd_sum(sumf[r]) * 0.5f;
                // Apply weight AFTER down, in router order, without an FMA.
                volatile float product = w * d;
                volatile float total = e == 0 ? product : acc[r] + product;
                acc[r] = total;
            }
        }
    }

    if (tiisg == 0) {
        for (int r = 0; r < NR; ++r) {
            if (r < nrows) {
                ((device float *) dst)[r0 + r] = acc[r];
            }
        }
    }
}

// Phase B, IQ4_XS down. Grid = (ceil(n_embd/NR), 1, 1), one simdgroup per NR output rows.
template <int NR>
kernel void kernel_hyv4_fused_phaseB_iq4_xs_t(
        constant ds4_metal_args_hy4_fused & args,
        device const char * h,        // f32 [n_ff, n_used]
        device const char * down,     // IQ4_XS [n_ff, n_embd, n_slots]
        device const char * weights,  // f32 [1, n_used, 1]
        device       char * dst,      // f32 [n_embd]
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float * shmem_f32 = (threadgroup float *) shmem;
    shmem_f32[tiisg] = ds4_metal_kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int r0 = tgpig.x * NR;
    if (r0 >= args.n_embd) {
        return;
    }
    const int nrows = min(NR, (int) (args.n_embd - r0));

    float acc[NR];
    for (int r = 0; r < NR; ++r) {
        acc[r] = 0.f;
    }

    for (int e = 0; e < args.n_used; ++e) {
        const int slot = args.slots[e];

        device const char  * dbase = down + slot*args.down_nb2 + (uint64_t) r0*args.down_nb1;
        device const float * he    = (device const float *) h + (uint64_t) e*args.n_ff;

        float sumf[NR];
        hyv4_iq4_xs_dot_rows<NR>(dbase, args.down_nb1, nrows, he, args.n_ff, shmem_f32, tiisg, sumf);

        const float w = *(device const float *)(weights + (uint64_t) e*sizeof(float));
        for (int r = 0; r < NR; ++r) {
            if (r < nrows) {
                const float d = simd_sum(sumf[r]);
                // Apply weight AFTER down, in router order, without an FMA.
                volatile float product = w * d;
                volatile float total = e == 0 ? product : acc[r] + product;
                acc[r] = total;
            }
        }
    }

    if (tiisg == 0) {
        for (int r = 0; r < NR; ++r) {
            if (r < nrows) {
                ((device float *) dst)[r0 + r] = acc[r];
            }
        }
    }
}

typedef decltype(kernel_hyv4_fused_phaseA_stq1_0_t<1>)  kernel_hyv4_fused_phaseA_stq1_0_fn;
typedef decltype(kernel_hyv4_fused_phaseA_iq2_xxs_t<1>) kernel_hyv4_fused_phaseA_iq2_xxs_fn;
typedef decltype(kernel_hyv4_fused_phaseB_iq3_xxs_t<1>) kernel_hyv4_fused_phaseB_iq3_xxs_fn;
typedef decltype(kernel_hyv4_fused_phaseB_iq4_xs_t<1>)  kernel_hyv4_fused_phaseB_iq4_xs_fn;

template [[host_name("kernel_hyv4_fused_phaseA_stq1_0")]]  kernel kernel_hyv4_fused_phaseA_stq1_0_fn  kernel_hyv4_fused_phaseA_stq1_0_t<4>;
template [[host_name("kernel_hyv4_fused_phaseA_iq2_xxs")]] kernel kernel_hyv4_fused_phaseA_iq2_xxs_fn kernel_hyv4_fused_phaseA_iq2_xxs_t<4>;
template [[host_name("kernel_hyv4_fused_phaseB_iq3_xxs")]] kernel kernel_hyv4_fused_phaseB_iq3_xxs_fn kernel_hyv4_fused_phaseB_iq3_xxs_t<4>;
template [[host_name("kernel_hyv4_fused_phaseB_iq4_xs")]]  kernel kernel_hyv4_fused_phaseB_iq4_xs_fn  kernel_hyv4_fused_phaseB_iq4_xs_t<4>;
