#define FC_FLASH_ATTN_EXT_PAD 100
#define FC_FLASH_ATTN_EXT_BLK 200
#define FC_FLASH_ATTN_EXT 300
#define FC_FLASH_ATTN_EXT_VEC 400
#define FC_FLASH_ATTN_EXT_VEC_REDUCE 500
#define OP_FLASH_ATTN_EXT_NQPSG 8
#define OP_FLASH_ATTN_EXT_NCPSG 64
#define OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#define OP_FLASH_ATTN_EXT_VEC_NCPSG 32

#ifndef PAD2
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))
#endif

template <typename type4>
void dequantize_f32_t4(device const float4 * src, short il, thread type4 & reg) {
    reg = (type4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg);

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg);

struct ds4_metal_args_flash_attn_ext_pad {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
};

struct ds4_metal_args_flash_attn_ext_blk {
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
};

struct ds4_metal_args_flash_attn_ext {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};

struct ds4_metal_args_flash_attn_ext_vec {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};

struct ds4_metal_args_flash_attn_ext_vec_reduce {
    int32_t nrows;
};

struct ds4_dspark_attn_phase_range_metal {
    ushort begin;
    ushort len;
    uchar  kind;
    uchar  lane;
    ushort reserved;
};

struct ds4_dspark_attn_mixed_phase_range_metal {
    ushort begin;
    ushort len;
    uchar  kind;
    uchar  lane;
    uchar  phase;
    uchar  row;
};

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];
constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// DS4 FlashAttention padding: pads the final partial K/V/mask cache block so the
// vector FlashAttention kernel can read full 32-row chunks.
kernel void kernel_flash_attn_ext_pad(
        constant ds4_metal_args_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// DS4 FlashAttention mask scan: marks blocks so the non-vector kernel can skip
// blocks that are entirely masked or entirely zero.
kernel void kernel_flash_attn_ext_blk(
        constant ds4_metal_args_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 ? 1 : 0;

    device const half * mask_src = (device const half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;

    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;

        FOR_UNROLL (short j = 0; j < Q; ++j) {
            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                mmin = min(mmin, mask_src[ii*NW]);
                mmax = max(mmax, mask_src[ii*NW]);
            }

            mask_src += args.nb31/2;
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}

constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];
constant bool FC_flash_attn_ext_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT + 4)]];

constant bool FC_flash_attn_ext_bc_mask [[function_constant(FC_FLASH_ATTN_EXT + 10)]];

constant int32_t FC_flash_attn_ext_ns10 [[function_constant(FC_FLASH_ATTN_EXT + 20)]];
constant int32_t FC_flash_attn_ext_ns20 [[function_constant(FC_FLASH_ATTN_EXT + 21)]];
constant int32_t FC_flash_attn_ext_nsg  [[function_constant(FC_FLASH_ATTN_EXT + 22)]];

// DS4 non-vector FlashAttention. The only exported instance uses the model's
// 512-wide F16 K/V rows; keeping the template body generic preserves the same
// arithmetic for dense and compressed-attention prefill.
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q,
    short C,
    short NSG>
void kernel_flash_attn_ext_impl(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
    constexpr short DV16 = DV/16;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C;

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV;

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T);
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T);
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK);
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T);
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T);

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS);

    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    device const half2 * pm2[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };

    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        float slope = 1.0f;

        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_has_mask) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -MAXHALF;
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = 1;

            if (FC_flash_attn_ext_has_mask) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (FC_flash_attn_ext_bc_mask) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                        } else {
                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
                        }

                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NW;
                    }
                }
            }

            if (is_same<kd4x4_t, k4x4_t>::value) {
                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    if (DK % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);
                            simdgroup_load(mq, pq + 8*i, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        #pragma unroll (MIN(DK8/2, 4*NSG))
                        for (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                        }
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            } else {
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    for (short ii = 0; ii < DK16; ii += 4) {
                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));

                        if (DK16%4 == 0) {
                            {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            FOR_UNROLL (short k = 0; k < 4; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        } else {
                            if (ii + tx < DK16) {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true);
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        }
                    }

                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                if (blk_cur != 2) {
                    if (FC_flash_attn_ext_has_bias) {
                        s2 += s2_t(sm2[j*SH + tiisg])*slope;
                    } else {
                        s2 += s2_t(sm2[j*SH + tiisg]);
                    }
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                if (is_same<vd4x4_t, v4x4_t>::value) {
                    static_assert(PV8 % NSG == 0, "");

                    constexpr short NO = PV8/NSG;

                    o8x8_t lo[NO];

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                        pv += 8*sgitg;

                        if (DV <= 64) {
                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                                s8x8_t vs;
                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[2];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                                }

                                pv  += 8*NS20;
                            }
                        } else {
                            constexpr short NC = (C/8)/2;

                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                                s8x8_t vs[2];

                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[4];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                                }

                                pv  += 2*8*NS20;
                            }
                        }
                    }

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }
                } else {
                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    for (short cc = 0; cc < C/8; ++cc) {
                        s8x8_t vs;
                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));

                            if (DV16%4 == 0) {
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                FOR_UNROLL (short k = 0; k < 4; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) {
            break;
        }

        device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;

        const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];

        if (DV4 % NW == 0) {
            FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                const short i = ii*NW + tiisg;

                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = (float4) so4[j*PV4 + i]*scale;
            }
        }
    }

#undef NS10
#undef NS20
}

// Batched FlashAttention for prompt/prefill rows. It computes QK, applies mask,
// sinks, ALiBi/softcap options when enabled, and multiplies by V without
// materializing the full attention matrix.
template<
    typename q_t,
    typename q4_t,
    typename q8x8_t,
    typename k_t,
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,
    typename qk8x8_t,
    typename s_t,
    typename s2_t,
    typename s8x8_t,
    typename o_t,
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t,
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,
    short DV,
    short Q  = OP_FLASH_ATTN_EXT_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_NCPSG>
kernel void kernel_flash_attn_ext(
        constant ds4_metal_args_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#define FWD_TMPL q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, Q, C
#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg
    switch (FC_flash_attn_ext_nsg) {
        case 4: kernel_flash_attn_ext_impl<FWD_TMPL, 4>(FWD_ARGS); break;
        case 8: kernel_flash_attn_ext_impl<FWD_TMPL, 8>(FWD_ARGS); break;
    }
#undef FWD_TMPL
#undef FWD_ARGS
}

#define FA_NONVEC_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8

typedef decltype(kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512>) flash_attn_ext_dk512_t;

// Host-visible prefill FlashAttention variant for DS4's 512-wide F16 K/V rows.
template [[host_name("kernel_flash_attn_ext_f16_dk512_dv512")]]
kernel flash_attn_ext_dk512_t kernel_flash_attn_ext<FA_NONVEC_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 512, 512>;

#undef FA_NONVEC_TYPES

constant bool FC_flash_attn_ext_vec_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 0)]];
constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];
constant bool FC_flash_attn_ext_vec_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 2)]];
constant bool FC_flash_attn_ext_vec_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 3)]];
constant bool FC_flash_attn_ext_vec_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 4)]];
constant int32_t FC_flash_attn_ext_vec_ns10 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 20)]];
constant int32_t FC_flash_attn_ext_vec_ns20 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 21)]];
constant int32_t FC_flash_attn_ext_vec_nsg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 22)]];
constant int32_t FC_flash_attn_ext_vec_nwg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 23)]];

// Decode FlashAttention for one query row. DS4 uses this in generation to scan
// raw and compressed KV cache chunks, optionally splitting long contexts across
// workgroups and writing partial softmax state for a later reduction.
template<
    typename q4_t,
    typename k4_t,
    typename v4_t,
    typename qk_t,
    typename s_t,
    typename s4_t,
    typename o4_t,
    typename kd4_t,
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t,
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,
    short DV,
    short NE = 4,
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>
kernel void kernel_flash_attn_ext_vec(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)
#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE;
    constexpr short SH  = 4*C;

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                      0*PK);
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + NSG*PK);
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + NSG*PK + NSG*SH);

    so4 += tiisg;

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    device const float4 * q4 = (device const float4 *) ((device const char *) q);

    if (iq1 < args.ne01) {
        for (short i = tiisg; i < PK4; i += NW) {
            if (i < DK4) {
                sq4[i] = (q4_t) q4[i];
            } else {
                sq4[i] = (q4_t) 0.0f;
            }
        }
    }

    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = (o4_t) 0.0f;
    }

    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        device const half * pm = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        float slope = 1.0f;

        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    if (ic + tiisg >= args.ne11) {
                        sm[tiisg] = -MAXHALF;
                    }
                } else {
                    pm = (device const half *) (mask) +
                        iq1*C +
                        (iq2%args.ne32)*(C*args.ne31) +
                        (iq3%args.ne33)*(C*args.ne31*args.ne32);
                }

                ic = 0;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                sm[tiisg] = pm[ic + tiisg];
            }

            if (simd_max(sm[tiisg]) <= -MAXHALF) {
                continue;
            }

            {
                device      const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);
                threadgroup const q4_t * pq4 = sq4;

                pk4 += ty*NS10/4 + tx;
                pq4 += tx;

                qk_t mqk[C/NE] = { [ 0 ... C/NE - 1] = 0.0f };

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            mqk[cc] += dot((float4) pk4[cc*NE*NS10/4 +  ii*NL], (float4) pq4[ii*NL]);
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            mqk[cc] += dot((float4) mk, (float4) sq4[i]);
                        }
                    }

                    if (NE == 1) {
                        mqk[cc] = simd_sum(mqk[cc]);
                    } else {
                        if (NE <= 1) {
                            mqk[cc] += simd_shuffle_down(mqk[cc], 16);
                        }
                        if (NE <= 2) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
                        }
                        if (NE <= 4) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
                        }
                        if (NE <= 8) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
                        }
                        if (NE <= 16) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
                        }

                        mqk[cc] = simd_shuffle(mqk[cc], NL*ty);
                    }
                }

                if (FC_flash_attn_ext_vec_has_mask &&
                   !FC_flash_attn_ext_vec_has_scap &&
                   !FC_flash_attn_ext_vec_has_bias) {
                    ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);
                } else {
                    mqk[tx] *= args.scale;

                    if (FC_flash_attn_ext_vec_has_scap) {
                        mqk[tx] = args.logit_softcap*precise::tanh(mqk[tx]);
                    }

                    if (FC_flash_attn_ext_vec_has_bias) {
                        mqk[tx] += (qk_t) sm[NE*tx + ty]*slope;
                    } else {
                        mqk[tx] += (qk_t) sm[NE*tx + ty];
                    }

                    ss[NE*tx + ty] = mqk[tx];
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                ss[tiisg] = vs;

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] *= ms;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    const auto sst = ss + ty;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            lo[ii] += o4_t(float4(pv4[cc*NE*NS20/4 + ii*NL])*float4(sst[cc*NE]));
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            lo[ii] += o4_t(float4(mv)*float4(ss[NE*cc + ty]));
                        }
                    }
                }

                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    if (NE > 1) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);
                    }

                    if (NE > 2) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  8);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  8);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  8);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  8);
                    }

                    if (NE > 4) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  4);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  4);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  4);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  4);
                    }

                    if (NE > 8) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  2);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  2);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  2);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  2);
                    }

                    if (NE > 16) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  1);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  1);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  1);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  1);
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] += lo[ii];
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            const float m = M;
            const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

            M = simd_max(max(M, s));

            const float ms = exp(m - M);
            const float vs = exp(s - M);

            S = S*ms + simd_sum(vs);

            if ((DV4/NL % NW == 0) || ty == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    so4[ii*NL] *= ms;
                }
            }
        }

        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[           0];
            const float S1 = ss[r*(SH/2) + 0];

            const float M0 = ss[           1];
            const float M1 = ss[r*(SH/2) + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            for (short i = tiisg; i < DV4; i += NW) {
                so4[i] = so4[i]*ms0 + so4[i + r*PV4]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid   = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

        const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f;

        for (short i = tiisg; i < DV4; i += NW) {
            dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i]*S;
        }

        if (NWG > 1) {
            if (tiisg == 0) {
                dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
                dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

// Host-visible decode FlashAttention variant for DS4's 512-wide F16 K/V rows.
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,  1, dequantize_f16_t4, half4,  1, dequantize_f16_t4, 512, 512, 1>;

// DSpark verifier prototype: process N<=5 query rows for one head/split-K group
// inside one threadgroup. Each row keeps its own online-softmax state and walks
// keys in the same order as kernel_flash_attn_ext_vec; this is a conservative
// landing point before explicitly shared K/V tile staging.
template<
    typename q4_t,
    typename k4_t,
    typename v4_t,
    typename qk_t,
    typename s_t,
    typename s4_t,
    typename o4_t,
    typename kd4_t,
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t,
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,
    short DV,
    short NE = 4,
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG,
    bool STAGE_K = false,
    bool STAGE_V = false>
kernel void kernel_flash_attn_ext_vec_rows5(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)
#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const ushort row = sgitg / NSG;
    const ushort lsg = sgitg - row * NSG;
    if (row >= args.ne01) {
        return;
    }

    const short iwg = tgpig[2] % NWG;
    const ushort iq3 = tgpig[2] / NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = row;

    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK / 4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV / 4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW / NE;
    constexpr short SH  = 4 * C;

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    threadgroup q4_t *sq_base = (threadgroup q4_t *) shmem_f16;
    threadgroup half *stage_base = shmem_f16 + args.ne01 * PK;
    threadgroup k4_t *sk4_base =
        (threadgroup k4_t *) (stage_base + lsg * C * DK);
    threadgroup half *sv_stage_base =
        stage_base + (STAGE_K ? NSG * C * DK : 0);
    threadgroup v4_t *sv4_base =
        (threadgroup v4_t *) (sv_stage_base + lsg * C * DV);
    threadgroup half *row_base =
        sv_stage_base + (STAGE_V ? NSG * C * DV : 0);
    threadgroup half *ss_half =
        row_base + row * (NSG * SH + 2 * NSG * PV) + lsg * SH;
    threadgroup half *so_half =
        row_base + row * (NSG * SH + 2 * NSG * PV) + NSG * SH + 2 * lsg * PV;

    threadgroup q4_t *sq4 = sq_base + row * PK4;
    threadgroup s_t  *ss  = (threadgroup s_t *) ss_half;
    threadgroup s4_t *ss4 = (threadgroup s4_t *) ss_half;
    threadgroup half *sm  = ss_half + 2 * C;
    threadgroup o4_t *so4 = (threadgroup o4_t *) so_half;

    so4 += tiisg;

    {
        q += iq1 * args.nb01 + iq2 * args.nb02 + iq3 * args.nb03;

        const short ikv2 = iq2 / (args.ne02 / args.ne_12_2);
        const short ikv3 = iq3 / (args.ne03 / args.ne_12_3);

        k += ikv2 * args.nb12 + ikv3 * args.nb13;
        v += ikv2 * args.nb22 + ikv3 * args.nb23;
    }

    device const float4 *q4 = (device const float4 *) ((device const char *) q);

    for (short i = tiisg; i < PK4; i += NW) {
        if (i < DK4) {
            sq4[i] = (q4_t) q4[i];
        } else {
            sq4[i] = (q4_t) 0.0f;
        }
    }

    for (short i = 0; i < DV4 / NL; ++i) {
        so4[i * NL] = (o4_t) 0.0f;
    }

    for (short i = tiisg; i < SH / 4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX / 2;

        const short tx = tiisg % NL;
        const short ty = tiisg / NL;

        device const half *pm = (device const half *) (mask + iq1 * args.nb31 +
            (iq2 % args.ne32) * args.nb32 + (iq3 % args.ne33) * args.nb33);

        float slope = 1.0f;

        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;
            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 :
                2 * (h - args.n_head_log2) + 1;
            slope = pow(base, exph);
        }

        for (int ic0 = iwg * NSG + lsg; ; ic0 += NWG * NSG) {
            int ic = ic0 * C;
            if (ic >= args.ne11) {
                break;
            }

            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11 * C * args.ne_12_2 * args.ne_12_3;
                mask = v + args.nb21 * C * args.ne_12_2 * args.ne_12_3;

                const short ikv2 = iq2 / (args.ne02 / args.ne_12_2);
                const short ikv3 = iq3 / (args.ne03 / args.ne_12_3);

                k += (ikv2 + ikv3 * args.ne_12_2) * args.nb11 * C;
                v += (ikv2 + ikv3 * args.ne_12_2) * args.nb21 * C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    if (ic + tiisg >= args.ne11) {
                        sm[tiisg] = -MAXHALF;
                    }
                } else {
                    pm = (device const half *) mask +
                        iq1 * C +
                        (iq2 % args.ne32) * (C * args.ne31) +
                        (iq3 % args.ne33) * (C * args.ne31 * args.ne32);
                }

                ic = 0;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                sm[tiisg] = pm[ic + tiisg];
            }

            if (STAGE_K && is_same<kd4_t, k4_t>::value) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (row == 0) {
                    device const k4_t *pk4_src =
                        (device const k4_t *) (k + ic * args.nb11);
                    for (short i = tiisg; i < C * DK4; i += NW) {
                        sk4_base[i] = pk4_src[i];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (STAGE_V && is_same<vd4_t, v4_t>::value) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (row == 0) {
                    device const v4_t *pv4_src =
                        (device const v4_t *) (v + ic * args.nb21);
                    for (short i = tiisg; i < C * DV4; i += NW) {
                        sv4_base[i] = pv4_src[i];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (simd_max(sm[tiisg]) <= -MAXHALF) {
                continue;
            }

            {
                threadgroup const q4_t *pq4 = sq4;
                pq4 += tx;

                qk_t mqk[C / NE] = { [0 ... C / NE - 1] = 0.0f };

                FOR_UNROLL (short cc = 0; cc < C / NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        if (STAGE_K) {
                            threadgroup const k4_t *pk4 = sk4_base + ty * NS10 / 4 + tx;
                            FOR_UNROLL (short ii = 0; ii < DK4 / NL; ++ii) {
                                mqk[cc] += dot((float4) pk4[cc * NE * NS10 / 4 + ii * NL],
                                               (float4) pq4[ii * NL]);
                            }
                        } else {
                            device const k4_t *pk4 =
                                (device const k4_t *) (k + ic * args.nb11);
                            pk4 += ty * NS10 / 4 + tx;
                            FOR_UNROLL (short ii = 0; ii < DK4 / NL; ++ii) {
                                mqk[cc] += dot((float4) pk4[cc * NE * NS10 / 4 + ii * NL],
                                               (float4) pq4[ii * NL]);
                            }
                        }
                    } else {
                        device const kd4_t *pk =
                            (device const kd4_t *) (k + ((ic + NE * cc + ty) * args.nb11));
                        k4_t mk;
                        FOR_UNROLL (short ii = 0; ii < DK4 / NL; ++ii) {
                            const short i = ii * NL + tx;
                            deq_k_t4(pk + i / nl_k, i % nl_k, mk);
                            mqk[cc] += dot((float4) mk, (float4) sq4[i]);
                        }
                    }

                    if (NE == 1) {
                        mqk[cc] = simd_sum(mqk[cc]);
                    } else {
                        if (NE <= 1) mqk[cc] += simd_shuffle_down(mqk[cc], 16);
                        if (NE <= 2) mqk[cc] += simd_shuffle_down(mqk[cc], 8);
                        if (NE <= 4) mqk[cc] += simd_shuffle_down(mqk[cc], 4);
                        if (NE <= 8) mqk[cc] += simd_shuffle_down(mqk[cc], 2);
                        if (NE <= 16) mqk[cc] += simd_shuffle_down(mqk[cc], 1);
                        mqk[cc] = simd_shuffle(mqk[cc], NL * ty);
                    }
                }

                if (FC_flash_attn_ext_vec_has_mask &&
                    !FC_flash_attn_ext_vec_has_scap &&
                    !FC_flash_attn_ext_vec_has_bias) {
                    ss[NE * tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE * tx + ty]);
                } else {
                    mqk[tx] *= args.scale;
                    if (FC_flash_attn_ext_vec_has_scap) {
                        mqk[tx] = args.logit_softcap * precise::tanh(mqk[tx]);
                    }
                    if (FC_flash_attn_ext_vec_has_bias) {
                        mqk[tx] += (qk_t) sm[NE * tx + ty] * slope;
                    } else {
                        mqk[tx] += (qk_t) sm[NE * tx + ty];
                    }
                    ss[NE * tx + ty] = mqk[tx];
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S * ms + simd_sum(vs);
                ss[tiisg] = vs;

                if ((DV4 / NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4 / NL; ++ii) {
                        so4[ii * NL] *= ms;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                o4_t lo[DV4 / NL];
                FOR_UNROLL (short ii = 0; ii < DV4 / NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t *pv4_device = (device const v4_t *) (v + ic * args.nb21);
                    threadgroup const v4_t *pv4_stage = sv4_base;
                    const bool use_stage_v = STAGE_V;
                    const auto sst = ss + ty;

                    FOR_UNROLL (short cc = 0; cc < C / NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4 / NL; ++ii) {
                            const v4_t vv = use_stage_v
                                ? pv4_stage[cc * NE * NS20 / 4 + ii * NL + ty * NS20 / 4 + tx]
                                : pv4_device[cc * NE * NS20 / 4 + ii * NL + ty * NS20 / 4 + tx];
                            lo[ii] += o4_t(float4(vv) *
                                           float4(sst[cc * NE]));
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C / NE; ++cc) {
                        device const vd4_t *pv4 =
                            (device const vd4_t *) (v + ((ic + NE * cc + ty) * args.nb21));
                        FOR_UNROLL (short ii = 0; ii < DV4 / NL; ++ii) {
                            const short i = ii * NL + tx;
                            v4_t mv;
                            deq_v_t4(pv4 + i / nl_v, i % nl_v, mv);
                            lo[ii] += o4_t(float4(mv) * float4(ss[NE * cc + ty]));
                        }
                    }
                }

                FOR_UNROLL (short ii = 0; ii < DV4 / NL; ++ii) {
                    if (NE > 1) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);
                    }
                    if (NE > 2) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 8);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 8);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 8);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 8);
                    }
                    if (NE > 4) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 4);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 4);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 4);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 4);
                    }
                    if (NE > 8) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 2);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 2);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 2);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 2);
                    }
                    if (NE > 16) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 1);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 1);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 1);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 1);
                    }
                }

                if ((DV4 / NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4 / NL; ++ii) {
                        so4[ii * NL] += lo[ii];
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && lsg == 0 && iwg == 0) {
            const float m = M;
            const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX / 2;

            M = simd_max(max(M, s));

            const float ms = exp(m - M);
            const float vs = exp(s - M);

            S = S * ms + simd_sum(vs);

            if ((DV4 / NL % NW == 0) || ty == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4 / NL; ++ii) {
                    so4[ii * NL] *= ms;
                }
            }
        }

        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short r = NSG / 2; r > 0; r >>= 1) {
        if (lsg < r) {
            const float S0 = ss[0];
            const float S1 = ss[r * (SH / 2) + 0];

            const float M0 = ss[1];
            const float M1 = ss[r * (SH / 2) + 1];

            const float M = max(M0, M1);
            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);
            const float S = S0 * ms0 + S1 * ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            for (short i = tiisg; i < DV4; i += NW) {
                so4[i] = so4[i] * ms0 + so4[i + r * PV4] * ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lsg == 0) {
        const int64_t nrows = args.ne3 * args.ne2 * args.ne1;
        const int64_t rid = iq3 * args.ne2 * args.ne1 + iq2 + iq1 * args.ne1;

        device float4 *dst4 = (device float4 *) dst;
        device float  *dst1 = (device float  *) dst + nrows * DV * NWG;

        const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f / ss[0]) : 1.0f;

        for (short i = tiisg; i < DV4; i += NW) {
            dst4[rid * DV4 * NWG + NWG * i + iwg] = (float4) so4[i] * S;
        }

        if (NWG > 1 && tiisg == 0) {
            dst1[rid * (2 * NWG) + 2 * iwg + 0] = ss[0];
            dst1[rid * (2 * NWG) + 2 * iwg + 1] = ss[1];
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

typedef decltype(kernel_flash_attn_ext_vec_rows5<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_rows5_t;
template [[host_name("kernel_flash_attn_ext_vec_rows5_f16_dk512_dv512")]] kernel flash_attn_ext_vec_rows5_t kernel_flash_attn_ext_vec_rows5<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_rows5_kstage_f16_dk512_dv512")]] kernel flash_attn_ext_vec_rows5_t kernel_flash_attn_ext_vec_rows5<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 512, 512, 1, OP_FLASH_ATTN_EXT_VEC_NQPSG, OP_FLASH_ATTN_EXT_VEC_NCPSG, true>;
template [[host_name("kernel_flash_attn_ext_vec_rows5_kvstage_f16_dk512_dv512")]] kernel flash_attn_ext_vec_rows5_t kernel_flash_attn_ext_vec_rows5<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 512, 512, 1, OP_FLASH_ATTN_EXT_VEC_NQPSG, OP_FLASH_ATTN_EXT_VEC_NCPSG, true, true>;

// DSpark strict verifier prototype: one dispatch for N<=5 rows, but each row
// keeps its own exact gathered key stream and true key count. This intentionally
// does not use raw-union masking or max-row padding as visibility; padding is
// only the same final 32-key chunk padding the scalar row path already uses.
kernel void kernel_flash_attn_varstream_rows5_f16_dk512_dv512(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char     * q,
        device const half4    * k,
        device const half4    * v,
        device const float    * sinks,
        device       char     * dst,
        device const uint32_t * row_offsets,
        device const uint32_t * row_n_keys,
        threadgroup  half     * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK = 512;
    constexpr short DV = 512;
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short C = OP_FLASH_ATTN_EXT_VEC_NCPSG;
    constexpr short NW = N_SIMDWIDTH;
    const short NWG = (short)((args.ne31 > 0) ? args.ne31 : 32);

    const ushort row = sgitg;
    if (row >= args.ne01) {
        return;
    }

    const short iwg = tgpig[2];
    const ushort head = tgpig[1];
    const uint32_t n_keys = row_n_keys[row];
    const uint32_t row_off = row_offsets[row];

    threadgroup half4 *sq4_base = (threadgroup half4 *)shmem_f16;
    threadgroup float  *ss_base =
        (threadgroup float *)(shmem_f16 + args.ne01 * DK);
    threadgroup float4 *so4_base =
        (threadgroup float4 *)(ss_base + args.ne01 * C);
    threadgroup half4 *sq4 = sq4_base + row * DK4;
    threadgroup float  *ss  = ss_base + row * C;
    threadgroup float4 *so4 = so4_base + row * DV4;

    device const float4 *q4 =
        (device const float4 *)(q + row * args.nb01 + head * args.nb02);

    for (short i = tiisg; i < DK4; i += NW) {
        sq4[i] = half4(q4[i]);
    }
    for (short i = tiisg; i < C; i += NW) {
        ss[i] = 0.0f;
    }
    for (short i = tiisg; i < DV4; i += NW) {
        so4[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;

    for (uint32_t ic = (uint32_t)iwg * C; ic < n_keys; ic += NWG * C) {
        float mqk[C] = { [0 ... C - 1] = 0.0f };

        FOR_UNROLL (short cc = 0; cc < C; ++cc) {
            const uint32_t key = ic + (uint32_t)cc;
            device const half4 *pk4 =
                k + ((uint64_t)row_off + key) * DK4 + tiisg;
            FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                const short qi = ii * NW + tiisg;
                half4 mk = half4(0.0h);
                if (key < n_keys) {
                    mk = pk4[ii * NW];
                }
                mqk[cc] += dot(float4(mk), float4(sq4[qi]));
            }
            mqk[cc] = simd_sum(mqk[cc]);
        }

        const bool valid = ic + tiisg < n_keys;
        float score = mqk[tiisg] * args.scale;
        score += valid ? 0.0f : -MAXHALF;
        ss[tiisg] = score;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float old_m = M;
        const float s = ss[tiisg];
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo[DV4 / NW] = { [0 ... DV4 / NW - 1] = float4(0.0f) };
        FOR_UNROLL (short cc = 0; cc < C; ++cc) {
            const uint32_t key = ic + (uint32_t)cc;
            device const half4 *pv4 =
                v + ((uint64_t)row_off + key) * DV4 + tiisg;
            const float weight = ss[cc];
            FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                half4 mv = half4(0.0h);
                if (key < n_keys) {
                    mv = pv4[ii * NW];
                }
                lo[ii] += float4(mv) * weight;
            }
        }
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] += lo[ii];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (iwg == 0) {
        const float old_m = M;
        const float s = tiisg == 0 ? sinks[head] : -FLT_MAX / 2;
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
    }

    const int64_t nrows = args.ne3 * args.ne2 * args.ne1;
    const int64_t rid = head + row * args.ne1;
    device float4 *dst4 = (device float4 *) dst;
    device float  *dst1 = (device float  *) dst + nrows * DV * NWG;

    FOR_UNROLL (short i = 0; i < DV4 / NW; ++i) {
        const short idx = i * NW + tiisg;
        dst4[rid * DV4 * NWG + NWG * idx + iwg] = so4[idx];
    }
    if (tiisg == 0) {
        dst1[rid * (2 * NWG) + 2 * iwg + 0] = S;
        dst1[rid * (2 * NWG) + 2 * iwg + 1] = M;
    }
}

// DSpark strict verifier prototype: one shared raw-union + comp-prefix stream
// for N<=5 rows. Each row still walks raw[0..n_raw) then comp[0..n_comp) in
// row-local order and writes split-K partials for the standard reduce kernel.
kernel void kernel_flash_attn_varmap_rows5_f16_dk512_dv512(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char     * q,
        device const half4    * k,
        device const half4    * v,
        device const float    * sinks,
        device       char     * dst,
        device const uint32_t * row_raw_base,
        device const uint32_t * row_n_raw,
        device const uint32_t * row_n_comp,
        constant uint32_t     & n_raw_union,
        constant uint32_t     & n_raw_common_full,
        constant uint32_t     & raw_common_base,
        constant uint32_t     & common_stage_mask,
        constant uint32_t     & raw_union_kstage_cap,
        constant uint32_t     & raw_union_vstage_cap,
        constant uint32_t     & raw_tile_intersection_kstage_cap,
        constant uint32_t     & raw_tile_intersection_vstage_cap,
        constant uint32_t     & comp_union_kstage_cap,
        constant uint32_t     & comp_union_vstage_cap,
        constant uint32_t     & mixed_chunk_union_kstage_cap,
        constant uint32_t     & mixed_chunk_union_vstage_cap,
        threadgroup  half     * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK = 512;
    constexpr short DV = 512;
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short C = OP_FLASH_ATTN_EXT_VEC_NCPSG;
    constexpr short NW = N_SIMDWIDTH;
    const short NWG = (short)((args.ne31 > 0) ? args.ne31 : 32);

    const ushort row = sgitg;
    if (row >= args.ne01) {
        return;
    }

    const short iwg = tgpig[2];
    const ushort head = tgpig[1];
    const uint32_t n_raw = row_n_raw[row];
    const uint32_t n_comp = row_n_comp[row];
    const uint32_t n_keys = n_raw + n_comp;
    const uint32_t raw_base = row_raw_base[row];

    threadgroup half4 *sq4_base = (threadgroup half4 *)shmem_f16;
    threadgroup float  *ss_base =
        (threadgroup float *)(shmem_f16 + args.ne01 * DK);
    threadgroup float4 *so4_base =
        (threadgroup float4 *)(ss_base + args.ne01 * C);
    threadgroup half4 *sk4_common_base =
        (threadgroup half4 *)(so4_base + args.ne01 * DV4);
    threadgroup half4 *sv4_common_base =
        sk4_common_base + ((common_stage_mask & 1u) ? C * DK4 : 0);
    threadgroup half4 *sk4_union_base =
        sv4_common_base + ((common_stage_mask & 2u) ? C * DV4 : 0);
    threadgroup half4 *sv4_union_base =
        sk4_union_base + (raw_union_kstage_cap ? raw_union_kstage_cap * DK4 : 0);
    threadgroup half4 *sk4_tile_intersection_base =
        sv4_union_base + (raw_union_vstage_cap ? raw_union_vstage_cap * DV4 : 0);
    threadgroup half4 *sv4_tile_intersection_base =
        sk4_tile_intersection_base +
        (raw_tile_intersection_kstage_cap ? raw_tile_intersection_kstage_cap * DK4 : 0);
    threadgroup half4 *sk4_comp_union_base =
        sv4_tile_intersection_base +
        (raw_tile_intersection_vstage_cap ? raw_tile_intersection_vstage_cap * DV4 : 0);
    threadgroup half4 *sv4_comp_union_base =
        sk4_comp_union_base + (comp_union_kstage_cap ? comp_union_kstage_cap * DK4 : 0);
    threadgroup half4 *sk4_mixed_chunk_union_base =
        sv4_comp_union_base + (comp_union_vstage_cap ? comp_union_vstage_cap * DV4 : 0);
    threadgroup half4 *sv4_mixed_chunk_union_base =
        sk4_mixed_chunk_union_base +
        (mixed_chunk_union_kstage_cap ? mixed_chunk_union_kstage_cap * DK4 : 0);
    threadgroup half4 *sq4 = sq4_base + row * DK4;
    threadgroup float  *ss  = ss_base + row * C;
    threadgroup float4 *so4 = so4_base + row * DV4;

    device const float4 *q4 =
        (device const float4 *)(q + row * args.nb01 + head * args.nb02);

    for (short i = tiisg; i < DK4; i += NW) {
        sq4[i] = half4(q4[i]);
    }
    for (short i = tiisg; i < C; i += NW) {
        ss[i] = 0.0f;
    }
    for (short i = tiisg; i < DV4; i += NW) {
        so4[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;

    for (uint32_t ic = (uint32_t)iwg * C; ic < n_keys; ic += NWG * C) {
        float mqk[C] = { [0 ... C - 1] = 0.0f };
        const bool common_k_stage =
            (common_stage_mask & 1u) &&
            n_raw_common_full >= C &&
            ic + (uint32_t)C <= n_raw_common_full;
        const bool common_v_stage =
            (common_stage_mask & 2u) &&
            n_raw_common_full >= C &&
            ic + (uint32_t)C <= n_raw_common_full;
        uint32_t raw_union_stage_begin = 0;
        uint32_t raw_union_stage_len = 0;
        bool raw_union_stage_ok =
            raw_union_kstage_cap >= C || raw_union_vstage_cap >= C;
        if (raw_union_stage_ok) {
            uint32_t union_begin = row_raw_base[0] + ic;
            uint32_t union_end = union_begin + C;
            FOR_UNROLL (short rr = 0; rr < 6; ++rr) {
                if ((uint32_t)rr < args.ne01) {
                    if (ic + (uint32_t)C > row_n_raw[rr]) {
                        raw_union_stage_ok = false;
                    }
                    const uint32_t row_begin = row_raw_base[rr] + ic;
                    const uint32_t row_end = row_begin + C;
                    if (row_begin < union_begin) union_begin = row_begin;
                    if (row_end > union_end) union_end = row_end;
                }
            }
            raw_union_stage_begin = union_begin;
            raw_union_stage_len = union_end - union_begin;
        }
        const bool raw_union_k_stage =
            raw_union_stage_ok &&
            raw_union_stage_len <= raw_union_kstage_cap &&
            !common_k_stage;
        const bool raw_union_v_stage =
            raw_union_stage_ok &&
            raw_union_stage_len <= raw_union_vstage_cap &&
            !common_v_stage;
        uint32_t raw_tile_intersection_begin = 0;
        uint32_t raw_tile_intersection_len = 0;
        bool raw_tile_intersection_valid =
            ((raw_tile_intersection_kstage_cap != 0u &&
              !common_k_stage &&
              !raw_union_k_stage) ||
             (raw_tile_intersection_vstage_cap != 0u &&
              !common_v_stage &&
              !raw_union_v_stage));
        if (raw_tile_intersection_valid) {
            uint32_t intersection_begin = row_raw_base[0] + ic;
            uint32_t intersection_end = intersection_begin + C;
            FOR_UNROLL (short rr = 0; rr < 6; ++rr) {
                if ((uint32_t)rr < args.ne01) {
                    if (ic + (uint32_t)C > row_n_raw[rr]) {
                        raw_tile_intersection_valid = false;
                    }
                    const uint32_t row_begin = row_raw_base[rr] + ic;
                    const uint32_t row_end = row_begin + C;
                    if (row_begin > intersection_begin) intersection_begin = row_begin;
                    if (row_end < intersection_end) intersection_end = row_end;
                }
            }
            if (intersection_end <= intersection_begin) {
                raw_tile_intersection_valid = false;
            }
            raw_tile_intersection_begin = intersection_begin;
            raw_tile_intersection_len = intersection_end - intersection_begin;
        }
        const bool raw_tile_intersection_k_stage =
            raw_tile_intersection_valid &&
            raw_tile_intersection_kstage_cap != 0u &&
            raw_tile_intersection_len <= raw_tile_intersection_kstage_cap &&
            !common_k_stage &&
            !raw_union_k_stage;
        const bool raw_tile_intersection_v_stage =
            raw_tile_intersection_valid &&
            raw_tile_intersection_vstage_cap != 0u &&
            raw_tile_intersection_len <= raw_tile_intersection_vstage_cap &&
            !common_v_stage &&
            !raw_union_v_stage;
        uint32_t comp_union_begin = 0;
        uint32_t comp_union_len = 0;
        bool comp_union_any = false;
        bool comp_union_all_rows_comp_tile = true;
        if (comp_union_kstage_cap != 0u || comp_union_vstage_cap != 0u) {
            uint32_t comp_union_end = 0;
            comp_union_begin = UINT_MAX;
            FOR_UNROLL (short rr = 0; rr < 6; ++rr) {
                if ((uint32_t)rr < args.ne01) {
                    const uint32_t rr_n_raw = row_n_raw[rr];
                    const uint32_t rr_n_keys = rr_n_raw + row_n_comp[rr];
                    // Row 0 performs the staged copy; every row must execute
                    // this tile so the threadgroup barrier and staged data are valid.
                    if (ic < rr_n_raw || ic >= rr_n_keys) {
                        comp_union_all_rows_comp_tile = false;
                    }
                    FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                        const uint32_t key = ic + (uint32_t)cc;
                        if (key < rr_n_keys && key >= rr_n_raw) {
                            const uint32_t comp_idx = key - rr_n_raw;
                            if (comp_idx < comp_union_begin) comp_union_begin = comp_idx;
                            if (comp_idx + 1u > comp_union_end) comp_union_end = comp_idx + 1u;
                            comp_union_any = true;
                        }
                    }
                }
            }
            if (comp_union_any) {
                comp_union_len = comp_union_end - comp_union_begin;
            }
        }
        const bool comp_union_k_stage =
            comp_union_any &&
            comp_union_all_rows_comp_tile &&
            comp_union_len <= comp_union_kstage_cap;
        const bool comp_union_v_stage =
            comp_union_any &&
            comp_union_all_rows_comp_tile &&
            comp_union_len <= comp_union_vstage_cap;
        uint32_t mixed_chunk_union_begin = UINT_MAX;
        uint32_t mixed_chunk_union_len = 0;
        bool mixed_chunk_union_any = false;
        bool mixed_chunk_union_all_rows_tile = true;
        if (mixed_chunk_union_kstage_cap != 0u ||
            mixed_chunk_union_vstage_cap != 0u) {
            uint32_t mixed_chunk_union_end = 0;
            FOR_UNROLL (short rr = 0; rr < 6; ++rr) {
                if ((uint32_t)rr < args.ne01) {
                    const uint32_t rr_n_raw = row_n_raw[rr];
                    const uint32_t rr_n_keys = rr_n_raw + row_n_comp[rr];
                    const uint32_t rr_raw_base = row_raw_base[rr];
                    if (ic >= rr_n_keys) {
                        mixed_chunk_union_all_rows_tile = false;
                    }
                    FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                        const uint32_t key = ic + (uint32_t)cc;
                        if (key < rr_n_keys) {
                            const uint32_t mapped_key = key < rr_n_raw
                                ? rr_raw_base + key
                                : n_raw_union + (key - rr_n_raw);
                            if (mapped_key < mixed_chunk_union_begin) {
                                mixed_chunk_union_begin = mapped_key;
                            }
                            if (mapped_key + 1u > mixed_chunk_union_end) {
                                mixed_chunk_union_end = mapped_key + 1u;
                            }
                            mixed_chunk_union_any = true;
                        }
                    }
                }
            }
            if (mixed_chunk_union_any) {
                mixed_chunk_union_len =
                    mixed_chunk_union_end - mixed_chunk_union_begin;
            }
        }
        const bool mixed_chunk_union_k_stage =
            mixed_chunk_union_any &&
            mixed_chunk_union_all_rows_tile &&
            mixed_chunk_union_len <= mixed_chunk_union_kstage_cap;
        const bool mixed_chunk_union_v_stage =
            mixed_chunk_union_any &&
            mixed_chunk_union_all_rows_tile &&
            mixed_chunk_union_len <= mixed_chunk_union_vstage_cap;

        if (common_k_stage ||
            common_v_stage ||
            raw_union_k_stage ||
            raw_union_v_stage ||
            raw_tile_intersection_k_stage ||
            raw_tile_intersection_v_stage ||
            comp_union_k_stage ||
            comp_union_v_stage ||
            mixed_chunk_union_k_stage ||
            mixed_chunk_union_v_stage) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row == 0) {
                if (common_k_stage) {
                    device const half4 *pk4_src =
                        k + ((uint64_t)raw_common_base + (uint64_t)ic) * DK4;
                    for (short i = tiisg; i < C * DK4; i += NW) {
                        sk4_common_base[i] = pk4_src[i];
                    }
                }
                if (common_v_stage) {
                    device const half4 *pv4_src =
                        v + ((uint64_t)raw_common_base + (uint64_t)ic) * DV4;
                    for (short i = tiisg; i < C * DV4; i += NW) {
                        sv4_common_base[i] = pv4_src[i];
                    }
                }
                if (raw_union_k_stage) {
                    device const half4 *pk4_src =
                        k + (uint64_t)raw_union_stage_begin * DK4;
                    for (uint i = tiisg;
                         i < raw_union_stage_len * (uint32_t)DK4;
                         i += NW) {
                        sk4_union_base[i] = pk4_src[i];
                    }
                }
                if (raw_union_v_stage) {
                    device const half4 *pv4_src =
                        v + (uint64_t)raw_union_stage_begin * DV4;
                    for (uint i = tiisg;
                         i < raw_union_stage_len * (uint32_t)DV4;
                         i += NW) {
                        sv4_union_base[i] = pv4_src[i];
                    }
                }
                if (raw_tile_intersection_k_stage) {
                    device const half4 *pk4_src =
                        k + (uint64_t)raw_tile_intersection_begin * DK4;
                    for (uint i = tiisg;
                         i < raw_tile_intersection_len * (uint32_t)DK4;
                         i += NW) {
                        sk4_tile_intersection_base[i] = pk4_src[i];
                    }
                }
                if (raw_tile_intersection_v_stage) {
                    device const half4 *pv4_src =
                        v + (uint64_t)raw_tile_intersection_begin * DV4;
                    for (uint i = tiisg;
                         i < raw_tile_intersection_len * (uint32_t)DV4;
                         i += NW) {
                        sv4_tile_intersection_base[i] = pv4_src[i];
                    }
                }
                if (comp_union_k_stage) {
                    device const half4 *pk4_src =
                        k + ((uint64_t)n_raw_union + comp_union_begin) * DK4;
                    for (uint i = tiisg;
                         i < comp_union_len * (uint32_t)DK4;
                         i += NW) {
                        sk4_comp_union_base[i] = pk4_src[i];
                    }
                }
                if (comp_union_v_stage) {
                    device const half4 *pv4_src =
                        v + ((uint64_t)n_raw_union + comp_union_begin) * DV4;
                    for (uint i = tiisg;
                         i < comp_union_len * (uint32_t)DV4;
                         i += NW) {
                        sv4_comp_union_base[i] = pv4_src[i];
                    }
                }
                if (mixed_chunk_union_k_stage) {
                    device const half4 *pk4_src =
                        k + (uint64_t)mixed_chunk_union_begin * DK4;
                    for (uint i = tiisg;
                         i < mixed_chunk_union_len * (uint32_t)DK4;
                         i += NW) {
                        sk4_mixed_chunk_union_base[i] = pk4_src[i];
                    }
                }
                if (mixed_chunk_union_v_stage) {
                    device const half4 *pv4_src =
                        v + (uint64_t)mixed_chunk_union_begin * DV4;
                    for (uint i = tiisg;
                         i < mixed_chunk_union_len * (uint32_t)DV4;
                         i += NW) {
                        sv4_mixed_chunk_union_base[i] = pv4_src[i];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        FOR_UNROLL (short cc = 0; cc < C; ++cc) {
            const uint32_t key = ic + (uint32_t)cc;
            const bool valid_key = key < n_keys;
            uint32_t mapped_key = 0;
            if (valid_key) {
                mapped_key = key < n_raw
                    ? raw_base + key
                    : n_raw_union + (key - n_raw);
            }
            FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                const short qi = ii * NW + tiisg;
                half4 mk = half4(0.0h);
                if (valid_key) {
                    if (mixed_chunk_union_k_stage &&
                        mapped_key >= mixed_chunk_union_begin &&
                        mapped_key < mixed_chunk_union_begin + mixed_chunk_union_len) {
                        const uint32_t union_off =
                            mapped_key - mixed_chunk_union_begin;
                        threadgroup const half4 *pk4 =
                            sk4_mixed_chunk_union_base +
                            (uint64_t)union_off * DK4 + tiisg;
                        mk = pk4[ii * NW];
                    } else if (common_k_stage) {
                        threadgroup const half4 *pk4 =
                            sk4_common_base + (uint64_t)cc * DK4 + tiisg;
                        mk = pk4[ii * NW];
                    } else if (raw_union_k_stage && key < n_raw) {
                        const uint32_t union_off =
                            mapped_key - raw_union_stage_begin;
                        threadgroup const half4 *pk4 =
                            sk4_union_base + (uint64_t)union_off * DK4 + tiisg;
                        mk = pk4[ii * NW];
                    } else if (raw_tile_intersection_k_stage &&
                               key < n_raw &&
                               mapped_key >= raw_tile_intersection_begin &&
                               mapped_key < raw_tile_intersection_begin + raw_tile_intersection_len) {
                        const uint32_t intersection_off =
                            mapped_key - raw_tile_intersection_begin;
                        threadgroup const half4 *pk4 =
                            sk4_tile_intersection_base +
                            (uint64_t)intersection_off * DK4 + tiisg;
                        mk = pk4[ii * NW];
                    } else if (comp_union_k_stage && key >= n_raw) {
                        const uint32_t comp_off = key - n_raw - comp_union_begin;
                        threadgroup const half4 *pk4 =
                            sk4_comp_union_base + (uint64_t)comp_off * DK4 + tiisg;
                        mk = pk4[ii * NW];
                    } else {
                        device const half4 *pk4 =
                            k + (uint64_t)mapped_key * DK4 + tiisg;
                        mk = pk4[ii * NW];
                    }
                }
                mqk[cc] += dot(float4(mk), float4(sq4[qi]));
            }
            mqk[cc] = simd_sum(mqk[cc]);
        }

        const bool valid = ic + tiisg < n_keys;
        float score = mqk[tiisg] * args.scale;
        score += valid ? 0.0f : -MAXHALF;
        ss[tiisg] = score;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float old_m = M;
        const float s = ss[tiisg];
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo[DV4 / NW] = { [0 ... DV4 / NW - 1] = float4(0.0f) };
        FOR_UNROLL (short cc = 0; cc < C; ++cc) {
            const uint32_t key = ic + (uint32_t)cc;
            const bool valid_key = key < n_keys;
            uint32_t mapped_key = 0;
            if (valid_key) {
                mapped_key = key < n_raw
                    ? raw_base + key
                    : n_raw_union + (key - n_raw);
            }
            const float weight = ss[cc];
            FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                half4 mv = half4(0.0h);
                if (valid_key) {
                    if (mixed_chunk_union_v_stage &&
                        mapped_key >= mixed_chunk_union_begin &&
                        mapped_key < mixed_chunk_union_begin + mixed_chunk_union_len) {
                        const uint32_t union_off =
                            mapped_key - mixed_chunk_union_begin;
                        threadgroup const half4 *pv4 =
                            sv4_mixed_chunk_union_base +
                            (uint64_t)union_off * DV4 + tiisg;
                        mv = pv4[ii * NW];
                    } else if (common_v_stage) {
                        threadgroup const half4 *pv4 =
                            sv4_common_base + (uint64_t)cc * DV4 + tiisg;
                        mv = pv4[ii * NW];
                    } else if (raw_union_v_stage && key < n_raw) {
                        const uint32_t union_off =
                            mapped_key - raw_union_stage_begin;
                        threadgroup const half4 *pv4 =
                            sv4_union_base + (uint64_t)union_off * DV4 + tiisg;
                        mv = pv4[ii * NW];
                    } else if (raw_tile_intersection_v_stage &&
                               key < n_raw &&
                               mapped_key >= raw_tile_intersection_begin &&
                               mapped_key < raw_tile_intersection_begin + raw_tile_intersection_len) {
                        const uint32_t intersection_off =
                            mapped_key - raw_tile_intersection_begin;
                        threadgroup const half4 *pv4 =
                            sv4_tile_intersection_base +
                            (uint64_t)intersection_off * DV4 + tiisg;
                        mv = pv4[ii * NW];
                    } else if (comp_union_v_stage && key >= n_raw) {
                        const uint32_t comp_off = key - n_raw - comp_union_begin;
                        threadgroup const half4 *pv4 =
                            sv4_comp_union_base + (uint64_t)comp_off * DV4 + tiisg;
                        mv = pv4[ii * NW];
                    } else {
                        device const half4 *pv4 =
                            v + (uint64_t)mapped_key * DV4 + tiisg;
                        mv = pv4[ii * NW];
                    }
                }
                lo[ii] += float4(mv) * weight;
            }
        }
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] += lo[ii];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (iwg == 0) {
        const float old_m = M;
        const float s = tiisg == 0 ? sinks[head] : -FLT_MAX / 2;
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
    }

    const int64_t nrows = args.ne3 * args.ne2 * args.ne1;
    const int64_t rid = head + row * args.ne1;
    device float4 *dst4 = (device float4 *) dst;
    device float  *dst1 = (device float  *) dst + nrows * DV * NWG;

    FOR_UNROLL (short i = 0; i < DV4 / NW; ++i) {
        const short idx = i * NW + tiisg;
        dst4[rid * DV4 * NWG + NWG * idx + iwg] = so4[idx];
    }
    if (tiisg == 0) {
        dst1[rid * (2 * NWG) + 2 * iwg + 0] = S;
        dst1[rid * (2 * NWG) + 2 * iwg + 1] = M;
    }
}

// DSpark strict verifier prototype: direct-resident variant of varmap rows5.
// It keeps the same row-local key order as varmap, but reads the raw ring and
// compressed cache directly instead of first materializing a shared F16 stream.
kernel void kernel_flash_attn_varmap_direct_rows5_f16_dk512_dv512(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char     * q,
        device const half4    * raw,
        device const half4    * comp,
        device const float    * sinks,
        device       char     * dst,
        device const uint32_t * row_raw_base,
        device const uint32_t * row_n_raw,
        device const uint32_t * row_n_comp,
        constant uint32_t     & raw_union_start,
        constant uint32_t     & n_raw_union,
        constant uint32_t     & raw_cap,
        threadgroup  half     * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK = 512;
    constexpr short DV = 512;
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short C = OP_FLASH_ATTN_EXT_VEC_NCPSG;
    constexpr short NW = N_SIMDWIDTH;
    const short NWG = (short)((args.ne31 > 0) ? args.ne31 : 32);

    const ushort row = sgitg;
    if (row >= args.ne01) {
        return;
    }

    const short iwg = tgpig[2];
    const ushort head = tgpig[1];
    const uint32_t n_raw = row_n_raw[row];
    const uint32_t n_comp = row_n_comp[row];
    const uint32_t n_keys = n_raw + n_comp;
    const uint32_t raw_base = row_raw_base[row];

    threadgroup half4 *sq4_base = (threadgroup half4 *)shmem_f16;
    threadgroup float  *ss_base =
        (threadgroup float *)(shmem_f16 + args.ne01 * DK);
    threadgroup float4 *so4_base =
        (threadgroup float4 *)(ss_base + args.ne01 * C);
    threadgroup half4 *sq4 = sq4_base + row * DK4;
    threadgroup float  *ss  = ss_base + row * C;
    threadgroup float4 *so4 = so4_base + row * DV4;

    device const float4 *q4 =
        (device const float4 *)(q + row * args.nb01 + head * args.nb02);

    for (short i = tiisg; i < DK4; i += NW) {
        sq4[i] = half4(q4[i]);
    }
    for (short i = tiisg; i < C; i += NW) {
        ss[i] = 0.0f;
    }
    for (short i = tiisg; i < DV4; i += NW) {
        so4[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;

    for (uint32_t ic = (uint32_t)iwg * C; ic < n_keys; ic += NWG * C) {
        float mqk[C] = { [0 ... C - 1] = 0.0f };

        FOR_UNROLL (short cc = 0; cc < C; ++cc) {
            const uint32_t key = ic + (uint32_t)cc;
            const bool valid_key = key < n_keys;
            const bool raw_key = valid_key && key < n_raw;
            uint32_t mapped_raw = 0;
            uint32_t mapped_comp = 0;
            if (valid_key) {
                if (raw_key) {
                    mapped_raw = raw_union_start + raw_base + key;
                    mapped_raw = mapped_raw % raw_cap;
                } else {
                    mapped_comp = key - n_raw;
                }
            }
            FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                const short qi = ii * NW + tiisg;
                half4 mk = half4(0.0h);
                if (valid_key) {
                    if (raw_key && raw_base + key < n_raw_union) {
                        mk = raw[(uint64_t)mapped_raw * DK4 + qi];
                    } else if (!raw_key) {
                        mk = comp[(uint64_t)mapped_comp * DK4 + qi];
                    }
                }
                mqk[cc] += dot(float4(mk), float4(sq4[qi]));
            }
            mqk[cc] = simd_sum(mqk[cc]);
        }

        const bool valid = ic + tiisg < n_keys;
        float score = mqk[tiisg] * args.scale;
        score += valid ? 0.0f : -MAXHALF;
        ss[tiisg] = score;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float old_m = M;
        const float s = ss[tiisg];
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo[DV4 / NW] = { [0 ... DV4 / NW - 1] = float4(0.0f) };
        FOR_UNROLL (short cc = 0; cc < C; ++cc) {
            const uint32_t key = ic + (uint32_t)cc;
            const bool valid_key = key < n_keys;
            const bool raw_key = valid_key && key < n_raw;
            uint32_t mapped_raw = 0;
            uint32_t mapped_comp = 0;
            if (valid_key) {
                if (raw_key) {
                    mapped_raw = raw_union_start + raw_base + key;
                    mapped_raw = mapped_raw % raw_cap;
                } else {
                    mapped_comp = key - n_raw;
                }
            }
            const float weight = ss[cc];
            FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                const short qi = ii * NW + tiisg;
                half4 mv = half4(0.0h);
                if (valid_key) {
                    if (raw_key && raw_base + key < n_raw_union) {
                        mv = raw[(uint64_t)mapped_raw * DV4 + qi];
                    } else if (!raw_key) {
                        mv = comp[(uint64_t)mapped_comp * DV4 + qi];
                    }
                }
                lo[ii] += float4(mv) * weight;
            }
        }
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] += lo[ii];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (iwg == 0) {
        const float old_m = M;
        const float s = tiisg == 0 ? sinks[head] : -FLT_MAX / 2;
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
    }

    const int64_t nrows = args.ne3 * args.ne2 * args.ne1;
    const int64_t rid = head + row * args.ne1;
    device float4 *dst4 = (device float4 *) dst;
    device float  *dst1 = (device float  *) dst + nrows * DV * NWG;

    FOR_UNROLL (short i = 0; i < DV4 / NW; ++i) {
        const short idx = i * NW + tiisg;
        dst4[rid * DV4 * NWG + NWG * idx + iwg] = so4[idx];
    }
    if (tiisg == 0) {
        dst1[rid * (2 * NWG) + 2 * iwg + 0] = S;
        dst1[rid * (2 * NWG) + 2 * iwg + 1] = M;
    }
}

kernel void kernel_flash_attn_varmap_direct_rows5_f32_dk512_dv512(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char     * q,
        device const float4   * raw,
        device const float4   * comp,
        device const float    * sinks,
        device       char     * dst,
        device const uint32_t * row_raw_base,
        device const uint32_t * row_n_raw,
        device const uint32_t * row_n_comp,
        constant uint32_t     & raw_union_start,
        constant uint32_t     & n_raw_union,
        constant uint32_t     & raw_cap,
        threadgroup  half     * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK = 512;
    constexpr short DV = 512;
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short C = OP_FLASH_ATTN_EXT_VEC_NCPSG;
    constexpr short NW = N_SIMDWIDTH;
    const short NWG = (short)((args.ne31 > 0) ? args.ne31 : 32);

    const ushort row = sgitg;
    if (row >= args.ne01) {
        return;
    }

    const short iwg = tgpig[2];
    const ushort head = tgpig[1];
    const uint32_t n_raw = row_n_raw[row];
    const uint32_t n_comp = row_n_comp[row];
    const uint32_t n_keys = n_raw + n_comp;
    const uint32_t raw_base = row_raw_base[row];

    threadgroup half4 *sq4_base = (threadgroup half4 *)shmem_f16;
    threadgroup float  *ss_base =
        (threadgroup float *)(shmem_f16 + args.ne01 * DK);
    threadgroup float4 *so4_base =
        (threadgroup float4 *)(ss_base + args.ne01 * C);
    threadgroup half4 *sq4 = sq4_base + row * DK4;
    threadgroup float  *ss  = ss_base + row * C;
    threadgroup float4 *so4 = so4_base + row * DV4;

    device const float4 *q4 =
        (device const float4 *)(q + row * args.nb01 + head * args.nb02);

    for (short i = tiisg; i < DK4; i += NW) {
        sq4[i] = half4(q4[i]);
    }
    for (short i = tiisg; i < C; i += NW) {
        ss[i] = 0.0f;
    }
    for (short i = tiisg; i < DV4; i += NW) {
        so4[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;

    for (uint32_t ic = (uint32_t)iwg * C; ic < n_keys; ic += NWG * C) {
        float mqk[C] = { [0 ... C - 1] = 0.0f };

        FOR_UNROLL (short cc = 0; cc < C; ++cc) {
            const uint32_t key = ic + (uint32_t)cc;
            const bool valid_key = key < n_keys;
            const bool raw_key = valid_key && key < n_raw;
            uint32_t mapped_raw = 0;
            uint32_t mapped_comp = 0;
            if (valid_key) {
                if (raw_key) {
                    mapped_raw = raw_union_start + raw_base + key;
                    mapped_raw = mapped_raw % raw_cap;
                } else {
                    mapped_comp = key - n_raw;
                }
            }
            FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                const short qi = ii * NW + tiisg;
                half4 mk = half4(0.0h);
                if (valid_key) {
                    if (raw_key && raw_base + key < n_raw_union) {
                        mk = half4(raw[(uint64_t)mapped_raw * DK4 + qi]);
                    } else if (!raw_key) {
                        mk = half4(comp[(uint64_t)mapped_comp * DK4 + qi]);
                    }
                }
                mqk[cc] += dot(float4(mk), float4(sq4[qi]));
            }
            mqk[cc] = simd_sum(mqk[cc]);
        }

        const bool valid = ic + tiisg < n_keys;
        float score = mqk[tiisg] * args.scale;
        score += valid ? 0.0f : -MAXHALF;
        ss[tiisg] = score;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float old_m = M;
        const float s = ss[tiisg];
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        ss[tiisg] = vs;
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 lo[DV4 / NW] = { [0 ... DV4 / NW - 1] = float4(0.0f) };
        FOR_UNROLL (short cc = 0; cc < C; ++cc) {
            const uint32_t key = ic + (uint32_t)cc;
            const bool valid_key = key < n_keys;
            const bool raw_key = valid_key && key < n_raw;
            uint32_t mapped_raw = 0;
            uint32_t mapped_comp = 0;
            if (valid_key) {
                if (raw_key) {
                    mapped_raw = raw_union_start + raw_base + key;
                    mapped_raw = mapped_raw % raw_cap;
                } else {
                    mapped_comp = key - n_raw;
                }
            }
            const float weight = ss[cc];
            FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                const short qi = ii * NW + tiisg;
                half4 mv = half4(0.0h);
                if (valid_key) {
                    if (raw_key && raw_base + key < n_raw_union) {
                        mv = half4(raw[(uint64_t)mapped_raw * DV4 + qi]);
                    } else if (!raw_key) {
                        mv = half4(comp[(uint64_t)mapped_comp * DV4 + qi]);
                    }
                }
                lo[ii] += float4(mv) * weight;
            }
        }
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] += lo[ii];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (iwg == 0) {
        const float old_m = M;
        const float s = tiisg == 0 ? sinks[head] : -FLT_MAX / 2;
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
    }

    const int64_t nrows = args.ne3 * args.ne2 * args.ne1;
    const int64_t rid = head + row * args.ne1;
    device float4 *dst4 = (device float4 *) dst;
    device float  *dst1 = (device float  *) dst + nrows * DV * NWG;

    FOR_UNROLL (short i = 0; i < DV4 / NW; ++i) {
        const short idx = i * NW + tiisg;
        dst4[rid * DV4 * NWG + NWG * idx + iwg] = so4[idx];
    }
    if (tiisg == 0) {
        dst1[rid * (2 * NWG) + 2 * iwg + 0] = S;
        dst1[rid * (2 * NWG) + 2 * iwg + 1] = M;
    }
}

// DSpark strict N<=6 direct-resident attention with shared K/V microtiles.
//
// The raw/compressed cache stores one 512-wide vector that is used for both K
// and V.  The ordinary direct-F32 kernel converts that vector independently in
// every verifier-row simdgroup.  This variant keeps the exact 32-key score /
// softmax / AV realization, keeps each lane's four half-Q values in registers,
// and stages fourteen keys at a time (plus at most the
// five-row positional span) once for all active rows.  Mixed raw/compressed
// boundary fragments and non-contiguous geometry retain the direct row-local
// read.  At N=6 the shared allocation is:
//   score + output state     = 13,056 B
//   19 half KV rows          = 19,456 B
//   total                    = 32,512 B
// so it remains below a 32 KiB threadgroup-memory budget.
kernel void kernel_flash_attn_varmap_direct_shared_rows6_f32_dk512_dv512(
        constant ds4_metal_args_flash_attn_ext_vec & args,
        device const char     * q,
        device const float4   * raw,
        device const float4   * comp,
        device const float    * sinks,
        device       char     * dst,
        device const uint32_t * row_raw_base,
        device const uint32_t * row_n_raw,
        device const uint32_t * row_n_comp,
        constant uint32_t     & raw_union_start,
        constant uint32_t     & n_raw_union,
        constant uint32_t     & raw_cap,
        threadgroup  half     * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint    tiitg[[thread_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK = 512;
    constexpr short DV = 512;
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short C = OP_FLASH_ATTN_EXT_VEC_NCPSG;
    constexpr short NW = N_SIMDWIDTH;
    constexpr uint32_t MICRO = 14;
    constexpr uint32_t MAX_ROWS = 6;
    constexpr uint32_t STAGE_ROWS = MICRO + MAX_ROWS - 1;
    const short NWG = (short)((args.ne31 > 0) ? args.ne31 : 32);
    const uint32_t tg_threads = (uint32_t)args.ne01 * (uint32_t)NW;

    const ushort row = sgitg;
    if (row >= args.ne01) {
        return;
    }

    const short iwg = tgpig[2];
    const ushort head = tgpig[1];
    const uint32_t n_raw = row_n_raw[row];
    const uint32_t n_comp = row_n_comp[row];
    const uint32_t n_keys = n_raw + n_comp;
    const uint32_t raw_base = row_raw_base[row];

    uint32_t max_n_keys = n_keys;
    FOR_UNROLL (short rr = 0; rr < (short)MAX_ROWS; ++rr) {
        if ((uint32_t)rr < (uint32_t)args.ne01) {
            const uint32_t row_keys = row_n_raw[rr] + row_n_comp[rr];
            if (row_keys > max_n_keys) max_n_keys = row_keys;
        }
    }

    threadgroup float  *ss_base = (threadgroup float *)shmem_f16;
    threadgroup float4 *so4_base =
        (threadgroup float4 *)(ss_base + args.ne01 * C);
    threadgroup half4 *skv4 =
        (threadgroup half4 *)(so4_base + args.ne01 * DV4);
    threadgroup float  *ss  = ss_base + row * C;
    threadgroup float4 *so4 = so4_base + row * DV4;

    device const float4 *q4 =
        (device const float4 *)(q + row * args.nb01 + head * args.nb02);

    half4 qreg[DK4 / NW];
    FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
        qreg[ii] = half4(q4[ii * NW + tiisg]);
    }
    for (short i = tiisg; i < C; i += NW) {
        ss[i] = 0.0f;
    }
    for (short i = tiisg; i < DV4; i += NW) {
        so4[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;

    for (uint32_t ic = (uint32_t)iwg * C; ic < max_n_keys; ic += NWG * C) {
        const bool row_chunk_active = ic < n_keys;
        float mqk[C] = { [0 ... C - 1] = 0.0f };

        // Score phase.  Microtiling changes only where the identical half KV
        // values are loaded from; all 32 scores are still reduced together.
        for (uint32_t mb = 0; mb < (uint32_t)C; mb += MICRO) {
            const uint32_t micro_len = min(MICRO, (uint32_t)C - mb);
            bool shared_raw = true;
            bool shared_comp = true;
            uint32_t union_begin = UINT_MAX;
            uint32_t union_end = 0;

            FOR_UNROLL (short rr = 0; rr < (short)MAX_ROWS; ++rr) {
                if ((uint32_t)rr < (uint32_t)args.ne01) {
                    const uint32_t rr_raw = row_n_raw[rr];
                    const uint32_t rr_keys = rr_raw + row_n_comp[rr];
                    const uint32_t rr_key = ic + mb;
                    const bool rr_full = rr_key + micro_len <= rr_keys;
                    const bool rr_is_raw = rr_full && rr_key + micro_len <= rr_raw;
                    const bool rr_is_comp = rr_full && rr_key >= rr_raw;
                    shared_raw = shared_raw && rr_is_raw;
                    shared_comp = shared_comp && rr_is_comp;
                    if (rr_is_raw) {
                        const uint32_t begin = row_raw_base[rr] + rr_key;
                        if (begin < union_begin) union_begin = begin;
                        if (begin + micro_len > union_end) union_end = begin + micro_len;
                    } else if (rr_is_comp) {
                        const uint32_t begin = rr_key - rr_raw;
                        if (begin < union_begin) union_begin = begin;
                        if (begin + micro_len > union_end) union_end = begin + micro_len;
                    }
                }
            }
            const bool shared_stage =
                (shared_raw || shared_comp) &&
                union_begin != UINT_MAX &&
                union_end >= union_begin &&
                union_end - union_begin <= STAGE_ROWS;

            if (shared_stage) {
                const uint32_t union_len = union_end - union_begin;
                const uint32_t stage_half4 = union_len * (uint32_t)DK4;
                for (uint32_t i = tiitg; i < stage_half4; i += tg_threads) {
                    const uint32_t stage_row = i / (uint32_t)DK4;
                    const uint32_t component = i - stage_row * (uint32_t)DK4;
                    if (shared_raw) {
                        const uint32_t logical = union_begin + stage_row;
                        const uint32_t physical = (raw_union_start + logical) % raw_cap;
                        skv4[i] = half4(raw[(uint64_t)physical * DK4 + component]);
                    } else {
                        skv4[i] = half4(comp[(uint64_t)(union_begin + stage_row) * DK4 + component]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row_chunk_active) {
                for (uint32_t mc = 0; mc < micro_len; mc++) {
                    const uint32_t cc = mb + mc;
                    const uint32_t key = ic + cc;
                    const bool valid_key = key < n_keys;
                    const bool raw_key = valid_key && key < n_raw;
                    uint32_t mapped_raw = 0;
                    uint32_t mapped_comp = 0;
                    if (valid_key) {
                        if (raw_key) {
                            mapped_raw = (raw_union_start + raw_base + key) % raw_cap;
                        } else {
                            mapped_comp = key - n_raw;
                        }
                    }
                    FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                        const short qi = ii * NW + tiisg;
                        half4 mk = half4(0.0h);
                        if (valid_key) {
                            if (shared_stage) {
                                const uint32_t logical = raw_key ? raw_base + key : mapped_comp;
                                const uint32_t stage_off = logical - union_begin;
                                mk = skv4[(uint64_t)stage_off * DK4 + qi];
                            } else if (raw_key && raw_base + key < n_raw_union) {
                                mk = half4(raw[(uint64_t)mapped_raw * DK4 + qi]);
                            } else if (!raw_key) {
                                mk = half4(comp[(uint64_t)mapped_comp * DK4 + qi]);
                            }
                        }
                        mqk[cc] += dot(float4(mk), float4(qreg[ii]));
                    }
                    mqk[cc] = simd_sum(mqk[cc]);
                }
            }
            if (shared_stage) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (row_chunk_active) {
            const bool valid = ic + tiisg < n_keys;
            float score = mqk[tiisg] * args.scale;
            score += valid ? 0.0f : -MAXHALF;
            ss[tiisg] = score;
            simdgroup_barrier(mem_flags::mem_threadgroup);

            const float old_m = M;
            const float s = ss[tiisg];
            M = simd_max(max(M, s));
            const float ms = exp(old_m - M);
            const float vs = exp(s - M);
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                so4[ii * NW + tiisg] *= ms;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        float4 lo[DV4 / NW] = { [0 ... DV4 / NW - 1] = float4(0.0f) };
        // AV phase.  Restage the same shared KV microtiles after all 32 softmax
        // weights are known; cc ordering remains identical to the direct path.
        for (uint32_t mb = 0; mb < (uint32_t)C; mb += MICRO) {
            const uint32_t micro_len = min(MICRO, (uint32_t)C - mb);
            bool shared_raw = true;
            bool shared_comp = true;
            uint32_t union_begin = UINT_MAX;
            uint32_t union_end = 0;

            FOR_UNROLL (short rr = 0; rr < (short)MAX_ROWS; ++rr) {
                if ((uint32_t)rr < (uint32_t)args.ne01) {
                    const uint32_t rr_raw = row_n_raw[rr];
                    const uint32_t rr_keys = rr_raw + row_n_comp[rr];
                    const uint32_t rr_key = ic + mb;
                    const bool rr_full = rr_key + micro_len <= rr_keys;
                    const bool rr_is_raw = rr_full && rr_key + micro_len <= rr_raw;
                    const bool rr_is_comp = rr_full && rr_key >= rr_raw;
                    shared_raw = shared_raw && rr_is_raw;
                    shared_comp = shared_comp && rr_is_comp;
                    if (rr_is_raw) {
                        const uint32_t begin = row_raw_base[rr] + rr_key;
                        if (begin < union_begin) union_begin = begin;
                        if (begin + micro_len > union_end) union_end = begin + micro_len;
                    } else if (rr_is_comp) {
                        const uint32_t begin = rr_key - rr_raw;
                        if (begin < union_begin) union_begin = begin;
                        if (begin + micro_len > union_end) union_end = begin + micro_len;
                    }
                }
            }
            const bool shared_stage =
                (shared_raw || shared_comp) &&
                union_begin != UINT_MAX &&
                union_end >= union_begin &&
                union_end - union_begin <= STAGE_ROWS;

            if (shared_stage) {
                const uint32_t union_len = union_end - union_begin;
                const uint32_t stage_half4 = union_len * (uint32_t)DV4;
                for (uint32_t i = tiitg; i < stage_half4; i += tg_threads) {
                    const uint32_t stage_row = i / (uint32_t)DV4;
                    const uint32_t component = i - stage_row * (uint32_t)DV4;
                    if (shared_raw) {
                        const uint32_t logical = union_begin + stage_row;
                        const uint32_t physical = (raw_union_start + logical) % raw_cap;
                        skv4[i] = half4(raw[(uint64_t)physical * DV4 + component]);
                    } else {
                        skv4[i] = half4(comp[(uint64_t)(union_begin + stage_row) * DV4 + component]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row_chunk_active) {
                for (uint32_t mc = 0; mc < micro_len; mc++) {
                    const uint32_t cc = mb + mc;
                    const uint32_t key = ic + cc;
                    const bool valid_key = key < n_keys;
                    const bool raw_key = valid_key && key < n_raw;
                    uint32_t mapped_raw = 0;
                    uint32_t mapped_comp = 0;
                    if (valid_key) {
                        if (raw_key) {
                            mapped_raw = (raw_union_start + raw_base + key) % raw_cap;
                        } else {
                            mapped_comp = key - n_raw;
                        }
                    }
                    const float weight = ss[cc];
                    FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                        const short qi = ii * NW + tiisg;
                        half4 mv = half4(0.0h);
                        if (valid_key) {
                            if (shared_stage) {
                                const uint32_t logical = raw_key ? raw_base + key : mapped_comp;
                                const uint32_t stage_off = logical - union_begin;
                                mv = skv4[(uint64_t)stage_off * DV4 + qi];
                            } else if (raw_key && raw_base + key < n_raw_union) {
                                mv = half4(raw[(uint64_t)mapped_raw * DV4 + qi]);
                            } else if (!raw_key) {
                                mv = half4(comp[(uint64_t)mapped_comp * DV4 + qi]);
                            }
                        }
                        lo[ii] += float4(mv) * weight;
                    }
                }
            }
            if (shared_stage) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (row_chunk_active) {
            FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                so4[ii * NW + tiisg] += lo[ii];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (iwg == 0) {
        const float old_m = M;
        const float s = tiisg == 0 ? sinks[head] : -FLT_MAX / 2;
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
    }

    const int64_t nrows = args.ne3 * args.ne2 * args.ne1;
    const int64_t rid = head + row * args.ne1;
    device float4 *dst4 = (device float4 *) dst;
    device float  *dst1 = (device float  *) dst + nrows * DV * NWG;

    FOR_UNROLL (short i = 0; i < DV4 / NW; ++i) {
        const short idx = i * NW + tiisg;
        dst4[rid * DV4 * NWG + NWG * idx + iwg] = so4[idx];
    }
    if (tiisg == 0) {
        dst1[rid * (2 * NWG) + 2 * iwg + 0] = S;
        dst1[rid * (2 * NWG) + 2 * iwg + 1] = M;
    }
}

#undef FA_TYPES
#undef FA_TYPES_F32

kernel void kernel_dspark_phase_range_probe(
        device const ds4_dspark_attn_phase_range_metal * ranges,
        constant uint32_t                              & n_ranges,
        device uint32_t                                * out,
        device const ushort4                           * k_bits,
        device const ushort4                           * v_bits,
        constant uint32_t                              & n_raw_union,
        constant uint32_t                              & row_half4,
        constant uint32_t                              & scan_kv,
        uint                                             tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    uint32_t raw_keys = 0;
    uint32_t raw_ranges = 0;
    uint32_t comp_keys = 0;
    uint32_t comp_ranges = 0;
    uint32_t lane_mask = 0;
    uint32_t max_run = 0;
    uint32_t checksum = 2166136261u;
    uint32_t bad = 0;
    uint32_t k_checksum = 2166136261u;
    uint32_t v_checksum = 2166136261u;
    uint32_t kv_reads = 0;
    uint32_t kv_bad = 0;

    for (uint32_t i = 0; i < n_ranges; i++) {
        const ds4_dspark_attn_phase_range_metal r = ranges[i];
        const uint32_t len = (uint32_t)r.len;
        const uint32_t kind = (uint32_t)r.kind;
        if (kind == 0u) {
            raw_keys += len;
            raw_ranges++;
        } else if (kind == 1u || kind == 2u) {
            comp_keys += len;
            comp_ranges++;
        } else {
            bad++;
        }
        if (len > max_run) max_run = len;
        lane_mask |= 1u << ((uint32_t)r.lane & 31u);
        checksum ^= ((uint32_t)r.begin) |
                    (len << 16) |
                    (kind << 28) |
                    (((uint32_t)r.lane & 15u) << 24);
        checksum *= 16777619u;

        if (scan_kv != 0u && row_half4 != 0u && (kind == 0u || kind == 1u)) {
            const uint32_t stream_begin =
                (kind == 0u) ? (uint32_t)r.begin : n_raw_union + (uint32_t)r.begin;
            for (uint32_t j = 0; j < len; j++) {
                const uint32_t stream_row = stream_begin + j;
                for (uint32_t h = 0; h < row_half4; h++) {
                    const ushort4 kb = k_bits[(uint64_t)stream_row * row_half4 + h];
                    const ushort4 vb = v_bits[(uint64_t)stream_row * row_half4 + h];
                    k_checksum ^= (uint32_t)kb.x | ((uint32_t)kb.y << 16);
                    k_checksum *= 16777619u;
                    k_checksum ^= (uint32_t)kb.z | ((uint32_t)kb.w << 16);
                    k_checksum *= 16777619u;
                    v_checksum ^= (uint32_t)vb.x | ((uint32_t)vb.y << 16);
                    v_checksum *= 16777619u;
                    v_checksum ^= (uint32_t)vb.z | ((uint32_t)vb.w << 16);
                    v_checksum *= 16777619u;
                    kv_reads++;
                }
            }
        } else if (scan_kv != 0u && row_half4 == 0u) {
            kv_bad++;
        }
    }

    out[0] = raw_keys;
    out[1] = raw_ranges;
    out[2] = comp_keys;
    out[3] = comp_ranges;
    out[4] = lane_mask;
    out[5] = max_run;
    out[6] = checksum;
    out[7] = bad;
    out[8] = k_checksum;
    out[9] = v_checksum;
    out[10] = kv_reads;
    out[11] = kv_bad;
}

kernel void kernel_dspark_mixed_prefix_range_probe(
        device const ds4_dspark_attn_mixed_phase_range_metal * ranges,
        constant uint32_t                                    & n_ranges,
        device uint32_t                                      * out,
        uint                                                   tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    out[15] = 0x4d495850u; // "MIXP"

    uint32_t raw_keys = 0;
    uint32_t raw_ranges = 0;
    uint32_t comp_keys = 0;
    uint32_t comp_ranges = 0;
    uint32_t shared_ranges = 0;
    uint32_t private_ranges = 0;
    uint32_t row_mask = 0;
    uint32_t lane_mask = 0;
    uint32_t bad = 0;
    uint32_t checksum = 2166136261u;

    for (uint32_t i = 0; i < n_ranges; i++) {
        const ds4_dspark_attn_mixed_phase_range_metal r = ranges[i];
        const uint32_t len = (uint32_t)r.len;
        const uint32_t kind = (uint32_t)r.kind;
        const uint32_t row = (uint32_t)r.row;
        if (kind == 0u) {
            raw_keys += len;
            raw_ranges++;
        } else if (kind == 1u || kind == 2u) {
            comp_keys += len;
            comp_ranges++;
        } else {
            bad++;
        }
        if (row == 31u) {
            shared_ranges++;
        } else if (row < 5u) {
            private_ranges++;
            row_mask |= 1u << row;
        } else {
            bad++;
        }
        lane_mask |= 1u << ((uint32_t)r.lane & 31u);
        checksum ^= ((uint32_t)r.begin) |
                    (len << 16) |
                    (kind << 28) |
                    (((uint32_t)r.lane & 15u) << 24);
        checksum *= 16777619u;
        checksum ^= ((uint32_t)r.phase) | (((uint32_t)r.row) << 8);
        checksum *= 16777619u;
    }

    out[0] = raw_keys;
    out[1] = raw_ranges;
    out[2] = comp_keys;
    out[3] = comp_ranges;
    out[4] = shared_ranges;
    out[5] = private_ranges;
    out[6] = row_mask;
    out[7] = bad;
    out[8] = checksum;
    out[9] = lane_mask;
    out[10] = n_ranges;
}

kernel void kernel_dspark_phase_softmax_probe(
        constant ds4_metal_args_flash_attn_ext_vec    & args,
        device const char                             * q,
        device const half4                            * k,
        device const ds4_dspark_attn_phase_range_metal * ranges,
        constant uint32_t                             & n_ranges,
        constant uint32_t                             & n_raw_union,
        constant uint32_t                             & nwg_arg,
        device float                                  * out,
        uint3                                           tgpig [[threadgroup_position_in_grid]],
        ushort                                          tiisg [[thread_index_in_simdgroup]]) {
    constexpr short DK = 512;
    constexpr short DK4 = DK / 4;
    constexpr short NW = N_SIMDWIDTH;
    const uint32_t row = tgpig.x;
    const uint32_t head = tgpig.y;
    const uint32_t NWG = min(max(nwg_arg, 1u), 32u);
    if (row >= (uint32_t)args.ne01 || head >= (uint32_t)args.ne02) {
        return;
    }

    float S_lane[32];
    float M_lane[32];
    if (tiisg == 0) {
        for (uint32_t i = 0; i < 32u; i++) {
            S_lane[i] = 0.0f;
            M_lane[i] = -FLT_MAX / 2;
        }
    }

    device const float4 *q4 =
        (device const float4 *)(q + row * args.nb01 + head * args.nb02);

    for (uint32_t ir = 0; ir < n_ranges; ir++) {
        const ds4_dspark_attn_phase_range_metal range = ranges[ir];
        const uint32_t lane = (uint32_t)range.lane;
        const uint32_t kind = (uint32_t)range.kind;
        if (lane >= NWG || (kind != 0u && kind != 1u)) {
            continue;
        }
        const uint32_t stream_begin =
            (kind == 0u) ? (uint32_t)range.begin : n_raw_union + (uint32_t)range.begin;
        for (uint32_t j = 0; j < (uint32_t)range.len; j++) {
            const uint32_t stream_row = stream_begin + j;
            float mqk = 0.0f;
            FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                const short qi = ii * NW + tiisg;
                const half4 mk = k[(uint64_t)stream_row * DK4 + qi];
                mqk += dot(float4(mk), float4(q4[qi]));
            }
            const float score = simd_sum(mqk) * args.scale;
            if (tiisg == 0) {
                const float old_m = M_lane[lane];
                const float new_m = max(old_m, score);
                const float ms = exp(old_m - new_m);
                const float vs = exp(score - new_m);
                S_lane[lane] = S_lane[lane] * ms + vs;
                M_lane[lane] = new_m;
            }
        }
    }

    if (tiisg == 0) {
        const uint32_t row_base = (row * (uint32_t)args.ne02 + head) * NWG * 2u;
        for (uint32_t lane = 0; lane < NWG; lane++) {
            out[row_base + lane * 2u + 0u] = S_lane[lane];
            out[row_base + lane * 2u + 1u] = M_lane[lane];
        }
    }
}

kernel void kernel_dspark_phase_head_probe(
        constant ds4_metal_args_flash_attn_ext_vec     & args,
        device const char                              * q,
        device const half4                             * k,
        device const half4                             * v,
        device const float                             * sinks,
        device const ds4_dspark_attn_phase_range_metal * ranges,
        constant uint32_t                              & n_ranges,
        constant uint32_t                              & n_raw_union,
        constant uint32_t                              & nwg_arg,
        device char                                    * dst,
        uint3                                            tgpig [[threadgroup_position_in_grid]],
        ushort                                           tiisg [[thread_index_in_simdgroup]]) {
    constexpr short DK = 512;
    constexpr short DV = 512;
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short NW = N_SIMDWIDTH;

    const uint32_t row = tgpig.x;
    const uint32_t head = tgpig.y;
    const uint32_t iwg = tgpig.z;
    const uint32_t NWG = min(max(nwg_arg, 1u), 32u);
    if (row >= (uint32_t)args.ne01 ||
        head >= (uint32_t)args.ne02 ||
        iwg >= NWG) {
        return;
    }

    device const float4 *q4 =
        (device const float4 *)(q + row * args.nb01 + head * args.nb02);

    float S = 0.0f;
    float M = -FLT_MAX / 2;
    float4 so4[DV4 / NW] = { [0 ... DV4 / NW - 1] = float4(0.0f) };

    for (uint32_t ir = 0; ir < n_ranges; ir++) {
        const ds4_dspark_attn_phase_range_metal range = ranges[ir];
        const uint32_t lane = (uint32_t)range.lane;
        const uint32_t kind = (uint32_t)range.kind;
        if (lane != iwg || (kind != 0u && kind != 1u)) {
            continue;
        }
        const uint32_t stream_begin =
            (kind == 0u) ? (uint32_t)range.begin : n_raw_union + (uint32_t)range.begin;
        for (uint32_t j = 0; j < (uint32_t)range.len; j++) {
            const uint32_t stream_row = stream_begin + j;
            float mqk = 0.0f;
            FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                const short qi = ii * NW + tiisg;
                const half4 mk = k[(uint64_t)stream_row * DK4 + qi];
                mqk += dot(float4(mk), float4(q4[qi]));
            }
            const float score = simd_sum(mqk) * args.scale;
            const float old_m = M;
            M = max(M, score);
            const float ms = exp(old_m - M);
            const float vs = exp(score - M);
            S = S * ms + vs;
            FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                so4[ii] *= ms;
            }

            FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                const short qi = ii * NW + tiisg;
                const half4 mv = v[(uint64_t)stream_row * DV4 + qi];
                so4[ii] += float4(mv) * vs;
            }
        }
    }

    if (iwg == 0) {
        const float old_m = M;
        const float s = tiisg == 0 ? sinks[head] : -FLT_MAX / 2;
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii] *= ms;
        }
    }

    const int64_t nrows = args.ne3 * args.ne2 * args.ne1;
    const int64_t rid = head + row * args.ne1;
    device float4 *dst4 = (device float4 *) dst;
    device float  *dst1 = (device float  *) dst + nrows * DV * NWG;

    FOR_UNROLL (short i = 0; i < DV4 / NW; ++i) {
        const short idx = i * NW + tiisg;
        dst4[rid * DV4 * NWG + NWG * idx + iwg] = so4[i];
    }
    if (tiisg == 0) {
        dst1[rid * (2 * NWG) + 2 * iwg + 0] = S;
        dst1[rid * (2 * NWG) + 2 * iwg + 1] = M;
    }
}

kernel void kernel_dspark_phase_head_shared_probe(
        constant ds4_metal_args_flash_attn_ext_vec      & args,
        device const char                               * q,
        device const half4                              * k,
        device const half4                              * v,
        device const float                              * sinks,
        device const ds4_dspark_attn_phase_range_metal  * ranges,
        constant uint32_t                               & n_ranges,
        constant uint32_t                               & n_raw_union,
        constant uint32_t                               & nwg_arg,
        device char                                     * dst,
        threadgroup half                                * shmem_f16 [[threadgroup(0)]],
        uint3                                             tgpig [[threadgroup_position_in_grid]],
        ushort                                            tiisg [[thread_index_in_simdgroup]],
        ushort                                            sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short DK = 512;
    constexpr short DV = 512;
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short C = OP_FLASH_ATTN_EXT_VEC_NCPSG;
    constexpr short NW = N_SIMDWIDTH;

    const ushort row = sgitg;
    if (row >= args.ne01) {
        return;
    }

    const uint32_t head = tgpig.y;
    const uint32_t iwg = tgpig.z;
    const uint32_t NWG = min(max(nwg_arg, 1u), 32u);
    if (head >= (uint32_t)args.ne02 || iwg >= NWG) {
        return;
    }

    threadgroup half4 *sq4_base = (threadgroup half4 *)shmem_f16;
    threadgroup float *ss_base =
        (threadgroup float *)(shmem_f16 + args.ne01 * DK);
    threadgroup float4 *so4_base =
        (threadgroup float4 *)(ss_base + args.ne01 * C);
    threadgroup half4 *sk4 =
        (threadgroup half4 *)(so4_base + args.ne01 * DV4);
    threadgroup half4 *sv4 = sk4 + C * DK4;
    threadgroup half4 *sq4 = sq4_base + row * DK4;
    threadgroup float *ss = ss_base + row * C;
    threadgroup float4 *so4 = so4_base + row * DV4;

    device const float4 *q4 =
        (device const float4 *)(q + row * args.nb01 + head * args.nb02);

    for (short i = tiisg; i < DK4; i += NW) {
        sq4[i] = half4(q4[i]);
    }
    for (short i = tiisg; i < C; i += NW) {
        ss[i] = 0.0f;
    }
    for (short i = tiisg; i < DV4; i += NW) {
        so4[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;

    for (uint32_t ir = 0; ir < n_ranges; ir++) {
        const ds4_dspark_attn_phase_range_metal range = ranges[ir];
        const uint32_t lane = (uint32_t)range.lane;
        const uint32_t kind = (uint32_t)range.kind;
        if (lane != iwg || (kind != 0u && kind != 1u)) {
            continue;
        }
        const uint32_t stream_begin =
            (kind == 0u) ? (uint32_t)range.begin : n_raw_union + (uint32_t)range.begin;
        for (uint32_t base = 0; base < (uint32_t)range.len; base += C) {
            const uint32_t chunk_len = min((uint32_t)C, (uint32_t)range.len - base);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row == 0) {
                device const half4 *pk4_src =
                    k + ((uint64_t)stream_begin + base) * DK4;
                device const half4 *pv4_src =
                    v + ((uint64_t)stream_begin + base) * DV4;
                for (uint i = tiisg; i < chunk_len * (uint32_t)DK4; i += NW) {
                    sk4[i] = pk4_src[i];
                }
                for (uint i = tiisg; i < chunk_len * (uint32_t)DV4; i += NW) {
                    sv4[i] = pv4_src[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float mqk[C] = { [0 ... C - 1] = 0.0f };
            FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                const bool valid_key = (uint32_t)cc < chunk_len;
                FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                    const short qi = ii * NW + tiisg;
                    half4 mk = half4(0.0h);
                    if (valid_key) {
                        threadgroup const half4 *pk4 =
                            sk4 + (uint64_t)cc * DK4 + tiisg;
                        mk = pk4[ii * NW];
                    }
                    mqk[cc] += dot(float4(mk), float4(sq4[qi]));
                }
                mqk[cc] = simd_sum(mqk[cc]);
            }

            const bool valid = tiisg < chunk_len;
            float score = mqk[tiisg] * args.scale;
            score += valid ? 0.0f : -MAXHALF;
            ss[tiisg] = score;
            simdgroup_barrier(mem_flags::mem_threadgroup);

            const float old_m = M;
            const float s = ss[tiisg];
            M = simd_max(max(M, s));
            const float ms = exp(old_m - M);
            const float vs = exp(s - M);
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                so4[ii * NW + tiisg] *= ms;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            float4 lo[DV4 / NW] = { [0 ... DV4 / NW - 1] = float4(0.0f) };
            FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                const bool valid_key = (uint32_t)cc < chunk_len;
                const float weight = ss[cc];
                FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                    half4 mv = half4(0.0h);
                    if (valid_key) {
                        threadgroup const half4 *pv4 =
                            sv4 + (uint64_t)cc * DV4 + tiisg;
                        mv = pv4[ii * NW];
                    }
                    lo[ii] += float4(mv) * weight;
                }
            }
            FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                so4[ii * NW + tiisg] += lo[ii];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (iwg == 0) {
        const float old_m = M;
        const float s = tiisg == 0 ? sinks[head] : -FLT_MAX / 2;
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
    }

    const int64_t nrows = args.ne3 * args.ne2 * args.ne1;
    const int64_t rid = head + row * args.ne1;
    device float4 *dst4 = (device float4 *) dst;
    device float  *dst1 = (device float  *) dst + nrows * DV * NWG;

    FOR_UNROLL (short i = 0; i < DV4 / NW; ++i) {
        const short idx = i * NW + tiisg;
        dst4[rid * DV4 * NWG + NWG * idx + iwg] = so4[idx];
    }
    if (tiisg == 0) {
        dst1[rid * (2 * NWG) + 2 * iwg + 0] = S;
        dst1[rid * (2 * NWG) + 2 * iwg + 1] = M;
    }
}

// Compare-only DSpark shared-prefix candidate. The range buffer contains raw
// and compressed descriptors split into private row work plus shared
// intersection work. Current host code keeps strict varmap heads authoritative
// and only compares this candidate against them.
kernel void kernel_dspark_mixed_prefix_head_probe(
        constant ds4_metal_args_flash_attn_ext_vec             & args,
        device const char                                      * q,
        device const half4                                     * k,
        device const half4                                     * v,
        device const float                                     * sinks,
        device const ds4_dspark_attn_mixed_phase_range_metal   * ranges,
        constant uint32_t                                      & n_ranges,
        constant uint32_t                                      & n_raw_union,
        constant uint32_t                                      & nwg_arg,
        device char                                            * dst,
        device const uint32_t                                  * row_raw_base,
        device const uint32_t                                  * row_n_raw,
        device const uint32_t                                  * row_n_comp,
        threadgroup half                                       * shmem_f16 [[threadgroup(0)]],
        uint3                                                    tgpig [[threadgroup_position_in_grid]],
        ushort                                                   tiisg [[thread_index_in_simdgroup]],
        ushort                                                   sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short DK = 512;
    constexpr short DV = 512;
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short C = OP_FLASH_ATTN_EXT_VEC_NCPSG;
    constexpr short NW = N_SIMDWIDTH;
    constexpr uchar ROW_SHARED = 31;
    constexpr uint KIND_ROWKEY = 2;

    const ushort row = sgitg;
    if (row >= args.ne01) {
        return;
    }

    const uint32_t head = tgpig.y;
    const uint32_t iwg = tgpig.z;
    const uint32_t NWG = min(max(nwg_arg, 1u), 32u);
    const bool direct_read = args.ne32 != 0;
    if (head >= (uint32_t)args.ne02 || iwg >= NWG) {
        return;
    }

    threadgroup half4 *sq4_base = (threadgroup half4 *)shmem_f16;
    threadgroup float *ss_base =
        (threadgroup float *)(shmem_f16 + args.ne01 * DK);
    threadgroup float4 *so4_base =
        (threadgroup float4 *)(ss_base + args.ne01 * C);
    threadgroup half4 *sk4 =
        (threadgroup half4 *)(so4_base + args.ne01 * DV4);
    threadgroup half4 *sv4 = sk4 + C * DK4;
    threadgroup half4 *sq4 = sq4_base + row * DK4;
    threadgroup float *ss = ss_base + row * C;
    threadgroup float4 *so4 = so4_base + row * DV4;

    device const float4 *q4 =
        (device const float4 *)(q + row * args.nb01 + head * args.nb02);

    for (short i = tiisg; i < DK4; i += NW) {
        sq4[i] = half4(q4[i]);
    }
    for (short i = tiisg; i < C; i += NW) {
        ss[i] = 0.0f;
    }
    for (short i = tiisg; i < DV4; i += NW) {
        so4[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;

    for (uint32_t ir = 0; ir < n_ranges; ir++) {
        const ds4_dspark_attn_mixed_phase_range_metal range = ranges[ir];
        const uint32_t lane = (uint32_t)range.lane;
        const uint32_t kind = (uint32_t)range.kind;
        const bool shared_range = range.row == ROW_SHARED;
        const bool row_range = (uint32_t)range.row == (uint32_t)row;
        const bool consume_range = shared_range || row_range;
        const bool load_range = shared_range ? (row == 0) : row_range;
        const bool rowkey_range = kind == KIND_ROWKEY;
        const bool staged_rowkey_range = rowkey_range && shared_range && !direct_read;
        if (lane != iwg || (kind != 0u && kind != 1u && !rowkey_range)) {
            continue;
        }

        uint32_t stream_begin =
            (kind == 0u) ? (uint32_t)range.begin : n_raw_union + (uint32_t)range.begin;
        if (staged_rowkey_range) {
            const uint32_t row_key = (uint32_t)range.begin;
            const uint32_t n_raw0 = row_n_raw[0];
            stream_begin = row_key < n_raw0
                ? row_raw_base[0] + row_key
                : n_raw_union + (row_key - n_raw0);
        } else if (rowkey_range) {
            stream_begin = 0u;
        }
        for (uint32_t base = 0; base < (uint32_t)range.len; base += C) {
            const uint32_t chunk_len = min((uint32_t)C, (uint32_t)range.len - base);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (load_range && !direct_read && (!rowkey_range || staged_rowkey_range)) {
                device const half4 *pk4_src =
                    k + ((uint64_t)stream_begin + base) * DK4;
                device const half4 *pv4_src =
                    v + ((uint64_t)stream_begin + base) * DV4;
                for (uint i = tiisg; i < chunk_len * (uint32_t)DK4; i += NW) {
                    sk4[i] = pk4_src[i];
                }
                for (uint i = tiisg; i < chunk_len * (uint32_t)DV4; i += NW) {
                    sv4[i] = pv4_src[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (consume_range) {
                float mqk[C] = { [0 ... C - 1] = 0.0f };
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    const bool valid_key = (uint32_t)cc < chunk_len;
                    FOR_UNROLL (short ii = 0; ii < DK4 / NW; ++ii) {
                        const short qi = ii * NW + tiisg;
                        half4 mk = half4(0.0h);
                        if (valid_key) {
                            if (direct_read || (rowkey_range && !staged_rowkey_range)) {
                                uint32_t stream_row = stream_begin + base + (uint32_t)cc;
                                if (rowkey_range) {
                                    const uint32_t row_key =
                                        (uint32_t)range.begin + base + (uint32_t)cc;
                                    const uint32_t n_raw = row_n_raw[row];
                                    stream_row = row_key < n_raw
                                        ? row_raw_base[row] + row_key
                                        : n_raw_union + (row_key - n_raw);
                                }
                                device const half4 *pk4 =
                                    k + (uint64_t)stream_row * DK4 + qi;
                                mk = *pk4;
                            } else {
                                threadgroup const half4 *pk4 =
                                    sk4 + (uint64_t)cc * DK4 + tiisg;
                                mk = pk4[ii * NW];
                            }
                        }
                        mqk[cc] += dot(float4(mk), float4(sq4[qi]));
                    }
                    mqk[cc] = simd_sum(mqk[cc]);
                }

                const bool valid = tiisg < chunk_len;
                float score = mqk[tiisg] * args.scale;
                score += valid ? 0.0f : -MAXHALF;
                ss[tiisg] = score;
                simdgroup_barrier(mem_flags::mem_threadgroup);

                const float old_m = M;
                const float s = ss[tiisg];
                M = simd_max(max(M, s));
                const float ms = exp(old_m - M);
                const float vs = exp(s - M);
                S = S * ms + simd_sum(vs);
                ss[tiisg] = vs;
                FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                    so4[ii * NW + tiisg] *= ms;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);

                float4 lo[DV4 / NW] = { [0 ... DV4 / NW - 1] = float4(0.0f) };
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    const bool valid_key = (uint32_t)cc < chunk_len;
                    const float weight = ss[cc];
                    FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                        half4 mv = half4(0.0h);
                        if (valid_key) {
                            if (direct_read || (rowkey_range && !staged_rowkey_range)) {
                                uint32_t stream_row = stream_begin + base + (uint32_t)cc;
                                if (rowkey_range) {
                                    const uint32_t row_key =
                                        (uint32_t)range.begin + base + (uint32_t)cc;
                                    const uint32_t n_raw = row_n_raw[row];
                                    stream_row = row_key < n_raw
                                        ? row_raw_base[row] + row_key
                                        : n_raw_union + (row_key - n_raw);
                                }
                                device const half4 *pv4 =
                                    v + (uint64_t)stream_row * DV4 + (ii * NW + tiisg);
                                mv = *pv4;
                            } else {
                                threadgroup const half4 *pv4 =
                                    sv4 + (uint64_t)cc * DV4 + tiisg;
                                mv = pv4[ii * NW];
                            }
                        }
                        lo[ii] += float4(mv) * weight;
                    }
                }
                FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
                    so4[ii * NW + tiisg] += lo[ii];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (iwg == 0) {
        const float old_m = M;
        const float s = tiisg == 0 ? sinks[head] : -FLT_MAX / 2;
        M = simd_max(max(M, s));
        const float ms = exp(old_m - M);
        const float vs = exp(s - M);
        S = S * ms + simd_sum(vs);
        FOR_UNROLL (short ii = 0; ii < DV4 / NW; ++ii) {
            so4[ii * NW + tiisg] *= ms;
        }
    }

    const int64_t nrows = args.ne3 * args.ne2 * args.ne1;
    const int64_t rid = head + row * args.ne1;
    device float4 *dst4 = (device float4 *) dst;
    device float  *dst1 = (device float  *) dst + nrows * DV * NWG;

    FOR_UNROLL (short i = 0; i < DV4 / NW; ++i) {
        const short idx = i * NW + tiisg;
        dst4[rid * DV4 * NWG + NWG * idx + iwg] = so4[idx];
    }
    if (tiisg == 0) {
        dst1[rid * (2 * NWG) + 2 * iwg + 0] = S;
        dst1[rid * (2 * NWG) + 2 * iwg + 1] = M;
    }
}

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

// Reduces split-K decode FlashAttention partials. It combines each workgroup's
// output vector and softmax (sum,max) pair into the final attention result.
kernel void kernel_flash_attn_ext_vec_reduce(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;
    const bool active_iwg = iwg < NWG;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = active_iwg ? ss[rid*(2*NWG) + 2*iwg + 0] : 0.0f;
    float M = active_iwg ? ss[rid*(2*NWG) + 2*iwg + 1] : -FLT_MAX/2;

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 h = active_iwg ? htmp4[i*NWG + iwg] : float4(0.0f);
        const float4 v = simd_sum(h*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV
}
