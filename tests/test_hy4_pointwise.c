/* Native HY4 gate/reduction parity. No model is opened; < 1 MiB per case.
 * Covers real widths, offset views, exact in-place gating, queued producer to
 * consumer use, non-FMA/reassociation witnesses, and rejected buffer aliases. */
#include "../ds4_gpu.h"
#include "../hy4/hy4_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static unsigned checks, cases;
static uint32_t random_state = UINT32_C(0x48593450);
static double worst_error;
static double worst_scaled;
static const float guard = -98765.25f;

#define CHECK(expr) do { \
    ++checks; \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

typedef struct test_tensor {
    ds4_gpu_tensor *base, *view;
    float *data;
    size_t n;
} test_tensor;

static float random_float(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return ((int32_t)(random_state % 2049u) - 1024) / 1024.0f;
}

static test_tensor tensor_new(size_t n) {
    test_tensor t = {.n = n};
    t.base = ds4_gpu_tensor_alloc((n + 8u) * sizeof(float));
    CHECK(t.base);
    float *all = ds4_gpu_tensor_contents(t.base);
    CHECK(all);
    for (size_t i = 0; i < n + 8u; ++i) all[i] = guard;
    t.view = ds4_gpu_tensor_view(t.base, 4u * sizeof(float), n * sizeof(float));
    CHECK(t.view);
    t.data = ds4_gpu_tensor_contents(t.view);
    CHECK(t.data == all + 4);
    return t;
}

static void tensor_ready(test_tensor *t) {
    CHECK(ds4_gpu_tensor_did_modify(t->base, 0, (t->n + 8u) * sizeof(float)));
}

static void tensor_free(test_tensor *t) {
    const float *all = ds4_gpu_tensor_contents(t->base);
    for (size_t i = 0; i < 4u; ++i) {
        CHECK(all[i] == guard);
        CHECK(all[t->n + 4u + i] == guard);
    }
    ds4_gpu_tensor_free(t->view);
    ds4_gpu_tensor_free(t->base);
}


static void compare_gate(const float *actual,const float *expected,uint32_t n) {
    for (uint32_t j=0;j<n;j++) {
        double delta=fabs((double)actual[j]-expected[j]);
        // CPU libm and Metal exp need not round identically. Budget four
        // float epsilons for exp, division and multiplication; the expert
        // reduction below still requires bit identity.
        const double tolerance=4.0*FLT_EPSILON*fmax(1.0,fabs(expected[j]));
        double scaled=delta/fmax(1.0,fabs(expected[j]));
        if(scaled>worst_scaled) worst_scaled=scaled;
        CHECK(isfinite(actual[j]));
        if(delta > tolerance)
            fprintf(stderr,"gate mismatch n=%u j=%u actual=%.9g expected=%.9g abs=%.9g scaled=%.9g\n",n,j,actual[j],expected[j],delta,delta/fmax(1.0,fabs(expected[j])));
        CHECK(delta <= tolerance);
        if (delta>worst_error) worst_error=delta;
    }
}
static void run_case(uint32_t n) {
    test_tensor x=tensor_new(n), gate=tensor_new(n), out=tensor_new(n);
    test_tensor down=tensor_new((size_t)8*n), weights=tensor_new(8);
    float *expected=malloc(n*sizeof(float)), *gated=malloc(n*sizeof(float));
    CHECK(expected && gated);
    const float logits[]={-100.f,-20.f,-1.f,-0.f,0.f,1.f,20.f,100.f};
    const float w[]={1.f,0x1.000002p0f,0.375f,-0.25f,0.f,0.5f,0.03125f,0.0625f};
    memcpy(weights.data,w,sizeof(w));
    for(uint32_t j=0;j<n;j++) {
        x.data[j]=random_float()*16.f;
        gate.data[j]=j%2 ? random_float()*10.f : logits[(j/2)%8];
        for(uint32_t k=0;k<8;k++) down.data[(size_t)k*n+j]=random_float()*32.f;
    }
    // A contracted multiply-add would produce 2^-46 instead of zero.
    for(uint32_t k=0;k<8;k++) down.data[(size_t)k*n]=0.f;
    down.data[0]=-0x1.000004p0f; down.data[n]=0x1.000002p0f;
    CHECK(fmaf(down.data[n],w[1],down.data[0])!=0.f);
    // An unordered tree sum can lose a later small contribution.
    if(n>1) { down.data[1]=0x1p80f; down.data[n+1]=-0x1p80f; }
    for(uint32_t j=0;j<n;j++) {
        expected[j]=hy4_f32_mul(down.data[j],w[0]);
        for(uint32_t k=1;k<8;k++) expected[j]=hy4_f32_add(expected[j],hy4_f32_mul(down.data[(size_t)k*n+j],w[k]));
    }
    CHECK(expected[0]==0.f);
    hy4_sigmoid_mul(gated,x.data,gate.data,n);
    tensor_ready(&x);tensor_ready(&gate);tensor_ready(&down);tensor_ready(&weights);
    CHECK(ds4_gpu_hy4_sigmoid_mul_tensor(out.view,x.view,gate.view,n));
    CHECK(ds4_gpu_synchronize());compare_gate(out.data,gated,n);
    CHECK(ds4_gpu_hy4_weighted_sum8_tensor(out.view,down.view,weights.view,n));
    CHECK(ds4_gpu_synchronize());
    for(uint32_t j=0;j<n;j++) CHECK(memcmp(out.data+j,expected+j,sizeof(float))==0);
    // Encode a reduction and in-place gate in one batch, without a CPU fence.
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_hy4_weighted_sum8_tensor(out.view,down.view,weights.view,n));
    CHECK(ds4_gpu_hy4_sigmoid_mul_tensor(out.view,out.view,gate.view,n));
    CHECK(ds4_gpu_synchronize());
    hy4_sigmoid_mul(gated,expected,gate.data,n);compare_gate(out.data,gated,n);
    CHECK(!ds4_gpu_hy4_weighted_sum8_tensor(down.view,down.view,weights.view,n));
    CHECK(!ds4_gpu_hy4_weighted_sum8_tensor(out.view,down.view,weights.view,0));
    CHECK(!ds4_gpu_hy4_weighted_sum8_tensor(out.view,down.view,NULL,n));
    CHECK(!ds4_gpu_hy4_sigmoid_mul_tensor(NULL,x.view,gate.view,n));
    CHECK(!ds4_gpu_hy4_sigmoid_mul_tensor(out.view,x.view,gate.view,n+1));
    ds4_gpu_tensor *short_w=ds4_gpu_tensor_view(weights.base,16,7*sizeof(float));
    ds4_gpu_tensor *partial=ds4_gpu_tensor_view(x.base,20,n*sizeof(float));
    ds4_gpu_tensor *unaligned=ds4_gpu_tensor_view(x.base,17,n*sizeof(float));
    CHECK(short_w && partial && unaligned);
    CHECK(!ds4_gpu_hy4_weighted_sum8_tensor(out.view,down.view,short_w,n));
    if(n>1) CHECK(!ds4_gpu_hy4_sigmoid_mul_tensor(partial,x.view,gate.view,n));
    CHECK(!ds4_gpu_hy4_sigmoid_mul_tensor(unaligned,x.view,gate.view,n));
    ds4_gpu_tensor_free(short_w);ds4_gpu_tensor_free(partial);ds4_gpu_tensor_free(unaligned);
    free(expected);free(gated);
    tensor_free(&weights);tensor_free(&down);tensor_free(&out);tensor_free(&gate);tensor_free(&x);
    ++cases;
}
int main(void) {
    CHECK(ds4_gpu_init());
    const uint32_t widths[]={1,31,32,33,127,128,129,6144,16384};
    for(unsigned i=0;i<sizeof(widths)/sizeof(widths[0]);i++) run_case(widths[i]);
    ds4_gpu_cleanup();
    printf("PASS HY4 Metal pointwise: %u cases, %u checks; ordered sum bit-exact, gate maxscaled %.3g; owned/batched, offsets, aliases, bounds\n",cases,checks,worst_scaled);
    return 0;
}
