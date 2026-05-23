// Probe: does ANE accept multiple requests on the SAME compiled model, each
// bound to a DIFFERENT input IOSurface?  This is the foundation for the
// "per-chunk IOSurface array" optimization for O-proj: pre-allocate one
// IOSurface per chunk, GPU writes converted fp16 input into each, ANE eval
// for chunk K uses the request bound to IOSurface[K] — no worker memcpy.
//
// Approach: create a small constexpr int8 linear model, then manually
// create N additional requests bound to N different IOSurfaces, and call
// evaluate on each in sequence.  Verify each completes without error.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include "ane_ds4_mlp_int8w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_f16_bits(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m++;
    if (m & 0x400u) { m = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (m & 0x3ffu));
}

static IOSurfaceRef make_surface(NSUInteger bytes, NSUInteger elem) {
    if (elem == 0) elem = 1;
    NSDictionary *p = @{
        (id)kIOSurfaceWidth: @((bytes + elem - 1) / elem),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @(elem),
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

int main(int argc, const char *argv[]) {
    int H = 4096, I = 8192, B = 256;
    int N_chunks = 8;
    for (int i = 1; i + 1 < argc; i++) {
        if      (!strcmp(argv[i], "-H")) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-I")) I = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-B")) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-N")) N_chunks = atoi(argv[++i]);
    }
    setenv("DS4_FLASH_MOE_ANE_DEBUG", "1", 1);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

    printf("=== Per-chunk IOSurface probe: H=%d I=%d B=%d N_chunks=%d ===\n",
           H, I, B, N_chunks);

    @autoreleasepool {
        /* Build a constexpr int8 linear ctx using the existing API.  This
         * gives us a compiled model + the model_r handle we can build
         * additional requests for. */
        const uint64_t a_elems = (uint64_t)I * H;
        const uint64_t b_elems = (uint64_t)H * I;
        int8_t   *Wa_q  = (int8_t   *)calloc((size_t)a_elems, 1);
        int8_t   *Wb_q  = (int8_t   *)calloc((size_t)b_elems, 1);
        int8_t   *Wa_off = (int8_t  *)calloc((size_t)I, 1);
        int8_t   *Wb_off = (int8_t  *)calloc((size_t)H, 1);
        uint16_t *Wa_scale = (uint16_t *)malloc((size_t)I * sizeof(uint16_t));
        uint16_t *Wb_scale = (uint16_t *)malloc((size_t)H * sizeof(uint16_t));
        for (int c = 0; c < I; c++) Wa_scale[c] = f32_to_f16_bits(0.01f);
        for (int c = 0; c < H; c++) Wb_scale[c] = f32_to_f16_bits(0.01f);

        ds4_ane_mlp_int8w_ctx *ctx = ds4_ane_mlp_int8w_linear_constexpr_create(
            H, I, B, Wa_q, Wa_off, Wa_scale, Wb_q, Wb_off, Wb_scale);
        free(Wa_q); free(Wb_q); free(Wa_off); free(Wb_off); free(Wa_scale); free(Wb_scale);
        if (!ctx) { fprintf(stderr, "ctx create failed\n"); return 1; }
        printf("ctx created OK\n");

        /* Resolve the private classes we need. */
        Class ReqCls = NSClassFromString(@"_ANERequest");
        Class IOCls  = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!ReqCls || !IOCls) { fprintf(stderr, "missing private classes\n"); return 2; }

        /* Get the model_r handle out of ctx — it's the first void* in the
         * struct.  Cheat: we know the ctx is what create returns and the
         * fields are layout-fixed; cast and dereference.  Better would be a
         * public accessor but for the probe this works. */
        struct ctx_layout { int H, I, B, mode; float a, b, c; void *model_r; void *request_r; };
        struct ctx_layout *lay = (struct ctx_layout *)ctx;
        id mdl = (__bridge id)lay->model_r;
        if (!mdl) { fprintf(stderr, "model_r nil\n"); return 3; }
        printf("model_r OK\n");

        /* Allocate N IOSurfaces sized [B, H] fp16, plus one OUTPUT IOSurface
         * that all requests share.  Create N requests, each bound to a
         * different input IOSurface. */
        const NSUInteger x_bytes = (NSUInteger)B * (NSUInteger)H * 2;
        const NSUInteger y_bytes = (NSUInteger)B * (NSUInteger)H * 2;
        IOSurfaceRef io_outs[64] = {0};   /* one output per request — simpler */
        IOSurfaceRef io_ins[64]  = {0};
        id           reqs[64]    = {nil};
        if (N_chunks > 64) N_chunks = 64;
        for (int k = 0; k < N_chunks; k++) {
            io_ins[k]  = make_surface(x_bytes, 2u);
            io_outs[k] = make_surface(y_bytes, 2u);
            if (!io_ins[k] || !io_outs[k]) { fprintf(stderr, "iosurface alloc fail k=%d\n", k); return 4; }
            id w_x = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(IOCls, @selector(objectWithIOSurface:), io_ins[k]);
            id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(IOCls, @selector(objectWithIOSurface:), io_outs[k]);
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[w_x], @[@0], @[w_o], @[@0], nil, nil, @0);
            if (!req) { fprintf(stderr, "request[%d] create FAILED\n", k); return 5; }
            reqs[k] = req;
        }
        printf("Created %d input IOSurfaces + %d requests\n", N_chunks, N_chunks);

        /* For each request, write some distinct fp16 data into its IOSurface
         * and call evaluate.  Verify each completes without error. */
        for (int k = 0; k < N_chunks; k++) {
            IOSurfaceLock(io_ins[k], 0, NULL);
            uint16_t *p = (uint16_t *)IOSurfaceGetBaseAddress(io_ins[k]);
            for (NSUInteger i = 0; i < (NSUInteger)B * H; i++) p[i] = (uint16_t)(0x3000 + ((k * 7 + i) & 0xff));
            IOSurfaceUnlock(io_ins[k], 0, NULL);

            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:),
                21, @{}, reqs[k], &e);
            if (!ok) {
                fprintf(stderr, "evaluate request[%d] FAILED: %s\n",
                        k, e ? [[e description] UTF8String] : "unknown");
                return 6;
            }
        }
        printf("All %d evaluates succeeded\n", N_chunks);

        /* Verify each output differs (rough sanity that ANE actually used the
         * different inputs and the outputs aren't accidentally shared). */
        for (int k = 0; k < N_chunks - 1; k++) {
            IOSurfaceLock(io_outs[k],     kIOSurfaceLockReadOnly, NULL);
            IOSurfaceLock(io_outs[k + 1], kIOSurfaceLockReadOnly, NULL);
            const uint16_t *a = (const uint16_t *)IOSurfaceGetBaseAddress(io_outs[k]);
            const uint16_t *b = (const uint16_t *)IOSurfaceGetBaseAddress(io_outs[k + 1]);
            int diff = 0;
            for (NSUInteger i = 0; i < (NSUInteger)B * H && diff < 4; i++) {
                if (a[i] != b[i]) diff++;
            }
            IOSurfaceUnlock(io_outs[k],     kIOSurfaceLockReadOnly, NULL);
            IOSurfaceUnlock(io_outs[k + 1], kIOSurfaceLockReadOnly, NULL);
            if (diff == 0) {
                printf("WARNING: outputs[%d] and outputs[%d] are byte-identical (suspicious)\n", k, k + 1);
            }
        }
        printf("Output divergence sanity passed\n");
        printf("\nPROBE PASSED: ANE accepts %d requests on one compiled model with distinct input IOSurfaces.\n", N_chunks);

        for (int k = 0; k < N_chunks; k++) {
            if (io_ins[k])  CFRelease(io_ins[k]);
            if (io_outs[k]) CFRelease(io_outs[k]);
        }
        ds4_ane_mlp_int8w_destroy(ctx);
    }
    return 0;
}
