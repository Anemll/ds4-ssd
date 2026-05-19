// ds4_mlp_inmem_bench.m
//
// End-to-end DSv4 MLP benchmark on the ANE via the private _ANEInMemoryModel
// API, with hidden states AND all three weight matrices passed as function
// inputs (not const + BLOBFILE).
//
// Graph:
//     input : fp16 [B, H]              hidden states
//     W_gate: fp16 [H, I]
//     W_up  : fp16 [H, I]
//     W_down: fp16 [I, H]
//
//     x_3d  = expand_dims(transpose(input, [1,0]), axes=[0])    // [1, H, B]
//     gate  = matmul(tx=true,  x_3d, W_gate)                    // [1, B, I]
//     up    = matmul(tx=true,  x_3d, W_up)                      // [1, B, I]
//     act   = silu(gate)
//     h     = act * up                                          // [1, B, I]
//     out   = matmul(tx=false, h,    W_down)                    // [1, B, H]
//
// Mirrors the post-compile MIL of
// ds4_mlp_matmul_fp16_weights_as_inputs.mlmodelc except the final matmul, which
// the high-level converter emits as `matmul(tx=false,tx_y=false, h, W_down)`
// with h shape [1,B,I] and W_down shape [I,H].  On the current test system the
// old tx=true down-projection shape fails ANE compilation, so this integrated
// bench uses the converter-style tx=false down projection.
//
// Build:
//   clang -fobjc-arc -O2 \
//       -framework Foundation -framework IOSurface \
//       ds4_mlp_inmem_bench.m -o ds4_mlp_inmem_bench

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// IEEE 754 fp32 -> fp16 (round-to-nearest-even, no subnormal trickery —
// values stay well within fp16 normal range for this bench).
static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0) {            // underflow -> zero
        return (uint16_t)sign;
    } else if (exp >= 31) {    // overflow -> inf
        return (uint16_t)(sign | 0x7C00);
    }
    uint32_t m = mant >> 13;
    if (mant & 0x1000) m += 1; // round half up
    if (m & 0x400) { m = 0; exp += 1; if (exp >= 31) return (uint16_t)(sign | 0x7C00); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (m & 0x3FF));
}

// 32-bit LCG used to fill verify buffers — deterministic across runs.
static uint32_t lcg(uint32_t s) { return s * 1664525u + 1013904223u; }

// Fill `n` fp16 elements with deterministic uniform values in [-0.02, 0.02].
static void fillDeterministicFP16(uint16_t *out, NSUInteger n, uint32_t seed) {
    uint32_t s = seed;
    for (NSUInteger i = 0; i < n; i++) {
        s = lcg(s);
        // Map to [-0.02, 0.02] — keeps sums of length 18432 well inside fp16.
        float v = (((int32_t)(s & 0xFFFFFF)) - (int32_t)0x800000) / (float)0x800000 * 0.02f;
        out[i] = f32_to_f16(v);
    }
}

static void writeBytesToPath(NSString *path, const void *data, NSUInteger bytes) {
    NSData *d = [NSData dataWithBytesNoCopy:(void*)data length:bytes freeWhenDone:NO];
    [d writeToFile:path atomically:YES];
}

// Generate MIL for the full DSv4 MLP with W and input all as function args.
NSString *genMIL_ds4_mlp(int H, int I, int B) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}})]\n"
        @"{\n"];
    // Function args in this order: W_gate, W_up, W_down, input.
    [m appendFormat:
        @"    func main<ios18>("
        @"tensor<fp16, [%d, %d]> W_gate, "
        @"tensor<fp16, [%d, %d]> W_up, "
        @"tensor<fp16, [%d, %d]> W_down, "
        @"tensor<fp16, [%d, %d]> input) {\n",
        H, I, H, I, I, H, B, H];
    [m appendString:
        @"            tensor<int32, [2]> perm0 = const()[name = string(\"perm0\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            tensor<int32, [1]> ax0  = const()[name = string(\"ax0\"),  val = tensor<int32, [1]>([0])];\n"
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"];
    // x [B,H] -> [H,B] -> [1,H,B]
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> in_t = transpose(perm = perm0, x = input)[name = string(\"in_t\")];\n",
        H, B];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> x_3d = expand_dims(axes = ax0, x = in_t)[name = string(\"x_3d\")];\n",
        H, B];
    // gate = matmul(tx=true, [1,H,B], W_gate[H,I]) -> [1,B,I]
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> gate = matmul(transpose_x = tx_t, transpose_y = tx_f, x = x_3d, y = W_gate)[name = string(\"gate\")];\n",
        B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> up   = matmul(transpose_x = tx_t, transpose_y = tx_f, x = x_3d, y = W_up)[name = string(\"up\")];\n",
        B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> act  = silu(x = gate)[name = string(\"act\")];\n", B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> h    = mul(x = act, y = up)[name = string(\"h\")];\n", B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> output = matmul(transpose_x = tx_f, transpose_y = tx_f, x = h, y = W_down)[name = string(\"output\")];\n",
        B, H];
    [m appendString:
        @"        } -> (output);\n"
        @"}\n"];
    return m;
}

static int runBench(int H, int I, int B, int warmup, int iters, BOOL printMIL) {
    @autoreleasepool {
        NSError *e = nil;
        NSString *milStr = genMIL_ds4_mlp(H, I, B);
        if (printMIL) printf("=== MIL ===\n%s\n", [milStr UTF8String]);
        NSData *milData = [[milStr dataUsingEncoding:NSUTF8StringEncoding] copy];

        Class D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IM = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO= NSClassFromString(@"_ANEIOSurfaceObject");
        if (!D || !IM || !AR || !AIO) {
            printf("FAIL: private ANE classes unavailable\n");
            return 1;
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            D, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);
        if (!desc) { printf("FAIL: descriptor nil\n"); return 2; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            IM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { printf("FAIL: model nil\n"); return 3; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];

        uint64_t tc0 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        double compile_ms = ticksToMs(mach_absolute_time() - tc0);
        if (!ok) {
            printf("FAIL: compile: %s\n",
                e ? [[e description] UTF8String] : "<nil>");
            [fm removeItemAtPath:td error:nil];
            return 4;
        }

        uint64_t tl0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        double load_ms = ticksToMs(mach_absolute_time() - tl0);
        if (!ok) {
            printf("FAIL: load: %s\n",
                e ? [[e description] UTF8String] : "<nil>");
            [fm removeItemAtPath:td error:nil];
            return 5;
        }
        printf("compile %.1f ms  |  load %.1f ms\n", compile_ms, load_ms);

        // IOSurface byte counts (fp16 = 2 bytes/element).
        NSUInteger inBytes   = (NSUInteger)B * H * 2;
        NSUInteger gateBytes = (NSUInteger)H * I * 2;
        NSUInteger upBytes   = (NSUInteger)H * I * 2;
        NSUInteger downBytes = (NSUInteger)I * H * 2;
        NSUInteger outBytes  = (NSUInteger)B * H * 2;

        NSDictionary *(^props)(NSUInteger) = ^NSDictionary *(NSUInteger n) {
            return @{
                (id)kIOSurfaceWidth: @(n),
                (id)kIOSurfaceHeight: @1,
                (id)kIOSurfaceBytesPerElement: @1,
                (id)kIOSurfaceBytesPerRow: @(n),
                (id)kIOSurfaceAllocSize: @(n),
                (id)kIOSurfacePixelFormat: @0,
            };
        };

        IOSurfaceRef io_in   = IOSurfaceCreate((__bridge CFDictionaryRef)props(inBytes));
        IOSurfaceRef io_g    = IOSurfaceCreate((__bridge CFDictionaryRef)props(gateBytes));
        IOSurfaceRef io_u    = IOSurfaceCreate((__bridge CFDictionaryRef)props(upBytes));
        IOSurfaceRef io_d    = IOSurfaceCreate((__bridge CFDictionaryRef)props(downBytes));
        IOSurfaceRef io_out  = IOSurfaceCreate((__bridge CFDictionaryRef)props(outBytes));
        if (!io_in || !io_g || !io_u || !io_d || !io_out) {
            printf("FAIL: IOSurfaceCreate\n");
            if (io_in)CFRelease(io_in); if (io_g)CFRelease(io_g);
            if (io_u)CFRelease(io_u); if (io_d)CFRelease(io_d);
            if (io_out)CFRelease(io_out);
            return 6;
        }

        id w_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_in);
        id w_g   = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_g);
        id w_u   = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_u);
        id w_d   = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_d);
        id w_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_out);

        // MIL function args: W_gate(0), W_up(1), W_down(2), input(3).
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[w_g, w_u, w_d, w_in], @[@0, @1, @2, @3],
            @[w_out], @[@0], nil, nil, @0);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:),
                21, @{}, req, &e);
            if (!ok) {
                printf("FAIL: warmup eval: %s\n",
                    e ? [[e description] UTF8String] : "<nil>");
                CFRelease(io_in); CFRelease(io_g); CFRelease(io_u);
                CFRelease(io_d); CFRelease(io_out);
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    mdl, @selector(unloadWithQoS:error:), 21, &e);
                [fm removeItemAtPath:td error:nil];
                return 7;
            }
        }

        // Timed runs
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:),
                21, @{}, req, &e);
        }
        double ms_per = ticksToMs(mach_absolute_time() - t0) / iters;

        // Stats
        double gf_mm = 2.0 * (double)B * H * I / 1e9;   // gate
        double gf_total = gf_mm + gf_mm + 2.0 * (double)B * I * H / 1e9; // gate+up+down
        double wMB = ((double)gateBytes + upBytes + downBytes) / (1024.0 * 1024.0);

        printf("eval (B=%d H=%d I=%d):  %.3f ms/iter  |  %.2f GFLOP/s overall  |  weights/iter %.1f MB\n",
            B, H, I, ms_per, gf_total / ms_per, wMB);

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(io_in); CFRelease(io_g); CFRelease(io_u);
        CFRelease(io_d); CFRelease(io_out);
        [fm removeItemAtPath:td error:nil];
        return 0;
    }
}

// Verify mode: deterministic inputs/weights, dump everything for Python compare.
static int runVerify(int H, int I, int B, NSString *outDir) {
    @autoreleasepool {
        NSError *e = nil;
        NSString *milStr = genMIL_ds4_mlp(H, I, B);
        NSData *milData = [[milStr dataUsingEncoding:NSUTF8StringEncoding] copy];

        Class D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IM = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            D, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            IM, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
        [fm createDirectoryAtPath:outDir withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            printf("verify: compile fail: %s\n", [[e description] UTF8String]); return 4;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            printf("verify: load fail: %s\n", [[e description] UTF8String]); return 5;
        }

        // Build host buffers with deterministic content.
        NSUInteger nIn = (NSUInteger)B * H;
        NSUInteger nG  = (NSUInteger)H * I;
        NSUInteger nU  = (NSUInteger)H * I;
        NSUInteger nD  = (NSUInteger)I * H;
        NSUInteger nO  = (NSUInteger)B * H;
        uint16_t *h_in = malloc(nIn * 2);
        uint16_t *h_g  = malloc(nG  * 2);
        uint16_t *h_u  = malloc(nU  * 2);
        uint16_t *h_d  = malloc(nD  * 2);
        uint16_t *h_o  = malloc(nO  * 2);
        fillDeterministicFP16(h_in, nIn, 0xA1B2C301);
        fillDeterministicFP16(h_g,  nG,  0xA1B2C302);
        fillDeterministicFP16(h_u,  nU,  0xA1B2C303);
        fillDeterministicFP16(h_d,  nD,  0xA1B2C304);

        // Write inputs/weights to outDir for Python.
        writeBytesToPath([outDir stringByAppendingPathComponent:@"input.bin"],  h_in, nIn*2);
        writeBytesToPath([outDir stringByAppendingPathComponent:@"W_gate.bin"], h_g,  nG*2);
        writeBytesToPath([outDir stringByAppendingPathComponent:@"W_up.bin"],   h_u,  nU*2);
        writeBytesToPath([outDir stringByAppendingPathComponent:@"W_down.bin"], h_d,  nD*2);
        NSString *meta = [NSString stringWithFormat:@"B=%d\nH=%d\nI=%d\n", B, H, I];
        [meta writeToFile:[outDir stringByAppendingPathComponent:@"meta.txt"]
              atomically:YES encoding:NSUTF8StringEncoding error:nil];

        // Create IOSurfaces and copy host data in.
        NSDictionary *(^props)(NSUInteger) = ^NSDictionary *(NSUInteger n) {
            return @{
                (id)kIOSurfaceWidth: @(n),
                (id)kIOSurfaceHeight: @1,
                (id)kIOSurfaceBytesPerElement: @1,
                (id)kIOSurfaceBytesPerRow: @(n),
                (id)kIOSurfaceAllocSize: @(n),
                (id)kIOSurfacePixelFormat: @0,
            };
        };
        IOSurfaceRef io_in  = IOSurfaceCreate((__bridge CFDictionaryRef)props(nIn*2));
        IOSurfaceRef io_g   = IOSurfaceCreate((__bridge CFDictionaryRef)props(nG*2));
        IOSurfaceRef io_u   = IOSurfaceCreate((__bridge CFDictionaryRef)props(nU*2));
        IOSurfaceRef io_d   = IOSurfaceCreate((__bridge CFDictionaryRef)props(nD*2));
        IOSurfaceRef io_out = IOSurfaceCreate((__bridge CFDictionaryRef)props(nO*2));
        IOSurfaceLock(io_in,  0, NULL); memcpy(IOSurfaceGetBaseAddress(io_in),  h_in, nIn*2); IOSurfaceUnlock(io_in,  0, NULL);
        IOSurfaceLock(io_g,   0, NULL); memcpy(IOSurfaceGetBaseAddress(io_g),   h_g,  nG*2);  IOSurfaceUnlock(io_g,   0, NULL);
        IOSurfaceLock(io_u,   0, NULL); memcpy(IOSurfaceGetBaseAddress(io_u),   h_u,  nU*2);  IOSurfaceUnlock(io_u,   0, NULL);
        IOSurfaceLock(io_d,   0, NULL); memcpy(IOSurfaceGetBaseAddress(io_d),   h_d,  nD*2);  IOSurfaceUnlock(io_d,   0, NULL);

        id w_in  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_in);
        id w_g   = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_g);
        id w_u   = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_u);
        id w_d   = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_d);
        id w_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_out);

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[w_g, w_u, w_d, w_in], @[@0, @1, @2, @3],
            @[w_out], @[@0], nil, nil, @0);

        if (!((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:),
                21, @{}, req, &e)) {
            printf("verify: eval fail: %s\n", [[e description] UTF8String]); return 6;
        }

        // Read output back from io_out.
        IOSurfaceLock(io_out, kIOSurfaceLockReadOnly, NULL);
        memcpy(h_o, IOSurfaceGetBaseAddress(io_out), nO*2);
        IOSurfaceUnlock(io_out, kIOSurfaceLockReadOnly, NULL);
        writeBytesToPath([outDir stringByAppendingPathComponent:@"output_ane.bin"], h_o, nO*2);

        printf("verify: dumped to %s  (B=%d H=%d I=%d, %lu bytes output)\n",
            [outDir UTF8String], B, H, I, (unsigned long)(nO*2));

        free(h_in); free(h_g); free(h_u); free(h_d); free(h_o);
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(io_in); CFRelease(io_g); CFRelease(io_u);
        CFRelease(io_d); CFRelease(io_out);
        [fm removeItemAtPath:td error:nil];
        return 0;
    }
}

static void runBenchList(int H, int I, const char *list, int warmup, int iters) {
    char *copy = strdup(list);
    char *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        int B = atoi(tok);
        if (B <= 0) continue;
        runBench(H, I, B, warmup, iters, NO);
    }
    free(copy);
}

int main(int argc, const char *argv[]) {
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
           RTLD_NOW);

    // Verify mode:  ./ds4_mlp_inmem_bench -verify  [H I B [outDir]]
    if (argc >= 2 && strcmp(argv[1], "-verify") == 0) {
        int H = (argc > 2) ? atoi(argv[2]) : 7168;
        int I = (argc > 3) ? atoi(argv[3]) : 18432;
        int B = (argc > 4) ? atoi(argv[4]) : 8;
        NSString *dir = (argc > 5) ? @(argv[5]) : @"/tmp/ds4_verify";
        return runVerify(H, I, B, dir);
    }
    if (argc >= 2 && strcmp(argv[1], "-bench-shape") == 0) {
        int H = (argc > 2) ? atoi(argv[2]) : 4096;
        int I = (argc > 3) ? atoi(argv[3]) : 2048;
        const char *batches = (argc > 4) ? argv[4] : "1,8,16,32,64,128";
        int warmup = (argc > 5) ? atoi(argv[5]) : 5;
        int iters = (argc > 6) ? atoi(argv[6]) : 30;
        runBenchList(H, I, batches, warmup, iters);
        return 0;
    }

    BOOL printMIL = (argc > 1 && strcmp(argv[1], "-v") == 0);

    printf("=== DSv4 MLP private-API benchmark (W + input as function args, fp16) ===\n\n");

    // 1. Smoke shape — quick structural sanity check.
    printf("--- smoke (H=256, I=512, B=32) ---\n");
    int rc = runBench(256, 512, 32, /*warmup=*/5, /*iters=*/50, printMIL);
    if (rc) { printf("smoke failed (rc=%d)\n", rc); return rc; }
    printf("\n");

    // 2. DSv4 Flash shapes at the canonical batch=32 the rest of the project uses.
    printf("--- DSv4 MLP, H=7168, I=18432, B=32 ---\n");
    rc = runBench(7168, 18432, 32, 5, 50, printMIL);
    if (rc) { printf("B=32 failed (rc=%d)\n", rc); return rc; }
    printf("\n");

    // 3. Sweep over a few batch sizes to see where the ANE saturates.
    int batches[] = {1, 8, 16, 32, 64, 96, 128, 256};
    for (int bi = 0; bi < (int)(sizeof(batches)/sizeof(batches[0])); bi++) {
        int B = batches[bi];
        printf("--- DSv4 MLP, H=7168, I=18432, B=%d ---\n", B);
        int rc2 = runBench(7168, 18432, B, 5, 50, NO);
        if (rc2) printf("B=%d failed (rc=%d)\n", B, rc2);
        printf("\n");
    }

    return 0;
}
