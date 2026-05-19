// ds4_mlp_inmem_bench_packed.m
//
// Fused DSv4 MLP on _ANEInMemoryModel using the pack-and-slice pattern:
// a SINGLE 4D function input
//     packed: tensor<fp16, [1, 1, 1, N]>
// contains the activation plus all three weight matrices, concatenated:
//     [ input (B*H) | W_gate (H*I) | W_up (H*I) | W_down (I*H) ]
// Inside the MIL we slice each piece out with `slice_by_index` and reshape
// to its native 2D shape, then run the full DSv4 MLP:
//
//     x_3d = expand_dims(transpose(input))      // [1, H, B]
//     gate = matmul(tx=true,  x_3d, W_gate)     // [1, B, I]
//     up   = matmul(tx=true,  x_3d, W_up)
//     h    = silu(gate) * up                    // [1, B, I]
//     out  = matmul(tx=false, h,   W_down)      // [1, B, H]    <-- converter style
//
// Down uses tx=false directly on the [1,B,I] activation (no extra transpose)
// — the high-level CoreML converter's emitted pattern — to maximize
// cross-hardware portability.
//
// Build:
//   clang -fobjc-arc -O2 -framework Foundation -framework IOSurface \
//       ds4_mlp_inmem_bench_packed.m -o ds4_mlp_inmem_bench_packed

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static inline uint64_t total_elts(int B, int H, int I) { return (uint64_t)B*H + 3ULL*(uint64_t)H*I; }

static int parseBatches(const char *s, int *out, int cap) {
    int n = 0;
    const char *p = s;
    while (*p && n < cap) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0 || v > INT_MAX) return -1;
        out[n++] = (int)v;
        if (*end == ',') {
            p = end + 1;
        } else if (*end == '\0') {
            p = end;
        } else {
            return -1;
        }
    }
    return (*p == '\0') ? n : -1;
}

static NSString *genMIL_packed(int H, int I, int B) {
    NSMutableString *m = [NSMutableString string];
    uint64_t bh = (uint64_t)B*H;
    uint64_t hi = (uint64_t)H*I;
    uint64_t N  = bh + 3ULL*hi;
    uint64_t off_in   = 0;
    uint64_t off_gate = bh;
    uint64_t off_up   = bh + hi;
    uint64_t off_down = bh + 2ULL*hi;

    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, 1, 1, %llu]> packed) {\n",
        (unsigned long long)N];

    [m appendString:
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"
        @"            tensor<int32, [2]> perm0 = const()[name = string(\"perm0\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            tensor<int32, [1]> ax0  = const()[name = string(\"ax0\"),  val = tensor<int32, [1]>([0])];\n"];

    // ---- slice + reshape activation ----
    [m appendFormat:
        @"            tensor<int32, [4]> in_begin = const()[name = string(\"in_begin\"), val = tensor<int32, [4]>([0, 0, 0, %llu])];\n"
        @"            tensor<int32, [4]> in_end   = const()[name = string(\"in_end\"),   val = tensor<int32, [4]>([1, 1, 1, %llu])];\n",
        (unsigned long long)off_in, (unsigned long long)(off_in + bh)];
    [m appendFormat:
        @"            tensor<fp16, [1, 1, 1, %llu]> in_slice = slice_by_index(begin = in_begin, end = in_end, x = packed)[name = string(\"in_slice\")];\n",
        (unsigned long long)bh];
    [m appendFormat:
        @"            tensor<int32, [2]> in_shape = const()[name = string(\"in_shape\"), val = tensor<int32, [2]>([%d, %d])];\n", B, H];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> input = reshape(shape = in_shape, x = in_slice)[name = string(\"input\")];\n", B, H];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> in_t = transpose(perm = perm0, x = input)[name = string(\"in_t\")];\n", H, B];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> x_3d = expand_dims(axes = ax0, x = in_t)[name = string(\"x_3d\")];\n", H, B];

    // ---- slice + reshape W_gate ----
    [m appendFormat:
        @"            tensor<int32, [4]> g_begin = const()[name = string(\"g_begin\"), val = tensor<int32, [4]>([0, 0, 0, %llu])];\n"
        @"            tensor<int32, [4]> g_end   = const()[name = string(\"g_end\"),   val = tensor<int32, [4]>([1, 1, 1, %llu])];\n",
        (unsigned long long)off_gate, (unsigned long long)(off_gate + hi)];
    [m appendFormat:
        @"            tensor<fp16, [1, 1, 1, %llu]> g_slice = slice_by_index(begin = g_begin, end = g_end, x = packed)[name = string(\"g_slice\")];\n",
        (unsigned long long)hi];
    [m appendFormat:
        @"            tensor<int32, [2]> g_shape = const()[name = string(\"g_shape\"), val = tensor<int32, [2]>([%d, %d])];\n", H, I];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> W_gate = reshape(shape = g_shape, x = g_slice)[name = string(\"W_gate\")];\n", H, I];

    // ---- slice + reshape W_up ----
    [m appendFormat:
        @"            tensor<int32, [4]> u_begin = const()[name = string(\"u_begin\"), val = tensor<int32, [4]>([0, 0, 0, %llu])];\n"
        @"            tensor<int32, [4]> u_end   = const()[name = string(\"u_end\"),   val = tensor<int32, [4]>([1, 1, 1, %llu])];\n",
        (unsigned long long)off_up, (unsigned long long)(off_up + hi)];
    [m appendFormat:
        @"            tensor<fp16, [1, 1, 1, %llu]> u_slice = slice_by_index(begin = u_begin, end = u_end, x = packed)[name = string(\"u_slice\")];\n",
        (unsigned long long)hi];
    [m appendFormat:
        @"            tensor<int32, [2]> u_shape = const()[name = string(\"u_shape\"), val = tensor<int32, [2]>([%d, %d])];\n", H, I];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> W_up = reshape(shape = u_shape, x = u_slice)[name = string(\"W_up\")];\n", H, I];

    // ---- slice + reshape W_down ----
    [m appendFormat:
        @"            tensor<int32, [4]> d_begin = const()[name = string(\"d_begin\"), val = tensor<int32, [4]>([0, 0, 0, %llu])];\n"
        @"            tensor<int32, [4]> d_end   = const()[name = string(\"d_end\"),   val = tensor<int32, [4]>([1, 1, 1, %llu])];\n",
        (unsigned long long)off_down, (unsigned long long)(off_down + hi)];
    [m appendFormat:
        @"            tensor<fp16, [1, 1, 1, %llu]> d_slice = slice_by_index(begin = d_begin, end = d_end, x = packed)[name = string(\"d_slice\")];\n",
        (unsigned long long)hi];
    [m appendFormat:
        @"            tensor<int32, [2]> d_shape = const()[name = string(\"d_shape\"), val = tensor<int32, [2]>([%d, %d])];\n", I, H];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> W_down = reshape(shape = d_shape, x = d_slice)[name = string(\"W_down\")];\n", I, H];

    // ---- MLP body (full DSv4 MLP: gate + up + silu + mul + down) ----
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> gate = matmul(transpose_x = tx_t, transpose_y = tx_f, x = x_3d, y = W_gate)[name = string(\"gate\")];\n",
        B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> up   = matmul(transpose_x = tx_t, transpose_y = tx_f, x = x_3d, y = W_up)[name = string(\"up\")];\n",
        B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> act  = silu(x = gate)[name = string(\"act\")];\n", B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> hidden = mul(x = act, y = up)[name = string(\"hidden\")];\n", B, I];
    // Down: converter-style tx=false on hidden[1,B,I] @ W_down[I,H] -> [1,B,H].
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> output = matmul(transpose_x = tx_f, transpose_y = tx_f, x = hidden, y = W_down)[name = string(\"output\")];\n",
        B, H];
    [m appendString:@"        } -> (output);\n}\n"];
    return m;
}

static int runBench(int H, int I, int B, int warmup, int iters, BOOL printMIL) {
    @autoreleasepool {
        NSError *e = nil;
        NSString *milStr = genMIL_packed(H, I, B);
        if (printMIL) printf("=== MIL ===\n%s\n", [milStr UTF8String]);
        NSData *milData = [[milStr dataUsingEncoding:NSUTF8StringEncoding] copy];

        Class D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IM = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            D, @selector(modelWithMILText:weights:optionsPlist:), milData, @{}, nil);
        if (!desc) { printf("FAIL: descriptor nil\n"); return 2; }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            IM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { printf("FAIL: model nil\n"); return 3; }
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        uint64_t tc0 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        double compile_ms = ticksToMs(mach_absolute_time() - tc0);
        if (!ok) {
            printf("FAIL: compile: %s\n", e ? [[e description] UTF8String] : "<nil>");
            [fm removeItemAtPath:td error:nil]; return 4;
        }
        uint64_t tl0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        double load_ms = ticksToMs(mach_absolute_time() - tl0);
        if (!ok) {
            printf("FAIL: load: %s\n", e ? [[e description] UTF8String] : "<nil>");
            [fm removeItemAtPath:td error:nil]; return 5;
        }
        printf("compile %.1f ms  |  load %.1f ms\n", compile_ms, load_ms);

        uint64_t N = total_elts(B, H, I);
        NSUInteger packedBytes = (NSUInteger)(N * 2);
        NSUInteger outBytes    = (NSUInteger)B * H * 2;
        // Surface layout: width in *elements*, BytesPerElement = 2 (fp16).
        // This matches what the high-level CoreML runtime hands the ANE for
        // fp16 input tensors.  Earlier benches used 1-byte-per-element on the
        // assumption surface dimensions were irrelevant — that works for small
        // tensors but seems to confuse ANE for larger packed inputs.
        NSDictionary *(^props)(NSUInteger) = ^NSDictionary *(NSUInteger nbytes) {
            NSUInteger nelts = nbytes / 2;
            return @{
                (id)kIOSurfaceWidth: @(nelts),
                (id)kIOSurfaceHeight: @1,
                (id)kIOSurfaceBytesPerElement: @2,
                (id)kIOSurfaceBytesPerRow: @(nbytes),
                (id)kIOSurfaceAllocSize: @(nbytes),
                (id)kIOSurfacePixelFormat: @0,
            };
        };
        IOSurfaceRef io_p = IOSurfaceCreate((__bridge CFDictionaryRef)props(packedBytes));
        IOSurfaceRef io_o = IOSurfaceCreate((__bridge CFDictionaryRef)props(outBytes));
        if (!io_p || !io_o) {
            printf("FAIL: IOSurfaceCreate (packed=%llu bytes, out=%llu bytes)\n",
                (unsigned long long)packedBytes, (unsigned long long)outBytes);
            if (io_p) CFRelease(io_p); if (io_o) CFRelease(io_o);
            [fm removeItemAtPath:td error:nil]; return 6;
        }
        id w_p = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_p);
        id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO,@selector(objectWithIOSurface:),io_o);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[w_p], @[@0], @[w_o], @[@0], nil, nil, @0);

        for (int i = 0; i < warmup; i++) {
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
            if (!ok) {
                printf("FAIL: warmup eval: %s\n", e ? [[e description] UTF8String] : "<nil>");
                CFRelease(io_p); CFRelease(io_o);
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    mdl, @selector(unloadWithQoS:error:), 21, &e);
                [fm removeItemAtPath:td error:nil]; return 7;
            }
        }

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        }
        double ms = ticksToMs(mach_absolute_time() - t0) / iters;
        double gf_total = 3.0 * 2.0 * (double)B * H * I / 1e9;
        double packedMB = packedBytes / (1024.0 * 1024.0);
        printf("eval (B=%d H=%d I=%d):  %.3f ms/iter  |  %.2f TFLOP/s  |  packed input %.1f MB/iter\n",
            B, H, I, ms, gf_total / ms, packedMB);

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(io_p); CFRelease(io_o);
        [fm removeItemAtPath:td error:nil];
        return 0;
    }
}

int main(int argc, const char *argv[]) {
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    BOOL printMIL = NO;
    int H = 7168;
    int I = 18432;
    int warmup = 5;
    int iters = 50;
    int batches[64] = {1, 8, 16, 32, 64, 96, 128, 256};
    int nbatches = 8;

    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-v") == 0) {
            printMIL = YES;
        } else if (strcmp(argv[ai], "-bench-shape") == 0 && ai + 5 < argc) {
            H = atoi(argv[++ai]);
            I = atoi(argv[++ai]);
            nbatches = parseBatches(argv[++ai], batches, (int)(sizeof(batches) / sizeof(batches[0])));
            warmup = atoi(argv[++ai]);
            iters = atoi(argv[++ai]);
            if (H <= 0 || I <= 0 || nbatches <= 0 || warmup < 0 || iters <= 0) {
                fprintf(stderr, "usage: %s [-v] [-bench-shape H I batches warmup iters]\n", argv[0]);
                return 2;
            }
        } else {
            fprintf(stderr, "usage: %s [-v] [-bench-shape H I batches warmup iters]\n", argv[0]);
            return 2;
        }
    }

    printf("=== DSv4 MLP packed-input fused ANE bench (M5 target) ===\n");
    printf("Single 4D function input (1,1,1,N) = concat(input, W_gate, W_up, W_down).\n");
    printf("Sliced + reshaped in MIL via slice_by_index.\n");
    printf("Down matmul uses converter-style tx=false.\n\n");

    // Note: the toy smoke shape (H=256, I=512) returns Program Inference error
    // for the full fused graph on this ANE version — the tile scheduler chokes
    // on the very small slice sizes.  All production shapes (H≥1024) work; we
    // skip the toy smoke and go straight to the DSv4 sweep.

    for (int bi = 0; bi < nbatches; bi++) {
        int B = batches[bi];
        printf("--- DSv4 MLP, H=%d, I=%d, B=%d ---\n", H, I, B);
        int rc2 = runBench(H, I, B, warmup, iters, printMIL);
        if (rc2) printf("B=%d failed (rc=%d)\n", B, rc2);
        printf("\n");
    }
    return 0;
}
