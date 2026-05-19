// ds4_mlp_inmem_bench_packed_split3.m
//
// Fully-split DSv4 MLP for ANE — works around the m3u (M3 Ultra) compile
// failure for the full DSv4 shape (H=7168, I=18432).
//
// Observation from TASK_BATCH_ENGINE_MOE.md: the m3u ANE compiler rejects any
// tensor shape that contains 18432 in a matmul-relevant axis (output of
// gate/up, K dim of down), regardless of byte count.  H=7168,I=16384 compiles
// at 672 MB packed; I=18432 fails even at smaller H.  So the fix isn't "more
// bytes"; it's "no tensor in the graph carries I=18432 anywhere."
//
// Strategy: split EVERYTHING along the I axis into 3 pieces (T=3).  The MLP
// becomes 3 parallel streams that don't merge until the final add.
//
//   For t in 0,1,2 (each I_t = I/T = 6144):
//     gate_t   = matmul(tx=true,  x_3d[1,H,B], W_gate_t[H,I_t])   -> [1,B,I_t]
//     up_t     = matmul(tx=true,  x_3d[1,H,B], W_up_t  [H,I_t])   -> [1,B,I_t]
//     hidden_t = silu(gate_t) * up_t                              -> [1,B,I_t]
//     d_t      = matmul(tx=false, hidden_t,    W_down_t[I_t,H])   -> [1,B,H]
//   output = d_0 + d_1 + d_2                                       -> [1,B,H]
//
// Largest I-axis tensor in the graph: I/T = 6144.  Well under the rejection
// boundary observed on m3u (I=16384 works, I=18432 fails).
//
// Packed input layout (caller materialises this each iteration):
//
//   packed: tensor<fp16, [1, 1, 1, N]> where
//     N = B*H + 6*(H*I/T) + 3*(I/T)*H = B*H + 9*H*(I/T) = B*H + 3*H*I
//
//   [ input
//   | W_gate_0 [H, I/T]   row-major
//   | W_gate_1 [H, I/T]
//   | W_gate_2 [H, I/T]
//   | W_up_0   [H, I/T]
//   | W_up_1   [H, I/T]
//   | W_up_2   [H, I/T]
//   | W_down_0 [I/T, H]   row-major
//   | W_down_1 [I/T, H]
//   | W_down_2 [I/T, H]
//   ]
//
//   W_gate/W_up split: in the original [H, I] layout, "column-block" t is
//   NOT contiguous in memory.  The caller must lay out W_gate as 3 separate
//   contiguous [H, I/T] blocks (equivalent to a one-time re-tile of W).  For
//   the DSv4 production weights this is a one-shot rearrange per expert load.
//
//   W_down split: original [I, H] row-major; "row-block" t IS contiguous
//   (rows [t*I/T, (t+1)*I/T)).
//
// Build:
//   clang -fobjc-arc -O2 -framework Foundation -framework IOSurface \
//       ds4_mlp_inmem_bench_packed_split3.m -o ds4_mlp_inmem_bench_packed_split3

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

#ifndef SPLIT_T
#define SPLIT_T 3
#endif

// total elements in packed input (T splits along I)
static inline uint64_t total_elts_split(int B, int H, int I, int T) {
    return (uint64_t)B*H + 3ULL * (uint64_t)H * (uint64_t)I;
}

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

static NSString *genMIL_split(int H, int I, int B, int T) {
    NSCAssert(I % T == 0, @"I must be divisible by T");
    NSMutableString *m = [NSMutableString string];
    uint64_t bh = (uint64_t)B*H;
    uint64_t hi = (uint64_t)H * (uint64_t)(I/T);      // elts per gate/up tile  ([H, I/T])
    uint64_t di = (uint64_t)(I/T) * (uint64_t)H;      // elts per down tile     ([I/T, H])  (== hi numerically)
    uint64_t N  = bh + (uint64_t)(2*T) * hi + (uint64_t)T * di;
    int It = I / T;

    uint64_t off_in      = 0;
    uint64_t off_gate0   = bh;
    uint64_t off_up0     = bh + (uint64_t)T * hi;
    uint64_t off_down0   = bh + (uint64_t)(2*T) * hi;

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

    // ---- 3 streams (t = 0..T-1) ----
    for (int t = 0; t < T; t++) {
        // --- W_gate_t [H, I/T] ---
        uint64_t off = off_gate0 + (uint64_t)t * hi;
        [m appendFormat:
            @"            tensor<int32, [4]> g%d_begin = const()[name = string(\"g%d_begin\"), val = tensor<int32, [4]>([0, 0, 0, %llu])];\n"
            @"            tensor<int32, [4]> g%d_end   = const()[name = string(\"g%d_end\"),   val = tensor<int32, [4]>([1, 1, 1, %llu])];\n",
            t, t, (unsigned long long)off,
            t, t, (unsigned long long)(off + hi)];
        [m appendFormat:
            @"            tensor<fp16, [1, 1, 1, %llu]> g%d_slice = slice_by_index(begin = g%d_begin, end = g%d_end, x = packed)[name = string(\"g%d_slice\")];\n",
            (unsigned long long)hi, t, t, t, t];
        [m appendFormat:
            @"            tensor<int32, [2]> g%d_shape = const()[name = string(\"g%d_shape\"), val = tensor<int32, [2]>([%d, %d])];\n",
            t, t, H, It];
        [m appendFormat:
            @"            tensor<fp16, [%d, %d]> W_gate_%d = reshape(shape = g%d_shape, x = g%d_slice)[name = string(\"W_gate_%d\")];\n",
            H, It, t, t, t, t];

        // --- W_up_t [H, I/T] ---
        off = off_up0 + (uint64_t)t * hi;
        [m appendFormat:
            @"            tensor<int32, [4]> u%d_begin = const()[name = string(\"u%d_begin\"), val = tensor<int32, [4]>([0, 0, 0, %llu])];\n"
            @"            tensor<int32, [4]> u%d_end   = const()[name = string(\"u%d_end\"),   val = tensor<int32, [4]>([1, 1, 1, %llu])];\n",
            t, t, (unsigned long long)off,
            t, t, (unsigned long long)(off + hi)];
        [m appendFormat:
            @"            tensor<fp16, [1, 1, 1, %llu]> u%d_slice = slice_by_index(begin = u%d_begin, end = u%d_end, x = packed)[name = string(\"u%d_slice\")];\n",
            (unsigned long long)hi, t, t, t, t];
        [m appendFormat:
            @"            tensor<int32, [2]> u%d_shape = const()[name = string(\"u%d_shape\"), val = tensor<int32, [2]>([%d, %d])];\n",
            t, t, H, It];
        [m appendFormat:
            @"            tensor<fp16, [%d, %d]> W_up_%d = reshape(shape = u%d_shape, x = u%d_slice)[name = string(\"W_up_%d\")];\n",
            H, It, t, t, t, t];

        // --- gate_t / up_t / silu / mul ---
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> gate_%d = matmul(transpose_x = tx_t, transpose_y = tx_f, x = x_3d, y = W_gate_%d)[name = string(\"gate_%d\")];\n",
            B, It, t, t, t];
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> up_%d   = matmul(transpose_x = tx_t, transpose_y = tx_f, x = x_3d, y = W_up_%d)[name = string(\"up_%d\")];\n",
            B, It, t, t, t];
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> act_%d  = silu(x = gate_%d)[name = string(\"act_%d\")];\n",
            B, It, t, t, t];
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> hidden_%d = mul(x = act_%d, y = up_%d)[name = string(\"hidden_%d\")];\n",
            B, It, t, t, t, t];

        // --- W_down_t [I/T, H] ---
        off = off_down0 + (uint64_t)t * di;
        [m appendFormat:
            @"            tensor<int32, [4]> d%d_begin = const()[name = string(\"d%d_begin\"), val = tensor<int32, [4]>([0, 0, 0, %llu])];\n"
            @"            tensor<int32, [4]> d%d_end   = const()[name = string(\"d%d_end\"),   val = tensor<int32, [4]>([1, 1, 1, %llu])];\n",
            t, t, (unsigned long long)off,
            t, t, (unsigned long long)(off + di)];
        [m appendFormat:
            @"            tensor<fp16, [1, 1, 1, %llu]> d%d_slice = slice_by_index(begin = d%d_begin, end = d%d_end, x = packed)[name = string(\"d%d_slice\")];\n",
            (unsigned long long)di, t, t, t, t];
        [m appendFormat:
            @"            tensor<int32, [2]> d%d_shape = const()[name = string(\"d%d_shape\"), val = tensor<int32, [2]>([%d, %d])];\n",
            t, t, It, H];
        [m appendFormat:
            @"            tensor<fp16, [%d, %d]> W_down_%d = reshape(shape = d%d_shape, x = d%d_slice)[name = string(\"W_down_%d\")];\n",
            It, H, t, t, t, t];

        // --- d_t partial down output ---
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> d%d = matmul(transpose_x = tx_f, transpose_y = tx_f, x = hidden_%d, y = W_down_%d)[name = string(\"d%d\")];\n",
            B, H, t, t, t, t];
    }

    // ---- sum partial outputs ----
    NSMutableString *prev = [NSMutableString stringWithString:@"d0"];
    for (int t = 1; t < T; t++) {
        NSString *sumName = (t == T - 1) ? @"output" : [NSString stringWithFormat:@"acc%d", t];
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> %@ = add(x = %@, y = d%d)[name = string(\"%@\")];\n",
            B, H, sumName, prev, t, sumName];
        [prev setString:sumName];
    }
    if (T == 1) {
        [m appendString:@"        } -> (d0);\n}\n"];
    } else {
        [m appendString:@"        } -> (output);\n}\n"];
    }
    return m;
}

static int runBench(int H, int I, int B, int T, int warmup, int iters, BOOL printMIL) {
    @autoreleasepool {
        NSError *e = nil;
        NSString *milStr = genMIL_split(H, I, B, T);
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

        uint64_t N = total_elts_split(B, H, I, T);
        NSUInteger packedBytes = (NSUInteger)(N * 2);
        NSUInteger outBytes    = (NSUInteger)B * H * 2;
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
        IOSurfaceRef io_p = IOSurfaceCreate((__bridge CFDictionaryRef)props(packedBytes));
        IOSurfaceRef io_o = IOSurfaceCreate((__bridge CFDictionaryRef)props(outBytes));
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
        printf("eval (B=%d H=%d I=%d T=%d):  %.3f ms/iter  |  %.2f TFLOP/s  |  packed input %.1f MB/iter\n",
            B, H, I, T, ms, gf_total / ms, packedMB);

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
    int T = SPLIT_T;
    int H = 7168;
    int I = 18432;
    int warmup = 5;
    int iters = 50;
    int batches[64] = {1, 8, 16, 32, 64, 96, 128};
    int nbatches = 7;

    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-v") == 0) {
            printMIL = YES;
        } else if (strcmp(argv[ai], "-bench-shape") == 0 && ai + 6 < argc) {
            H = atoi(argv[++ai]);
            I = atoi(argv[++ai]);
            T = atoi(argv[++ai]);
            nbatches = parseBatches(argv[++ai], batches, (int)(sizeof(batches) / sizeof(batches[0])));
            warmup = atoi(argv[++ai]);
            iters = atoi(argv[++ai]);
            if (H <= 0 || I <= 0 || T <= 0 || (I % T) != 0 ||
                nbatches <= 0 || warmup < 0 || iters <= 0) {
                fprintf(stderr, "usage: %s [-v] [-bench-shape H I T batches warmup iters]\n", argv[0]);
                return 2;
            }
        } else {
            fprintf(stderr, "usage: %s [-v] [-bench-shape H I T batches warmup iters]\n", argv[0]);
            return 2;
        }
    }

    printf("=== DSv4 MLP packed-input fused ANE bench, FULL I-axis split (T=%d) ===\n", T);
    printf("No tensor in the graph has the full I axis; everything is split into T pieces of I/T = %d.\n",
        I / T);
    printf("Target: m3u (M3 Ultra) — H=%d I=%d prefill shape.\n\n", H, I);

    for (int bi = 0; bi < nbatches; bi++) {
        int B = batches[bi];
        printf("--- DSv4 MLP, H=%d, I=%d, B=%d, T=%d ---\n", H, I, B, T);
        int rc2 = runBench(H, I, B, T, warmup, iters, printMIL && bi == 0);
        if (rc2) printf("B=%d failed (rc=%d)\n", B, rc2);
        printf("\n");
    }
    return 0;
}
