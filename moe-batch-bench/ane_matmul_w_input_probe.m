// test_matmul_w_input.m
//
// Goal: Determine whether the private _ANEInMemoryModel API will compile and
// run a matmul where the weight tensor is a function INPUT (not a const +
// BLOBFILE).  Mirrors the working transpose_x=true / transpose_y=false pattern
// from inmem_peak_matmul.m on DSv4 MLP gate-projection shapes
// ([batch=32, hidden=7168] -> [batch=32, intermediate=18432]).
//
// If this compiles and runs on the ANE it directly answers Open Questions
// Q2 / Q4 in DSv4_MLP_ANE_Matmul_Investigation.md:
//   - W as a normal MIL function input is a viable alternative to the
//     "pack + slice" trick from ane-prefill-bench.
//   - The high-level CoreML mlpackage's success (anemll-profile already shows
//     100% ANE placement) is reproducible at the raw-MIL / private-API layer.
//
// Build (Apple silicon, system frameworks):
//   clang -fobjc-arc -O2 \
//       -framework Foundation -framework IOSurface \
//       test_matmul_w_input.m -o test_matmul_w_input

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Builds a MIL program that exactly mirrors the compiled mlmodelc of
// ds4_mlp_matmul_fp16_weights_as_inputs (which anemll-profile shows runs 100%
// on ANE).  Signature:
//     main(tensor<fp16, [hidden, interm]> W, tensor<fp16, [batch, hidden]> input)
// producing
//     tensor<fp16, [1, batch, interm]> output
//
// Key differences from the original inmem_peak_matmul.m template:
//   - input is 2D fp16, NOT 4D fp32 with a cast (the high-level CoreML pipeline
//     emits the simpler 2D fp16 form, and the ANE compiler is happy with it).
//   - no cast_in / cast_out / squeeze / expand_dims wrappers.
//   - input alias `x` is produced by transpose([1,0]) + expand_dims(axis=0)
//     so matmul gets the [1, hidden, batch] view its transpose_x=true expects.
NSString *genMIL_matmul_w_input(int hidden, int interm, int batch) {
    BOOL direct_x = hidden > interm;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}})]\n"
        @"{\n"];
    [m appendFormat:
        @"    func main<ios18>(tensor<fp16, [%d, %d]> W, tensor<fp16, [%d, %d]> input) {\n",
        hidden, interm, batch, hidden];
    [m appendString:
        @"            tensor<int32, [1]> ax0 = const()[name = string(\"ax0\"), val = tensor<int32, [1]>([0])];\n"
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"];
    if (direct_x) {
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> x_3d = expand_dims(axes = ax0, x = input)[name = string(\"x_3d\")];\n",
            batch, hidden];
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> output = matmul(transpose_x = tx_f, transpose_y = tx_f, x = x_3d, y = W)[name = string(\"output\")];\n",
            batch, interm];
    } else {
        [m appendString:
            @"            tensor<int32, [2]> perm0 = const()[name = string(\"perm0\"), val = tensor<int32, [2]>([1, 0])];\n"];
        [m appendFormat:
            @"            tensor<fp16, [%d, %d]> in_t = transpose(perm = perm0, x = input)[name = string(\"in_t\")];\n",
            hidden, batch];
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> x_3d = expand_dims(axes = ax0, x = in_t)[name = string(\"x_3d\")];\n",
            hidden, batch];
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> output = matmul(transpose_x = tx_t, transpose_y = tx_f, x = x_3d, y = W)[name = string(\"output\")];\n",
            batch, interm];
    }
    [m appendString:
        @"        } -> (output);\n"
        @"}\n"];
    return m;
}

static int runOnce(int hidden, int interm, int batch, BOOL verboseMIL) {
    @autoreleasepool {
        NSError *e = nil;
        NSString *milStr = genMIL_matmul_w_input(hidden, interm, batch);
        if (verboseMIL) {
            printf("=== MIL ===\n%s\n", [milStr UTF8String]);
        }
        NSData *milData = [[milStr dataUsingEncoding:NSUTF8StringEncoding] copy];

        Class D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class I  = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO= NSClassFromString(@"_ANEIOSurfaceObject");
        if (!D || !I || !AR || !AIO) {
            printf("FAIL: private ANE classes not available\n");
            return 1;
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            D, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);
        if (!desc) { printf("FAIL: descriptor nil\n"); return 2; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            I, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { printf("FAIL: model nil\n"); return 3; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];

        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            printf("FAIL: compile: %s\n",
                e ? [[e description] UTF8String] : "<nil>");
            [fm removeItemAtPath:td error:nil];
            return 4;
        }
        printf("compile: OK\n");

        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            printf("FAIL: load: %s\n",
                e ? [[e description] UTF8String] : "<nil>");
            [fm removeItemAtPath:td error:nil];
            return 5;
        }
        printf("load: OK\n");

        // IOSurfaces — all fp16, matching the 2D layout in the MIL.
        NSUInteger wBytes = (NSUInteger)hidden * interm * 2; // fp16 [hidden,interm]
        NSUInteger xBytes = (NSUInteger)batch  * hidden * 2; // fp16 [batch,hidden]
        NSUInteger oBytes = (NSUInteger)batch  * interm * 2; // fp16 [1,batch,interm]

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

        IOSurfaceRef ioX = IOSurfaceCreate((__bridge CFDictionaryRef)props(xBytes));
        IOSurfaceRef ioW = IOSurfaceCreate((__bridge CFDictionaryRef)props(wBytes));
        IOSurfaceRef ioO = IOSurfaceCreate((__bridge CFDictionaryRef)props(oBytes));
        if (!ioX || !ioW || !ioO) {
            printf("FAIL: IOSurfaceCreate (x=%p w=%p o=%p)\n", ioX, ioW, ioO);
            if (ioX) CFRelease(ioX); if (ioW) CFRelease(ioW); if (ioO) CFRelease(ioO);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return 6;
        }

        id wX = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            AIO, @selector(objectWithIOSurface:), ioX);
        id wW = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            AIO, @selector(objectWithIOSurface:), ioW);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            AIO, @selector(objectWithIOSurface:), ioO);

        // MIL declares W as func arg 0, input as func arg 1.  Mirror that order.
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wW, wX], @[@0, @1], @[wO], @[@0], nil, nil, @0);

        // Warmup
        for (int i = 0; i < 10; i++) {
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:),
                21, @{}, req, &e);
            if (!ok) {
                printf("FAIL: warmup eval: %s\n",
                    e ? [[e description] UTF8String] : "<nil>");
                CFRelease(ioX); CFRelease(ioW); CFRelease(ioO);
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    mdl, @selector(unloadWithQoS:error:), 21, &e);
                [fm removeItemAtPath:td error:nil];
                return 7;
            }
        }

        int it = 50;
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < it; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:),
                21, @{}, req, &e);
        }
        double ms = ticksToMs(mach_absolute_time() - t0) / it;
        double gf = 2.0 * (double)hidden * interm * batch / 1e9;
        printf("eval: %.3f ms/iter  |  %.2f GFLOP/s  (1 matmul, "
               "batch=%d hidden=%d interm=%d)\n",
               ms, gf / ms, batch, hidden, interm);

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioX); CFRelease(ioW); CFRelease(ioO);
        [fm removeItemAtPath:td error:nil];
        return 0;
    }
}

int main(int argc, const char *argv[]) {
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
           RTLD_NOW);

    if (argc >= 4) {
        const int hidden = atoi(argv[1]);
        const int interm = atoi(argv[2]);
        const int batch = atoi(argv[3]);
        return runOnce(hidden, interm, batch, /*verboseMIL=*/argc > 4);
    }

    printf("=== ane_matmul_w_input_probe ===\n");
    printf("Usage: ane_matmul_w_input_probe hidden interm batch [verboseMIL]\n\n");

    // Small smoke first so any structural failure prints quickly.
    printf("--- smoke: small shape (h=256, i=512, b=32) ---\n");
    int rc = runOnce(256, 512, 32, /*verboseMIL=*/argc > 1);
    if (rc != 0) {
        printf("Smoke failed (rc=%d). Stopping before DSv4 shape.\n", rc);
        return rc;
    }

    printf("\n--- DSv4 MLP gate-shape (h=7168, i=18432, b=32) ---\n");
    rc = runOnce(7168, 18432, 32, /*verboseMIL=*/argc > 1);
    if (rc != 0) {
        printf("DSv4 shape failed (rc=%d).\n", rc);
        return rc;
    }

    int down_batches[] = {1, 8, 16, 32};
    for (int i = 0; i < (int)(sizeof(down_batches) / sizeof(down_batches[0])); i++) {
        printf("\n--- DSv4 MLP down-shape (h=18432, i=7168, b=%d) ---\n", down_batches[i]);
        rc = runOnce(18432, 7168, down_batches[i], /*verboseMIL=*/argc > 1);
        if (rc != 0) printf("down b=%d failed rc=%d\n", down_batches[i], rc);
    }
    return 0;
}
