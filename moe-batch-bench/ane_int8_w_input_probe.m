// ane_int8_w_input_probe.m
//
// Probe streamed int8 weights and int8 activations on _ANEInMemoryModel.
// Inputs:
//   Wq [H, I] int8
//   Xq [B, H] int8
// MIL dequantizes both to fp16, then uses the known-good W-as-input matmul:
//   output = dequant(Xq) @ dequant(Wq)
//
// This answers whether "dynamic weights" can at least reduce input bandwidth
// with int8 tensors. It does not prove native int8 GEMM; the proven native int8
// ANE path remains conv1x1 + const/BLOBFILE + quantize/dequantize hint.

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static NSString *genMIL_int8_w_input_matmul(int H, int I, int B) {
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
        @"    func main<ios18>(tensor<int8, [%d, %d]> Wq, tensor<int8, [%d, %d]> Xq) {\n",
        H, I, B, H];
    [m appendString:
        @"            fp16 scale = const()[name = string(\"scale\"), val = fp16(0x1p-3)];\n"
        @"            int8 zp = const()[name = string(\"zp\"), val = int8(0)];\n"
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"
        @"            tensor<int32, [2]> perm0 = const()[name = string(\"perm0\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            tensor<int32, [1]> ax0 = const()[name = string(\"ax0\"), val = tensor<int32, [1]>([0])];\n"];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> W = dequantize(input = Wq, scale = scale, zero_point = zp)[name = string(\"W\")];\n",
        H, I];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> X = dequantize(input = Xq, scale = scale, zero_point = zp)[name = string(\"X\")];\n",
        B, H];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm0, x = X)[name = string(\"Xt\")];\n",
        H, B];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = Xt)[name = string(\"X3\")];\n",
        H, B];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = W)[name = string(\"Y\")];\n",
        B, I];
    [m appendString:
        @"        } -> (Y);\n"
        @"}\n"];
    return m;
}

static NSString *genMIL_int8_w_input_mlp(int H, int I, int B) {
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
        @"    func main<ios18>(tensor<int8, [%d, %d]> Wgq, tensor<int8, [%d, %d]> Wuq, tensor<int8, [%d, %d]> Wdq, tensor<int8, [%d, %d]> Xq) {\n",
        H, I, H, I, I, H, B, H];
    [m appendString:
        @"            fp16 scale = const()[name = string(\"scale\"), val = fp16(0x1p-3)];\n"
        @"            int8 zp = const()[name = string(\"zp\"), val = int8(0)];\n"
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"
        @"            tensor<int32, [2]> perm0 = const()[name = string(\"perm0\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            tensor<int32, [1]> ax0 = const()[name = string(\"ax0\"), val = tensor<int32, [1]>([0])];\n"];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> Wg = dequantize(input = Wgq, scale = scale, zero_point = zp)[name = string(\"Wg\")];\n",
        H, I];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> Wu = dequantize(input = Wuq, scale = scale, zero_point = zp)[name = string(\"Wu\")];\n",
        H, I];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> Wd = dequantize(input = Wdq, scale = scale, zero_point = zp)[name = string(\"Wd\")];\n",
        I, H];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> X = dequantize(input = Xq, scale = scale, zero_point = zp)[name = string(\"X\")];\n",
        B, H];
    [m appendFormat:
        @"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm0, x = X)[name = string(\"Xt\")];\n",
        H, B];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = Xt)[name = string(\"X3\")];\n",
        H, B];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> gate = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = Wg)[name = string(\"gate\")];\n",
        B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> up = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = Wu)[name = string(\"up\")];\n",
        B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> act = silu(x = gate)[name = string(\"act\")];\n",
        B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> hidden = mul(x = act, y = up)[name = string(\"hidden\")];\n",
        B, I];
    [m appendFormat:
        @"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_f, transpose_y = tx_f, x = hidden, y = Wd)[name = string(\"Y\")];\n",
        B, H];
    [m appendString:
        @"        } -> (Y);\n"
        @"}\n"];
    return m;
}

static void fill_i8(IOSurfaceRef s) {
    IOSurfaceLock(s, 0, NULL);
    int8_t *p = (int8_t *)IOSurfaceGetBaseAddress(s);
    size_t n = IOSurfaceGetAllocSize(s);
    for (size_t i = 0; i < n; i++) p[i] = (int8_t)((i * 13u + 17u) & 0x7f);
    IOSurfaceUnlock(s, 0, NULL);
}

static int runOnce(int H, int I, int B, int warmup, int iters, BOOL printMIL, BOOL mlp) {
    @autoreleasepool {
        NSError *e = nil;
        NSString *milStr = mlp ? genMIL_int8_w_input_mlp(H, I, B) :
                                 genMIL_int8_w_input_matmul(H, I, B);
        if (printMIL) printf("=== MIL ===\n%s\n", [milStr UTF8String]);
        NSData *milData = [[milStr dataUsingEncoding:NSUTF8StringEncoding] copy];

        Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IM = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!D || !IM || !AR || !AIO) {
            printf("FAIL: private ANE classes unavailable\n");
            return 1;
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            D, @selector(modelWithMILText:weights:optionsPlist:), milData, @{}, nil);
        if (!desc) { printf("FAIL: descriptor nil\n"); return 2; }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(IM, @selector(inMemoryModelWithDescriptor:), desc);
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
            printf("compile FAIL: %s\n", e ? [[e description] UTF8String] : "<nil>");
            [fm removeItemAtPath:td error:nil];
            return 4;
        }

        uint64_t tl0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        double load_ms = ticksToMs(mach_absolute_time() - tl0);
        if (!ok) {
            printf("load FAIL: %s\n", e ? [[e description] UTF8String] : "<nil>");
            [fm removeItemAtPath:td error:nil];
            return 5;
        }

        NSUInteger wBytes = (NSUInteger)H * (NSUInteger)I;
        NSUInteger xBytes = (NSUInteger)B * (NSUInteger)H;
        NSUInteger wdBytes = (NSUInteger)I * (NSUInteger)H;
        NSUInteger yBytes = (NSUInteger)B * (NSUInteger)(mlp ? H : I) * 2u;
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
        IOSurfaceRef ioW = IOSurfaceCreate((__bridge CFDictionaryRef)props(wBytes));
        IOSurfaceRef ioWu = mlp ? IOSurfaceCreate((__bridge CFDictionaryRef)props(wBytes)) : NULL;
        IOSurfaceRef ioWd = mlp ? IOSurfaceCreate((__bridge CFDictionaryRef)props(wdBytes)) : NULL;
        IOSurfaceRef ioX = IOSurfaceCreate((__bridge CFDictionaryRef)props(xBytes));
        IOSurfaceRef ioY = IOSurfaceCreate((__bridge CFDictionaryRef)props(yBytes));
        if (!ioW || !ioX || !ioY || (mlp && (!ioWu || !ioWd))) {
            printf("FAIL: IOSurfaceCreate\n");
            return 6;
        }
        fill_i8(ioW);
        if (ioWu) fill_i8(ioWu);
        if (ioWd) fill_i8(ioWd);
        fill_i8(ioX);

        id wW = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioW);
        id wWu = ioWu ? ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioWu) : nil;
        id wWd = ioWd ? ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioWd) : nil;
        id wX = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioX);
        id wY = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioY);
        id req = mlp ?
            ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                AR, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wW, wWu, wWd, wX], @[@0, @1, @2, @3], @[wY], @[@0], nil, nil, @0) :
            ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                AR, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wW, wX], @[@0, @1], @[wY], @[@0], nil, nil, @0);

        for (int i = 0; i < warmup; i++) {
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
            if (!ok) {
                printf("eval FAIL: %s\n", e ? [[e description] UTF8String] : "<nil>");
                break;
            }
        }

        double ms = -1.0;
        if (ok) {
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                if (!ok) break;
            }
            ms = ticksToMs(mach_absolute_time() - t0) / (double)iters;
        }
        if (!ok) {
            printf("timed eval FAIL: %s\n", e ? [[e description] UTF8String] : "<nil>");
        } else {
            double gflop = (mlp ? 6.0 : 2.0) * (double)B * (double)H * (double)I / 1e9;
            printf("compile %.1f ms | load %.1f ms\n", compile_ms, load_ms);
            printf("eval int8-input-W %s: %.3f ms/iter | %.2f GFLOP/s | W %.1f MB int8, X %.3f MB int8\n",
                   mlp ? "mlp" : "matmul", ms, gflop / (ms / 1000.0),
                   (double)(mlp ? (2u * wBytes + wdBytes) : wBytes) / 1e6,
                   (double)xBytes / 1e6);
        }

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioW);
        if (ioWu) CFRelease(ioWu);
        if (ioWd) CFRelease(ioWd);
        CFRelease(ioX); CFRelease(ioY);
        [fm removeItemAtPath:td error:nil];
        return ok ? 0 : 7;
    }
}

int main(int argc, const char *argv[]) {
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    int H = 4096, I = 2048, B = 64, warmup = 5, iters = 50;
    BOOL printMIL = NO;
    BOOL mlp = NO;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--mil")) {
            printMIL = YES;
        } else if (!strcmp(argv[a], "--mlp")) {
            mlp = YES;
        } else if (!strcmp(argv[a], "--shape") && a + 3 < argc) {
            H = atoi(argv[++a]);
            I = atoi(argv[++a]);
            B = atoi(argv[++a]);
        } else if (!strcmp(argv[a], "--iters") && a + 1 < argc) {
            iters = atoi(argv[++a]);
        } else if (!strcmp(argv[a], "--warmup") && a + 1 < argc) {
            warmup = atoi(argv[++a]);
        } else {
            fprintf(stderr, "usage: %s [--shape H I B] [--warmup N] [--iters N] [--mlp] [--mil]\n", argv[0]);
            return 2;
        }
    }
    printf("=== ANE int8 W/input function-input %s probe ===\n", mlp ? "MLP" : "matmul");
    printf("H=%d I=%d B=%d warmup=%d iters=%d\n", H, I, B, warmup, iters);
    return runOnce(H, I, B, warmup, iters, printMIL, mlp);
}
