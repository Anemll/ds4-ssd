// Value-check a single ANE fp16 matmul with dynamic W input.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m++;
    if (m & 0x400u) {
        m = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (m & 0x3ffu));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        bits = sign;
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static IOSurfaceRef make_surface(NSUInteger bytes) {
    NSDictionary *p = @{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

static int write_surface(IOSurfaceRef s, const void *src, NSUInteger bytes) {
    if (IOSurfaceLock(s, 0, NULL) != kIOReturnSuccess) return 0;
    memcpy(IOSurfaceGetBaseAddress(s), src, bytes);
    IOSurfaceUnlock(s, 0, NULL);
    return 1;
}

static int read_surface(IOSurfaceRef s, void *dst, NSUInteger bytes) {
    if (IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return 0;
    memcpy(dst, IOSurfaceGetBaseAddress(s), bytes);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return 1;
}

static NSString *gen_mil(int H, int O, int B, int mode) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [%d, %d]> W, tensor<fp16, [%d, %d]> X) {\n",
                    H, O, B, H];
    [m appendString:
        @"            tensor<int32, [1]> ax0 = const()[name = string(\"ax0\"), val = tensor<int32, [1]>([0])];\n"
        @"            tensor<int32, [2]> perm2 = const()[name = string(\"perm2\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"];
    if (mode == 0) {
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Y = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X, y = W)[name = string(\"Y\")];\n", B, O];
    } else if (mode == 1) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = X)[name = string(\"X3\")];\n", B, H];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = W)[name = string(\"Y\")];\n", B, O];
    } else {
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm2, x = X)[name = string(\"Xt\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = Xt)[name = string(\"X3\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = W)[name = string(\"Y\")];\n", B, O];
    }
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

static int run(int H, int O, int B, int mode) {
    @autoreleasepool {
        Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class M = NSClassFromString(@"_ANEInMemoryModel");
        Class R = NSClassFromString(@"_ANERequest");
        Class IO = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!D || !M || !R || !IO) return 2;
        NSString *mil = gen_mil(H, O, B, mode);
        NSData *milData = [[mil dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSError *e = nil;
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D, @selector(modelWithMILText:weights:optionsPlist:), milData, @{}, nil);
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(M, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            printf("compile failed %s\n", e.localizedDescription.UTF8String);
            return 3;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) return 4;

        size_t wN = (size_t)H * O, xN = (size_t)B * H, yN = (size_t)B * O;
        uint16_t *W = calloc(wN, sizeof(uint16_t));
        uint16_t *X = calloc(xN, sizeof(uint16_t));
        uint16_t *Y = calloc(yN, sizeof(uint16_t));
        float *ref = calloc(yN, sizeof(float));
        for (size_t i = 0; i < wN; i++) W[i] = f32_to_f16((float)((int)(i * 17 % 31) - 15) * 0.05f);
        for (size_t i = 0; i < xN; i++) X[i] = f32_to_f16((float)((int)(i * 11 % 29) - 14) * 0.04f);
        for (int b = 0; b < B; b++) {
            for (int o = 0; o < O; o++) {
                double s = 0.0;
                for (int h = 0; h < H; h++) s += (double)f16_to_f32(X[b * H + h]) * f16_to_f32(W[h * O + o]);
                ref[b * O + o] = (float)s;
            }
        }
        IOSurfaceRef ioW = make_surface(wN * 2), ioX = make_surface(xN * 2), ioY = make_surface(yN * 2);
        write_surface(ioW, W, wN * 2);
        write_surface(ioX, X, xN * 2);
        id wW = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(IO, @selector(objectWithIOSurface:), ioW);
        id wX = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(IO, @selector(objectWithIOSurface:), ioX);
        id wY = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(IO, @selector(objectWithIOSurface:), ioY);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            R, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wW, wX], @[@0, @1], @[wY], @[@0], nil, nil, @0);
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        if (!ok) {
            printf("eval failed %s\n", e.localizedDescription.UTF8String);
            return 5;
        }
        read_surface(ioY, Y, yN * 2);
        double sq = 0.0, rsq = 0.0;
        float max_abs = 0.0f;
        int worst = 0;
        for (size_t i = 0; i < yN; i++) {
            float a = f16_to_f32(Y[i]);
            float err = fabsf(a - ref[i]);
            sq += (double)err * err;
            rsq += (double)ref[i] * ref[i];
            if (err > max_abs) { max_abs = err; worst = (int)i; }
        }
        printf("matmul H=%d O=%d B=%d mode=%d max_abs=%g rel_rms=%g worst=%d ref=%g ane=%g\n",
               H, O, B, mode, max_abs, sqrt(sq / fmax(rsq, 1e-30)), worst, ref[worst], f16_to_f32(Y[worst]));
        for (int i = 0; i < 8 && i < (int)yN; i++) printf("i=%d ref=%g ane=%g\n", i, ref[i], f16_to_f32(Y[i]));
        return 0;
    }
}

int main(int argc, const char **argv) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    int H = argc > 1 ? atoi(argv[1]) : 128;
    int O = argc > 2 ? atoi(argv[2]) : 64;
    int B = argc > 3 ? atoi(argv[3]) : 2;
    int mode = argc > 4 ? atoi(argv[4]) : 1;
    return run(H, O, B, mode);
}
