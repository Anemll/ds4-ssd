// Experimental ANE DS4 MLP with fp16 activations and int8 dynamic weights.

#import "ane_ds4_mlp_int8w.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct ds4_ane_mlp_int8w_ctx {
    int H, I, B;
    int mode;
    float w_scale;
    float x_scale;
    float mid_scale;
    void *model_r;
    void *request_r;
    void *model_down_r;
    void *request_down_r;
    void *tmpDir_r;
    void *tmpDir_down_r;
    IOSurfaceRef io_gate;
    IOSurfaceRef io_up;
    IOSurfaceRef io_down;
    IOSurfaceRef io_x;
    IOSurfaceRef io_mid;
    IOSurfaceRef io_hidden;
    IOSurfaceRef io_route;
    IOSurfaceRef io_out;
    NSUInteger gate_bytes;
    NSUInteger down_bytes;
    NSUInteger x_bytes;
    NSUInteger mid_bytes;
    NSUInteger hidden_bytes;
    NSUInteger route_bytes;
    NSUInteger out_bytes;
};

static dispatch_once_t g_classes_once;
static Class g_DescCls = nil;
static Class g_ModelCls = nil;
static Class g_ReqCls = nil;
static Class g_IOCls = nil;

static bool ane_int8w_debug_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_DEBUG");
    return env && env[0] && atoi(env) != 0;
}

static bool ane_int8w_stats_enabled(void) {
    const char *env = getenv("DS4_FLASH_MOE_ANE_STATS");
    return (env && env[0] && atoi(env) != 0) || ane_int8w_debug_enabled();
}

static uint64_t g_i8i8_hidden_values;
static uint64_t g_i8i8_hidden_saturated;
static float g_i8i8_hidden_abs_max;

void ds4_ane_mlp_int8w_quant_stats_reset(void) {
    g_i8i8_hidden_values = 0;
    g_i8i8_hidden_saturated = 0;
    g_i8i8_hidden_abs_max = 0.0f;
}

void ds4_ane_mlp_int8w_quant_stats(uint64_t *hidden_values,
                                   uint64_t *hidden_saturated,
                                   float *hidden_abs_max) {
    if (hidden_values) *hidden_values = g_i8i8_hidden_values;
    if (hidden_saturated) *hidden_saturated = g_i8i8_hidden_saturated;
    if (hidden_abs_max) *hidden_abs_max = g_i8i8_hidden_abs_max;
}

static int ane_mlp_test_mode(void) {
    const char *env = getenv("DS4_ANE_MLP_TEST_MODE");
    return env && env[0] ? atoi(env) : 0;
}

static void resolve_classes(void) {
    dispatch_once(&g_classes_once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
               RTLD_NOW);
        g_DescCls  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_ModelCls = NSClassFromString(@"_ANEInMemoryModel");
        g_ReqCls   = NSClassFromString(@"_ANERequest");
        g_IOCls    = NSClassFromString(@"_ANEIOSurfaceObject");
    });
}

static IOSurfaceRef make_surface_typed(NSUInteger bytes, NSUInteger elem) {
    if (elem == 0) elem = 1;
    const NSUInteger width = (bytes + elem - 1u) / elem;
    NSDictionary *p = @{
        (id)kIOSurfaceWidth: @(width),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @(elem),
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

static bool write_surface(IOSurfaceRef s, const void *src, NSUInteger bytes) {
    if (!s || !src) return false;
    if (IOSurfaceGetAllocSize(s) < bytes) return false;
    if (IOSurfaceLock(s, 0, NULL) != kIOReturnSuccess) return false;
    memcpy(IOSurfaceGetBaseAddress(s), src, bytes);
    IOSurfaceUnlock(s, 0, NULL);
    return true;
}

static bool read_surface(IOSurfaceRef s, void *dst, NSUInteger bytes) {
    if (!s || !dst) return false;
    if (IOSurfaceGetAllocSize(s) < bytes) return false;
    if (IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
    memcpy(dst, IOSurfaceGetBaseAddress(s), bytes);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return true;
}

static uint16_t ane_f32_to_f16_bits(float f) {
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

static float ane_f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static NSString *gen_mil_fp16_matmul(int H, int O, int B) {
    BOOL direct_x = H > O;
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
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"];
    if (direct_x) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = X)[name = string(\"X3\")];\n", B, H];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = W)[name = string(\"Y\")];\n", B, O];
    } else {
        [m appendString:@"            tensor<int32, [2]> perm0 = const()[name = string(\"perm0\"), val = tensor<int32, [2]>([1, 0])];\n"];
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm0, x = X)[name = string(\"Xt\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = Xt)[name = string(\"X3\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = W)[name = string(\"Y\")];\n", B, O];
    }
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

static NSString *gen_mil_i8w_fp16x_matmul(int H, int O, int B, float w_scale) {
    BOOL direct_x = H > O;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<int8, [%d, %d]> Wq, tensor<fp16, [%d, %d]> X) {\n",
                    H, O, B, H];
    [m appendFormat:@"            fp16 wscale = const()[name = string(\"wscale\"), val = fp16(%a)];\n",
                    (double)w_scale];
    [m appendString:
        @"            int8 zp = const()[name = string(\"zp\"), val = int8(0)];\n"
        @"            tensor<int32, [1]> ax0 = const()[name = string(\"ax0\"), val = tensor<int32, [1]>([0])];\n"
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> W = dequantize(input = Wq, scale = wscale, zero_point = zp)[name = string(\"W\")];\n", H, O];
    if (direct_x) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = X)[name = string(\"X3\")];\n", B, H];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = W)[name = string(\"Y\")];\n", B, O];
    } else {
        [m appendString:@"            tensor<int32, [2]> perm0 = const()[name = string(\"perm0\"), val = tensor<int32, [2]>([1, 0])];\n"];
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm0, x = X)[name = string(\"Xt\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = Xt)[name = string(\"X3\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = W)[name = string(\"Y\")];\n", B, O];
    }
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

static NSString *gen_mil_i8w_i8x_matmul(int H, int O, int B, float w_scale, float x_scale) {
    BOOL direct_x = H > O;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<int8, [%d, %d]> Wq, tensor<int8, [%d, %d]> Xq) {\n",
                    H, O, B, H];
    [m appendFormat:@"            fp16 wscale = const()[name = string(\"wscale\"), val = fp16(%a)];\n",
                    (double)w_scale];
    [m appendFormat:@"            fp16 xscale = const()[name = string(\"xscale\"), val = fp16(%a)];\n",
                    (double)x_scale];
    [m appendString:
        @"            int8 zp = const()[name = string(\"zp\"), val = int8(0)];\n"
        @"            tensor<int32, [1]> ax0 = const()[name = string(\"ax0\"), val = tensor<int32, [1]>([0])];\n"
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> W = dequantize(input = Wq, scale = wscale, zero_point = zp)[name = string(\"W\")];\n", H, O];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> X = dequantize(input = Xq, scale = xscale, zero_point = zp)[name = string(\"X\")];\n", B, H];
    if (direct_x) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = X)[name = string(\"X3\")];\n", B, H];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = W)[name = string(\"Y\")];\n", B, O];
    } else {
        [m appendString:@"            tensor<int32, [2]> perm0 = const()[name = string(\"perm0\"), val = tensor<int32, [2]>([1, 0])];\n"];
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm0, x = X)[name = string(\"Xt\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = Xt)[name = string(\"X3\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = W)[name = string(\"Y\")];\n", B, O];
    }
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

static NSString *gen_mil_i8w_i8x_gateup(int H, int I, int B, float w_scale, float x_scale) {
    const BOOL direct_x = H > I;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:
        @"    func main<ios18>(tensor<int8, [%d, %d]> Wgq, tensor<int8, [%d, %d]> Wuq, tensor<int8, [%d, %d]> Xq) {\n",
        H, I, H, I, B, H];
    [m appendFormat:@"            fp16 wscale = const()[name = string(\"wscale\"), val = fp16(%a)];\n",
                    (double)w_scale];
    [m appendFormat:@"            fp16 xscale = const()[name = string(\"xscale\"), val = fp16(%a)];\n",
                    (double)x_scale];
    [m appendString:
        @"            int8 zp = const()[name = string(\"zp\"), val = int8(0)];\n"
        @"            tensor<int32, [1]> ax0 = const()[name = string(\"ax0\"), val = tensor<int32, [1]>([0])];\n"
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wg = dequantize(input = Wgq, scale = wscale, zero_point = zp)[name = string(\"Wg\")];\n", H, I];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wu = dequantize(input = Wuq, scale = wscale, zero_point = zp)[name = string(\"Wu\")];\n", H, I];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> X = dequantize(input = Xq, scale = xscale, zero_point = zp)[name = string(\"X\")];\n", B, H];
    if (direct_x) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = X)[name = string(\"X3\")];\n", B, H];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> gate = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = Wg)[name = string(\"gate\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> up = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = Wu)[name = string(\"up\")];\n", B, I];
    } else {
        [m appendString:@"            tensor<int32, [2]> perm0 = const()[name = string(\"perm0\"), val = tensor<int32, [2]>([1, 0])];\n"];
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm0, x = X)[name = string(\"Xt\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = Xt)[name = string(\"X3\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> gate = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = Wg)[name = string(\"gate\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> up = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = Wu)[name = string(\"up\")];\n", B, I];
    }
    [m appendString:@"        } -> (gate, up);\n}\n"];
    return m;
}

static NSString *gen_mil_i8w_i8x_fused(int H, int I, int B, float w_scale, float x_scale, float mid_scale) {
    const int test_mode = ane_mlp_test_mode();
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:
        @"    func main<ios18>(tensor<int8, [%d, %d]> Wgq, tensor<int8, [%d, %d]> Wuq, tensor<int8, [%d, %d]> Wdq, tensor<int8, [%d, %d]> Xq) {\n",
        H, I, H, I, I, H, B, H];
    [m appendFormat:@"            fp16 wscale = const()[name = string(\"wscale\"), val = fp16(%a)];\n",
                    (double)w_scale];
    [m appendFormat:@"            fp16 xscale = const()[name = string(\"xscale\"), val = fp16(%a)];\n",
                    (double)x_scale];
    [m appendFormat:@"            fp16 midscale = const()[name = string(\"midscale\"), val = fp16(%a)];\n",
                    (double)mid_scale];
    [m appendString:
        @"            int8 zp = const()[name = string(\"zp\"), val = int8(0)];\n"
        @"            string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"
        @"            fp16 clamp_hi = const()[name = string(\"clamp_hi\"), val = fp16(0x1.4p+3)];\n"
        @"            fp16 clamp_lo = const()[name = string(\"clamp_lo\"), val = fp16(-0x1.4p+3)];\n"
        @"            fp16 one = const()[name = string(\"one\"), val = fp16(0x1p+0)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wg = dequantize(input = Wgq, scale = wscale, zero_point = zp)[name = string(\"Wg\")];\n", H, I];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wu = dequantize(input = Wuq, scale = wscale, zero_point = zp)[name = string(\"Wu\")];\n", H, I];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wd = dequantize(input = Wdq, scale = wscale, zero_point = zp)[name = string(\"Wd\")];\n", I, H];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> X = dequantize(input = Xq, scale = xscale, zero_point = zp)[name = string(\"X\")];\n", B, H];
    [m appendFormat:@"            tensor<int32, [4]> x_shape = const()[name = string(\"x_shape\"), val = tensor<int32, [4]>([1, 1, %d, %d])];\n", B, H];
    [m appendFormat:@"            tensor<int32, [4]> wgu_shape = const()[name = string(\"wgu_shape\"), val = tensor<int32, [4]>([1, 1, %d, %d])];\n", H, 2 * I];
    [m appendFormat:@"            tensor<int32, [4]> wd_shape = const()[name = string(\"wd_shape\"), val = tensor<int32, [4]>([1, 1, %d, %d])];\n", I, H];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wgu2 = concat(axis = int32(1), interleave = bool(false), values = (Wg, Wu))[name = string(\"Wgu2\")];\n", H, 2 * I];
    [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> X4 = reshape(shape = x_shape, x = X)[name = string(\"X4\")];\n", B, H];
    [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> Wgu = reshape(shape = wgu_shape, x = Wgu2)[name = string(\"Wgu\")];\n", H, 2 * I];
    [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> Wd4 = reshape(shape = wd_shape, x = Wd)[name = string(\"Wd4\")];\n", I, H];
    [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> gu = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X4, y = Wgu)[name = string(\"gu\")];\n", B, 2 * I];
    [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> gate = slice_by_index(begin = tensor<int32, [4]>([0, 0, 0, 0]), end = tensor<int32, [4]>([1, 1, %d, %d]), end_mask = tensor<bool, [4]>([false, false, false, false]), stride = tensor<int32, [4]>([1, 1, 1, 1]), x = gu)[name = string(\"gate\")];\n",
                    B, I, B, I];
    [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> up = slice_by_index(begin = tensor<int32, [4]>([0, 0, 0, %d]), end = tensor<int32, [4]>([1, 1, %d, %d]), end_mask = tensor<bool, [4]>([false, false, false, false]), stride = tensor<int32, [4]>([1, 1, 1, 1]), x = gu)[name = string(\"up\")];\n",
                    B, I, I, B, 2 * I];
    if (test_mode == 4) {
        [m appendString:@"        } -> (gate);\n}\n"];
        return m;
    }
    [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> gate_c = clip(x = gate, alpha = clamp_lo, beta = clamp_hi)[name = string(\"gate_c\")];\n", B, I];
    [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> up_c = clip(x = up, alpha = clamp_lo, beta = clamp_hi)[name = string(\"up_c\")];\n", B, I];
    if (test_mode == 1) {
        [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> hidden_fp = mul(x = gate_c, y = one)[name = string(\"hidden_fp\")];\n", B, I];
    } else if (test_mode == 2) {
        [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> hidden_fp = mul(x = gate_c, y = up_c)[name = string(\"hidden_fp\")];\n", B, I];
    } else if (test_mode == 3) {
        [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> hidden_fp = silu(x = gate_c)[name = string(\"hidden_fp\")];\n", B, I];
    } else {
        [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> act = silu(x = gate_c)[name = string(\"act\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> hidden_fp = mul(x = act, y = up_c)[name = string(\"hidden_fp\")];\n", B, I];
    }
    if (test_mode == 5) {
        [m appendString:@"        } -> (hidden_fp);\n}\n"];
        return m;
    }
    [m appendFormat:@"            tensor<int8, [1, 1, %d, %d]> hidden_q = quantize(input = hidden_fp, output_dtype = q_dtype, scale = midscale, zero_point = zp)[name = string(\"hidden_q\")];\n", B, I];
    [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> hidden = dequantize(input = hidden_q, scale = midscale, zero_point = zp)[name = string(\"hidden\")];\n", B, I];
    [m appendFormat:@"            tensor<fp16, [1, 1, %d, %d]> Y = matmul(transpose_x = tx_f, transpose_y = tx_f, x = hidden, y = Wd4)[name = string(\"Y\")];\n", B, H];
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

static NSString *gen_mil_i8w_i8x_tiled_fused(int H, int I, int B, float w_scale, float x_scale, float mid_scale) {
    const int tile_i = 256;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:
        @"    func main<ios18>(tensor<int8, [%d, %d]> Wgq, tensor<int8, [%d, %d]> Wuq, tensor<int8, [%d, %d]> Wdq, tensor<int8, [%d, %d]> Xq) {\n",
        H, I, H, I, I, H, B, H];
    [m appendFormat:@"            fp16 wscale = const()[name = string(\"wscale\"), val = fp16(%a)];\n",
                    (double)w_scale];
    [m appendFormat:@"            fp16 xscale = const()[name = string(\"xscale\"), val = fp16(%a)];\n",
                    (double)x_scale];
    [m appendFormat:@"            fp16 midscale = const()[name = string(\"midscale\"), val = fp16(%a)];\n",
                    (double)mid_scale];
    [m appendString:
        @"            int8 zp = const()[name = string(\"zp\"), val = int8(0)];\n"
        @"            string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"
        @"            fp16 clamp_hi = const()[name = string(\"clamp_hi\"), val = fp16(0x1.4p+3)];\n"
        @"            fp16 clamp_lo = const()[name = string(\"clamp_lo\"), val = fp16(-0x1.4p+3)];\n"
        @"            tensor<int32, [1]> ax0 = const()[name = string(\"ax0\"), val = tensor<int32, [1]>([0])];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> X = dequantize(input = Xq, scale = xscale, zero_point = zp)[name = string(\"X\")];\n", B, H];
    [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = X)[name = string(\"X3\")];\n", B, H];

    NSString *prev_y = nil;
    for (int start = 0; start < I; start += tile_i) {
        const int end = start + tile_i < I ? start + tile_i : I;
        const int T = end - start;
        NSString *tag = [NSString stringWithFormat:@"i%d_%d", start, end];
        [m appendFormat:@"            tensor<int8, [%d, %d]> Wgq_%@ = slice_by_index(begin = tensor<int32, [2]>([0, %d]), end = tensor<int32, [2]>([%d, %d]), x = Wgq)[name = string(\"Wgq_%@\")];\n",
                        H, T, tag, start, H, end, tag];
        [m appendFormat:@"            tensor<int8, [%d, %d]> Wuq_%@ = slice_by_index(begin = tensor<int32, [2]>([0, %d]), end = tensor<int32, [2]>([%d, %d]), x = Wuq)[name = string(\"Wuq_%@\")];\n",
                        H, T, tag, start, H, end, tag];
        [m appendFormat:@"            tensor<int8, [%d, %d]> Wdq_%@ = slice_by_index(begin = tensor<int32, [2]>([%d, 0]), end = tensor<int32, [2]>([%d, %d]), x = Wdq)[name = string(\"Wdq_%@\")];\n",
                        T, H, tag, start, end, H, tag];
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Wg_%@ = dequantize(input = Wgq_%@, scale = wscale, zero_point = zp)[name = string(\"Wg_%@\")];\n",
                        H, T, tag, tag, tag];
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Wu_%@ = dequantize(input = Wuq_%@, scale = wscale, zero_point = zp)[name = string(\"Wu_%@\")];\n",
                        H, T, tag, tag, tag];
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Wd_%@ = dequantize(input = Wdq_%@, scale = wscale, zero_point = zp)[name = string(\"Wd_%@\")];\n",
                        T, H, tag, tag, tag];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> gate_%@ = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = Wg_%@)[name = string(\"gate_%@\")];\n",
                        B, T, tag, tag, tag];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> up_%@ = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = Wu_%@)[name = string(\"up_%@\")];\n",
                        B, T, tag, tag, tag];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> gate_c_%@ = clip(x = gate_%@, alpha = clamp_lo, beta = clamp_hi)[name = string(\"gate_c_%@\")];\n",
                        B, T, tag, tag, tag];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> up_c_%@ = clip(x = up_%@, alpha = clamp_lo, beta = clamp_hi)[name = string(\"up_c_%@\")];\n",
                        B, T, tag, tag, tag];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> act_%@ = silu(x = gate_c_%@)[name = string(\"act_%@\")];\n",
                        B, T, tag, tag, tag];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden_fp_%@ = mul(x = act_%@, y = up_c_%@)[name = string(\"hidden_fp_%@\")];\n",
                        B, T, tag, tag, tag, tag];
        [m appendFormat:@"            tensor<int8, [1, %d, %d]> hidden_q_%@ = quantize(input = hidden_fp_%@, output_dtype = q_dtype, scale = midscale, zero_point = zp)[name = string(\"hidden_q_%@\")];\n",
                        B, T, tag, tag, tag];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden_%@ = dequantize(input = hidden_q_%@, scale = midscale, zero_point = zp)[name = string(\"hidden_%@\")];\n",
                        B, T, tag, tag, tag];
        NSString *y_name = [NSString stringWithFormat:@"Y_%@", tag];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> %@ = matmul(transpose_x = tx_f, transpose_y = tx_f, x = hidden_%@, y = Wd_%@)[name = string(\"%@\")];\n",
                        B, H, y_name, tag, tag, y_name];
        if (prev_y) {
            NSString *acc_name = [NSString stringWithFormat:@"Y_acc_%@", tag];
            [m appendFormat:@"            tensor<fp16, [1, %d, %d]> %@ = add(x = %@, y = %@)[name = string(\"%@\")];\n",
                            B, H, acc_name, prev_y, y_name, acc_name];
            prev_y = acc_name;
        } else {
            prev_y = y_name;
        }
    }
    [m appendFormat:@"        } -> (%@);\n}\n", prev_y ?: @"X3"];
    return m;
}

static NSString *gen_mil_int8w(int H, int I, int B, float w_scale, float x_scale) {
    const BOOL gate_direct = H > I;
    const BOOL down_direct = I > H;
    const int test_mode = ane_mlp_test_mode();
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}})]\n{\n"];
    [m appendFormat:
        @"    func main<ios18>(tensor<int8, [%d, %d]> Wgq, tensor<int8, [%d, %d]> Wuq, tensor<int8, [%d, %d]> Wdq, tensor<int8, [%d, %d]> Xq) {\n",
        H, I, H, I, I, H, B, H];
    [m appendFormat:
        @"            fp16 wscale = const()[name = string(\"wscale\"), val = fp16(%a)];\n",
        (double)w_scale];
    [m appendFormat:
        @"            fp16 xscale = const()[name = string(\"xscale\"), val = fp16(%a)];\n",
        (double)x_scale];
    [m appendString:
        @"            int8 zp = const()[name = string(\"zp\"), val = int8(0)];\n"
        @"            fp16 clamp_hi = const()[name = string(\"clamp_hi\"), val = fp16(0x1.4p+3)];\n"
        @"            fp16 clamp_lo = const()[name = string(\"clamp_lo\"), val = fp16(-0x1.4p+3)];\n"
        @"            fp16 one = const()[name = string(\"one\"), val = fp16(0x1p+0)];\n"
        @"            tensor<int32, [1]> ax0 = const()[name = string(\"ax0\"), val = tensor<int32, [1]>([0])];\n"
        @"            tensor<int32, [2]> perm2 = const()[name = string(\"perm2\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            tensor<int32, [3]> perm3 = const()[name = string(\"perm3\"), val = tensor<int32, [3]>([0, 2, 1])];\n"
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"
    ];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wg = dequantize(input = Wgq, scale = wscale, zero_point = zp)[name = string(\"Wg\")];\n", H, I];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wu = dequantize(input = Wuq, scale = wscale, zero_point = zp)[name = string(\"Wu\")];\n", H, I];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wd = dequantize(input = Wdq, scale = wscale, zero_point = zp)[name = string(\"Wd\")];\n", I, H];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> X = dequantize(input = Xq, scale = xscale, zero_point = zp)[name = string(\"X\")];\n", B, H];
    if (gate_direct) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = X)[name = string(\"X3\")];\n", B, H];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> gate = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = Wg)[name = string(\"gate\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> up = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = Wu)[name = string(\"up\")];\n", B, I];
    } else {
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm2, x = X)[name = string(\"Xt\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = Xt)[name = string(\"X3\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> gate = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = Wg)[name = string(\"gate\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> up = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = Wu)[name = string(\"up\")];\n", B, I];
    }
    if (test_mode == 4) {
        [m appendString:@"        } -> (gate);\n}\n"];
        return m;
    }
    if (test_mode == 1) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden = mul(x = gate, y = one)[name = string(\"hidden\")];\n", B, I];
    } else if (test_mode == 2) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden = mul(x = gate, y = up)[name = string(\"hidden\")];\n", B, I];
    } else if (test_mode == 3) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> act = silu(x = gate)[name = string(\"act\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden = mul(x = act, y = one)[name = string(\"hidden\")];\n", B, I];
    } else {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> gate_c = clip(x = gate, alpha = clamp_lo, beta = clamp_hi)[name = string(\"gate_c\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> up_c = clip(x = up, alpha = clamp_lo, beta = clamp_hi)[name = string(\"up_c\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> act = silu(x = gate_c)[name = string(\"act\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden = mul(x = act, y = up_c)[name = string(\"hidden\")];\n", B, I];
    }
    if (down_direct) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_f, transpose_y = tx_f, x = hidden, y = Wd)[name = string(\"Y\")];\n", B, H];
    } else {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden_t = transpose(perm = perm3, x = hidden)[name = string(\"hidden_t\")];\n", I, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_t, transpose_y = tx_f, x = hidden_t, y = Wd)[name = string(\"Y\")];\n", B, H];
    }
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

static NSString *gen_mil_fp16w(int H, int I, int B) {
    const BOOL gate_direct = H > I;
    const BOOL down_direct = I > H;
    const int test_mode = ane_mlp_test_mode();
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}})]\n{\n"];
    [m appendFormat:
        @"    func main<ios18>(tensor<fp16, [%d, %d]> Wg, tensor<fp16, [%d, %d]> Wu, tensor<fp16, [%d, %d]> Wd, tensor<fp16, [%d, %d]> X) {\n",
        H, I, H, I, I, H, B, H];
    [m appendString:
        @"            fp16 clamp_hi = const()[name = string(\"clamp_hi\"), val = fp16(0x1.4p+3)];\n"
        @"            fp16 clamp_lo = const()[name = string(\"clamp_lo\"), val = fp16(-0x1.4p+3)];\n"
        @"            fp16 one = const()[name = string(\"one\"), val = fp16(0x1p+0)];\n"
        @"            tensor<int32, [1]> ax0 = const()[name = string(\"ax0\"), val = tensor<int32, [1]>([0])];\n"
        @"            tensor<int32, [2]> perm2 = const()[name = string(\"perm2\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            tensor<int32, [3]> perm3 = const()[name = string(\"perm3\"), val = tensor<int32, [3]>([0, 2, 1])];\n"
        @"            bool tx_t = const()[name = string(\"tx_t\"), val = bool(true)];\n"
        @"            bool tx_f = const()[name = string(\"tx_f\"), val = bool(false)];\n"];
    if (gate_direct) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = X)[name = string(\"X3\")];\n", B, H];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> gate = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = Wg)[name = string(\"gate\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> up = matmul(transpose_x = tx_f, transpose_y = tx_f, x = X3, y = Wu)[name = string(\"up\")];\n", B, I];
    } else {
        [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm2, x = X)[name = string(\"Xt\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> X3 = expand_dims(axes = ax0, x = Xt)[name = string(\"X3\")];\n", H, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> gate = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = Wg)[name = string(\"gate\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> up = matmul(transpose_x = tx_t, transpose_y = tx_f, x = X3, y = Wu)[name = string(\"up\")];\n", B, I];
    }
    if (test_mode == 4) {
        [m appendString:@"        } -> (gate);\n}\n"];
        return m;
    }
    if (test_mode == 1) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden = mul(x = gate, y = one)[name = string(\"hidden\")];\n", B, I];
    } else if (test_mode == 2) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden = mul(x = gate, y = up)[name = string(\"hidden\")];\n", B, I];
    } else if (test_mode == 3) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> act = silu(x = gate)[name = string(\"act\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden = mul(x = act, y = one)[name = string(\"hidden\")];\n", B, I];
    } else {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> gate_c = clip(x = gate, alpha = clamp_lo, beta = clamp_hi)[name = string(\"gate_c\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> up_c = clip(x = up, alpha = clamp_lo, beta = clamp_hi)[name = string(\"up_c\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> act = silu(x = gate_c)[name = string(\"act\")];\n", B, I];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden = mul(x = act, y = up_c)[name = string(\"hidden\")];\n", B, I];
    }
    if (down_direct) {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_f, transpose_y = tx_f, x = hidden, y = Wd)[name = string(\"Y\")];\n", B, H];
    } else {
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> hidden_t = transpose(perm = perm3, x = hidden)[name = string(\"hidden_t\")];\n", I, B];
        [m appendFormat:@"            tensor<fp16, [1, %d, %d]> Y = matmul(transpose_x = tx_t, transpose_y = tx_f, x = hidden_t, y = Wd)[name = string(\"Y\")];\n", B, H];
    }
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

static bool compile_and_load_mil(NSString *mil,
                                 const char *label,
                                 void **model_r,
                                 void **tmpDir_r) {
    const bool dbg = ane_int8w_debug_enabled();
    NSError *e = nil;
    NSData *milData = [[mil dataUsingEncoding:NSUTF8StringEncoding] copy];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_DescCls, @selector(modelWithMILText:weights:optionsPlist:), milData, @{}, nil);
    if (!desc) {
        if (dbg) fprintf(stderr, "ds4: ANE %s descriptor create failed\n", label);
        return false;
    }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_ModelCls, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) {
        if (dbg) fprintf(stderr, "ds4: ANE %s inMemoryModel create failed\n", label);
        return false;
    }
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        if (dbg) fprintf(stderr, "ds4: ANE %s compile failed: %s\n",
                         label, e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
        [fm removeItemAtPath:td error:nil];
        return false;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        if (dbg) fprintf(stderr, "ds4: ANE %s load failed: %s\n",
                         label, e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
        [fm removeItemAtPath:td error:nil];
        return false;
    }
    if (model_r) *model_r = (void *)CFBridgingRetain(mdl);
    if (tmpDir_r) *tmpDir_r = (void *)CFBridgingRetain([td copy]);
    return true;
}

static ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_create_common(int H, int I, int B, float w_scale, float x_scale, float mid_scale, int mode) {
    if (H <= 0 || I <= 0 || B <= 0) return NULL;
    if (mode == 0 && (!(w_scale > 0.0f) || !(x_scale > 0.0f))) return NULL;
    if (mode == 3 && (!(w_scale > 0.0f) || !(x_scale > 0.0f) || !(mid_scale > 0.0f))) return NULL;
    if (mode == 4 && (!(w_scale > 0.0f) || !(x_scale > 0.0f) || !(mid_scale > 0.0f))) return NULL;
    if (mode == 5 && (!(w_scale > 0.0f) || !(x_scale > 0.0f) || !(mid_scale > 0.0f))) return NULL;
    if (mode == 6 && (!(w_scale > 0.0f) || !(x_scale > 0.0f) || !(mid_scale > 0.0f))) return NULL;
    resolve_classes();
    const bool dbg = ane_int8w_debug_enabled();
    if (!g_DescCls || !g_ModelCls || !g_ReqCls || !g_IOCls) {
        if (dbg) fprintf(stderr, "ds4: ANE int8w missing private classes desc=%p model=%p req=%p io=%p\n",
                         g_DescCls, g_ModelCls, g_ReqCls, g_IOCls);
        return NULL;
    }
    @autoreleasepool {
        if ((mode == 1 || mode == 2 || mode == 3 || mode == 5) && ane_mlp_test_mode() == 0) {
            ds4_ane_mlp_int8w_ctx *ctx = (ds4_ane_mlp_int8w_ctx *)calloc(1, sizeof(*ctx));
            if (!ctx) return NULL;
            ctx->H = H; ctx->I = I; ctx->B = B; ctx->mode = mode; ctx->w_scale = w_scale; ctx->x_scale = x_scale;
            ctx->mid_scale = mid_scale;
            const NSUInteger w_elem = mode == 1 ? 2u : 1u;
            ctx->gate_bytes = (NSUInteger)H * (NSUInteger)I * w_elem;
            ctx->down_bytes = (NSUInteger)I * (NSUInteger)H * w_elem;
            ctx->x_bytes = (NSUInteger)B * (NSUInteger)H * (mode == 3 ? sizeof(int8_t) : sizeof(uint16_t));
            if (mode == 5) ctx->x_bytes = (NSUInteger)B * (NSUInteger)H * sizeof(int8_t);
            ctx->mid_bytes = (NSUInteger)B * (NSUInteger)I * sizeof(uint16_t);
            ctx->hidden_bytes = (NSUInteger)B * (NSUInteger)I * (mode == 3 ? sizeof(int8_t) : sizeof(uint16_t));
            if (mode == 5) ctx->hidden_bytes = (NSUInteger)B * (NSUInteger)I * sizeof(int8_t);
            ctx->route_bytes = mode == 5 ? (NSUInteger)B * (NSUInteger)I * sizeof(uint16_t) : 0u;
            ctx->out_bytes = (NSUInteger)B * (NSUInteger)H * sizeof(uint16_t);
            ctx->io_gate = make_surface_typed(ctx->gate_bytes, 1u);
            ctx->io_up = mode == 5 ? make_surface_typed(ctx->gate_bytes, 1u) : NULL;
            ctx->io_down = make_surface_typed(ctx->down_bytes, 1u);
            ctx->io_x = make_surface_typed(ctx->x_bytes, 1u);
            ctx->io_mid = make_surface_typed(ctx->mid_bytes, 1u);
            ctx->io_hidden = make_surface_typed(ctx->hidden_bytes, 1u);
            ctx->io_route = mode == 5 ? make_surface_typed(ctx->route_bytes, 1u) : NULL;
            ctx->io_out = make_surface_typed(ctx->out_bytes, 1u);
            if (!ctx->io_gate || (mode == 5 && !ctx->io_up) ||
                !ctx->io_down || !ctx->io_x || !ctx->io_mid ||
                !ctx->io_hidden || (mode == 5 && !ctx->io_route) || !ctx->io_out) {
                if (dbg) fprintf(stderr, "ds4: ANE fp16 split IOSurface allocation failed\n");
                ds4_ane_mlp_int8w_destroy(ctx);
                return NULL;
            }
            NSString *gate_mil = mode == 5
                ? gen_mil_i8w_i8x_gateup(H, I, B, w_scale, x_scale)
                : (mode == 2
                   ? gen_mil_i8w_fp16x_matmul(H, I, B, w_scale)
                   : (mode == 3
                      ? gen_mil_i8w_i8x_matmul(H, I, B, w_scale, x_scale)
                      : gen_mil_fp16_matmul(H, I, B)));
            NSString *down_mil = mode == 2
                ? gen_mil_i8w_fp16x_matmul(I, H, B, w_scale)
                : ((mode == 3 || mode == 5)
                   ? gen_mil_i8w_i8x_matmul(I, H, B, w_scale, mid_scale)
                   : gen_mil_fp16_matmul(I, H, B));
            const char *mode_name = mode == 5 ? "i8w-i8x-gateup-fused" :
                (mode == 3 ? "i8w-i8x" : (mode == 2 ? "i8w-fp16x" : "fp16"));
            if (!compile_and_load_mil(gate_mil,
                                      mode == 5 ? "i8w-i8x gateup fused" :
                                      (mode == 3 ? "i8w-i8x split gate/up" : (mode == 2 ? "i8w-fp16x split gate/up" : "fp16 split gate/up")),
                                      &ctx->model_r, &ctx->tmpDir_r) ||
                !compile_and_load_mil(down_mil,
                                      (mode == 3 || mode == 5) ? "i8w-i8x split down" : (mode == 2 ? "i8w-fp16x split down" : "fp16 split down"),
                                      &ctx->model_down_r, &ctx->tmpDir_down_r)) {
                ds4_ane_mlp_int8w_destroy(ctx);
                return NULL;
            }
            id w_g = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_gate);
            id w_u = ctx->io_up ? ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_up) : nil;
            id w_x = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_x);
            id w_m = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_mid);
            id w_r = ctx->io_route ? ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_route) : nil;
            id req = nil;
            if (mode == 5) {
                req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                    g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[w_g, w_u, w_x], @[@0, @1, @2], @[w_m, w_r], @[@0, @1], nil, nil, @0);
            } else {
                req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                    g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[w_g, w_x], @[@0, @1], @[w_m], @[@0], nil, nil, @0);
            }
            id w_d = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_down);
            id w_h = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_hidden);
            id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_out);
            id req_down = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[w_d, w_h], @[@0, @1], @[w_o], @[@0], nil, nil, @0);
            if (!req || !req_down) {
                if (dbg) fprintf(stderr, "ds4: ANE %s split request create failed\n",
                                 mode_name);
                ds4_ane_mlp_int8w_destroy(ctx);
                return NULL;
            }
            ctx->request_r = (void *)CFBridgingRetain(req);
            ctx->request_down_r = (void *)CFBridgingRetain(req_down);
            if (dbg) fprintf(stderr, "ds4: ANE %s split create ok H=%d I=%d B=%d w_scale=%g x_scale=%g mid_scale=%g\n",
                             mode_name, H, I, B, w_scale, x_scale, mid_scale);
            return ctx;
        }

        NSError *e = nil;
        NSString *mil = mode == 1 ? gen_mil_fp16w(H, I, B) :
            (mode == 4 ? gen_mil_i8w_i8x_fused(H, I, B, w_scale, x_scale, mid_scale) :
             (mode == 6 ? gen_mil_i8w_i8x_tiled_fused(H, I, B, w_scale, x_scale, mid_scale) :
                          gen_mil_int8w(H, I, B, w_scale, x_scale)));
        NSData *milData = [[mil dataUsingEncoding:NSUTF8StringEncoding] copy];
        if (dbg) fprintf(stderr, "ds4: ANE %s create H=%d I=%d B=%d w_scale=%g x_scale=%g mid_scale=%g\n",
                         mode == 1 ? "fp16w" : (mode == 4 ? "i8w-i8x-fused" : (mode == 6 ? "i8w-i8x-tiled-fused" : "int8w")),
                         H, I, B, w_scale, x_scale, mid_scale);
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_DescCls, @selector(modelWithMILText:weights:optionsPlist:), milData, @{}, nil);
        if (!desc) {
            if (dbg) fprintf(stderr, "ds4: ANE descriptor create failed\n");
            return NULL;
        }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_ModelCls, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            if (dbg) fprintf(stderr, "ds4: ANE inMemoryModel create failed\n");
            return NULL;
        }
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            if (dbg) fprintf(stderr, "ds4: ANE compile failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        if (dbg) fprintf(stderr, "ds4: ANE compile ok\n");
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            if (dbg) fprintf(stderr, "ds4: ANE load failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        if (dbg) fprintf(stderr, "ds4: ANE int8w load ok\n");

        ds4_ane_mlp_int8w_ctx *ctx = (ds4_ane_mlp_int8w_ctx *)calloc(1, sizeof(*ctx));
        if (!ctx) return NULL;
        ctx->H = H; ctx->I = I; ctx->B = B; ctx->mode = mode; ctx->w_scale = w_scale; ctx->x_scale = x_scale; ctx->mid_scale = mid_scale;
        const NSUInteger elem = mode == 1 ? 2u : 1u;
        ctx->gate_bytes = (NSUInteger)H * (NSUInteger)I * elem;
        ctx->down_bytes = (NSUInteger)I * (NSUInteger)H * elem;
        ctx->x_bytes = (NSUInteger)B * (NSUInteger)H * elem;
        ctx->route_bytes = (NSUInteger)B * 2u;
        ctx->out_bytes = (NSUInteger)B * (NSUInteger)H * 2u;
        ctx->io_gate = make_surface_typed(ctx->gate_bytes, elem);
        ctx->io_up = make_surface_typed(ctx->gate_bytes, elem);
        ctx->io_down = make_surface_typed(ctx->down_bytes, elem);
        ctx->io_x = make_surface_typed(ctx->x_bytes, elem);
        ctx->io_out = make_surface_typed(ctx->out_bytes, 2u);
        if (!ctx->io_gate || !ctx->io_up || !ctx->io_down || !ctx->io_x || !ctx->io_out) {
            if (dbg) fprintf(stderr, "ds4: ANE IOSurface allocation failed\n");
            ds4_ane_mlp_int8w_destroy(ctx);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        id w_g = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_gate);
        id w_u = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_up);
        id w_d = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_down);
        id w_x = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_x);
        id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_out);
        NSArray *req_inputs = (mode == 4 || mode == 6) ? @[w_d, w_g, w_u, w_x] : @[w_g, w_u, w_d, w_x];
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            req_inputs, @[@0, @1, @2, @3], @[w_o], @[@0], nil, nil, @0);
        if (!req) {
            if (dbg) fprintf(stderr, "ds4: ANE request create failed\n");
            ds4_ane_mlp_int8w_destroy(ctx);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        ctx->model_r = (void *)CFBridgingRetain(mdl);
        ctx->request_r = (void *)CFBridgingRetain(req);
        ctx->tmpDir_r = (void *)CFBridgingRetain([td copy]);
        return ctx;
    }
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_int8w_create(int H, int I, int B, float w_scale, float x_scale) {
    return ds4_ane_mlp_create_common(H, I, B, w_scale, x_scale, x_scale, 0);
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_fp16w_create(int H, int I, int B) {
    return ds4_ane_mlp_create_common(H, I, B, 1.0f, 1.0f, 1.0f, 1);
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_fp16x_create(int H, int I, int B, float w_scale) {
    return ds4_ane_mlp_create_common(H, I, B, w_scale, 1.0f, 1.0f, 2);
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale) {
    return ds4_ane_mlp_create_common(H, I, B, w_scale, x_scale, mid_scale, 3);
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_fused_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale) {
    return ds4_ane_mlp_create_common(H, I, B, w_scale, x_scale, mid_scale, 4);
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_gateup_fused_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale) {
    return ds4_ane_mlp_create_common(H, I, B, w_scale, x_scale, mid_scale, 5);
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_tiled_fused_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale) {
    return ds4_ane_mlp_create_common(H, I, B, w_scale, x_scale, mid_scale, 6);
}

bool ds4_ane_mlp_int8w_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    const uint16_t *route_f16,
    uint16_t *output_f16)
{
    if (!ctx || !Wgate_i8 || !Wup_i8 || !Wdown_i8 || !input_i8 || !route_f16 || !output_f16) return false;
    if (ctx->mode != 0) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_gate, Wgate_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_up, Wup_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_down, Wdown_i8, ctx->down_bytes) ||
            !write_surface(ctx->io_x, input_i8, ctx->x_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE int8w IOSurface write failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE int8w evaluate failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE int8w IOSurface read failed\n");
            return false;
        }
        return true;
    }
}

bool ds4_ane_mlp_i8w_i8x_fused_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16)
{
    if (!ctx || !Wgate_i8 || !Wup_i8 || !Wdown_i8 || !input_i8 || !output_f16) return false;
    if (ctx->mode != 4) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_gate, Wgate_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_up, Wup_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_down, Wdown_i8, ctx->down_bytes) ||
            !write_surface(ctx->io_x, input_i8, ctx->x_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x fused IOSurface write failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x fused evaluate failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x fused IOSurface read failed\n");
            return false;
        }
        return true;
    }
}

bool ds4_ane_mlp_i8w_i8x_tiled_fused_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16)
{
    if (!ctx || !Wgate_i8 || !Wup_i8 || !Wdown_i8 || !input_i8 || !output_f16) return false;
    if (ctx->mode != 6) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_gate, Wgate_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_up, Wup_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_down, Wdown_i8, ctx->down_bytes) ||
            !write_surface(ctx->io_x, input_i8, ctx->x_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused IOSurface write failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused evaluate failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused IOSurface read failed\n");
            return false;
        }
        return true;
    }
}

bool ds4_ane_mlp_i8w_i8x_tiled_fused_eval_to_surface(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8)
{
    if (!ctx || !Wgate_i8 || !Wup_i8 || !Wdown_i8 || !input_i8) return false;
    if (ctx->mode != 6) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_gate, Wgate_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_up, Wup_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_down, Wdown_i8, ctx->down_bytes) ||
            !write_surface(ctx->io_x, input_i8, ctx->x_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused IOSurface write failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused evaluate failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        return true;
    }
}

const uint16_t *ds4_ane_mlp_int8w_lock_output_f16(ds4_ane_mlp_int8w_ctx *ctx, uint64_t *elems)
{
    if (elems) *elems = 0;
    if (!ctx || !ctx->io_out || ctx->out_bytes == 0) return NULL;
    if (IOSurfaceGetAllocSize(ctx->io_out) < ctx->out_bytes) return NULL;
    if (IOSurfaceLock(ctx->io_out, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return NULL;
    if (elems) *elems = (uint64_t)(ctx->out_bytes / sizeof(uint16_t));
    return (const uint16_t *)IOSurfaceGetBaseAddress(ctx->io_out);
}

void ds4_ane_mlp_int8w_unlock_output(ds4_ane_mlp_int8w_ctx *ctx)
{
    if (!ctx || !ctx->io_out) return;
    IOSurfaceUnlock(ctx->io_out, kIOSurfaceLockReadOnly, NULL);
}

bool ds4_ane_mlp_i8w_i8x_gateup_fused_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16)
{
    if (!ctx || !Wgate_i8 || !Wup_i8 || !Wdown_i8 || !input_i8 || !output_f16) return false;
    if (ctx->mode != 5 || !ctx->model_down_r || !(ctx->mid_scale > 0.0f)) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        uint16_t *gate = (uint16_t *)malloc(ctx->mid_bytes);
        uint16_t *up = (uint16_t *)malloc(ctx->route_bytes);
        int8_t *hidden = (int8_t *)malloc(ctx->hidden_bytes);
        if (!gate || !up || !hidden) {
            free(gate);
            free(up);
            free(hidden);
            return false;
        }
        NSError *e = nil;
        if (!write_surface(ctx->io_x, input_i8, ctx->x_bytes) ||
            !write_surface(ctx->io_gate, Wgate_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_up, Wup_i8, ctx->gate_bytes)) {
            free(gate);
            free(up);
            free(hidden);
            return false;
        }
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok ||
            !read_surface(ctx->io_mid, gate, ctx->mid_bytes) ||
            !read_surface(ctx->io_route, up, ctx->route_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x gateup fused eval failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            free(gate);
            free(up);
            free(hidden);
            return false;
        }
        const int B = ctx->B;
        const int I = ctx->I;
        const float mid_qscale = 1.0f / ctx->mid_scale;
        const bool stats = ane_int8w_stats_enabled();
        uint64_t hidden_values = 0;
        uint64_t hidden_saturated = 0;
        float hidden_abs_max = 0.0f;
        for (int b = 0; b < B; b++) {
            for (int j = 0; j < I; j++) {
                const size_t idx = (size_t)b * (size_t)I + (size_t)j;
                float g = ane_f16_bits_to_f32(gate[idx]);
                float u = ane_f16_bits_to_f32(up[idx]);
                if (g < -10.0f) g = -10.0f;
                if (g > 10.0f) g = 10.0f;
                if (u < -10.0f) u = -10.0f;
                if (u > 10.0f) u = 10.0f;
                const float h = (g / (1.0f + expf(-g))) * u;
                if (stats) {
                    const float ah = fabsf(h);
                    if (ah > hidden_abs_max) hidden_abs_max = ah;
                    hidden_values++;
                }
                float v = h * mid_qscale;
                v = nearbyintf(v);
                if (v < -128.0f) {
                    v = -128.0f;
                    if (stats) hidden_saturated++;
                }
                if (v > 127.0f) {
                    v = 127.0f;
                    if (stats) hidden_saturated++;
                }
                hidden[idx] = (int8_t)v;
            }
        }
        if (stats) {
            g_i8i8_hidden_values += hidden_values;
            g_i8i8_hidden_saturated += hidden_saturated;
            if (hidden_abs_max > g_i8i8_hidden_abs_max) g_i8i8_hidden_abs_max = hidden_abs_max;
        }
        if (!write_surface(ctx->io_down, Wdown_i8, ctx->down_bytes) ||
            !write_surface(ctx->io_hidden, hidden, ctx->hidden_bytes)) {
            free(gate);
            free(up);
            free(hidden);
            return false;
        }
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_down_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_down_r, &e);
        const bool read_ok = ok && read_surface(ctx->io_out, output_f16, ctx->out_bytes);
        if (!read_ok && dbg) {
            fprintf(stderr, "ds4: ANE i8w-i8x gateup fused down eval failed: %s\n",
                    e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
        }
        free(gate);
        free(up);
        free(hidden);
        return read_ok;
    }
}

bool ds4_ane_mlp_fp16w_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const uint16_t *Wgate_f16,
    const uint16_t *Wup_f16,
    const uint16_t *Wdown_f16,
    const uint16_t *input_f16,
    uint16_t *output_f16)
{
    if (!ctx || !Wgate_f16 || !Wup_f16 || !Wdown_f16 || !input_f16 || !output_f16) return false;
    if (ctx->mode != 1) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (ctx->model_down_r) {
            uint16_t *gate = (uint16_t *)malloc(ctx->mid_bytes);
            uint16_t *up = (uint16_t *)malloc(ctx->mid_bytes);
            uint16_t *hidden = (uint16_t *)malloc(ctx->hidden_bytes);
            if (!gate || !up || !hidden) {
                free(gate);
                free(up);
                free(hidden);
                return false;
            }
            NSError *e = nil;
            if (!write_surface(ctx->io_x, input_f16, ctx->x_bytes) ||
                !write_surface(ctx->io_gate, Wgate_f16, ctx->gate_bytes)) {
                free(gate); free(up); free(hidden);
                return false;
            }
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                (__bridge id)ctx->model_r,
                @selector(evaluateWithQoS:options:request:error:),
                21, @{}, (__bridge id)ctx->request_r, &e);
            if (!ok || !read_surface(ctx->io_mid, gate, ctx->mid_bytes)) {
                if (dbg) fprintf(stderr, "ds4: ANE fp16 split gate eval failed: %s\n",
                                 e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
                free(gate); free(up); free(hidden);
                return false;
            }
            if (!write_surface(ctx->io_gate, Wup_f16, ctx->gate_bytes)) {
                free(gate); free(up); free(hidden);
                return false;
            }
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                (__bridge id)ctx->model_r,
                @selector(evaluateWithQoS:options:request:error:),
                21, @{}, (__bridge id)ctx->request_r, &e);
            if (!ok || !read_surface(ctx->io_mid, up, ctx->mid_bytes)) {
                if (dbg) fprintf(stderr, "ds4: ANE fp16 split up eval failed: %s\n",
                                 e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
                free(gate); free(up); free(hidden);
                return false;
            }
            const int B = ctx->B;
            const int I = ctx->I;
            for (int i = 0; i < B * I; i++) {
                float g = ane_f16_bits_to_f32(gate[i]);
                float u = ane_f16_bits_to_f32(up[i]);
                if (g < -10.0f) g = -10.0f;
                if (g > 10.0f) g = 10.0f;
                if (u < -10.0f) u = -10.0f;
                if (u > 10.0f) u = 10.0f;
                const float h = (g / (1.0f + expf(-g))) * u;
                hidden[i] = ane_f32_to_f16_bits(h);
            }
            if (!write_surface(ctx->io_down, Wdown_f16, ctx->down_bytes) ||
                !write_surface(ctx->io_hidden, hidden, ctx->hidden_bytes)) {
                free(gate); free(up); free(hidden);
                return false;
            }
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                (__bridge id)ctx->model_down_r,
                @selector(evaluateWithQoS:options:request:error:),
                21, @{}, (__bridge id)ctx->request_down_r, &e);
            const bool read_ok = ok && read_surface(ctx->io_out, output_f16, ctx->out_bytes);
            if (!read_ok && dbg) {
                fprintf(stderr, "ds4: ANE fp16 split down eval failed: %s\n",
                        e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            }
            free(gate);
            free(up);
            free(hidden);
            return read_ok;
        }
        if (!write_surface(ctx->io_gate, Wgate_f16, ctx->gate_bytes) ||
            !write_surface(ctx->io_up, Wup_f16, ctx->gate_bytes) ||
            !write_surface(ctx->io_down, Wdown_f16, ctx->down_bytes) ||
            !write_surface(ctx->io_x, input_f16, ctx->x_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w IOSurface write failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w evaluate failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w IOSurface read failed\n");
            return false;
        }
        return true;
    }
}

bool ds4_ane_mlp_i8w_fp16x_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const uint16_t *input_f16,
    uint16_t *output_f16)
{
    if (!ctx || !Wgate_i8 || !Wup_i8 || !Wdown_i8 || !input_f16 || !output_f16) return false;
    if (ctx->mode != 2 || !ctx->model_down_r) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        uint16_t *gate = (uint16_t *)malloc(ctx->mid_bytes);
        uint16_t *up = (uint16_t *)malloc(ctx->mid_bytes);
        uint16_t *hidden = (uint16_t *)malloc(ctx->hidden_bytes);
        if (!gate || !up || !hidden) {
            free(gate);
            free(up);
            free(hidden);
            return false;
        }
        NSError *e = nil;
        if (!write_surface(ctx->io_x, input_f16, ctx->x_bytes) ||
            !write_surface(ctx->io_gate, Wgate_i8, ctx->gate_bytes)) {
            free(gate); free(up); free(hidden);
            return false;
        }
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok || !read_surface(ctx->io_mid, gate, ctx->mid_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-fp16x split gate eval failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            free(gate); free(up); free(hidden);
            return false;
        }
        if (!write_surface(ctx->io_gate, Wup_i8, ctx->gate_bytes)) {
            free(gate); free(up); free(hidden);
            return false;
        }
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok || !read_surface(ctx->io_mid, up, ctx->mid_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-fp16x split up eval failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            free(gate); free(up); free(hidden);
            return false;
        }
        const int elems = ctx->B * ctx->I;
        for (int i = 0; i < elems; i++) {
            float g = ane_f16_bits_to_f32(gate[i]);
            float u = ane_f16_bits_to_f32(up[i]);
            if (g < -10.0f) g = -10.0f;
            if (g > 10.0f) g = 10.0f;
            if (u < -10.0f) u = -10.0f;
            if (u > 10.0f) u = 10.0f;
            hidden[i] = ane_f32_to_f16_bits((g / (1.0f + expf(-g))) * u);
        }
        if (!write_surface(ctx->io_down, Wdown_i8, ctx->down_bytes) ||
            !write_surface(ctx->io_hidden, hidden, ctx->hidden_bytes)) {
            free(gate); free(up); free(hidden);
            return false;
        }
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_down_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_down_r, &e);
        const bool read_ok = ok && read_surface(ctx->io_out, output_f16, ctx->out_bytes);
        if (!read_ok && dbg) {
            fprintf(stderr, "ds4: ANE i8w-fp16x split down eval failed: %s\n",
                    e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
        }
        free(gate);
        free(up);
        free(hidden);
        return read_ok;
    }
}

bool ds4_ane_mlp_i8w_i8x_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    uint16_t *output_f16)
{
    if (!ctx || !Wgate_i8 || !Wup_i8 || !Wdown_i8 || !input_i8 || !output_f16) return false;
    if (ctx->mode != 3 || !ctx->model_down_r || !(ctx->mid_scale > 0.0f)) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        uint16_t *gate = (uint16_t *)malloc(ctx->mid_bytes);
        uint16_t *up = (uint16_t *)malloc(ctx->mid_bytes);
        int8_t *hidden = (int8_t *)malloc(ctx->hidden_bytes);
        if (!gate || !up || !hidden) {
            free(gate);
            free(up);
            free(hidden);
            return false;
        }
        NSError *e = nil;
        if (!write_surface(ctx->io_x, input_i8, ctx->x_bytes) ||
            !write_surface(ctx->io_gate, Wgate_i8, ctx->gate_bytes)) {
            free(gate); free(up); free(hidden);
            return false;
        }
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok || !read_surface(ctx->io_mid, gate, ctx->mid_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x split gate eval failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            free(gate); free(up); free(hidden);
            return false;
        }
        if (!write_surface(ctx->io_gate, Wup_i8, ctx->gate_bytes)) {
            free(gate); free(up); free(hidden);
            return false;
        }
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok || !read_surface(ctx->io_mid, up, ctx->mid_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x split up eval failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            free(gate); free(up); free(hidden);
            return false;
        }
        const int elems = ctx->B * ctx->I;
        const float mid_qscale = 1.0f / ctx->mid_scale;
        const bool stats = ane_int8w_stats_enabled();
        uint64_t hidden_values = 0;
        uint64_t hidden_saturated = 0;
        float hidden_abs_max = 0.0f;
        for (int i = 0; i < elems; i++) {
            float g = ane_f16_bits_to_f32(gate[i]);
            float u = ane_f16_bits_to_f32(up[i]);
            if (g < -10.0f) g = -10.0f;
            if (g > 10.0f) g = 10.0f;
            if (u < -10.0f) u = -10.0f;
            if (u > 10.0f) u = 10.0f;
            const float h = (g / (1.0f + expf(-g))) * u;
            if (stats) {
                const float ah = fabsf(h);
                if (ah > hidden_abs_max) hidden_abs_max = ah;
                hidden_values++;
            }
            float v = h * mid_qscale;
            v = nearbyintf(v);
            if (v < -128.0f) {
                v = -128.0f;
                if (stats) hidden_saturated++;
            }
            if (v > 127.0f) {
                v = 127.0f;
                if (stats) hidden_saturated++;
            }
            hidden[i] = (int8_t)v;
        }
        if (stats) {
            g_i8i8_hidden_values += hidden_values;
            g_i8i8_hidden_saturated += hidden_saturated;
            if (hidden_abs_max > g_i8i8_hidden_abs_max) g_i8i8_hidden_abs_max = hidden_abs_max;
        }
        if (!write_surface(ctx->io_down, Wdown_i8, ctx->down_bytes) ||
            !write_surface(ctx->io_hidden, hidden, ctx->hidden_bytes)) {
            free(gate); free(up); free(hidden);
            return false;
        }
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_down_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_down_r, &e);
        const bool read_ok = ok && read_surface(ctx->io_out, output_f16, ctx->out_bytes);
        if (!read_ok && dbg) {
            fprintf(stderr, "ds4: ANE i8w-i8x split down eval failed: %s\n",
                    e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
        }
        free(gate);
        free(up);
        free(hidden);
        return read_ok;
    }
}

void ds4_ane_mlp_int8w_destroy(ds4_ane_mlp_int8w_ctx *ctx) {
    if (!ctx) return;
    @autoreleasepool {
        NSError *e = nil;
        id model = ctx->model_r ? CFBridgingRelease(ctx->model_r) : nil;
        id request = ctx->request_r ? CFBridgingRelease(ctx->request_r) : nil;
        id model_down = ctx->model_down_r ? CFBridgingRelease(ctx->model_down_r) : nil;
        id request_down = ctx->request_down_r ? CFBridgingRelease(ctx->request_down_r) : nil;
        NSString *tmpDir = ctx->tmpDir_r ? CFBridgingRelease(ctx->tmpDir_r) : nil;
        NSString *tmpDirDown = ctx->tmpDir_down_r ? CFBridgingRelease(ctx->tmpDir_down_r) : nil;
        (void)request;
        (void)request_down;
        if (model) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &e);
        }
        if (model_down) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model_down, @selector(unloadWithQoS:error:), 21, &e);
        }
        if (ctx->io_gate) CFRelease(ctx->io_gate);
        if (ctx->io_up) CFRelease(ctx->io_up);
        if (ctx->io_down) CFRelease(ctx->io_down);
        if (ctx->io_x) CFRelease(ctx->io_x);
        if (ctx->io_mid) CFRelease(ctx->io_mid);
        if (ctx->io_hidden) CFRelease(ctx->io_hidden);
        if (ctx->io_route) CFRelease(ctx->io_route);
        if (ctx->io_out) CFRelease(ctx->io_out);
        if (tmpDir) [[NSFileManager defaultManager] removeItemAtPath:tmpDir error:nil];
        if (tmpDirDown) [[NSFileManager defaultManager] removeItemAtPath:tmpDirDown error:nil];
        free(ctx);
    }
}

int ds4_ane_mlp_int8w_H(const ds4_ane_mlp_int8w_ctx *ctx) { return ctx ? ctx->H : 0; }
int ds4_ane_mlp_int8w_I(const ds4_ane_mlp_int8w_ctx *ctx) { return ctx ? ctx->I : 0; }
int ds4_ane_mlp_int8w_B(const ds4_ane_mlp_int8w_ctx *ctx) { return ctx ? ctx->B : 0; }
float ds4_ane_mlp_int8w_scale(const ds4_ane_mlp_int8w_ctx *ctx) { return ctx ? ctx->w_scale : 0.0f; }
float ds4_ane_mlp_int8w_x_scale(const ds4_ane_mlp_int8w_ctx *ctx) { return ctx ? ctx->x_scale : 0.0f; }
float ds4_ane_mlp_int8w_mid_scale(const ds4_ane_mlp_int8w_ctx *ctx) { return ctx ? ctx->mid_scale : 0.0f; }
int ds4_ane_mlp_int8w_mode(const ds4_ane_mlp_int8w_ctx *ctx) { return ctx ? ctx->mode : -1; }
