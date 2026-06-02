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
    /* Per-chunk-IOSurface variant (mode 11, GPU-conversion O-proj path):
     * one request per externally-supplied input IOSurface.  All chunk
     * requests share the ctx's io_out.  When n_chunk_requests > 0,
     * eval_at_chunk uses chunk_requests[k] instead of request_r. */
    int n_chunk_requests;
    void *chunk_requests[64];
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

static bool ane_tmp_cleanup_debug_enabled(void) {
    const char *env = getenv("DS4_ANE_TMP_CLEANUP_DEBUG");
    return env && env[0] && atoi(env) != 0;
}

static double ane_tmp_cleanup_age_sec(void) {
    const char *env = getenv("DS4_ANE_TMP_CLEANUP_AGE_SEC");
    long age = (env && env[0]) ? atol(env) : 2 * 60 * 60;
    if (age < 60) age = 60;
    if (age > 30L * 24L * 60L * 60L) age = 30L * 24L * 60L * 60L;
    return (double)age;
}

static bool ane_tmp_cleanup_candidate(NSString *name) {
    if (!name.length) return false;
    if ([name hasPrefix:@"ds4-ane-"]) return true;
    /* Old _ANEInMemoryModel temp directories are named by the private model
     * hexStringIdentifier.  This middle hash is stable across the DS4 ANE MIL
     * descriptors observed in the runtime and benchmark artifacts. */
    return [name containsString:@"_DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_"];
}

static void ane_cleanup_stale_tmp_dirs_once(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        @autoreleasepool {
            const char *enabled = getenv("DS4_ANE_TMP_CLEANUP");
            if (enabled && enabled[0] && atoi(enabled) == 0) return;

            NSFileManager *fm = [NSFileManager defaultManager];
            NSString *tmp = NSTemporaryDirectory();
            NSError *err = nil;
            NSArray *names = [fm contentsOfDirectoryAtPath:tmp error:&err];
            if (!names) {
                if (ane_tmp_cleanup_debug_enabled()) {
                    fprintf(stderr, "ds4: ANE tmp cleanup list failed: %s\n",
                            err ? [[err description] UTF8String] : "unknown");
                }
                return;
            }

            const double age_sec = ane_tmp_cleanup_age_sec();
            NSDate *cutoff = [NSDate dateWithTimeIntervalSinceNow:-age_sec];
            NSUInteger removed = 0;
            NSUInteger skipped_recent = 0;
            NSUInteger failed = 0;
            for (NSString *name in names) {
                if (!ane_tmp_cleanup_candidate(name)) continue;
                NSString *path = [tmp stringByAppendingPathComponent:name];
                NSDictionary *attrs = [fm attributesOfItemAtPath:path error:nil];
                if (![attrs[NSFileType] isEqualToString:NSFileTypeDirectory]) continue;
                NSDate *mtime = attrs[NSFileModificationDate];
                if (mtime && [mtime compare:cutoff] == NSOrderedDescending) {
                    skipped_recent++;
                    continue;
                }
                NSError *rm_err = nil;
                if ([fm removeItemAtPath:path error:&rm_err]) {
                    removed++;
                } else {
                    failed++;
                    if (ane_tmp_cleanup_debug_enabled()) {
                        fprintf(stderr, "ds4: ANE tmp cleanup failed %s: %s\n",
                                [path UTF8String],
                                rm_err ? [[rm_err description] UTF8String] : "unknown");
                    }
                }
            }
            if (ane_tmp_cleanup_debug_enabled() && (removed || failed || skipped_recent)) {
                fprintf(stderr,
                        "ds4: ANE tmp cleanup removed=%lu failed=%lu skipped_recent=%lu age_sec=%.0f root=%s\n",
                        (unsigned long)removed, (unsigned long)failed,
                        (unsigned long)skipped_recent, age_sec, [tmp UTF8String]);
            }
        }
    });
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

/* CoreML MIL weight-blob format (matches MILBlob/Blob/StorageFormat.hpp):
 *   - 64-byte storage_header (count, version=2, reserved)
 *   - per blob:
 *       - 64-byte blob_metadata (sentinel=0xDEADBEEF, dtype, sizeInBytes, offset to data)
 *       - raw data (64-aligned)
 * The MIL `BLOBFILE(... offset = uint64(N))` references the metadata, not the
 * data: the reader follows metadata.offset to locate the raw bytes. */
typedef struct __attribute__((packed)) {
    uint32_t count;
    uint32_t version;
    uint64_t reserved[7];
} ane_blob_storage_header;
typedef struct __attribute__((packed)) {
    uint32_t sentinel;
    uint32_t mil_dtype;          /* 1 = Float16 */
    uint64_t size_in_bytes;
    uint64_t offset;             /* file offset of raw data */
    uint64_t padding_size_in_bits;
    uint64_t reserved[4];
} ane_blob_metadata;
_Static_assert(sizeof(ane_blob_storage_header) == 64, "storage_header must be 64 bytes");
_Static_assert(sizeof(ane_blob_metadata) == 64, "blob_metadata must be 64 bytes");

/* Build an NSData containing three fp16 tensors back-to-back in the CoreML
 * MIL weight-blob format.  Returns the three metadata offsets (the values MIL's
 * BLOBFILE references) via out_off_*.  The same NSData must be (a) passed to
 * modelWithMILText:weights:options: keyed by the @model_path/... path string,
 * AND (b) written to disk at <hexIdTmpDir>/weights/weight.bin — the private
 * compiler reads both paths. */
static NSData *ane_build_fp16_blob_3(const uint16_t *t0, NSUInteger t0_bytes,
                                     const uint16_t *t1, NSUInteger t1_bytes,
                                     const uint16_t *t2, NSUInteger t2_bytes,
                                     uint64_t *out_off_0,
                                     uint64_t *out_off_1,
                                     uint64_t *out_off_2) {
    if (!t0 || !t1 || !t2) return nil;
    if ((t0_bytes & 63u) || (t1_bytes & 63u) || (t2_bytes & 63u)) return nil;
    const uint64_t off_meta_0 = 64;
    const uint64_t off_data_0 = off_meta_0 + 64;
    const uint64_t off_meta_1 = off_data_0 + t0_bytes;
    const uint64_t off_data_1 = off_meta_1 + 64;
    const uint64_t off_meta_2 = off_data_1 + t1_bytes;
    const uint64_t off_data_2 = off_meta_2 + 64;
    const uint64_t total = off_data_2 + t2_bytes;
    NSMutableData *buf = [NSMutableData dataWithLength:(NSUInteger)total];
    if (!buf) return nil;
    uint8_t *p = (uint8_t *)buf.mutableBytes;

    ane_blob_storage_header hdr = {0};
    hdr.count = 3;
    hdr.version = 2;
    memcpy(p, &hdr, sizeof(hdr));

    const uint8_t *srcs[3]    = { (const uint8_t *)t0, (const uint8_t *)t1, (const uint8_t *)t2 };
    const uint64_t sizes[3]   = { t0_bytes, t1_bytes, t2_bytes };
    const uint64_t metas[3]   = { off_meta_0, off_meta_1, off_meta_2 };
    const uint64_t datas[3]   = { off_data_0, off_data_1, off_data_2 };
    for (int i = 0; i < 3; i++) {
        ane_blob_metadata m = {0};
        m.sentinel = 0xDEADBEEFu;
        m.mil_dtype = 1;  /* Float16 */
        m.size_in_bytes = sizes[i];
        m.offset = datas[i];
        memcpy(p + metas[i], &m, sizeof(m));
        memcpy(p + datas[i], srcs[i], sizes[i]);
    }
    if (out_off_0) *out_off_0 = off_meta_0;
    if (out_off_1) *out_off_1 = off_meta_1;
    if (out_off_2) *out_off_2 = off_meta_2;
    return buf;
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

static NSString *gen_mil_i8w_i8x_tiled_fused_common(int H, int I, int B, float w_scale, float x_scale, float mid_scale, bool routed) {
    const int tile_i = 256;
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    if (routed) {
        [m appendFormat:
            @"    func main<ios18>(tensor<int8, [%d, %d]> Wgq, tensor<int8, [%d, %d]> Wuq, tensor<int8, [%d, %d]> Wdq, tensor<int8, [%d, %d]> Xq, tensor<fp16, [1, %d, %d]> R) {\n",
            H, I, H, I, I, H, B, H, B, I];
    } else {
        [m appendFormat:
            @"    func main<ios18>(tensor<int8, [%d, %d]> Wgq, tensor<int8, [%d, %d]> Wuq, tensor<int8, [%d, %d]> Wdq, tensor<int8, [%d, %d]> Xq) {\n",
            H, I, H, I, I, H, B, H];
    }
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
        @"            tensor<int32, [1]> ax1 = const()[name = string(\"ax1\"), val = tensor<int32, [1]>([1])];\n"
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
        NSString *hidden_src = [NSString stringWithFormat:@"hidden_fp_%@", tag];
        if (routed) {
            NSString *route_tile = [NSString stringWithFormat:@"R_%@", tag];
            [m appendFormat:@"            tensor<fp16, [1, %d, %d]> %@ = slice_by_index(begin = tensor<int32, [3]>([0, 0, %d]), end = tensor<int32, [3]>([1, %d, %d]), x = R)[name = string(\"%@\")];\n",
                            B, T, route_tile, start, B, end, route_tile];
            NSString *hidden_wr = [NSString stringWithFormat:@"hidden_wr_%@", tag];
            [m appendFormat:@"            tensor<fp16, [1, %d, %d]> %@ = mul(x = hidden_fp_%@, y = %@)[name = string(\"%@\")];\n",
                            B, T, hidden_wr, tag, route_tile, hidden_wr];
            hidden_src = hidden_wr;
        }
        [m appendFormat:@"            tensor<int8, [1, %d, %d]> hidden_q_%@ = quantize(input = %@, output_dtype = q_dtype, scale = midscale, zero_point = zp)[name = string(\"hidden_q_%@\")];\n",
                        B, T, tag, hidden_src, tag];
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

static NSString *gen_mil_i8w_i8x_tiled_fused(int H, int I, int B, float w_scale, float x_scale, float mid_scale) {
    return gen_mil_i8w_i8x_tiled_fused_common(H, I, B, w_scale, x_scale, mid_scale, false);
}

static NSString *gen_mil_i8w_i8x_tiled_fused_routed(int H, int I, int B, float w_scale, float x_scale, float mid_scale) {
    return gen_mil_i8w_i8x_tiled_fused_common(H, I, B, w_scale, x_scale, mid_scale, true);
}

/* Variant of gen_mil_i8w_i8x_tiled_fused that ends with a `quantize` so the
 * model's output tensor is int8 instead of fp16.  Halves the output IOSurface
 * size (B*H bytes vs B*H*2) and the corresponding read_surface memcpy each
 * call — useful for measuring whether output-bandwidth is bounding aggregate
 * ANE TFLOP/s.  The output scale is `mid_scale` (reused for simplicity). */
static NSString *gen_mil_i8w_i8x_tiled_fused_i8out(int H, int I, int B,
                                                    float w_scale, float x_scale,
                                                    float mid_scale) {
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
    /* Quantize the final accumulator to int8 so the model output IOSurface is
     * B*H bytes instead of B*H*2.  Reuses midscale as the output scale. */
    if (prev_y) {
        [m appendFormat:@"            tensor<int8, [1, %d, %d]> Y_out_q = quantize(input = %@, output_dtype = q_dtype, scale = midscale, zero_point = zp)[name = string(\"Y_out_q\")];\n",
                        B, H, prev_y];
        [m appendString:@"        } -> (Y_out_q);\n}\n"];
    } else {
        [m appendString:@"        } -> (X3);\n}\n"];
    }
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

/* Single-call fused fp16-weight MLP lowered as conv2d-1x1, the canonical
 * ANE-friendly pattern for linear layers (matches Apple's own LLM exports).
 * Activations land in NCHW [1, C, 1, W]; weights in [O, I, 1, 1].  The W and X
 * layout transforms are emitted in MIL so the host-side IOSurface contract is
 * identical to gen_mil_fp16w (Wg/Wu [H,I], Wd [I,H], X [B,H], Y [B,H]). */
static NSString *gen_mil_fp16w_fused_conv(int H, int I, int B) {
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
        @"            tensor<int32, [2]> perm2 = const()[name = string(\"perm2\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            string pad_type = const()[name = string(\"pad_type\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> strides = const()[name = string(\"strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> pad = const()[name = string(\"pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> dilations = const()[name = string(\"dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 groups = const()[name = string(\"groups\"), val = int32(1)];\n"];
    /* Reshape host-side [H,I] / [I,H] / [B,H] inputs into ANE-native NCHW.
     * Weight layout target: [O, I, 1, 1].  Activation target: [1, C, 1, B]. */
    [m appendFormat:@"            tensor<int32, [4]> wg_shape = const()[name = string(\"wg_shape\"), val = tensor<int32, [4]>([%d, %d, 1, 1])];\n", I, H];
    [m appendFormat:@"            tensor<int32, [4]> wu_shape = const()[name = string(\"wu_shape\"), val = tensor<int32, [4]>([%d, %d, 1, 1])];\n", I, H];
    [m appendFormat:@"            tensor<int32, [4]> wd_shape = const()[name = string(\"wd_shape\"), val = tensor<int32, [4]>([%d, %d, 1, 1])];\n", H, I];
    [m appendFormat:@"            tensor<int32, [4]> x_shape = const()[name = string(\"x_shape\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n", H, B];
    [m appendFormat:@"            tensor<int32, [2]> y_shape = const()[name = string(\"y_shape\"), val = tensor<int32, [2]>([%d, %d])];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wgt = transpose(perm = perm2, x = Wg)[name = string(\"Wgt\")];\n", I, H];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wut = transpose(perm = perm2, x = Wu)[name = string(\"Wut\")];\n", I, H];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Wdt = transpose(perm = perm2, x = Wd)[name = string(\"Wdt\")];\n", H, I];
    [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> Wgk = reshape(shape = wg_shape, x = Wgt)[name = string(\"Wgk\")];\n", I, H];
    [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> Wuk = reshape(shape = wu_shape, x = Wut)[name = string(\"Wuk\")];\n", I, H];
    [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> Wdk = reshape(shape = wd_shape, x = Wdt)[name = string(\"Wdk\")];\n", H, I];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm2, x = X)[name = string(\"Xt\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> X4 = reshape(shape = x_shape, x = Xt)[name = string(\"X4\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wgk, x = X4)[name = string(\"gate\")];\n", I, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wuk, x = X4)[name = string(\"up\")];\n", I, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> act = silu(x = gate)[name = string(\"act\")];\n", I, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> hidden = mul(x = act, y = up)[name = string(\"hidden\")];\n", I, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> Y4 = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wdk, x = hidden)[name = string(\"Y4\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Yt = reshape(shape = y_shape, x = Y4)[name = string(\"Yt\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Y = transpose(perm = perm2, x = Yt)[name = string(\"Y\")];\n", B, H];
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

/* Single-call fused fp16w MLP with weights as constexpr fp16 baked into the
 * MIL via a side-loaded blob file.  This matches Apple's own ANE LLM exports —
 * weights live as conv-native [O,I,1,1] constants, only X is uploaded per call.
 * Wg/Wu have shape [I, H, 1, 1], Wd has shape [H, I, 1, 1] (ggml-native [O,I]
 * orientation matches conv weight layout, so no host or in-MIL transpose). */
static NSString *gen_mil_fp16w_constexpr_conv(int H, int I, int B,
                                              NSString *blob_path,
                                              uint64_t off_g, uint64_t off_u, uint64_t off_d) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [%d, %d]> X) {\n", B, H];
    [m appendString:
        @"            tensor<int32, [2]> perm2 = const()[name = string(\"perm2\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            string pad_type = const()[name = string(\"pad_type\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> strides = const()[name = string(\"strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> pad = const()[name = string(\"pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> dilations = const()[name = string(\"dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 groups = const()[name = string(\"groups\"), val = int32(1)];\n"];
    [m appendFormat:@"            tensor<int32, [4]> x_shape = const()[name = string(\"x_shape\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n", H, B];
    [m appendFormat:@"            tensor<int32, [2]> y_shape = const()[name = string(\"y_shape\"), val = tensor<int32, [2]>([%d, %d])];\n", H, B];
    /* Weight constants backed by BLOBFILE.  No transpose: blob bytes are already
     * laid out as [O, I, 1, 1] fp16 row-major. */
    [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> Wg = const()[name = string(\"Wg\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    I, H, I, H, blob_path, (unsigned long long)off_g];
    [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> Wu = const()[name = string(\"Wu\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    I, H, I, H, blob_path, (unsigned long long)off_u];
    [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> Wd = const()[name = string(\"Wd\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    H, I, H, I, blob_path, (unsigned long long)off_d];
    /* Activation reshape into NCHW. */
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm2, x = X)[name = string(\"Xt\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> X4 = reshape(shape = x_shape, x = Xt)[name = string(\"X4\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wg, x = X4)[name = string(\"gate\")];\n", I, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wu, x = X4)[name = string(\"up\")];\n", I, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> act = silu(x = gate)[name = string(\"act\")];\n", I, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> hidden = mul(x = act, y = up)[name = string(\"hidden\")];\n", I, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> Y4 = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wd, x = hidden)[name = string(\"Y4\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Yt = reshape(shape = y_shape, x = Y4)[name = string(\"Yt\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Y = transpose(perm = perm2, x = Yt)[name = string(\"Y\")];\n", B, H];
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

/* Build a CoreML MIL weight blob with N (data, dtype) pairs.  dtypes per
 * BlobDataType: 1=Float16, 4=Int8.  Sizes must be 64-aligned. */
static NSData *ane_build_blob_n(const uint8_t * const *data_ptrs,
                                const NSUInteger      *bytes_per,
                                const uint32_t        *dtypes,
                                int                    count,
                                uint64_t              *out_offsets) {
    if (!data_ptrs || !bytes_per || !dtypes || count <= 0) return nil;
    uint64_t total = 64;  /* storage_header */
    for (int i = 0; i < count; i++) {
        if (bytes_per[i] & 63u) return nil;
        total += 64 + bytes_per[i];
    }
    NSMutableData *buf = [NSMutableData dataWithLength:(NSUInteger)total];
    if (!buf) return nil;
    uint8_t *p = (uint8_t *)buf.mutableBytes;
    ane_blob_storage_header hdr = {0};
    hdr.count = (uint32_t)count;
    hdr.version = 2;
    memcpy(p, &hdr, sizeof(hdr));
    uint64_t cur = 64;
    for (int i = 0; i < count; i++) {
        const uint64_t meta_off = cur;
        const uint64_t data_off = cur + 64;
        ane_blob_metadata m = {0};
        m.sentinel = 0xDEADBEEFu;
        m.mil_dtype = dtypes[i];
        m.size_in_bytes = bytes_per[i];
        m.offset = data_off;
        memcpy(p + meta_off, &m, sizeof(m));
        memcpy(p + data_off, data_ptrs[i], (size_t)bytes_per[i]);
        if (out_offsets) out_offsets[i] = meta_off;
        cur = data_off + bytes_per[i];
    }
    return buf;
}

/* Two-stage linear MLP with int8 weights + per-output-channel fp16 scales,
 * lowered as conv2d-1x1.  Uses constexpr_blockwise_shift_scale with the
 * degenerate "block = full in_dim" config (ANE accepts this as per-channel
 * scale).  Inputs/outputs same as the fp16 linear constexpr conv (X [B, H]
 * in, Y [B, H] out).  Mode 11.  Blob chunk order: Wa_q, Wa_off, Wa_scale,
 * Wb_q, Wb_off, Wb_scale. */
static NSString *gen_mil_int8w_linear_constexpr_conv(int H, int I, int B,
                                                     NSString *blob_path,
                                                     uint64_t off_wa_q, uint64_t off_wa_off, uint64_t off_wa_scale,
                                                     uint64_t off_wb_q, uint64_t off_wb_off, uint64_t off_wb_scale,
                                                     bool input_i8, float x_scale) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}})]\n{\n"];
    if (input_i8) {
        [m appendFormat:@"    func main<ios18>(tensor<int8, [%d, %d]> Xq) {\n", B, H];
    } else {
        [m appendFormat:@"    func main<ios18>(tensor<fp16, [%d, %d]> X) {\n", B, H];
    }
    [m appendString:
        @"            tensor<int32, [2]> perm2 = const()[name = string(\"perm2\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            string pad_type = const()[name = string(\"pad_type\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> strides = const()[name = string(\"strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> pad = const()[name = string(\"pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> dilations = const()[name = string(\"dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 groups = const()[name = string(\"groups\"), val = int32(1)];\n"
        @"            int8 zp = const()[name = string(\"zp\"), val = int8(0)];\n"];
    if (input_i8) {
        [m appendFormat:@"            fp16 xscale = const()[name = string(\"xscale\"), val = fp16(%a)];\n",
                        (double)x_scale];
        [m appendFormat:@"            tensor<fp16, [%d, %d]> X = dequantize(input = Xq, scale = xscale, zero_point = zp)[name = string(\"X\")];\n",
                        B, H];
    }
    [m appendFormat:@"            tensor<int32, [4]> x_shape = const()[name = string(\"x_shape\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n", H, B];
    [m appendFormat:@"            tensor<int32, [2]> y_shape = const()[name = string(\"y_shape\"), val = tensor<int32, [2]>([%d, %d])];\n", H, B];

    [m appendFormat:@"            tensor<int8, [%d, %d, 1, 1]> Wa_q = const()[name = string(\"Wa_q\"), val = tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    I, H, I, H, blob_path, (unsigned long long)off_wa_q];
    [m appendFormat:@"            tensor<int8, [%d, 1, 1, 1]> Wa_off = const()[name = string(\"Wa_off\"), val = tensor<int8, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    I, I, blob_path, (unsigned long long)off_wa_off];
    [m appendFormat:@"            tensor<fp16, [%d, 1, 1, 1]> Wa_scale = const()[name = string(\"Wa_scale\"), val = tensor<fp16, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    I, I, blob_path, (unsigned long long)off_wa_scale];
    [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> Wa = constexpr_blockwise_shift_scale(data = Wa_q, offset = Wa_off, scale = Wa_scale)[name = string(\"Wa\")];\n",
                    I, H];

    [m appendFormat:@"            tensor<int8, [%d, %d, 1, 1]> Wb_q = const()[name = string(\"Wb_q\"), val = tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    H, I, H, I, blob_path, (unsigned long long)off_wb_q];
    [m appendFormat:@"            tensor<int8, [%d, 1, 1, 1]> Wb_off = const()[name = string(\"Wb_off\"), val = tensor<int8, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    H, H, blob_path, (unsigned long long)off_wb_off];
    [m appendFormat:@"            tensor<fp16, [%d, 1, 1, 1]> Wb_scale = const()[name = string(\"Wb_scale\"), val = tensor<fp16, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    H, H, blob_path, (unsigned long long)off_wb_scale];
    [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> Wb = constexpr_blockwise_shift_scale(data = Wb_q, offset = Wb_off, scale = Wb_scale)[name = string(\"Wb\")];\n",
                    H, I];

    [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm2, x = X)[name = string(\"Xt\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> X4 = reshape(shape = x_shape, x = Xt)[name = string(\"X4\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> mid = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wa, x = X4)[name = string(\"mid\")];\n", I, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> Y4 = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wb, x = mid)[name = string(\"Y4\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Yt = reshape(shape = y_shape, x = Y4)[name = string(\"Yt\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Y = transpose(perm = perm2, x = Yt)[name = string(\"Y\")];\n", B, H];
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

/* Linear two-stage MLP lowered as conv2d-1x1 with constexpr fp16 weights.
 * No activation between the two convs — for LoRA-style projections like the
 * DSv4 attention output (W_a × W_b).  Input X [B, H], W_a → [I, H, 1, 1],
 * W_b → [H, I, 1, 1], Output Y [B, H].  Weights live in a side-loaded blob
 * referenced via BLOBFILE; main takes only X, returns Y. */
static NSString *gen_mil_fp16w_linear_constexpr_conv(int H, int I, int B,
                                                     NSString *blob_path,
                                                     uint64_t off_a, uint64_t off_b) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3520.4.1\"}, "
        @"{\"coremlc-version\", \"3520.5.1\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [%d, %d]> X) {\n", B, H];
    [m appendString:
        @"            tensor<int32, [2]> perm2 = const()[name = string(\"perm2\"), val = tensor<int32, [2]>([1, 0])];\n"
        @"            string pad_type = const()[name = string(\"pad_type\"), val = string(\"valid\")];\n"
        @"            tensor<int32, [2]> strides = const()[name = string(\"strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            tensor<int32, [4]> pad = const()[name = string(\"pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"            tensor<int32, [2]> dilations = const()[name = string(\"dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"            int32 groups = const()[name = string(\"groups\"), val = int32(1)];\n"];
    [m appendFormat:@"            tensor<int32, [4]> x_shape = const()[name = string(\"x_shape\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n", H, B];
    [m appendFormat:@"            tensor<int32, [2]> y_shape = const()[name = string(\"y_shape\"), val = tensor<int32, [2]>([%d, %d])];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> Wa = const()[name = string(\"Wa\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    I, H, I, H, blob_path, (unsigned long long)off_a];
    [m appendFormat:@"            tensor<fp16, [%d, %d, 1, 1]> Wb = const()[name = string(\"Wb\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"%@\"), offset = uint64(%llu)))];\n",
                    H, I, H, I, blob_path, (unsigned long long)off_b];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Xt = transpose(perm = perm2, x = X)[name = string(\"Xt\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> X4 = reshape(shape = x_shape, x = Xt)[name = string(\"X4\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> mid = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wa, x = X4)[name = string(\"mid\")];\n", I, B];
    [m appendFormat:@"            tensor<fp16, [1, %d, 1, %d]> Y4 = conv(dilations = dilations, groups = groups, pad = pad, pad_type = pad_type, strides = strides, weight = Wb, x = mid)[name = string(\"Y4\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Yt = reshape(shape = y_shape, x = Y4)[name = string(\"Yt\")];\n", H, B];
    [m appendFormat:@"            tensor<fp16, [%d, %d]> Y = transpose(perm = perm2, x = Yt)[name = string(\"Y\")];\n", B, H];
    [m appendString:@"        } -> (Y);\n}\n"];
    return m;
}

/* Build an NSData blob containing two fp16 tensors (storage_header + two
 * metadata+data records).  Same wire format as ane_build_fp16_blob_3 but
 * with count=2 — used by the LoRA-pair / linear constexpr path. */
static NSData *ane_build_fp16_blob_2(const uint16_t *t0, NSUInteger t0_bytes,
                                     const uint16_t *t1, NSUInteger t1_bytes,
                                     uint64_t *out_off_0,
                                     uint64_t *out_off_1) {
    if (!t0 || !t1) return nil;
    if ((t0_bytes & 63u) || (t1_bytes & 63u)) return nil;
    const uint64_t off_meta_0 = 64;
    const uint64_t off_data_0 = off_meta_0 + 64;
    const uint64_t off_meta_1 = off_data_0 + t0_bytes;
    const uint64_t off_data_1 = off_meta_1 + 64;
    const uint64_t total = off_data_1 + t1_bytes;
    NSMutableData *buf = [NSMutableData dataWithLength:(NSUInteger)total];
    if (!buf) return nil;
    uint8_t *p = (uint8_t *)buf.mutableBytes;
    ane_blob_storage_header hdr = {0};
    hdr.count = 2;
    hdr.version = 2;
    memcpy(p, &hdr, sizeof(hdr));
    const uint8_t *srcs[2]  = { (const uint8_t *)t0, (const uint8_t *)t1 };
    const uint64_t sizes[2] = { t0_bytes, t1_bytes };
    const uint64_t metas[2] = { off_meta_0, off_meta_1 };
    const uint64_t datas[2] = { off_data_0, off_data_1 };
    for (int i = 0; i < 2; i++) {
        ane_blob_metadata m = {0};
        m.sentinel = 0xDEADBEEFu;
        m.mil_dtype = 1;
        m.size_in_bytes = sizes[i];
        m.offset = datas[i];
        memcpy(p + metas[i], &m, sizeof(m));
        memcpy(p + datas[i], srcs[i], sizes[i]);
    }
    if (out_off_0) *out_off_0 = off_meta_0;
    if (out_off_1) *out_off_1 = off_meta_1;
    return buf;
}

static bool compile_and_load_mil(NSString *mil,
                                 const char *label,
                                 void **model_r,
                                 void **tmpDir_r) {
    const bool dbg = ane_int8w_debug_enabled();
    ane_cleanup_stale_tmp_dirs_once();
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
    if (mode == 13 && (!(w_scale > 0.0f) || !(x_scale > 0.0f) || !(mid_scale > 0.0f))) return NULL;
    ane_cleanup_stale_tmp_dirs_once();
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
              (mode == 7 ? gen_mil_i8w_i8x_tiled_fused_i8out(H, I, B, w_scale, x_scale, mid_scale) :
               (mode == 8 ? gen_mil_fp16w_fused_conv(H, I, B) :
                (mode == 13 ? gen_mil_i8w_i8x_tiled_fused_routed(H, I, B, w_scale, x_scale, mid_scale) :
                          gen_mil_int8w(H, I, B, w_scale, x_scale))))));
        NSData *milData = [[mil dataUsingEncoding:NSUTF8StringEncoding] copy];
        if (dbg) fprintf(stderr, "ds4: ANE %s create H=%d I=%d B=%d w_scale=%g x_scale=%g mid_scale=%g\n",
                         mode == 1 ? "fp16w" :
                            (mode == 4 ? "i8w-i8x-fused" :
                             (mode == 6 ? "i8w-i8x-tiled-fused" :
                              (mode == 7 ? "i8w-i8x-tiled-fused-i8out" :
                               (mode == 8 ? "fp16w-fused-conv" :
                                (mode == 13 ? "i8w-i8x-tiled-fused-routed" : "int8w"))))),
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
        /* fp16 weights for mode 1 (legacy split path uses other branch above) and
         * mode 8 (fused conv); int8 weights for all other single-model modes. */
        const NSUInteger elem = (mode == 1 || mode == 8) ? 2u : 1u;
        /* mode==7 outputs int8 ([1,B,H]) instead of fp16 ([1,B,H]) → half the
         * output bytes + half the read_surface memcpy per call. */
        const NSUInteger out_elem = (mode == 7) ? 1u : 2u;
        ctx->gate_bytes = (NSUInteger)H * (NSUInteger)I * elem;
        ctx->down_bytes = (NSUInteger)I * (NSUInteger)H * elem;
        ctx->x_bytes = (NSUInteger)B * (NSUInteger)H * elem;
        ctx->route_bytes = (mode == 13) ? (NSUInteger)B * (NSUInteger)I * 2u : (NSUInteger)B * 2u;
        ctx->out_bytes = (NSUInteger)B * (NSUInteger)H * out_elem;
        ctx->io_gate = make_surface_typed(ctx->gate_bytes, elem);
        ctx->io_up = make_surface_typed(ctx->gate_bytes, elem);
        ctx->io_down = make_surface_typed(ctx->down_bytes, elem);
        ctx->io_x = make_surface_typed(ctx->x_bytes, elem);
        ctx->io_route = (mode == 13) ? make_surface_typed(ctx->route_bytes, 2u) : NULL;
        ctx->io_out = make_surface_typed(ctx->out_bytes, out_elem);
        if (!ctx->io_gate || !ctx->io_up || !ctx->io_down || !ctx->io_x ||
            (mode == 13 && !ctx->io_route) || !ctx->io_out) {
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
        id w_r = ctx->io_route ? ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_route) : nil;
        id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_out);
        NSArray *req_inputs = nil;
        if (mode == 13) {
            const char *order_env = getenv("DS4_ANE_ROUTED_ORDER");
            const int order = (order_env && order_env[0]) ? atoi(order_env) : 0;
            switch (order) {
                case 1: req_inputs = @[w_d, w_g, w_u, w_x, w_r]; break;
                case 2: req_inputs = @[w_g, w_u, w_d, w_x, w_r]; break;
                case 3: req_inputs = @[w_g, w_u, w_d, w_r, w_x]; break;
                case 4: req_inputs = @[w_d, w_g, w_u, w_r, w_x]; break;
                case 5: req_inputs = @[w_r, w_g, w_u, w_d, w_x]; break;
                default: req_inputs = @[w_r, w_d, w_g, w_u, w_x]; break;
            }
        } else {
            req_inputs = (mode == 4 || mode == 6 || mode == 7) ? @[w_d, w_g, w_u, w_x] : @[w_g, w_u, w_d, w_x];
        }
        NSArray *req_indices = mode == 13 ? @[@0, @1, @2, @3, @4] : @[@0, @1, @2, @3];
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            req_inputs, req_indices, @[w_o], @[@0], nil, nil, @0);
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

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_tiled_fused_routed_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale) {
    return ds4_ane_mlp_create_common(H, I, B, w_scale, x_scale, mid_scale, 13);
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_i8x_tiled_fused_i8out_create(int H, int I, int B, float w_scale, float x_scale, float mid_scale) {
    return ds4_ane_mlp_create_common(H, I, B, w_scale, x_scale, mid_scale, 7);
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_fp16w_fused_conv_create(int H, int I, int B) {
    return ds4_ane_mlp_create_common(H, I, B, 1.0f, 1.0f, 1.0f, 8);
}

/* Per-layer compiled ctx with weights baked as fp16 constexpr via BLOBFILE.
 * Wg/Wu must be supplied in [I, H, 1, 1] row-major layout (mid_dim × in_dim);
 * Wd must be in [H, I, 1, 1] (in_dim × mid_dim).  This is ggml's native [O, I]
 * orientation for these tensors, so callers that just dequantized Q8_0 row-
 * major can pass the buffer directly without an extra transpose. */
/* Per-layer constexpr-weight linear two-stage MLP (no activation).  Wa is the
 * first matmul's weight in [I, H] orientation; Wb is the second in [H, I].
 * Mode 10 — sibling of mode 9 but for LoRA-style pairs (DSv4 O-proj). */
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_fp16w_linear_constexpr_create(int H, int I, int B,
                                                                  const uint16_t *Wa_OI,
                                                                  const uint16_t *Wb_OI) {
    if (H <= 0 || I <= 0 || B <= 0 || !Wa_OI || !Wb_OI) return NULL;
    ane_cleanup_stale_tmp_dirs_once();
    resolve_classes();
    const bool dbg = ane_int8w_debug_enabled();
    if (!g_DescCls || !g_ModelCls || !g_ReqCls || !g_IOCls) {
        if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear constexpr missing classes\n");
        return NULL;
    }
    @autoreleasepool {
        NSFileManager *fm = [NSFileManager defaultManager];
        const NSUInteger a_bytes = (NSUInteger)I * (NSUInteger)H * sizeof(uint16_t);
        const NSUInteger b_bytes = (NSUInteger)H * (NSUInteger)I * sizeof(uint16_t);
        NSString *blob_path_in_mil = @"@model_path/weights/weight.bin";
        const uint64_t off_a = 64;
        const uint64_t off_b = 64 + 64 + (uint64_t)a_bytes;
        uint64_t off_a_chk = 0, off_b_chk = 0;
        NSData *blob = ane_build_fp16_blob_2(Wa_OI, a_bytes, Wb_OI, b_bytes,
                                              &off_a_chk, &off_b_chk);
        if (!blob || off_a_chk != off_a || off_b_chk != off_b) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear constexpr blob build failed\n");
            return NULL;
        }
        NSString *mil = gen_mil_fp16w_linear_constexpr_conv(H, I, B,
                                                             blob_path_in_mil, off_a, off_b);
        NSData *milData = [[mil dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSError *e = nil;
        NSDictionary *weights = @{
            blob_path_in_mil: @{ @"offset": @(0), @"data": blob }
        };
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_DescCls, @selector(modelWithMILText:weights:optionsPlist:),
            milData, weights, nil);
        if (!desc) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear constexpr descriptor failed\n");
            return NULL;
        }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ModelCls, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear constexpr inMemoryModel failed\n");
            return NULL;
        }
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSString *weights_dir = [td stringByAppendingPathComponent:@"weights"];
        if (![fm createDirectoryAtPath:weights_dir withIntermediateDirectories:YES
                            attributes:nil error:nil]) {
            return NULL;
        }
        NSString *blob_path = [weights_dir stringByAppendingPathComponent:@"weight.bin"];
        if (![blob writeToFile:blob_path atomically:YES]) {
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear constexpr compile failed: %s\n",
                             e ? [[e description] UTF8String] : "unknown");
            if (!dbg) [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear constexpr load failed: %s\n",
                             e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        ds4_ane_mlp_int8w_ctx *ctx = (ds4_ane_mlp_int8w_ctx *)calloc(1, sizeof(*ctx));
        if (!ctx) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        ctx->H = H; ctx->I = I; ctx->B = B; ctx->mode = 10;
        ctx->w_scale = 1.0f; ctx->x_scale = 1.0f; ctx->mid_scale = 1.0f;
        ctx->x_bytes  = (NSUInteger)B * (NSUInteger)H * sizeof(uint16_t);
        ctx->out_bytes = (NSUInteger)B * (NSUInteger)H * sizeof(uint16_t);
        ctx->io_x   = make_surface_typed(ctx->x_bytes, 2u);
        ctx->io_out = make_surface_typed(ctx->out_bytes, 2u);
        if (!ctx->io_x || !ctx->io_out) {
            ds4_ane_mlp_int8w_destroy(ctx);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        id w_x = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_x);
        id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_out);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[w_x], @[@0], @[w_o], @[@0], nil, nil, @0);
        if (!req) {
            ds4_ane_mlp_int8w_destroy(ctx);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        ctx->model_r   = (void *)CFBridgingRetain(mdl);
        ctx->request_r = (void *)CFBridgingRetain(req);
        ctx->tmpDir_r  = (void *)CFBridgingRetain([td copy]);
        return ctx;
    }
}

/* Per-layer constexpr-weight linear two-stage MLP with int8 weights + per-
 * channel fp16 scales (mode 11).  Wa_q/Wb_q already in ggml-native [O, I]
 * layout (rows = out channels).  Wa/Wb_scale: per-output-channel fp16 scalars.
 * Wa/Wb_off: per-output-channel int8 zero points (all zero for symmetric). */
static ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_int8w_linear_constexpr_create_common(int H, int I, int B,
                                                                                bool input_i8,
                                                                                float x_scale,
                                                                                const int8_t   *Wa_q_OI,
                                                                                const int8_t   *Wa_off_O,
                                                                                const uint16_t *Wa_scale_f16_O,
                                                                                const int8_t   *Wb_q_OI,
                                                                                const int8_t   *Wb_off_O,
                                                                                const uint16_t *Wb_scale_f16_O) {
    if (H <= 0 || I <= 0 || B <= 0) return NULL;
    if (input_i8 && (!(x_scale > 0.0f) || !isfinite(x_scale))) return NULL;
    if (!Wa_q_OI || !Wa_off_O || !Wa_scale_f16_O ||
        !Wb_q_OI || !Wb_off_O || !Wb_scale_f16_O) return NULL;
    ane_cleanup_stale_tmp_dirs_once();
    resolve_classes();
    const bool dbg = ane_int8w_debug_enabled();
    if (!g_DescCls || !g_ModelCls || !g_ReqCls || !g_IOCls) return NULL;
    @autoreleasepool {
        NSFileManager *fm = [NSFileManager defaultManager];
        const NSUInteger a_q_bytes     = (NSUInteger)I * (NSUInteger)H;            /* int8 */
        const NSUInteger a_off_bytes   = (NSUInteger)I;                            /* int8 [I,1,1,1] */
        const NSUInteger a_scale_bytes = (NSUInteger)I * sizeof(uint16_t);         /* fp16 [I,1,1,1] */
        const NSUInteger b_q_bytes     = (NSUInteger)H * (NSUInteger)I;
        const NSUInteger b_off_bytes   = (NSUInteger)H;
        const NSUInteger b_scale_bytes = (NSUInteger)H * sizeof(uint16_t);
        /* All these must be 64-aligned for the blob writer.  Small ones (I=8192
         * → 8192 bytes int8 = 128*64; 16384 fp16 = 256*64) are; the big int8
         * weight blocks (32 MB / 32 MB) are 64-aligned by construction. */
        const uint8_t *data_ptrs[6] = {
            (const uint8_t *)Wa_q_OI, (const uint8_t *)Wa_off_O,
            (const uint8_t *)Wa_scale_f16_O,
            (const uint8_t *)Wb_q_OI, (const uint8_t *)Wb_off_O,
            (const uint8_t *)Wb_scale_f16_O,
        };
        const NSUInteger sizes[6] = {
            a_q_bytes, a_off_bytes, a_scale_bytes,
            b_q_bytes, b_off_bytes, b_scale_bytes,
        };
        const uint32_t dtypes[6] = { 4, 4, 1, 4, 4, 1 };
        uint64_t offs[6] = {0};
        NSData *blob = ane_build_blob_n(data_ptrs, sizes, dtypes, 6, offs);
        if (!blob) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-linear constexpr blob build failed (likely an int8-OFF size that's not 64-aligned)\n");
            return NULL;
        }
        NSString *blob_path_in_mil = @"@model_path/weights/weight.bin";
        NSString *mil = gen_mil_int8w_linear_constexpr_conv(
            H, I, B, blob_path_in_mil,
            offs[0], offs[1], offs[2],
            offs[3], offs[4], offs[5],
            input_i8, x_scale);
        NSData *milData = [[mil dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSError *e = nil;
        NSDictionary *weights = @{
            blob_path_in_mil: @{ @"offset": @(0), @"data": blob }
        };
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_DescCls, @selector(modelWithMILText:weights:optionsPlist:),
            milData, weights, nil);
        if (!desc) return NULL;
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ModelCls, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) return NULL;
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSString *weights_dir = [td stringByAppendingPathComponent:@"weights"];
        if (![fm createDirectoryAtPath:weights_dir withIntermediateDirectories:YES attributes:nil error:nil]) return NULL;
        NSString *blob_path = [weights_dir stringByAppendingPathComponent:@"weight.bin"];
        if (![blob writeToFile:blob_path atomically:YES]) {
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-linear constexpr compile failed: %s\n  (debug: keeping tmpdir %s)\n",
                             e ? [[e description] UTF8String] : "unknown", [td UTF8String]);
            if (!dbg) [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-linear constexpr load failed: %s\n",
                             e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        ds4_ane_mlp_int8w_ctx *ctx = (ds4_ane_mlp_int8w_ctx *)calloc(1, sizeof(*ctx));
        if (!ctx) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        ctx->H = H; ctx->I = I; ctx->B = B; ctx->mode = input_i8 ? 12 : 11;
        ctx->w_scale = 1.0f; ctx->x_scale = input_i8 ? x_scale : 1.0f; ctx->mid_scale = 1.0f;
        ctx->x_bytes  = (NSUInteger)B * (NSUInteger)H * (input_i8 ? sizeof(int8_t) : sizeof(uint16_t));
        ctx->out_bytes = (NSUInteger)B * (NSUInteger)H * sizeof(uint16_t);
        ctx->io_x   = make_surface_typed(ctx->x_bytes, input_i8 ? 1u : 2u);
        ctx->io_out = make_surface_typed(ctx->out_bytes, 2u);
        if (!ctx->io_x || !ctx->io_out) {
            ds4_ane_mlp_int8w_destroy(ctx);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        id w_x = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_x);
        id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_out);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[w_x], @[@0], @[w_o], @[@0], nil, nil, @0);
        if (!req) {
            ds4_ane_mlp_int8w_destroy(ctx);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        ctx->model_r   = (void *)CFBridgingRetain(mdl);
        ctx->request_r = (void *)CFBridgingRetain(req);
        ctx->tmpDir_r  = (void *)CFBridgingRetain([td copy]);
        return ctx;
    }
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_int8w_linear_constexpr_create(int H, int I, int B,
                                                                  const int8_t   *Wa_q_OI,
                                                                  const int8_t   *Wa_off_O,
                                                                  const uint16_t *Wa_scale_f16_O,
                                                                  const int8_t   *Wb_q_OI,
                                                                  const int8_t   *Wb_off_O,
                                                                  const uint16_t *Wb_scale_f16_O) {
    return ds4_ane_mlp_int8w_linear_constexpr_create_common(
        H, I, B, false, 1.0f,
        Wa_q_OI, Wa_off_O, Wa_scale_f16_O,
        Wb_q_OI, Wb_off_O, Wb_scale_f16_O);
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_int8w_i8x_linear_constexpr_create(int H, int I, int B,
                                                                      float x_scale,
                                                                      const int8_t   *Wa_q_OI,
                                                                      const int8_t   *Wa_off_O,
                                                                      const uint16_t *Wa_scale_f16_O,
                                                                      const int8_t   *Wb_q_OI,
                                                                      const int8_t   *Wb_off_O,
                                                                      const uint16_t *Wb_scale_f16_O) {
    return ds4_ane_mlp_int8w_linear_constexpr_create_common(
        H, I, B, true, x_scale,
        Wa_q_OI, Wa_off_O, Wa_scale_f16_O,
        Wb_q_OI, Wb_off_O, Wb_scale_f16_O);
}

/* Attach N additional input IOSurfaces to an existing mode-11/mode-12 ctx by
 * creating N requests, each bound to one of the supplied input IOSurfaces (with the
 * ctx's existing io_out as the shared output).  After this, eval_at_chunk(k)
 * uses chunk_requests[k] instead of request_r — ANE reads the matching
 * external IOSurface for that chunk.  Caller owns the IOSurfaces' lifetime. */
bool ds4_ane_mlp_int8w_linear_constexpr_attach_chunks(
        ds4_ane_mlp_int8w_ctx *ctx,
        const IOSurfaceRef *chunk_input_iosurfaces,
        int n_chunks) {
    if (!ctx || !chunk_input_iosurfaces || n_chunks <= 0) return false;
    if (n_chunks > 64) return false;
    if (ctx->mode != 11 && ctx->mode != 12) return false;
    resolve_classes();
    if (!g_ReqCls || !g_IOCls) return false;
    @autoreleasepool {
        id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_IOCls, @selector(objectWithIOSurface:), ctx->io_out);
        if (!w_o) return false;
        for (int k = 0; k < n_chunks; k++) {
            id w_x = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_IOCls, @selector(objectWithIOSurface:), chunk_input_iosurfaces[k]);
            if (!w_x) return false;
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[w_x], @[@0], @[w_o], @[@0], nil, nil, @0);
            if (!req) return false;
            ctx->chunk_requests[k] = (void *)CFBridgingRetain(req);
        }
        ctx->n_chunk_requests = n_chunks;
        return true;
    }
}

/* Eval using the per-chunk request bound to chunk_idx's IOSurface — caller
 * has already written the input (typically via a GPU dispatch into the
 * external IOSurface), and reads fp16 output from the shared io_out. */
bool ds4_ane_mlp_int8w_linear_constexpr_eval_at_chunk(
        ds4_ane_mlp_int8w_ctx *ctx,
        int chunk_idx,
        uint16_t *output_f16) {
    if (!ctx || !output_f16) return false;
    if (ctx->mode != 11 && ctx->mode != 12) return false;
    if (chunk_idx < 0 || chunk_idx >= ctx->n_chunk_requests) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->chunk_requests[chunk_idx], &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-linear constexpr eval_at_chunk %d failed: %s\n",
                             chunk_idx, e ? [[e description] UTF8String] : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) return false;
        return true;
    }
}

bool ds4_ane_mlp_int8w_linear_constexpr_eval(ds4_ane_mlp_int8w_ctx *ctx,
                                              const uint16_t *input_f16,
                                              uint16_t *output_f16) {
    if (!ctx || !input_f16 || !output_f16) return false;
    if (ctx->mode != 11) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_x, input_f16, ctx->x_bytes)) return false;
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-linear constexpr eval failed: %s\n",
                             e ? [[e description] UTF8String] : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) return false;
        return true;
    }
}

bool ds4_ane_mlp_int8w_i8x_linear_constexpr_eval(ds4_ane_mlp_int8w_ctx *ctx,
                                                  const int8_t *input_i8,
                                                  uint16_t *output_f16) {
    if (!ctx || !input_i8 || !output_f16) return false;
    if (ctx->mode != 12) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_x, input_i8, ctx->x_bytes)) return false;
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x-linear constexpr eval failed: %s\n",
                             e ? [[e description] UTF8String] : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) return false;
        return true;
    }
}

bool ds4_ane_mlp_fp16w_linear_constexpr_eval(ds4_ane_mlp_int8w_ctx *ctx,
                                              const uint16_t *input_f16,
                                              uint16_t *output_f16) {
    if (!ctx || !input_f16 || !output_f16) return false;
    if (ctx->mode != 10) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_x, input_f16, ctx->x_bytes)) return false;
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear constexpr eval failed: %s\n",
                             e ? [[e description] UTF8String] : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) return false;
        return true;
    }
}

ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_fp16w_constexpr_create(int H, int I, int B,
                                                          const uint16_t *Wgate_OI,
                                                          const uint16_t *Wup_OI,
                                                          const uint16_t *Wdown_OI) {
    if (H <= 0 || I <= 0 || B <= 0 || !Wgate_OI || !Wup_OI || !Wdown_OI) return NULL;
    ane_cleanup_stale_tmp_dirs_once();
    resolve_classes();
    const bool dbg = ane_int8w_debug_enabled();
    if (!g_DescCls || !g_ModelCls || !g_ReqCls || !g_IOCls) {
        if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr missing private classes\n");
        return NULL;
    }
    @autoreleasepool {
        NSFileManager *fm = [NSFileManager defaultManager];
        const NSUInteger gate_bytes = (NSUInteger)I * (NSUInteger)H * sizeof(uint16_t);
        const NSUInteger down_bytes = (NSUInteger)H * (NSUInteger)I * sizeof(uint16_t);

        /* Canonical form: BLOBFILE path is the literal "@model_path/...".  The
         * private compiler resolves @model_path/ via two channels:
         *   (1) An entry in the `weights:` dict passed to the descriptor, keyed
         *       by the full @model_path/... string, value {offset, data}.
         *   (2) An on-disk file at NSTemporaryDirectory()/<hexId>/weights/...
         * We provide BOTH (the in-memory dict for the descriptor's hash, and
         * the file for the post-descriptor compile pass). */
        NSString *blob_path_in_mil = @"@model_path/weights/weight.bin";
        const uint64_t off_g = 64;
        const uint64_t off_u = 64 + 64 + (uint64_t)gate_bytes;
        const uint64_t off_d = 64 + 64 + (uint64_t)gate_bytes + 64 + (uint64_t)gate_bytes;

        uint64_t off_g_chk = 0, off_u_chk = 0, off_d_chk = 0;
        NSData *blob = ane_build_fp16_blob_3(Wgate_OI, gate_bytes,
                                              Wup_OI,   gate_bytes,
                                              Wdown_OI, down_bytes,
                                              &off_g_chk, &off_u_chk, &off_d_chk);
        if (!blob || off_g_chk != off_g || off_u_chk != off_u || off_d_chk != off_d) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr blob build failed\n");
            return NULL;
        }

        NSString *mil = gen_mil_fp16w_constexpr_conv(H, I, B, blob_path_in_mil, off_g, off_u, off_d);
        NSData *milData = [[mil dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSError *e = nil;
        NSDictionary *weights = @{
            blob_path_in_mil: @{
                @"offset": @(0),
                @"data": blob,
            }
        };
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_DescCls, @selector(modelWithMILText:weights:optionsPlist:),
            milData, weights, nil);
        if (!desc) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr descriptor failed\n");
            return NULL;
        }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ModelCls, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr inMemoryModel failed\n");
            return NULL;
        }
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSString *weights_dir = [td stringByAppendingPathComponent:@"weights"];
        if (![fm createDirectoryAtPath:weights_dir withIntermediateDirectories:YES
                            attributes:nil error:nil]) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr mkdir %s failed\n",
                             [weights_dir UTF8String]);
            return NULL;
        }
        NSString *blob_path = [weights_dir stringByAppendingPathComponent:@"weight.bin"];
        if (![blob writeToFile:blob_path atomically:YES]) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr blob writeToFile failed: %s\n",
                             [blob_path UTF8String]);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr compile failed: %s\n  (debug: keeping tmpdir %s for inspection)\n",
                             e ? [[e description] UTF8String] : "unknown",
                             [td UTF8String]);
            if (!dbg) [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr load failed: %s\n",
                             e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        ds4_ane_mlp_int8w_ctx *ctx = (ds4_ane_mlp_int8w_ctx *)calloc(1, sizeof(*ctx));
        if (!ctx) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        ctx->H = H; ctx->I = I; ctx->B = B; ctx->mode = 9;
        ctx->w_scale = 1.0f; ctx->x_scale = 1.0f; ctx->mid_scale = 1.0f;
        /* Only X input + Y output surfaces — weights live in the model. */
        ctx->x_bytes  = (NSUInteger)B * (NSUInteger)H * sizeof(uint16_t);
        ctx->out_bytes = (NSUInteger)B * (NSUInteger)H * sizeof(uint16_t);
        ctx->io_x   = make_surface_typed(ctx->x_bytes, 2u);
        ctx->io_out = make_surface_typed(ctx->out_bytes, 2u);
        if (!ctx->io_x || !ctx->io_out) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr IOSurface alloc failed\n");
            ds4_ane_mlp_int8w_destroy(ctx);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        id w_x = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls,
            @selector(objectWithIOSurface:), ctx->io_x);
        id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls,
            @selector(objectWithIOSurface:), ctx->io_out);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[w_x], @[@0], @[w_o], @[@0], nil, nil, @0);
        if (!req) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr request create failed\n");
            ds4_ane_mlp_int8w_destroy(ctx);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        ctx->model_r   = (void *)CFBridgingRetain(mdl);
        ctx->request_r = (void *)CFBridgingRetain(req);
        ctx->tmpDir_r  = (void *)CFBridgingRetain([td copy]);
        return ctx;
    }
}

bool ds4_ane_mlp_fp16w_constexpr_eval(ds4_ane_mlp_int8w_ctx *ctx,
                                       const uint16_t *input_f16,
                                       uint16_t *output_f16) {
    if (!ctx || !input_f16 || !output_f16) return false;
    if (ctx->mode != 9) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_x, input_f16, ctx->x_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr write X failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr evaluate failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w constexpr read Y failed\n");
            return false;
        }
        return true;
    }
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

/* Dedicated create for the linear two-matmul int8 path.  Unlike the mode-2
 * create_common (which bakes ONE w_scale into both gate and down models), this
 * lets gate and down have DIFFERENT scales — needed for LoRA-style projections
 * where Wa and Wb absmax differ. */
ds4_ane_mlp_int8w_ctx *ds4_ane_mlp_i8w_fp16x_linear_create(int H, int I, int B,
                                                           float a_scale, float b_scale) {
    if (H <= 0 || I <= 0 || B <= 0 || !(a_scale > 0.0f) || !(b_scale > 0.0f)) return NULL;
    resolve_classes();
    const bool dbg = ane_int8w_debug_enabled();
    if (!g_DescCls || !g_ModelCls || !g_ReqCls || !g_IOCls) {
        if (dbg) fprintf(stderr, "ds4: ANE i8w-fp16x-linear missing classes\n");
        return NULL;
    }
    @autoreleasepool {
        ds4_ane_mlp_int8w_ctx *ctx = (ds4_ane_mlp_int8w_ctx *)calloc(1, sizeof(*ctx));
        if (!ctx) return NULL;
        ctx->H = H; ctx->I = I; ctx->B = B; ctx->mode = 2;
        ctx->w_scale = a_scale;     /* gate-side scale */
        ctx->mid_scale = b_scale;   /* down-side scale stored in mid_scale slot */
        ctx->x_scale = 1.0f;
        /* int8 weights → 1 byte/elem; fp16 activations → 2 bytes/elem. */
        ctx->gate_bytes   = (NSUInteger)H * (NSUInteger)I * 1u;
        ctx->down_bytes   = (NSUInteger)I * (NSUInteger)H * 1u;
        ctx->x_bytes      = (NSUInteger)B * (NSUInteger)H * sizeof(uint16_t);
        ctx->mid_bytes    = (NSUInteger)B * (NSUInteger)I * sizeof(uint16_t);
        ctx->hidden_bytes = (NSUInteger)B * (NSUInteger)I * sizeof(uint16_t);
        ctx->out_bytes    = (NSUInteger)B * (NSUInteger)H * sizeof(uint16_t);
        ctx->io_gate   = make_surface_typed(ctx->gate_bytes, 1u);
        ctx->io_down   = make_surface_typed(ctx->down_bytes, 1u);
        ctx->io_x      = make_surface_typed(ctx->x_bytes, 1u);
        ctx->io_mid    = make_surface_typed(ctx->mid_bytes, 1u);
        ctx->io_hidden = make_surface_typed(ctx->hidden_bytes, 1u);
        ctx->io_out    = make_surface_typed(ctx->out_bytes, 1u);
        if (!ctx->io_gate || !ctx->io_down || !ctx->io_x ||
            !ctx->io_mid  || !ctx->io_hidden || !ctx->io_out) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-fp16x-linear IOSurface alloc failed\n");
            ds4_ane_mlp_int8w_destroy(ctx);
            return NULL;
        }
        NSString *gate_mil = gen_mil_i8w_fp16x_matmul(H, I, B, a_scale);
        NSString *down_mil = gen_mil_i8w_fp16x_matmul(I, H, B, b_scale);
        if (!compile_and_load_mil(gate_mil, "i8w-fp16x-linear A", &ctx->model_r, &ctx->tmpDir_r) ||
            !compile_and_load_mil(down_mil, "i8w-fp16x-linear B", &ctx->model_down_r, &ctx->tmpDir_down_r)) {
            ds4_ane_mlp_int8w_destroy(ctx);
            return NULL;
        }
        id w_g = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_gate);
        id w_x = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_x);
        id w_m = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_mid);
        id w_d = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_down);
        id w_h = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_hidden);
        id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls, @selector(objectWithIOSurface:), ctx->io_out);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[w_g, w_x], @[@0, @1], @[w_m], @[@0], nil, nil, @0);
        id req_down = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[w_d, w_h], @[@0, @1], @[w_o], @[@0], nil, nil, @0);
        if (!req || !req_down) {
            ds4_ane_mlp_int8w_destroy(ctx);
            return NULL;
        }
        ctx->request_r = (void *)CFBridgingRetain(req);
        ctx->request_down_r = (void *)CFBridgingRetain(req_down);
        if (dbg) fprintf(stderr, "ds4: ANE i8w-fp16x-linear create H=%d I=%d B=%d a_scale=%g b_scale=%g\n",
                         H, I, B, a_scale, b_scale);
        return ctx;
    }
}

/* Linear two-matmul i8w-fp16x eval: int8 weights with per-tensor scale baked
 * into the compiled MIL.  Reuses the mode-2 split ctx (model_r is gate-shape
 * [B, H] @ [H, I], model_down_r is down-shape [B, I] @ [I, H]).  Two ANE
 * evals, no activation between.  Per-call upload is half the fp16 path's
 * (Wa+Wb together = HI bytes instead of 2*HI). */
bool ds4_ane_mlp_i8w_fp16x_linear_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t   *Wa_i8,
    const int8_t   *Wb_i8,
    const uint16_t *input_f16,
    uint16_t       *output_f16)
{
    if (!ctx || !Wa_i8 || !Wb_i8 || !input_f16 || !output_f16) return false;
    if (ctx->mode != 2 || !ctx->model_down_r) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_x, input_f16, ctx->x_bytes) ||
            !write_surface(ctx->io_gate, Wa_i8, ctx->gate_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-fp16x-linear A write failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-fp16x-linear A eval failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        uint16_t *mid = (uint16_t *)malloc(ctx->mid_bytes);
        if (!mid) return false;
        if (!read_surface(ctx->io_mid, mid, ctx->mid_bytes) ||
            !write_surface(ctx->io_hidden, mid, ctx->hidden_bytes) ||
            !write_surface(ctx->io_down, Wb_i8, ctx->down_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-fp16x-linear A→B staging failed\n");
            free(mid);
            return false;
        }
        free(mid);
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_down_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_down_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-fp16x-linear B eval failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) return false;
        return true;
    }
}

/* Linear two-matmul fp16w eval: input → W_a → mid → W_b → output, no
 * activation between.  Reuses the mode-1 split ctx (model_r for the A matmul
 * with shape [B, H] @ [H, I], model_down_r for the B matmul with shape
 * [B, I] @ [I, H]).  Designed for LoRA-style projections like the attention
 * output (attn_output_a × attn_output_b in DSv4). */
bool ds4_ane_mlp_fp16w_linear_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const uint16_t *Wa_f16,
    const uint16_t *Wb_f16,
    const uint16_t *input_f16,
    uint16_t *output_f16)
{
    if (!ctx || !Wa_f16 || !Wb_f16 || !input_f16 || !output_f16) return false;
    if (ctx->mode != 1 || !ctx->model_down_r) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_x, input_f16, ctx->x_bytes) ||
            !write_surface(ctx->io_gate, Wa_f16, ctx->gate_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear A-stage write failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear A-stage eval failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        /* Stage the A-stage output into io_hidden so model_down_r can read it.
         * mid_bytes == hidden_bytes (both = B*I*2) for the mode-1 split ctx. */
        uint16_t *mid = (uint16_t *)malloc(ctx->mid_bytes);
        if (!mid) return false;
        if (!read_surface(ctx->io_mid, mid, ctx->mid_bytes) ||
            !write_surface(ctx->io_hidden, mid, ctx->hidden_bytes) ||
            !write_surface(ctx->io_down, Wb_f16, ctx->down_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear A→B staging failed\n");
            free(mid);
            return false;
        }
        free(mid);
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_down_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_down_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear B-stage eval failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w-linear out read failed\n");
            return false;
        }
        return true;
    }
}

bool ds4_ane_mlp_fp16w_fused_conv_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const uint16_t *Wgate_f16,
    const uint16_t *Wup_f16,
    const uint16_t *Wdown_f16,
    const uint16_t *input_f16,
    uint16_t *output_f16)
{
    if (!ctx || !Wgate_f16 || !Wup_f16 || !Wdown_f16 || !input_f16 || !output_f16) return false;
    if (ctx->mode != 8) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_gate, Wgate_f16, ctx->gate_bytes) ||
            !write_surface(ctx->io_up, Wup_f16, ctx->gate_bytes) ||
            !write_surface(ctx->io_down, Wdown_f16, ctx->down_bytes) ||
            !write_surface(ctx->io_x, input_f16, ctx->x_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w fused conv IOSurface write failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w fused conv evaluate failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE fp16w fused conv IOSurface read failed\n");
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

bool ds4_ane_mlp_i8w_i8x_tiled_fused_routed_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    const uint16_t *route_f16,
    uint16_t *output_f16)
{
    if (!ctx || !Wgate_i8 || !Wup_i8 || !Wdown_i8 || !input_i8 || !route_f16 || !output_f16) return false;
    if (ctx->mode != 13) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_gate, Wgate_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_up, Wup_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_down, Wdown_i8, ctx->down_bytes) ||
            !write_surface(ctx->io_x, input_i8, ctx->x_bytes) ||
            !write_surface(ctx->io_route, route_f16, ctx->route_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused routed IOSurface write failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused routed evaluate failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused routed IOSurface read failed\n");
            return false;
        }
        return true;
    }
}

/* Int8-output variant of tiled_fused_eval.  Identical pipeline; only the
 * output IOSurface dtype + read_surface size differs (B*H bytes int8 vs
 * B*H*2 bytes fp16).  Returns whatever int8 values the ANE produced — caller
 * is responsible for dequantizing using ctx->mid_scale if needed. */
bool ds4_ane_mlp_i8w_i8x_tiled_fused_i8out_eval(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *Wgate_i8,
    const int8_t *Wup_i8,
    const int8_t *Wdown_i8,
    const int8_t *input_i8,
    int8_t       *output_i8)
{
    if (!ctx || !Wgate_i8 || !Wup_i8 || !Wdown_i8 || !input_i8 || !output_i8) return false;
    if (ctx->mode != 7) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_gate, Wgate_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_up, Wup_i8, ctx->gate_bytes) ||
            !write_surface(ctx->io_down, Wdown_i8, ctx->down_bytes) ||
            !write_surface(ctx->io_x, input_i8, ctx->x_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused i8out IOSurface write failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused i8out evaluate failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_i8, ctx->out_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused i8out IOSurface read failed\n");
            return false;
        }
        return true;
    }
}

bool ds4_ane_mlp_i8w_i8x_tiled_fused_eval_xonly(
    ds4_ane_mlp_int8w_ctx *ctx,
    const int8_t *input_i8,
    uint16_t *output_f16)
{
    if (!ctx || !input_i8 || !output_f16) return false;
    if (ctx->mode != 6) return false;
    const bool dbg = ane_int8w_debug_enabled();
    @autoreleasepool {
        if (!write_surface(ctx->io_x, input_i8, ctx->x_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused xonly write_x failed\n");
            return false;
        }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, (__bridge id)ctx->request_r, &e);
        if (!ok) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused xonly evaluate failed: %s\n",
                             e.localizedDescription ? e.localizedDescription.UTF8String : "unknown");
            return false;
        }
        if (!read_surface(ctx->io_out, output_f16, ctx->out_bytes)) {
            if (dbg) fprintf(stderr, "ds4: ANE i8w-i8x tiled fused xonly read failed\n");
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
        for (int k = 0; k < ctx->n_chunk_requests; k++) {
            if (ctx->chunk_requests[k]) {
                CFBridgingRelease(ctx->chunk_requests[k]);
                ctx->chunk_requests[k] = NULL;
            }
        }
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
