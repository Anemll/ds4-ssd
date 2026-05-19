// ane_ds4_mlp_split3.m  — implementation of ane_ds4_mlp_split3.h
//
// Wraps the split-3 packed MLP graph from
// `ds4_mlp_inmem_bench_packed_split3.m` behind a C API that the production
// MoE expert path can call directly.
//
// Build (as a translation unit linked into a larger program):
//   clang -fobjc-arc -O2 -c ane_ds4_mlp_split3.m
// Or directly into the final binary along with main:
//   clang -fobjc-arc -O2 ... ane_ds4_mlp_split3.m main.m \
//       -framework Foundation -framework IOSurface

#import "ane_ds4_mlp_split3.h"

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

#define SPLIT3_T 3

static inline uint64_t total_elts_(int B, int H, int I) {
    return (uint64_t)B*H + 3ULL * (uint64_t)H * (uint64_t)I;
}

// Under ARC, ObjC `id` can't live in a plain C struct (the compiler can't
// emit the right retain/release for malloc'd memory).  Store as `void *` with
// CFBridgingRetain / CFBridgingRelease at the lifecycle boundaries instead.
struct ds4_ane_mlp_split3_ctx {
    int          H, I, B;
    void        *model_r;      // retained _ANEInMemoryModel*
    void        *request_r;    // retained _ANERequest*
    IOSurfaceRef io_packed;
    IOSurfaceRef io_out;
    void        *tmpDir_r;     // retained NSString*
    NSUInteger   packed_bytes;
    NSUInteger   output_bytes;
};

static dispatch_once_t g_classes_once;
static Class g_DescCls = nil;
static Class g_ModelCls = nil;
static Class g_ReqCls = nil;
static Class g_IOCls = nil;

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

// MIL generator (identical structure to ds4_mlp_inmem_bench_packed_split3.m).
static NSString *gen_mil(int H, int I, int B) {
    NSMutableString *m = [NSMutableString string];
    uint64_t bh = (uint64_t)B*H;
    uint64_t hi = (uint64_t)H * (uint64_t)(I/SPLIT3_T);  // [H, I/T] tile size
    uint64_t di = (uint64_t)(I/SPLIT3_T) * (uint64_t)H;  // [I/T, H] tile size (== hi)
    uint64_t N  = bh + (uint64_t)(2*SPLIT3_T)*hi + (uint64_t)SPLIT3_T*di;
    int It = I / SPLIT3_T;

    uint64_t off_in    = 0;
    uint64_t off_gate0 = bh;
    uint64_t off_up0   = bh + (uint64_t)SPLIT3_T * hi;
    uint64_t off_down0 = bh + (uint64_t)(2*SPLIT3_T) * hi;

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

    // Activation slice + reshape -> x_3d [1, H, B]
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

    for (int t = 0; t < SPLIT3_T; t++) {
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
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> d%d = matmul(transpose_x = tx_f, transpose_y = tx_f, x = hidden_%d, y = W_down_%d)[name = string(\"d%d\")];\n",
            B, H, t, t, t, t];
    }

    NSMutableString *prev = [NSMutableString stringWithString:@"d0"];
    for (int t = 1; t < SPLIT3_T; t++) {
        NSString *sumName = (t == SPLIT3_T - 1) ? @"output" : [NSString stringWithFormat:@"acc%d", t];
        [m appendFormat:
            @"            tensor<fp16, [1, %d, %d]> %@ = add(x = %@, y = d%d)[name = string(\"%@\")];\n",
            B, H, sumName, prev, t, sumName];
        [prev setString:sumName];
    }
    [m appendString:@"        } -> (output);\n}\n"];
    return m;
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

ds4_ane_mlp_split3_ctx *ds4_ane_mlp_split3_create(int H, int I, int B) {
    if (I % SPLIT3_T != 0 || H <= 0 || B <= 0) return NULL;
    resolve_classes();
    if (!g_DescCls || !g_ModelCls || !g_ReqCls || !g_IOCls) return NULL;

    @autoreleasepool {
        NSError *e = nil;
        NSString *mil = gen_mil(H, I, B);
        NSData *milData = [[mil dataUsingEncoding:NSUTF8StringEncoding] copy];

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_DescCls, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);
        if (!desc) return NULL;
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ModelCls, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) return NULL;
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        uint64_t N = total_elts_(B, H, I);
        NSUInteger packed_bytes = (NSUInteger)(N * 2);
        NSUInteger output_bytes = (NSUInteger)B * H * 2;
        IOSurfaceRef io_p = make_surface(packed_bytes);
        IOSurfaceRef io_o = make_surface(output_bytes);
        if (!io_p || !io_o) {
            if (io_p) CFRelease(io_p);
            if (io_o) CFRelease(io_o);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        id w_p = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls,@selector(objectWithIOSurface:),io_p);
        id w_o = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOCls,@selector(objectWithIOSurface:),io_o);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ReqCls, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[w_p], @[@0], @[w_o], @[@0], nil, nil, @0);

        ds4_ane_mlp_split3_ctx *ctx = (ds4_ane_mlp_split3_ctx *)calloc(1, sizeof(*ctx));
        if (!ctx) {
            CFRelease(io_p); CFRelease(io_o);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        ctx->H = H; ctx->I = I; ctx->B = B;
        ctx->model_r   = (void *)CFBridgingRetain(mdl);
        ctx->request_r = (void *)CFBridgingRetain(req);
        ctx->io_packed = io_p;
        ctx->io_out    = io_o;
        ctx->tmpDir_r  = (void *)CFBridgingRetain([td copy]);
        ctx->packed_bytes = packed_bytes;
        ctx->output_bytes = output_bytes;
        return ctx;
    }
}

bool ds4_ane_mlp_split3_eval(
    ds4_ane_mlp_split3_ctx *ctx,
    const void *input_fp16,
    const void *Wgate_t3_fp16,
    const void *Wup_t3_fp16,
    const void *Wdown_t3_fp16,
    void *output_fp16)
{
    if (!ctx || !input_fp16 || !Wgate_t3_fp16 || !Wup_t3_fp16 || !Wdown_t3_fp16 || !output_fp16) return false;
    @autoreleasepool {
        NSError *e = nil;
        // Layout: [input | W_gate (3 tiles) | W_up (3 tiles) | W_down (3 tiles)]
        // We just memcpy each region into the packed IOSurface; the MIL slices
        // by element offset, which matches our byte layout since fp16=2 bytes.
        NSUInteger bh_b = (NSUInteger)ctx->B * ctx->H * 2;
        NSUInteger w_b  = (NSUInteger)ctx->H * ctx->I * 2;
        NSUInteger wd_b = (NSUInteger)ctx->I * ctx->H * 2;
        // sanity:
        if (bh_b + w_b + w_b + wd_b != ctx->packed_bytes) return false;

        IOSurfaceLock(ctx->io_packed, 0, NULL);
        uint8_t *p = (uint8_t *)IOSurfaceGetBaseAddress(ctx->io_packed);
        memcpy(p, input_fp16, bh_b);                       p += bh_b;
        memcpy(p, Wgate_t3_fp16, w_b);                     p += w_b;
        memcpy(p, Wup_t3_fp16,   w_b);                     p += w_b;
        memcpy(p, Wdown_t3_fp16, wd_b);
        IOSurfaceUnlock(ctx->io_packed, 0, NULL);

        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{},
            (__bridge id)ctx->request_r,
            &e);
        if (!ok) return false;

        IOSurfaceLock(ctx->io_out, kIOSurfaceLockReadOnly, NULL);
        memcpy(output_fp16, IOSurfaceGetBaseAddress(ctx->io_out), ctx->output_bytes);
        IOSurfaceUnlock(ctx->io_out, kIOSurfaceLockReadOnly, NULL);
        return true;
    }
}

void *ds4_ane_mlp_split3_begin_input(ds4_ane_mlp_split3_ctx *ctx) {
    if (!ctx || !ctx->io_packed) return NULL;
    if (IOSurfaceLock(ctx->io_packed, 0, NULL) != kIOReturnSuccess) return NULL;
    return IOSurfaceGetBaseAddress(ctx->io_packed);
}

void ds4_ane_mlp_split3_end_input(ds4_ane_mlp_split3_ctx *ctx) {
    if (!ctx || !ctx->io_packed) return;
    IOSurfaceUnlock(ctx->io_packed, 0, NULL);
}

bool ds4_ane_mlp_split3_eval_packed(
    ds4_ane_mlp_split3_ctx *ctx,
    void *output_fp16)
{
    if (!ctx || !output_fp16) return false;
    @autoreleasepool {
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)ctx->model_r,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{},
            (__bridge id)ctx->request_r,
            &e);
        if (!ok) return false;
        IOSurfaceLock(ctx->io_out, kIOSurfaceLockReadOnly, NULL);
        memcpy(output_fp16, IOSurfaceGetBaseAddress(ctx->io_out), ctx->output_bytes);
        IOSurfaceUnlock(ctx->io_out, kIOSurfaceLockReadOnly, NULL);
        return true;
    }
}

void ds4_ane_mlp_split3_destroy(ds4_ane_mlp_split3_ctx *ctx) {
    if (!ctx) return;
    @autoreleasepool {
        NSError *e = nil;
        // Take back ownership; ARC will release at scope end.
        id model   = ctx->model_r   ? CFBridgingRelease(ctx->model_r)   : nil;
        id request = ctx->request_r ? CFBridgingRelease(ctx->request_r) : nil;
        NSString *tmpDir = ctx->tmpDir_r ? CFBridgingRelease(ctx->tmpDir_r) : nil;
        (void)request;
        if (model) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &e);
        }
        if (ctx->io_packed) CFRelease(ctx->io_packed);
        if (ctx->io_out)    CFRelease(ctx->io_out);
        if (tmpDir) {
            [[NSFileManager defaultManager] removeItemAtPath:tmpDir error:nil];
        }
        free(ctx);
    }
}

int ds4_ane_mlp_split3_H(const ds4_ane_mlp_split3_ctx *ctx) { return ctx ? ctx->H : 0; }
int ds4_ane_mlp_split3_I(const ds4_ane_mlp_split3_ctx *ctx) { return ctx ? ctx->I : 0; }
int ds4_ane_mlp_split3_B(const ds4_ane_mlp_split3_ctx *ctx) { return ctx ? ctx->B : 0; }
