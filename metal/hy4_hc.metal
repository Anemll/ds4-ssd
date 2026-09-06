// HY4 independent Hyper-Connections; graph layout from source 34cccef.
// No DS4 Sinkhorn/comb transform. Mix dots use a parallel F32 reduction;
// stream weighting and residual updates preserve separate F32 mul/add order.
struct ds4_metal_args_hy4_hc { uint emb, head; };
kernel void kernel_hy4_hc_mix(
        constant ds4_metal_args_hy4_hc &a [[buffer(0)]],
        device const float *streams [[buffer(1)]],
        device const float *fn [[buffer(2)]],
        device float *mix [[buffer(3)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
#pragma clang fp reassociate(off) contract(off)
    threadgroup float partial[256];
    const uint flat=4*a.emb;
    float square=0;
    for(uint j=tid;j<flat;j+=256) square+=streams[j]*streams[j];
    partial[tid]=square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=128;step;step>>=1) {
        if(tid<step) partial[tid]+=partial[tid+step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv=precise::rsqrt(partial[0]/float(flat)+1e-5f);
    // All lanes must consume the RMS reduction before reusing its scratch.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0;
    for(uint j=tid;j<flat;j+=256) {
        const float normalized=streams[j]*inv;
        sum+=fn[ulong(row)*flat+j]*normalized;
    }
    partial[tid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=128;step;step>>=1) {
        if(tid<step) partial[tid]+=partial[tid+step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0) mix[row]=partial[0];
}
kernel void kernel_hy4_hc_reduce(
        constant ds4_metal_args_hy4_hc &a [[buffer(0)]],
        device const float *streams [[buffer(1)]],
        device const float *mix [[buffer(2)]],
        device const float *scale [[buffer(3)]],
        device const float *base [[buffer(4)]],
        device float *out [[buffer(5)]],
        device float *post [[buffer(6)]],
        uint j [[thread_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
#pragma clang fp reassociate(off) contract(off)
    threadgroup float pre[4];
    if(tid<4) {
        volatile float scaled=mix[tid]*scale[0];
        const float logit=scaled+base[tid];
        pre[tid]=precise::divide(1.0f,1.0f+precise::exp(-logit))+1e-6f;
    }
    if(!a.head && j<4) {
        volatile float scaled=mix[4+j]*scale[1];
        const float logit=scaled+base[4+j];
        volatile float probability=precise::divide(1.0f,1.0f+precise::exp(-logit));
        volatile float magnitude=probability*2.0f;
        post[j]=magnitude+1e-6f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(j>=a.emb) return;
    volatile float sum=streams[j]*pre[0];
    for(uint k=1;k<4;k++) {
        volatile float product=streams[ulong(k)*a.emb+j]*pre[k];
        sum=sum+product;
    }
    out[j]=sum;
}
kernel void kernel_hy4_hc_post(
        constant uint &emb [[buffer(0)]],
        device float *streams [[buffer(1)]],
        device const float *x [[buffer(2)]],
        device const float *post [[buffer(3)]],
        uint j [[thread_position_in_grid]]) {
#pragma clang fp reassociate(off) contract(off)
    if(j>=4*emb) return;
    volatile float product=x[j%emb]*post[j/emb];
    streams[j]=streams[j]+product;
}
