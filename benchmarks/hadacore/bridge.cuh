#include <cuda_fp16.h>
__global__ void hadacore_scale(__half* output, uint32_t count, float factor) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2half_rn(__half2float(output[i]) * factor);
}
extern "C" int hadacore_execute(void* input, void* output, uint32_t count,
                                uint32_t size, float scale, cudaStream_t stream) {
    if (!input || !output || count == 0 || count > 2147483392U || count % 256 ||
        size < 2 || size > 32768 || (size & (size-1)) || count % size) return -1;
    try {
        run_fht<torch::ScalarType::Half>(input, output, count, size, stream);
        hadacore_scale<<<(count+255)/256, 256, 0, stream>>>((__half*)output, count, scale);
        return (int)cudaGetLastError();
    } catch (...) { return -2; }
}
