// Appended to a mechanically generated upstream translation unit.
#include <climits>
#include <vector>

struct BaselinePlan {
    tcfftHandle fft{};
    int *reverse = nullptr;
    half2 *scratch = nullptr;
};

__global__ void baseline_reorder(const half2 *source, half2 *destination,
                                 const int *reverse, int n, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) destination[i] = source[(i / n) * n + reverse[i % n]];
}

static cudaError_t baseline_free(BaselinePlan *p) {
    cudaError_t result = cudaSuccess;
    for (void *ptr : {static_cast<void *>(p->reverse), static_cast<void *>(p->scratch),
                      static_cast<void *>(p->fft.F_real), static_cast<void *>(p->fft.F_imag)}) {
        cudaError_t code = cudaFree(ptr);
        if (result == cudaSuccess) result = code;
    }
    free(p->fft.F_real_tmp);
    free(p->fft.F_imag_tmp);
    delete p;
    return result;
}

extern "C" int tcfft_baseline_create(int n, int batch, void **result) {
    if (!result) return cudaErrorInvalidValue;
    *result = nullptr;
    if (batch < 1 || n < 1 || static_cast<long long>(n) * batch > INT_MAX ||
        (static_cast<long long>(n) * batch) % 8192) return cudaErrorInvalidValue;
    switch (n) {
    case 256: case 512: case 1024: case 131072: case 262144: case 524288:
    case 16777216: case 33554432: case 67108864: case 134217728: break;
    default: return cudaErrorInvalidValue;
    }
    auto *p = new BaselinePlan;
    tcfftCreate(&p->fft, n, batch);
    cudaError_t code = cudaGetLastError();
    if (!p->fft.F_real || !p->fft.F_imag || code != cudaSuccess) {
        baseline_free(p);
        return code == cudaSuccess ? cudaErrorMemoryAllocation : code;
    }
    const int cont[] = {32, 16, 8};
    int step = 1;
    for (int i = 0; i < p->fft.n_mergings; ++i) {
        int m = p->fft.mergings[i], radix = 256 << m;
        if (i && n * batch / step / radix > 65535) {
            baseline_free(p);
            return cudaErrorInvalidConfiguration;
        }
        auto function = i ? reinterpret_cast<const void *>(p->fft.layer_1[m])
                          : reinterpret_cast<const void *>(p->fft.layer_0[m]);
        code = cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    radix * sizeof(half2) * cont[m]);
        if (code != cudaSuccess) { baseline_free(p); return code; }
        step *= radix;
    }
    std::vector<int> a(n), b(n);
    for (int i = 0; i < n; ++i) a[i] = i;
    int current = n;
    for (int i = p->fft.n_radices - 1; i >= 0; --i) {
        int radix = p->fft.radices[i];
        for (int j = 0; j < n; j += current)
            for (int k = 0; k < radix; ++k)
                for (int l = 0; l < current / radix; ++l)
                    b[j + l + k * (current / radix)] = a[j + l * radix + k];
        current /= radix;
        a.swap(b);
    }
    code = cudaMalloc(&p->reverse, sizeof(int) * static_cast<size_t>(n));
    if (code == cudaSuccess) code = cudaMalloc(&p->scratch, sizeof(half2) * static_cast<size_t>(n) * batch);
    if (code == cudaSuccess) code = cudaMemcpy(p->reverse, a.data(), sizeof(int) * static_cast<size_t>(n), cudaMemcpyHostToDevice);
    if (code != cudaSuccess) { baseline_free(p); return code; }
    *result = p;
    return cudaSuccess;
}

extern "C" int tcfft_baseline_execute(void *handle, const void *source,
                                      void *destination, void *stream_pointer) {
    if (!handle || !source || !destination) return cudaErrorInvalidValue;
    auto *p = static_cast<BaselinePlan *>(handle);
    auto stream = static_cast<cudaStream_t>(stream_pointer);
    int total = p->fft.N * p->fft.N_batch;
    baseline_reorder<<<(static_cast<size_t>(total) + 255) / 256, 256, 0, stream>>>(
        static_cast<const half2 *>(source), p->scratch, p->reverse, p->fft.N, total);
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) return code;
    tcfftExecStream(p->fft, reinterpret_cast<half *>(p->scratch), stream);
    code = cudaGetLastError();
    if (code != cudaSuccess) return code;
    return cudaMemcpyAsync(destination, p->scratch, sizeof(half2) * static_cast<size_t>(total),
                           cudaMemcpyDeviceToDevice, stream);
}

extern "C" int tcfft_baseline_destroy(void *handle) {
    return handle ? baseline_free(static_cast<BaselinePlan *>(handle)) : cudaSuccess;
}
