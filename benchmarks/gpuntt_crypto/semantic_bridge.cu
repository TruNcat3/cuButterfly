// Prepared rank-one cyclic GPU-NTT bridge; arithmetic is the pinned vendor core.
#include <cuda_runtime.h>
#include <cuntt/ntt.hpp>
#include <gpuntt/ntt_merge/ntt.cuh>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
constexpr const char* source_revision = "d03c5eaeadaa780d153496afcb3b6a9b79a13a63";

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::uint64_t validate(unsigned log_n, std::size_t batch, unsigned bits,
                       std::uint64_t modulus, int inverse) {
    if (log_n < 1 || log_n > 28) throw std::invalid_argument("GPU-NTT logN must be in [1,28]");
    if (!batch || batch > 65535) throw std::invalid_argument("GPU-NTT full batch must be in [1,65535]");
    if (bits != 32 && bits != 64) throw std::invalid_argument("GPU-NTT word bits must be 32 or 64");
    if (inverse != 0 && inverse != 1) throw std::invalid_argument("inverse must be 0 or 1");
    // The pinned GPU-NTT Barrett implementation is only exact for Data64
    // moduli below 2^61.  At 62-bit moduli its correction path can return a
    // value larger than q, so expose those cases as unsupported baselines.
    const auto limit = std::uint64_t{1} << (bits == 32 ? 30 : 61);
    if (modulus <= 2 || modulus >= limit) throw std::invalid_argument("modulus outside GPU-NTT arithmetic range");
    if (bits == 64 && Modulus<std::uint64_t>(modulus).bit > 61)
        throw std::invalid_argument("GPU-NTT floating log2 modulus metadata exceeds exact Data64 shift range");
    // These are the exact frozen cuButterfly helpers used by the crypto adapter.
    return cuntt::find_primitive_power_of_two_root(log_n, modulus);
}

template <class Function> int guarded(Function&& function) noexcept {
    try {
        last_error.clear();
        function();
        return 0;
    } catch (const std::invalid_argument& error) {
        last_error = error.what();
        return 1;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 2;
    } catch (...) {
        last_error = "unknown GPU-NTT bridge exception";
        return 3;
    }
}

template <typename Word> struct Buffer {
    Word* pointer = nullptr;
    explicit Buffer(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(Word))
            throw std::invalid_argument("GPU-NTT allocation size overflows size_t");
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(Word)), "cudaMalloc prepared buffer");
    }
    ~Buffer() { if (pointer) cudaFree(pointer); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

// Identical index mapping to the existing crypto adapter's bit_reverse_copy_kernel.
template <typename Word>
__global__ void bit_reverse_copy_kernel(const Word* source, Word* destination,
                                        std::size_t n, unsigned log_n,
                                        std::size_t count, bool source_is_bit_reversed) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::size_t batch = index >> log_n;
    const std::size_t coefficient = index & (n - 1);
    const std::size_t reversed = static_cast<std::size_t>(
        __brev(static_cast<unsigned>(coefficient)) >> (32U - log_n));
    if (source_is_bit_reversed)
        destination[batch * n + coefficient] = source[batch * n + reversed];
    else
        destination[batch * n + reversed] = source[index];
}

struct Base {
    int device = 0;
    virtual ~Base() = default;
    virtual void execute(void*, void*, cudaStream_t) = 0;
};

template <typename Word> struct Prepared : Base {
    unsigned log_n;
    std::size_t batch, n, count;
    bool inverse;
    Buffer<Word> work;
    Buffer<Root<Word>> roots;
    Buffer<Modulus<Word>> modulus;
    Buffer<Ninverse<Word>> n_inverse;
    unsigned permutation_grid;

    Prepared(unsigned log, std::size_t batches, std::uint64_t q, bool inv,
             std::uint64_t root, cudaStream_t stream)
        : log_n(log), batch(batches), n(std::size_t{1} << log), count(n * batch), inverse(inv),
          work(count), roots(n / 2), modulus(1), n_inverse(1) {
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        const auto grid = (count + 255) / 256;
        if (grid > static_cast<std::size_t>(properties.maxGridSize[0]))
            throw std::invalid_argument("full-batch bit reversal exceeds device grid.x; no batch reduction");
        permutation_grid = static_cast<unsigned>(grid);
        const auto selected_root = inverse ? cuntt::mod_pow(root, q - 2, q) : root;
        std::vector<Root<Word>> host_roots(n / 2);
        std::uint64_t power = 1;
        for (std::size_t index = 0; index < host_roots.size(); ++index) {
            std::size_t reversed = 0;
            for (unsigned bit = 0; bit < log_n - 1; ++bit)
                reversed = (reversed << 1) | ((index >> bit) & 1U);
            host_roots[reversed] = static_cast<Word>(power);
            power = static_cast<std::uint64_t>((static_cast<unsigned __int128>(power) * selected_root) % q);
        }
        Modulus<Word> host_modulus(static_cast<Word>(q));
        const auto host_inverse = static_cast<Ninverse<Word>>(cuntt::mod_pow(n, q - 2, q));
        check(cudaMemcpyAsync(roots.pointer, host_roots.data(), host_roots.size() * sizeof(Root<Word>),
                              cudaMemcpyHostToDevice, stream), "prepare roots");
        check(cudaMemcpyAsync(modulus.pointer, &host_modulus, sizeof(host_modulus),
                              cudaMemcpyHostToDevice, stream), "prepare modulus");
        check(cudaMemcpyAsync(n_inverse.pointer, &host_inverse, sizeof(host_inverse),
                              cudaMemcpyHostToDevice, stream), "prepare inverse-N");
        check(cudaStreamSynchronize(stream), "finish prepared copies");
    }

    void execute(void* input, void* output, cudaStream_t stream) override {
        int current_device = -1;
        check(cudaGetDevice(&current_device), "cudaGetDevice");
        if (current_device != device) throw std::invalid_argument("GPU-NTT plan device differs from caller device");
        if (!input || !output) throw std::invalid_argument("input and output device pointers are required");
        gpuntt::ntt_rns_configuration<Word> config{};
        config.n_power = static_cast<int>(log_n);
        config.ntt_type = inverse ? gpuntt::INVERSE : gpuntt::FORWARD;
        config.ntt_layout = gpuntt::PerPolynomial;
        config.reduction_poly = gpuntt::ReductionPolynomial::X_N_minus;
        config.zero_padding = false;
        config.mod_inverse = n_inverse.pointer;
        config.stream = stream;
        if (inverse) {
            bit_reverse_copy_kernel<Word><<<permutation_grid, 256, 0, stream>>>(
                static_cast<Word*>(input), work.pointer, n, log_n, count, false);
            check(cudaGetLastError(), "inverse input bit reversal");
            gpuntt::GPU_INTT<Word>(work.pointer, static_cast<Word*>(output), roots.pointer,
                                  modulus.pointer, config, static_cast<int>(batch), 1);
        } else {
            gpuntt::GPU_NTT<Word>(static_cast<Word*>(input), work.pointer, roots.pointer,
                                 modulus.pointer, config, static_cast<int>(batch), 1);
            check(cudaGetLastError(), "GPU-NTT forward");
            bit_reverse_copy_kernel<Word><<<permutation_grid, 256, 0, stream>>>(
                work.pointer, static_cast<Word*>(output), n, log_n, count, true);
        }
        check(cudaGetLastError(), "GPU-NTT complete call");
    }
};
}  // namespace

extern "C" {
int gpunttSemanticAbiVersion() { return 1; }
const char* gpunttSemanticSourceRevision() { return source_revision; }
const char* gpunttSemanticLastError() { return last_error.c_str(); }

int gpunttSemanticValidate(unsigned log_n, std::size_t batch, unsigned bits,
                           std::uint64_t modulus, int inverse, std::uint64_t* root) {
    return guarded([&] {
        if (!root) throw std::invalid_argument("root result pointer is required");
        *root = validate(log_n, batch, bits, modulus, inverse);
    });
}

int gpunttSemanticCreate(void** handle, unsigned log_n, std::size_t batch, unsigned bits,
                         std::uint64_t modulus, int inverse, void* caller_stream,
                         std::uint64_t* root, std::size_t* workspace_bytes) {
    return guarded([&] {
        if (!handle || !root || !workspace_bytes) throw std::invalid_argument("creation result pointers are required");
        *handle = nullptr;
        const auto selected_root = validate(log_n, batch, bits, modulus, inverse);
        const auto stream = static_cast<cudaStream_t>(caller_stream);
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(stream, &capture), "cudaStreamIsCapturing");
        if (capture != cudaStreamCaptureStatusNone) throw std::invalid_argument("create must precede CUDA graph capture");
        std::unique_ptr<Base> prepared;
        if (bits == 32) prepared = std::make_unique<Prepared<std::uint32_t>>(log_n, batch, modulus, inverse, selected_root, stream);
        else prepared = std::make_unique<Prepared<std::uint64_t>>(log_n, batch, modulus, inverse, selected_root, stream);
        *root = selected_root;
        *workspace_bytes = ((std::size_t{1} << log_n) * batch + (std::size_t{1} << (log_n - 1)) + 1) * (bits / 8)
                           + (bits == 32 ? sizeof(Modulus<std::uint32_t>) : sizeof(Modulus<std::uint64_t>));
        *handle = prepared.release();
    });
}

int gpunttSemanticExecute(void* handle, void* input, void* output, void* caller_stream) {
    return guarded([&] {
        if (!handle) throw std::invalid_argument("plan handle is required");
        static_cast<Base*>(handle)->execute(input, output, static_cast<cudaStream_t>(caller_stream));
    });
}

int gpunttSemanticDestroy(void* handle) {
    return guarded([&] {
        if (!handle) return;
        auto* prepared = static_cast<Base*>(handle);
        int device;
        check(cudaGetDevice(&device), "cudaGetDevice before destroy");
        if (device != prepared->device) throw std::invalid_argument("destroy must run on the plan device");
        delete prepared;
    });
}
}
