#include "xd_pair_core.hpp"
#include "xd_pair_fft.hpp"
#include "xd_cooperative_pair_fft.hpp"
#include "xd_stockham_pair_fft.hpp"

#include <stdexcept>

namespace cuntt::detail {

bool xd_pair_core_available(XdPairCore core, std::size_t rows,
                            std::size_t cols, bool fp64, unsigned layout) noexcept {
    switch (core) {
        case XdPairCore::Thread:
            return layout == 0 && xd_pair_fft_available(rows, cols, fp64);
        case XdPairCore::CooperativeRegister:
            return xd_cooperative_pair_fft_available(rows, cols, fp64, layout);
        case XdPairCore::Radix4Register: {
            if (layout != 0 || !xd_stockham_pair_fft_available(rows, cols, fp64))
                return false;
            int device = 0, threads = 0, shared = 0, warp = 0;
            return cudaGetDevice(&device) == cudaSuccess &&
                cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, device) == cudaSuccess &&
                cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlock, device) == cudaSuccess &&
                cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, device) == cudaSuccess &&
                threads >= 128 && shared >= 32 * 33 * (fp64 ? 16 : 8) && warp == 32;
        }
    }
    return false;
}

void prepare_xd_pair_core(XdPairCore core, std::size_t rows,
                         std::size_t cols, bool fp64, bool inverse, unsigned layout) {
    if (!xd_pair_core_available(core, rows, cols, fp64, layout))
        throw std::invalid_argument("requested xD pair core, shape, precision or layout is unavailable");
    if (core == XdPairCore::CooperativeRegister)
        prepare_xd_cooperative_pair_fft(rows, cols, fp64, inverse, layout);
}

void launch_xd_pair_core(XdPairCore core, const void* input, void* output,
                        std::size_t rows, std::size_t cols, std::size_t planes,
                        bool fp64, bool inverse, double scale,
                        cudaStream_t stream, unsigned layout) {
    // This dispatch adds no queries or allocations. The new radix-4 provider
    // completes capability checks during construction; the legacy cooperative
    // launcher still performs its own device validation before submission.
    switch (core) {
        case XdPairCore::Thread:
            if (layout) throw std::invalid_argument("Thread pair requires padded layout");
            launch_xd_pair_fft(input, output, rows, cols, planes, fp64, inverse, scale, stream);
            return;
        case XdPairCore::CooperativeRegister:
            launch_xd_cooperative_pair_fft(input, output, rows, cols, planes, fp64,
                                          inverse, scale, stream, layout);
            return;
        case XdPairCore::Radix4Register:
            if (layout) throw std::invalid_argument("radix-4 pair requires padded layout");
            launch_xd_stockham_pair_fft(input, output, rows, cols, planes, fp64,
                inverse, scale, stream, XdStagePairVariant::Radix4Register);
            return;
    }
    throw std::invalid_argument("unknown xD pair core");
}

const char* xd_pair_execution_name(XdPairCore core) {
    switch (core) {
        case XdPairCore::Thread: return "cufftdx-pair-tiled";
        case XdPairCore::CooperativeRegister: return "register-cooperative-pair";
        case XdPairCore::Radix4Register: return "radix4-register-pair";
    }
    throw std::invalid_argument("unknown xD pair core");
}

const char* xd_pair_twiddle_policy(XdPairCore core) {
    switch (core) {
        case XdPairCore::Thread: return "core-defined";
        case XdPairCore::CooperativeRegister: return "constant-roots-128";
        case XdPairCore::Radix4Register: return "constant-roots-32";
    }
    throw std::invalid_argument("unknown xD pair core");
}

unsigned xd_pair_cooperative_lanes(XdPairCore core) {
    switch (core) {
        case XdPairCore::Thread: return 0;  // Preserve existing serialization.
        case XdPairCore::CooperativeRegister: return 4;
        case XdPairCore::Radix4Register: return 1;
    }
    throw std::invalid_argument("unknown xD pair core");
}

}  // namespace cuntt::detail
