#include "general_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuntt::detail {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct alignas(16) Bytes16 { std::uint64_t lo; std::uint64_t hi; };

struct DevicePermutation {
    std::uint32_t rank = 0;
    std::uint32_t axis[kGeneralMaxRank]{};
};

struct DeviceLogicalOrder {
    std::uint32_t bits[kGeneralMaxRank]{};
};

template <typename T, bool Gather>
__global__ void ordered_boundary_kernel(const T* input, T* output,
                                        GeneralNdLayout logical, DeviceLogicalOrder order) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= logical.points * logical.batch)
        return;
    std::size_t within = linear % logical.points;
    std::size_t physical = (linear / logical.points) * logical.batch_stride;
    for (std::uint32_t axis = logical.rank; axis-- > 0;) {
        std::size_t coordinate = within % logical.extents[axis];
        within /= logical.extents[axis];
        if (order.bits[axis] != 0) {
            std::size_t reversed = 0;
            for (std::uint32_t bit = 0; bit < order.bits[axis]; ++bit) {
                reversed = (reversed << 1) | (coordinate & 1);
                coordinate >>= 1;
            }
            coordinate = reversed;
        }
        physical += coordinate * logical.strides[axis];
    }
    if constexpr (Gather)
        output[linear] = input[physical];
    else
        output[physical] = input[linear];
}

// These rank-3 launch records keep all coordinate mapping decisions on the
// host.  In particular, the kernels must not index a runtime-sized stride
// array: doing so makes nvcc place the array in local memory and destroys the
// coalesced access pattern the tiled path is intended to provide.
struct Rank3TiledParams {
    std::size_t fixed_extent = 0;
    std::size_t input_batch_stride = 0;
    std::size_t input_fixed_stride = 0;
    std::size_t input_row_extent = 0;
    std::size_t input_col_extent = 0;
    std::size_t input_row_stride = 0;
    std::size_t input_col_stride = 0;
    std::size_t output_batch_stride = 0;
    std::size_t output_fixed_stride = 0;
    std::size_t output_row_stride = 0;
    std::size_t output_col_stride = 0;
};

struct Rank3SameInnerParams {
    std::size_t output_extent0 = 0;
    std::size_t output_extent1 = 0;
    std::size_t inner_extent = 0;
    std::size_t input_batch_stride = 0;
    std::size_t input_outer_stride0 = 0;
    std::size_t input_outer_stride1 = 0;
    std::size_t input_inner_stride = 0;
    std::size_t output_batch_stride = 0;
};

template <typename T>
__global__ void pack_1d_kernel(const T* input, T* packed, std::size_t logical_n,
                               std::size_t physical_n, std::size_t batch,
                               std::size_t input_stride, std::size_t input_batch_stride) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = logical_n * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / logical_n;
    const std::size_t index = linear - transform * logical_n;
    packed[transform * physical_n + index] = input[transform * input_batch_stride + index * input_stride];
}

template <typename T>
__global__ void scatter_1d_kernel(const T* packed, T* output, std::size_t output_n,
                                  std::size_t physical_n, std::size_t batch,
                                  std::size_t output_stride, std::size_t output_batch_stride) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = output_n * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / output_n;
    const std::size_t index = linear - transform * output_n;
    output[transform * output_batch_stride + index * output_stride] = packed[transform * physical_n + index];
}

template <typename T>
__global__ void pack_2d_kernel(const T* input, T* packed,
                               std::size_t logical_rows, std::size_t logical_cols,
                               std::size_t physical_rows, std::size_t physical_cols,
                               std::size_t batch, std::size_t input_row_stride,
                               std::size_t input_col_stride, std::size_t input_batch_stride) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t matrix_points = logical_rows * logical_cols;
    const std::size_t count = matrix_points * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / matrix_points;
    const std::size_t within = linear - transform * matrix_points;
    const std::size_t row = within / logical_cols;
    const std::size_t col = within - row * logical_cols;
    packed[(transform * physical_rows + row) * physical_cols + col] =
        input[transform * input_batch_stride + row * input_row_stride + col * input_col_stride];
}

template <typename T>
__global__ void scatter_2d_kernel(const T* packed, T* output,
                                  std::size_t output_rows, std::size_t output_cols,
                                  std::size_t physical_rows, std::size_t physical_cols,
                                  std::size_t batch, std::size_t output_row_stride,
                                  std::size_t output_col_stride, std::size_t output_batch_stride) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t matrix_points = output_rows * output_cols;
    const std::size_t count = matrix_points * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / matrix_points;
    const std::size_t within = linear - transform * matrix_points;
    const std::size_t row = within / output_cols;
    const std::size_t col = within - row * output_cols;
    output[transform * output_batch_stride + row * output_row_stride + col * output_col_stride] =
        packed[(transform * physical_rows + row) * physical_cols + col];
}

template <typename T>
__global__ void transpose_2d_kernel(const T* input, T* output,
                                    std::size_t rows, std::size_t cols, std::size_t batch) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t matrix_points = rows * cols;
    const std::size_t count = matrix_points * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / matrix_points;
    const std::size_t within = linear - transform * matrix_points;
    const std::size_t row = within / cols;
    const std::size_t col = within - row * cols;
    output[(transform * cols + col) * rows + row] = input[linear];
}

// The generic permutation kernel below is intentionally scalar because it
// must support arbitrary ranks and strides.  Axis exchange is much more
// common in the xD lowering, however, and its scalar coordinate decode turns
// a perfectly regular transpose into a collection of uncoalesced loads and
// stores.  Use the standard shared-memory tiled transpose for contiguous
// rank-2 layouts.  The extra column avoids shared-memory bank conflicts while
// retaining the same logical layout contract as transpose_2d_kernel.
template <typename T, unsigned Tile = 32, unsigned BlockRows = 8>
__global__ void tiled_transpose_2d_kernel(const T* input, T* output,
                                          std::size_t rows, std::size_t cols) {
    __shared__ T tile[Tile][Tile + 1];
    const std::size_t batch = static_cast<std::size_t>(blockIdx.z);
    const std::size_t input_x = static_cast<std::size_t>(blockIdx.x) * Tile + threadIdx.x;
    const std::size_t input_y = static_cast<std::size_t>(blockIdx.y) * Tile + threadIdx.y;
    const std::size_t input_base = batch * rows * cols;

    for (unsigned offset = 0; offset < Tile; offset += BlockRows) {
        const std::size_t y = input_y + offset;
        if (input_x < cols && y < rows)
            tile[threadIdx.y + offset][threadIdx.x] = input[input_base + y * cols + input_x];
    }
    __syncthreads();

    const std::size_t output_x = static_cast<std::size_t>(blockIdx.y) * Tile + threadIdx.x;
    const std::size_t output_y = static_cast<std::size_t>(blockIdx.x) * Tile + threadIdx.y;
    const std::size_t output_base = batch * rows * cols;
    for (unsigned offset = 0; offset < Tile; offset += BlockRows) {
        const std::size_t y = output_y + offset;
        if (output_x < rows && y < cols)
            output[output_base + y * rows + output_x] = tile[threadIdx.x][threadIdx.y + offset];
    }
}

// Rank-3 packed permutations have one useful common shape: the input's
// physical innermost axis (axis 2) is contiguous, while the output's
// innermost axis is the input axis named by permutation.axis[2].  For the
// four permutations where those axes differ, each fixed (batch, remaining
// axis) slab is a matrix with input rows along output-axis permutation.axis[2]
// and input columns along axis 2.  Load that matrix coalesced, transpose it in
// shared memory, and store rows coalesced in the output order.  Strides are
// carried through for the input so a padded leading dimension remains valid.
template <typename T, unsigned Tile = 32, unsigned BlockRows = 8>
__global__ void tiled_permute_rank3_kernel(const T* input, T* output,
                                           Rank3TiledParams params) {
    __shared__ T tile[Tile][Tile + 1];

    const std::size_t slab = static_cast<std::size_t>(blockIdx.z);
    const std::size_t batch = slab / params.fixed_extent;
    const std::size_t fixed = slab - batch * params.fixed_extent;

    const std::size_t input_row =
        static_cast<std::size_t>(blockIdx.y) * Tile + threadIdx.y;
    const std::size_t input_col =
        static_cast<std::size_t>(blockIdx.x) * Tile + threadIdx.x;
    const std::size_t input_base = batch * params.input_batch_stride +
                                   fixed * params.input_fixed_stride;

    for (unsigned offset = 0; offset < Tile; offset += BlockRows) {
        const std::size_t row = input_row + offset;
        if (row < params.input_row_extent && input_col < params.input_col_extent) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[input_base + row * params.input_row_stride +
                      input_col * params.input_col_stride];
        }
    }
    __syncthreads();

    const std::size_t output_col =
        static_cast<std::size_t>(blockIdx.y) * Tile + threadIdx.x;
    const std::size_t output_base = batch * params.output_batch_stride +
                                    fixed * params.output_fixed_stride;
    for (unsigned offset = 0; offset < Tile; offset += BlockRows) {
        const std::size_t output_row =
            static_cast<std::size_t>(blockIdx.x) * Tile + threadIdx.y + offset;
        if (output_row < params.input_col_extent &&
            output_col < params.input_row_extent) {
            output[output_base + output_row * params.output_row_stride +
                   output_col * params.output_col_stride] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

// If the input and output retain the same innermost axis, no shared-memory
// transpose is needed.  One block covers a small contiguous run of that axis
// for each output row; decoding the two outer coordinates handles the one
// nontrivial rank-3 permutation (the outer-axis exchange) with coalesced loads
// and stores.
template <typename T, unsigned Tile = 32, unsigned BlockRows = 8>
__global__ void permute_rank3_same_inner_kernel(const T* input, T* output,
                                                Rank3SameInnerParams params) {
    const std::size_t output_rows = params.output_extent0 * params.output_extent1;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * BlockRows + threadIdx.y;
    const std::size_t inner = static_cast<std::size_t>(blockIdx.x) * Tile + threadIdx.x;
    if (row >= output_rows || inner >= params.inner_extent)
        return;

    const std::size_t output_coord0 = row / params.output_extent1;
    const std::size_t output_coord1 = row - output_coord0 * params.output_extent1;
    const std::size_t input_index =
        static_cast<std::size_t>(blockIdx.z) * params.input_batch_stride +
        output_coord0 * params.input_outer_stride0 +
        output_coord1 * params.input_outer_stride1 +
        inner * params.input_inner_stride;
    const std::size_t output_index =
        static_cast<std::size_t>(blockIdx.z) * params.output_batch_stride +
        row * params.inner_extent + inner;
    output[output_index] = input[input_index];
}

template <typename T>
__global__ void pack_nd_kernel(const T* input, T* packed,
                               GeneralNdLayout logical, GeneralNdLayout physical) {
    const std::size_t logical_points = logical.points;
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = logical_points * logical.batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / logical_points;
    std::size_t within = linear - transform * logical_points;
    std::size_t packed_index = transform * physical.batch_stride;
    std::size_t input_index = transform * logical.batch_stride;
    for (std::uint32_t axis = logical.rank; axis-- > 0;) {
        const std::size_t coordinate = within % logical.extents[axis];
        within /= logical.extents[axis];
        packed_index += coordinate * physical.strides[axis];
        input_index += coordinate * logical.strides[axis];
    }
    packed[packed_index] = input[input_index];
}

template <typename T>
__global__ void scatter_nd_kernel(const T* packed, T* output,
                                  GeneralNdLayout physical, GeneralNdLayout logical) {
    const std::size_t logical_points = logical.points;
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = logical_points * logical.batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / logical_points;
    std::size_t within = linear - transform * logical_points;
    std::size_t packed_index = transform * physical.batch_stride;
    std::size_t output_index = transform * logical.batch_stride;
    for (std::uint32_t axis = logical.rank; axis-- > 0;) {
        const std::size_t coordinate = within % logical.extents[axis];
        within /= logical.extents[axis];
        packed_index += coordinate * physical.strides[axis];
        output_index += coordinate * logical.strides[axis];
    }
    output[output_index] = packed[packed_index];
}

template <typename T>
__global__ void permute_nd_kernel(const T* input, T* output,
                                  GeneralNdLayout input_layout,
                                  DevicePermutation permutation) {
    std::size_t points = 1;
    for (std::uint32_t axis = 0; axis < permutation.rank; ++axis)
        points *= input_layout.extents[permutation.axis[axis]];
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = points * input_layout.batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / points;
    std::size_t within = linear - transform * points;
    std::size_t input_index = transform * input_layout.batch_stride;
    std::size_t output_index = transform * points;
    std::size_t output_stride = points;
    // Decode output coordinates from the slowest axis to the fastest axis.
    for (std::uint32_t out_axis = 0; out_axis < permutation.rank; ++out_axis) {
        output_stride = 1;
        for (std::uint32_t tail = out_axis + 1; tail < permutation.rank; ++tail)
            output_stride *= input_layout.extents[permutation.axis[tail]];
        const std::size_t coordinate = within / output_stride;
        within -= coordinate * output_stride;
        input_index += coordinate * input_layout.strides[permutation.axis[out_axis]];
    }
    output[output_index + (linear - transform * points)] = input[input_index];
}

template <typename Function>
void dispatch_size(std::size_t bytes, Function&& function) {
    switch (bytes) {
        case 2: function(static_cast<std::uint16_t*>(nullptr)); return;
        case 4: function(static_cast<std::uint32_t*>(nullptr)); return;
        case 8: function(static_cast<std::uint64_t*>(nullptr)); return;
        case 16: function(static_cast<Bytes16*>(nullptr)); return;
        default: throw std::invalid_argument("general transform element size must be 2, 4, 8, or 16 bytes");
    }
}

unsigned int blocks_for(std::size_t count) {
    constexpr std::size_t threads = 256;
    const std::size_t blocks = (count + threads - 1) / threads;
    if (blocks > static_cast<std::size_t>(UINT32_MAX))
        throw std::overflow_error("general transform copy grid exceeds CUDA grid.x");
    return static_cast<unsigned int>(blocks);
}

// CUDA guarantees a larger x dimension than y/z, but the portable minimum
// for y and z remains 65535.  Keep these checks on the host so oversized
// shapes take the scalar fallback instead of producing an invalid launch.
constexpr std::size_t kPortableGridX = 2147483647ULL;
constexpr std::size_t kPortableGridYZ = 65535;

bool tiled_grid_dimension_fits(std::size_t extent, std::size_t tile,
                               std::size_t limit) {
    if (tile == 0 || limit == 0)
        return false;
    const std::size_t quotient = extent / tile;
    return quotient < limit || (quotient == limit && extent % tile == 0);
}

}  // namespace

void launch_pack_1d(const void* input, void* packed, std::size_t logical_n, std::size_t physical_n,
                    std::size_t batch, std::size_t input_stride, std::size_t input_batch_stride,
                    std::size_t bytes, cudaStream_t stream) {
    check_cuda(cudaMemsetAsync(packed, 0, physical_n * batch * bytes, stream), "clear packed rank-1 buffer");
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        pack_1d_kernel<<<blocks_for(logical_n * batch), 256, 0, stream>>>(
            static_cast<const T*>(input), static_cast<T*>(packed), logical_n, physical_n,
            batch, input_stride, input_batch_stride);
    });
    check_cuda(cudaGetLastError(), "launch rank-1 pack");
}

void launch_scatter_1d(const void* packed, void* output, std::size_t output_n, std::size_t physical_n,
                       std::size_t batch, std::size_t output_stride, std::size_t output_batch_stride,
                       std::size_t bytes, cudaStream_t stream) {
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        scatter_1d_kernel<<<blocks_for(output_n * batch), 256, 0, stream>>>(
            static_cast<const T*>(packed), static_cast<T*>(output), output_n, physical_n,
            batch, output_stride, output_batch_stride);
    });
    check_cuda(cudaGetLastError(), "launch rank-1 scatter");
}

void launch_pack_2d(const void* input, void* packed, std::size_t logical_rows, std::size_t logical_cols,
                    std::size_t physical_rows, std::size_t physical_cols, std::size_t batch,
                    std::size_t input_row_stride, std::size_t input_col_stride,
                    std::size_t input_batch_stride, std::size_t bytes, cudaStream_t stream) {
    check_cuda(cudaMemsetAsync(packed, 0, physical_rows * physical_cols * batch * bytes, stream),
               "clear packed rank-2 buffer");
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        pack_2d_kernel<<<blocks_for(logical_rows * logical_cols * batch), 256, 0, stream>>>(
            static_cast<const T*>(input), static_cast<T*>(packed), logical_rows, logical_cols,
            physical_rows, physical_cols, batch, input_row_stride, input_col_stride, input_batch_stride);
    });
    check_cuda(cudaGetLastError(), "launch rank-2 pack");
}

void launch_scatter_2d(const void* packed, void* output, std::size_t output_rows, std::size_t output_cols,
                       std::size_t physical_rows, std::size_t physical_cols, std::size_t batch,
                       std::size_t output_row_stride, std::size_t output_col_stride,
                       std::size_t output_batch_stride, std::size_t bytes, cudaStream_t stream) {
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        scatter_2d_kernel<<<blocks_for(output_rows * output_cols * batch), 256, 0, stream>>>(
            static_cast<const T*>(packed), static_cast<T*>(output), output_rows, output_cols,
            physical_rows, physical_cols, batch, output_row_stride, output_col_stride, output_batch_stride);
    });
    check_cuda(cudaGetLastError(), "launch rank-2 scatter");
}

void launch_transpose_2d(const void* input, void* output, std::size_t rows, std::size_t cols,
                         std::size_t batch, std::size_t bytes, cudaStream_t stream) {
    // A three-dimensional launch keeps each batch independent and lets the
    // tile retain coalesced global accesses.  65535 is the portable minimum
    // grid.z limit for CUDA devices supported by this project.  Avoid querying
    // device properties on every execute: this function is on the hot path and
    // the general scalar implementation remains available for larger batches.
    constexpr std::size_t kPortableTiledBatchLimit = 65535;
    const bool tiled = batch != 0 && batch <= kPortableTiledBatchLimit;
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        if (tiled) {
            constexpr unsigned tile = 32;
            const dim3 block(tile, 8, 1);
            const dim3 grid(static_cast<unsigned>((cols + tile - 1) / tile),
                            static_cast<unsigned>((rows + tile - 1) / tile),
                            static_cast<unsigned>(batch));
            tiled_transpose_2d_kernel<T><<<grid, block, 0, stream>>>(
                static_cast<const T*>(input), static_cast<T*>(output), rows, cols);
        } else {
            transpose_2d_kernel<<<blocks_for(rows * cols * batch), 256, 0, stream>>>(
                static_cast<const T*>(input), static_cast<T*>(output), rows, cols, batch);
        }
    });
    check_cuda(cudaGetLastError(), "launch rank-2 transpose");
}

template <typename Function>
void dispatch_nd_size(std::size_t points, std::size_t batch, Function&& function) {
    if (points == 0 || batch == 0 || points > static_cast<std::size_t>(-1) / batch)
        throw std::overflow_error("general transform rank-N point count overflows size_t");
    function(points * batch);
}

namespace {

std::size_t checked_order_product(std::size_t a, std::size_t b) {
    if (b != 0 && a > static_cast<std::size_t>(-1) / b)
        throw std::overflow_error("ordered boundary size overflows size_t");
    return a * b;
}

template <bool Gather>
void launch_ordered_boundary(const void* input, void* output,
                             const GeneralNdLayout& logical,
                             const std::uint32_t* flags,
                             std::size_t bytes, cudaStream_t stream) {
    if (logical.rank == 0 || logical.rank > kGeneralMaxRank || flags == nullptr)
        throw std::invalid_argument("unsupported ordered boundary rank or order array");
    if (input == nullptr || output == nullptr || input == output)
        throw std::invalid_argument("ordered boundary requires distinct nonnull buffers");
    DeviceLogicalOrder order;
    std::size_t points = 1;
    std::size_t last = 0;
    for (std::uint32_t axis = 0; axis < logical.rank; ++axis) {
        const auto extent = logical.extents[axis];
        if (extent == 0 || flags[axis] > 1 || (extent > 1 && logical.strides[axis] == 0))
            throw std::invalid_argument("invalid ordered boundary extent, stride or order");
        if (flags[axis] != 0) {
            if ((extent & (extent - 1)) != 0)
                throw std::invalid_argument("bit-reversed ordered boundary extent must be power of two");
            for (auto n = extent; n > 1; n >>= 1)
                ++order.bits[axis];
        }
        points = checked_order_product(points, extent);
        const auto offset = checked_order_product(extent - 1, logical.strides[axis]);
        if (offset > static_cast<std::size_t>(-1) - last)
            throw std::overflow_error("ordered boundary address overflows size_t");
        last += offset;
    }
    if (points != logical.points || logical.batch == 0 ||
        (logical.batch > 1 && logical.batch_stride == 0))
        throw std::invalid_argument("invalid ordered boundary points or batch");
    const auto batch_offset = checked_order_product(logical.batch - 1, logical.batch_stride);
    if (batch_offset >= static_cast<std::size_t>(-1) - last)
        throw std::overflow_error("ordered boundary address overflows size_t");
    checked_order_product(batch_offset + last + 1, bytes);
    const auto count = checked_order_product(points, logical.batch);
    checked_order_product(count, bytes);
    const auto blocks = count / 256 + (count % 256 != 0);
    if (blocks > kPortableGridX)
        throw std::overflow_error("ordered boundary copy grid exceeds CUDA grid.x");
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        ordered_boundary_kernel<T, Gather><<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
            static_cast<const T*>(input), static_cast<T*>(output), logical, order);
    });
    check_cuda(cudaGetLastError(), "launch ordered rank-N boundary");
}

}  // namespace

void launch_gather_ordered_nd(const void* input, void* packed,
                              const GeneralNdLayout& logical, const std::uint32_t* flags,
                              std::size_t bytes, cudaStream_t stream) {
    launch_ordered_boundary<true>(input, packed, logical, flags, bytes, stream);
}

void launch_scatter_ordered_nd(const void* packed, void* output,
                               const GeneralNdLayout& logical, const std::uint32_t* flags,
                               std::size_t bytes, cudaStream_t stream) {
    launch_ordered_boundary<false>(packed, output, logical, flags, bytes, stream);
}

void launch_pack_nd(const void* input, void* packed,
                    const GeneralNdLayout& logical,
                    const GeneralNdLayout& physical,
                    std::size_t bytes, cudaStream_t stream) {
    if (logical.rank == 0 || logical.rank > kGeneralMaxRank || physical.rank != logical.rank)
        throw std::invalid_argument("unsupported rank-N pack rank");
    check_cuda(cudaMemsetAsync(packed, 0, physical.batch_stride * physical.batch * bytes, stream),
               "clear packed rank-N buffer");
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        dispatch_nd_size(logical.points, logical.batch, [&](std::size_t count) {
            pack_nd_kernel<<<blocks_for(count), 256, 0, stream>>>(
                static_cast<const T*>(input), static_cast<T*>(packed), logical, physical);
        });
    });
    check_cuda(cudaGetLastError(), "launch rank-N pack");
}

void launch_scatter_nd(const void* packed, void* output,
                       const GeneralNdLayout& physical,
                       const GeneralNdLayout& logical,
                       std::size_t bytes, cudaStream_t stream) {
    if (logical.rank == 0 || logical.rank > kGeneralMaxRank || physical.rank != logical.rank)
        throw std::invalid_argument("unsupported rank-N scatter rank");
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        dispatch_nd_size(logical.points, logical.batch, [&](std::size_t count) {
            scatter_nd_kernel<<<blocks_for(count), 256, 0, stream>>>(
                static_cast<const T*>(packed), static_cast<T*>(output), physical, logical);
        });
    });
    check_cuda(cudaGetLastError(), "launch rank-N scatter");
}

void launch_permute_nd(const void* input, void* output,
                       const GeneralNdLayout& input_layout,
                       const std::uint32_t* permutation,
                       std::size_t bytes, cudaStream_t stream) {
    if (input_layout.rank == 0 || input_layout.rank > kGeneralMaxRank || permutation == nullptr)
        throw std::invalid_argument("unsupported rank-N permutation");
    DevicePermutation perm;
    perm.rank = input_layout.rank;
    bool seen[kGeneralMaxRank]{};
    for (std::uint32_t axis = 0; axis < perm.rank; ++axis) {
        if (permutation[axis] >= perm.rank || seen[permutation[axis]])
            throw std::invalid_argument("rank-N permutation is not bijective");
        perm.axis[axis] = permutation[axis];
        seen[permutation[axis]] = true;
    }
    // Packed xD axis lowering represents a rank-2 axis exchange as a
    // contiguous matrix transpose.  Route that case through the tiled
    // implementation so the optimization also applies to packed real FFT
    // and arbitrary-stride plans after their layout boundary.  Non-contiguous
    // rank-2 layouts, unsupported rank-3 shapes, and other ranks continue to
    // use the fully general kernel.
    const bool contiguous_rank2 =
        perm.rank == 2 && input_layout.strides[1] == 1 &&
        input_layout.strides[0] == input_layout.extents[1] &&
        input_layout.batch_stride == input_layout.extents[0] * input_layout.extents[1] &&
        perm.axis[0] == 1 && perm.axis[1] == 0;
    if (contiguous_rank2) {
        launch_transpose_2d(input, output, input_layout.extents[0], input_layout.extents[1],
                            input_layout.batch, bytes, stream);
        return;
    }
    std::size_t points = 1;
    for (std::uint32_t axis = 0; axis < perm.rank; ++axis)
        points *= input_layout.extents[perm.axis[axis]];
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        // Packed rank-3 tensors have a contiguous input axis 2.  If the
        // output's fastest axis differs, the tiled kernel exchanges those
        // two axes for each independent (batch, remaining-axis) slab.  The
        // slab count is kept within portable grid.z limits; oversized cases
        // deliberately retain the generic rank-N implementation below.
        const bool rank3_contiguous_inner = perm.rank == 3 && input_layout.strides[2] == 1;
        if (rank3_contiguous_inner && perm.axis[2] != 2) {
            std::uint32_t fixed_axis = 0;
            while (fixed_axis == 2 || fixed_axis == perm.axis[2])
                ++fixed_axis;
            const std::size_t fixed_extent = input_layout.extents[fixed_axis];
            const bool slab_grid_supported =
                fixed_extent != 0 && fixed_extent <= kPortableGridYZ &&
                input_layout.batch != 0 &&
                input_layout.batch <= kPortableGridYZ / fixed_extent &&
                tiled_grid_dimension_fits(input_layout.extents[2], 32, kPortableGridX) &&
                tiled_grid_dimension_fits(input_layout.extents[perm.axis[2]], 32,
                                          kPortableGridYZ);
            if (slab_grid_supported) {
                dispatch_nd_size(points, input_layout.batch, [&](std::size_t) {
                    constexpr unsigned tile = 32;
                    constexpr unsigned block_rows = 8;
                    std::size_t output_strides[3]{};
                    std::size_t output_stride = 1;
                    for (int output_axis = 2; output_axis >= 0; --output_axis) {
                        output_strides[perm.axis[output_axis]] = output_stride;
                        output_stride *= input_layout.extents[perm.axis[output_axis]];
                    }
                    Rank3TiledParams params;
                    params.fixed_extent = fixed_extent;
                    params.input_batch_stride = input_layout.batch_stride;
                    params.input_fixed_stride = input_layout.strides[fixed_axis];
                    params.input_row_extent = input_layout.extents[perm.axis[2]];
                    params.input_col_extent = input_layout.extents[2];
                    params.input_row_stride = input_layout.strides[perm.axis[2]];
                    params.input_col_stride = input_layout.strides[2];
                    params.output_batch_stride = points;
                    params.output_fixed_stride = output_strides[fixed_axis];
                    params.output_row_stride = output_strides[2];
                    params.output_col_stride = output_strides[perm.axis[2]];
                    const dim3 block(tile, block_rows, 1);
                    const dim3 grid(
                        static_cast<unsigned>((input_layout.extents[2] + tile - 1) / tile),
                        static_cast<unsigned>((input_layout.extents[perm.axis[2]] + tile - 1) / tile),
                        static_cast<unsigned>(input_layout.batch * fixed_extent));
                    tiled_permute_rank3_kernel<T><<<grid, block, 0, stream>>>(
                        static_cast<const T*>(input), static_cast<T*>(output), params);
                });
                return;
            }
        }

        // Swapping only the two outer axes retains a contiguous innermost
        // axis.  Copy each output row through a 2-D tiled launch so both the
        // input and output accesses remain coalesced.  Identity also uses
        // this path; its mapping is the same and remains safe row by row.
        // The existing permutation contract is out-of-place for reordered
        // rows; unsupported layouts still use generic decode.
        const bool same_inner_output_rows_valid =
            input_layout.extents[perm.axis[1]] != 0 &&
            input_layout.extents[perm.axis[0]] <=
                static_cast<std::size_t>(-1) / input_layout.extents[perm.axis[1]];
        const std::size_t same_inner_output_rows = same_inner_output_rows_valid
                                                       ? input_layout.extents[perm.axis[0]] *
                                                             input_layout.extents[perm.axis[1]]
                                                       : 0;
        const bool rank3_same_inner =
            rank3_contiguous_inner && perm.axis[2] == 2 && input_layout.batch != 0 &&
            input_layout.batch <= kPortableGridYZ &&
            tiled_grid_dimension_fits(input_layout.extents[2], 32, kPortableGridX) &&
            same_inner_output_rows_valid &&
            tiled_grid_dimension_fits(same_inner_output_rows, 8, kPortableGridYZ);
        if (rank3_same_inner) {
            dispatch_nd_size(points, input_layout.batch, [&](std::size_t) {
                constexpr unsigned tile = 32;
                constexpr unsigned block_rows = 8;
                Rank3SameInnerParams params;
                params.output_extent0 = input_layout.extents[perm.axis[0]];
                params.output_extent1 = input_layout.extents[perm.axis[1]];
                params.inner_extent = input_layout.extents[2];
                params.input_batch_stride = input_layout.batch_stride;
                params.input_outer_stride0 = input_layout.strides[perm.axis[0]];
                params.input_outer_stride1 = input_layout.strides[perm.axis[1]];
                params.input_inner_stride = input_layout.strides[2];
                params.output_batch_stride = points;
                const dim3 block(tile, block_rows, 1);
                const dim3 grid(
                    static_cast<unsigned>((input_layout.extents[2] + tile - 1) / tile),
                    static_cast<unsigned>((same_inner_output_rows + block_rows - 1) / block_rows),
                    static_cast<unsigned>(input_layout.batch));
                permute_rank3_same_inner_kernel<T><<<grid, block, 0, stream>>>(
                    static_cast<const T*>(input), static_cast<T*>(output), params);
            });
            return;
        }

        dispatch_nd_size(points, input_layout.batch, [&](std::size_t count) {
            permute_nd_kernel<<<blocks_for(count), 256, 0, stream>>>(
                static_cast<const T*>(input), static_cast<T*>(output), input_layout, perm);
        });
    });
    check_cuda(cudaGetLastError(), "launch rank-N permutation");
}

}  // namespace cuntt::detail
