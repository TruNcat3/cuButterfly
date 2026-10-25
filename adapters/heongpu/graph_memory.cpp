#include "graph_memory.hpp"

#include <heongpu/util/memorypool.cuh>

#include <stdexcept>
#include <string>
#include <cstring>

extern "C" rmm::mr::device_memory_resource*
__real__ZNK7heongpu10MemoryPool19get_device_resourceEv(const heongpu::MemoryPool* pool);
extern "C" heongpu::MemoryPool& __real__ZN7heongpu10MemoryPool8instanceEv();
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t,
                                            cudaMemcpyKind, cudaStream_t);

namespace {
thread_local heongpu::MemoryPool* prepared_pool = nullptr;
thread_local bool arena_active = false;
}

extern "C" heongpu::MemoryPool& __wrap__ZN7heongpu10MemoryPool8instanceEv() {
    // Upstream instance() calls cudaFree(nullptr) on EVERY access. CUDA forbids
    // that during capture. Initialize eagerly, then return that same singleton
    // during record/capture. Both compared backends use the identical shim.
    if (arena_active) return *prepared_pool;
    return __real__ZN7heongpu10MemoryPool8instanceEv();
}

namespace cubutterfly_heongpu {

thread_local GraphMemory* GraphMemory::active_ = nullptr;

GraphMemory::GraphMemory(rmm::mr::device_memory_resource* upstream)
    : upstream_(upstream) {}

GraphMemory::~GraphMemory() {
    if (active_ == this) {
        active_ = nullptr;
        arena_active = false;
    }
    // The caller has already synchronized and retired all graph/output users.
    for (auto it = records_.rbegin(); it != records_.rend(); ++it)
        upstream_->deallocate(it->pointer, it->bytes, it->stream);
    for (auto it = host_copies_.rbegin(); it != host_copies_.rend(); ++it)
        if (it->snapshot) cudaFreeHost(it->snapshot);
}

void GraphMemory::recording() {
    if (!records_.empty() || !host_copies_.empty())
        throw std::logic_error("GraphMemory already has a recorded allocation/copy sequence");
    mode_ = Mode::Record;
    cursor_ = 0;
    host_cursor_ = 0;
}

void GraphMemory::replaying() {
    mode_ = Mode::Replay;
    cursor_ = 0;
    host_cursor_ = 0;
}

void GraphMemory::verify_replay_complete() {
    if (mode_ != Mode::Replay)
        throw std::logic_error("GraphMemory is not replaying");
    if (cursor_ != records_.size())
        throw std::runtime_error("GraphMemory replay consumed " + std::to_string(cursor_) +
                                 " of " + std::to_string(records_.size()) + " allocations");
    if (host_cursor_ != host_copies_.size())
        throw std::runtime_error("GraphMemory replay consumed " + std::to_string(host_cursor_) +
                                 " of " + std::to_string(host_copies_.size()) + " H2D copies");
}

std::size_t GraphMemory::allocations() const { return records_.size(); }
std::size_t GraphMemory::host_snapshots() const { return host_copies_.size(); }

const void* GraphMemory::retain_host_copy(const void* source, std::size_t bytes,
                                         cudaStream_t stream) {
    if (bytes && !source) throw std::invalid_argument("GraphMemory H2D source is null");
    if (mode_ == Mode::Replay) {
        if (host_cursor_ == host_copies_.size())
            throw std::runtime_error("GraphMemory replay requested an unrecorded H2D copy");
        const auto& copy = host_copies_[host_cursor_];
        if (bytes != copy.bytes || stream != copy.stream)
            throw std::runtime_error("GraphMemory H2D copy " + std::to_string(host_cursor_) +
                                     " size/stream mismatch");
        if (bytes && std::memcmp(source, copy.snapshot, bytes) != 0)
            throw std::runtime_error("GraphMemory H2D copy " + std::to_string(host_cursor_) +
                                     " host metadata changed since preparation");
        ++host_cursor_;
        return bytes ? copy.snapshot : source;
    }

    cudaStreamCaptureStatus capture;
    auto status = cudaStreamIsCapturing(stream, &capture);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("GraphMemory H2D capture query: ") + cudaGetErrorString(status));
    if (capture != cudaStreamCaptureStatusNone)
        throw std::runtime_error("GraphMemory cannot record new H2D snapshots during capture");
    void* snapshot = nullptr;
    if (bytes) {
        status = cudaHostAlloc(&snapshot, bytes, cudaHostAllocDefault);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("GraphMemory pinned H2D snapshot: ") + cudaGetErrorString(status));
        std::memcpy(snapshot, source, bytes);
    }
    try {
        host_copies_.push_back({snapshot, bytes, stream});
    } catch (...) {
        if (snapshot) cudaFreeHost(snapshot);
        throw;
    }
    return bytes ? snapshot : source;
}

void GraphMemory::activate(GraphMemory* arena) {
    if (arena) prepared_pool = &__real__ZN7heongpu10MemoryPool8instanceEv();
    arena_active = arena != nullptr;
    active_ = arena;
}

void* GraphMemory::do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) {
    if (mode_ == Mode::Replay) {
        if (cursor_ == records_.size())
            throw std::runtime_error("GraphMemory replay requested an unrecorded allocation");
        const auto& record = records_[cursor_];
        if (bytes != record.bytes)
            throw std::runtime_error("GraphMemory allocation " + std::to_string(cursor_) +
                                     " size mismatch: recorded " + std::to_string(record.bytes) +
                                     ", replay requested " + std::to_string(bytes));
        ++cursor_;
        return record.pointer;
    }

    if (!upstream_)
        upstream_ = __real__ZNK7heongpu10MemoryPool19get_device_resourceEv(
            &heongpu::MemoryPool::instance());
    if (!upstream_ || upstream_ == this)
        throw std::logic_error("GraphMemory needs an independent upstream resource");
    void* pointer = upstream_->allocate(bytes, stream);
    try {
        records_.push_back({pointer, bytes, stream});
    } catch (...) {
        upstream_->deallocate(pointer, bytes, stream);
        throw;
    }
    return pointer;
}

void GraphMemory::do_deallocate(void*, std::size_t, rmm::cuda_stream_view) {
    // Both preparation and capture destroy temporary DeviceVectors. Returning
    // their storage upstream here would invalidate captured scratch addresses.
}

bool GraphMemory::do_is_equal(const rmm::mr::device_memory_resource& other) const noexcept {
    return this == &other;
}

} // namespace cubutterfly_heongpu

extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* destination, const void* source,
                                            std::size_t bytes, cudaMemcpyKind kind,
                                            cudaStream_t stream) {
    auto* arena = cubutterfly_heongpu::GraphMemory::active_;
    if (arena && kind == cudaMemcpyHostToDevice)
        source = arena->retain_host_copy(source, bytes, stream);
    return __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
}

extern "C" rmm::mr::device_memory_resource*
__wrap__ZNK7heongpu10MemoryPool19get_device_resourceEv(const heongpu::MemoryPool* pool) {
    auto* arena = cubutterfly_heongpu::GraphMemory::active_;
    if (!arena) return __real__ZNK7heongpu10MemoryPool19get_device_resourceEv(pool);
    if (!arena->upstream_)
        arena->upstream_ = __real__ZNK7heongpu10MemoryPool19get_device_resourceEv(pool);
    return arena;
}
