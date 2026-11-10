#pragma once

#include <rmm/mr/device/device_memory_resource.hpp>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

namespace heongpu {
class MemoryPool;
}

extern "C" rmm::mr::device_memory_resource*
__wrap__ZNK7heongpu10MemoryPool19get_device_resourceEv(const heongpu::MemoryPool* pool);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* destination, const void* source,
                                            std::size_t bytes, cudaMemcpyKind kind,
                                            cudaStream_t stream);

namespace cubutterfly_heongpu {

// Link with --wrap=_ZNK7heongpu10MemoryPool19get_device_resourceEv and
// --wrap=_ZN7heongpu10MemoryPool8instanceEv. The second hook returns the already
// initialized singleton, avoiding upstream's capture-invalid cudaFree(nullptr).
// Also link --wrap=cudaMemcpyAsync. H2D copies made while active retain pinned
// snapshots of CPU-readable host metadata. Metadata must be stable at submission
// and identical between preparation and capture. Device/managed sources and
// changing application inputs are outside this metadata snapshot contract.
// This applies to host ciphertext sources too: changed bytes are rejected,
// rather than silently replaced by old ciphertext. Keep changing inputs in
// device buffers updated outside the captured call sequence. Replays of a
// captured graph keep using its prepared metadata until the arena is destroyed.
// Use one arena per captured public-call sequence, on one host thread/device:
//   activate(&arena); recording(); run eagerly; synchronize; destroy outputs;
//   replaying(); begin capture; run the same calls; end capture;
//   verify_replay_complete(); activate(nullptr);
// On any exception, deactivate the arena and end/retire an active capture before
// unwinding its resources. The enclosing public-call harness owns that cleanup.
// Keep the arena, upstream resource, and recorded streams alive for all graph
// replays and output use. Before destroying the arena, synchronize, destroy the
// graph/graph executable, and destroy every ciphertext allocated through it.
// Clear capture outputs before starting another allocation-sequence replay.
// Deallocation only retires a logical allocation; scratch remains owned here.
class GraphMemory final : public rmm::mr::device_memory_resource {
public:
    explicit GraphMemory(rmm::mr::device_memory_resource* upstream = nullptr);
    ~GraphMemory() override;

    GraphMemory(const GraphMemory&) = delete;
    GraphMemory& operator=(const GraphMemory&) = delete;
    GraphMemory(GraphMemory&&) = delete;
    GraphMemory& operator=(GraphMemory&&) = delete;

    // Recording is a one-time preparation phase. Create another arena to
    // record a different sequence; existing records are never discarded.
    void recording();
    void replaying();
    void verify_replay_complete();
    std::size_t allocations() const;
    std::size_t host_snapshots() const;
    static void activate(GraphMemory* arena);

private:
    struct Allocation {
        void* pointer;
        std::size_t bytes;
        rmm::cuda_stream_view stream;
    };
    struct HostCopy {
        void* snapshot;
        std::size_t bytes;
        cudaStream_t stream;
    };
    enum class Mode { Record, Replay };

    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;
    void do_deallocate(void* pointer, std::size_t bytes,
                       rmm::cuda_stream_view stream) override;
    bool do_is_equal(const rmm::mr::device_memory_resource& other) const noexcept override;
    const void* retain_host_copy(const void* source, std::size_t bytes, cudaStream_t stream);

    static thread_local GraphMemory* active_;
    rmm::mr::device_memory_resource* upstream_;
    std::vector<Allocation> records_;
    std::vector<HostCopy> host_copies_;
    std::size_t cursor_ = 0;
    std::size_t host_cursor_ = 0;
    Mode mode_ = Mode::Record;

    friend rmm::mr::device_memory_resource*
    ::__wrap__ZNK7heongpu10MemoryPool19get_device_resourceEv(const heongpu::MemoryPool* pool);
    friend cudaError_t (::__wrap_cudaMemcpyAsync)(void*, const void*, std::size_t,
                                                cudaMemcpyKind, cudaStream_t);
};

} // namespace cubutterfly_heongpu
