#pragma once
#include "resident_mapping.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cuntt::detail {
struct SharedOrganization {
    std::vector<std::uint32_t> stage_partition;
    std::uint32_t threads = 128;
    bool writer_aligned = true;
    LocalPartitions local_stage_partitions;
    std::uint32_t batch_tile_count = 1;
    std::vector<std::uint32_t> logical_stage_partition;
    std::vector<bool> fused_edges;
};

struct SharedOrganizationCapabilities {
    std::vector<std::uint32_t> thread_options;
    std::uint32_t max_threads = 256;
    bool resident_available = false;
    std::uint32_t max_bulk_batch_tiles = 1;
    bool resident_boundary_fusion = false;
    // Keep the established launch-width priors before additional seed axes.
    std::size_t preferred_thread_count = 3;
    // Shared output already has final ownership: exchange chunking does
    // not apply to this organization, independently of the arithmetic trait.
    static constexpr bool exchange_chunking = false;
};

// The caller supplies executable launch widths, independently of operator
// semantics. Additional local shapes vary one axis at a representative point.
inline std::vector<SharedOrganization> shared_organization_candidates(
    const std::vector<std::vector<std::uint32_t>>& partitions,
    const SharedOrganizationCapabilities& capabilities, std::size_t batch = 1) {
    std::vector<SharedOrganization> out, additional;
    for (const auto& partition : partitions) {
        if (partition.empty()) continue;
        for (std::size_t index=0;index<capabilities.thread_options.size();++index) {
            const auto threads=capabilities.thread_options[index];
            if (threads < 32 || threads % 32 || threads > capabilities.max_threads) continue;
            auto& destination=index<capabilities.preferred_thread_count ? out : additional;
            for (bool aligned : {true, false}) {
                SharedOrganization point{partition, threads, aligned, {}};
                destination.push_back(point);
                if (!capabilities.resident_available) continue;
                std::vector<LocalPartitions> seen;
                for (unsigned width : {2U, 3U, 4U, 5U}) {
                    auto parts = resident_partition_seed(partition, width);
                    if (std::find(seen.begin(), seen.end(), parts) != seen.end()) continue;
                    seen.push_back(parts);
                    point.local_stage_partitions = std::move(parts);
                    destination.push_back(point);
                }
            }
        }
        const auto anchor = std::find(capabilities.thread_options.begin(), capabilities.thread_options.end(), 128U);
        if (anchor == capabilities.thread_options.end() || capabilities.max_threads < 128) continue;
        if (capabilities.resident_available) {
            if (std::any_of(partition.begin(),partition.end(),[](auto stages){return stages > 1;}))
                additional.push_back({partition,128,true,resident_partition_seed(partition,1)});
            auto parts = resident_partition_seed(partition, 5);
            for (auto& local : parts) std::reverse(local.begin(), local.end());
            if (parts != resident_partition_seed(partition, 5))
                additional.push_back({partition, 128, true, std::move(parts)});
        }
        for (auto tile : {2U,4U}) {
            if (tile > batch || tile > capabilities.max_bulk_batch_tiles) continue;
            SharedOrganization point{partition,128,true,{}};
            point.batch_tile_count=tile;
            additional.push_back(std::move(point));
        }
        if (capabilities.resident_boundary_fusion) {
            SharedOrganization point{partition,128,true,{}};
            for (auto stages : partition) {
                if (!point.logical_stage_partition.empty()) point.fused_edges.push_back(false);
                if (stages > 1) {
                    point.logical_stage_partition.push_back(stages/2);
                    point.logical_stage_partition.push_back(stages-stages/2);
                    point.fused_edges.push_back(true);
                } else point.logical_stage_partition.push_back(stages);
            }
            if (point.logical_stage_partition != partition) additional.push_back(std::move(point));
        }
    }
    out.insert(out.end(),additional.begin(),additional.end());
    return out;
}

inline std::vector<std::uint32_t> shared_overlap_tiles(std::size_t batch, std::size_t groups) {
    std::vector<std::uint32_t> out;
    if (groups < 2) return out;
    for (std::uint64_t tile = 1; tile < batch && tile <= UINT32_MAX; tile *= 2)
        out.push_back(static_cast<std::uint32_t>(tile));
    // A full-batch tile is a valid one-tile pipeline.  Keep it in the
    // search space because it retains the stage dependency schedule while
    // avoiding an artificial omission at powers of two (and at the largest
    // requested tile for a non-power-of-two batch).
    if (batch > 1 && batch <= UINT32_MAX)
        out.push_back(static_cast<std::uint32_t>(batch));
    return out;
}
} // namespace cuntt::detail
