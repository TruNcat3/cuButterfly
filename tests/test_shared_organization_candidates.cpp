#include "shared_organization_candidates.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() try {
    using namespace cuntt::detail;
    const std::vector<std::vector<std::uint32_t>> partitions{{8,4},{1,1,2}};
    SharedOrganizationCapabilities capabilities{{128,256,64,32,512,1024},1024,true,4,true};
    const auto points=shared_organization_candidates(partitions,capabilities,5);
    bool saw_wide=false,saw_single_stage_units=false,saw_uneven=false;
    bool saw_bulk_two=false,saw_bulk_four=false,saw_fused=false;
    for (const auto& point : points) {
        require(point.threads>=32 && point.threads%32==0 && point.threads<=1024,"invalid launch width");
        validate_shared_resident_axes(point.local_stage_partitions,{},point.stage_partition);
        saw_wide |= point.threads==1024;
        saw_single_stage_units |= point.local_stage_partitions==resident_partition_seed(partitions[0],1);
        saw_uneven |= point.local_stage_partitions==LocalPartitions{{3,5},{4}};
        saw_bulk_two |= point.batch_tile_count==2;
        saw_bulk_four |= point.batch_tile_count==4;
        if (!point.logical_stage_partition.empty()) {
            require(point.fused_edges.size()+1==point.logical_stage_partition.size(),"invalid logical edges");
            std::vector<std::uint32_t> physical;
            unsigned sum=0;
            for (std::size_t i=0;i<point.logical_stage_partition.size();++i) {
                sum+=point.logical_stage_partition[i];
                if (i+1==point.logical_stage_partition.size() || !point.fused_edges[i]) {
                    physical.push_back(sum); sum=0;
                }
            }
            require(physical==point.stage_partition,"fusion changes physical grouping");
            saw_fused=true;
        }
    }
    require(saw_wide && saw_single_stage_units && saw_uneven,"missing thread/resident basis");
    require(saw_bulk_two && saw_bulk_four && saw_fused,"missing bulk/fusion basis");
    require(points.size()<=partitions.size()*(capabilities.thread_options.size()*10+5),"seed inventory exploded");
    for (auto threads : {128U,256U,64U}) for (bool aligned : {true,false})
        for (unsigned width : {0U,2U,3U,4U,5U}) {
            const auto parts=width ? resident_partition_seed(partitions[0],width) : LocalPartitions{};
            require(std::any_of(points.begin(),points.end(),[&](const auto& point) {
                return point.stage_partition==partitions[0] && point.threads==threads &&
                    point.writer_aligned==aligned && point.local_stage_partitions==parts &&
                    point.batch_tile_count==1 && point.logical_stage_partition.empty();
            }),"lost previous shared candidate");
        }
    capabilities.max_threads=64;
    capabilities.resident_available=false;
    capabilities.max_bulk_batch_tiles=1;
    capabilities.resident_boundary_fusion=false;
    for (const auto& point : shared_organization_candidates(partitions,capabilities,5)) {
        require(point.threads<=64 && point.local_stage_partitions.empty(),"ignored capability restriction");
        require(point.batch_tile_count==1 && point.logical_stage_partition.empty(),"emitted disabled mechanism");
    }
    require(!SharedOrganizationCapabilities::exchange_chunking,"shared final ownership falsely exposes chunking");
    require(shared_overlap_tiles(5,3)==std::vector<std::uint32_t>{1,2,4,5},"missing overlap tile basis");
    require(shared_overlap_tiles(16,3)==std::vector<std::uint32_t>{1,2,4,8,16},"missing full-batch overlap tile");
    require(shared_overlap_tiles(5,1).empty() && shared_overlap_tiles(1,3).empty(),"invalid overlap domain");
    std::cout << "Shared organization candidate contract passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
