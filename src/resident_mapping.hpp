#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cuntt::detail {
using LocalPartitions = std::vector<std::vector<std::uint32_t>>;
inline bool resident_axes_requested(const LocalPartitions& parts,
                                    const std::vector<std::uint32_t>& chunks) {
    return std::any_of(parts.begin(),parts.end(),[](const auto& p){return !p.empty();}) ||
           std::any_of(chunks.begin(),chunks.end(),[](auto c){return c!=0;});
}
inline void validate_resident_axis_counts(const LocalPartitions& parts,
                                         const std::vector<std::uint32_t>& chunks,
                                         std::size_t groups) {
    if ((!parts.empty() && parts.size()!=groups) || (!chunks.empty() && chunks.size()!=groups))
        throw std::invalid_argument("resident axes must have one entry per physical execution group");
}
inline void validate_shared_resident_axes(const LocalPartitions& parts,
                                          const std::vector<std::uint32_t>& chunks,
                                          const std::vector<std::uint32_t>& stages) {
    validate_resident_axis_counts(parts,chunks,stages.size());
    for (auto chunk:chunks) if(chunk)
        throw std::invalid_argument("shared iterative output already has final ownership; exchange chunks are unsupported");
    for (std::size_t g=0;g<parts.size();++g) {
        if(parts[g].empty()) continue;
        unsigned sum=0;
        for(auto s:parts[g]) {
            if(!s || s>5 || sum>stages[g] || s>stages[g]-sum)
                throw std::invalid_argument("shared resident processing units require 1..5 stages covering the physical group");
            sum+=s;
        }
        if(sum!=stages[g]) throw std::invalid_argument("local stage partition must cover its physical group");
    }
}
// Bounded seed shapes, not a restriction on explicit replay or GA mutations.
inline LocalPartitions resident_partition_seed(const std::vector<std::uint32_t>& groups,unsigned width) {
    LocalPartitions result;
    for(auto stages:groups) {
        std::vector<std::uint32_t> part;
        while(stages) {const auto s=std::min(stages,width);part.push_back(s);stages-=s;}
        result.push_back(std::move(part));
    }
    return result;
}
}
