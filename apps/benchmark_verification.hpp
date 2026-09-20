#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

// Zero checks every transform. A bounded check includes both ends of the
// actual batch, so the full launch and its partial final tile still execute.
inline std::vector<std::size_t> verification_batches(std::size_t batch, std::size_t limit) {
    const auto count = limit ? std::min(batch, limit) : batch;
    std::vector<std::size_t> indices;
    indices.reserve(count);
    for (std::size_t i=0; i<count; ++i)
        indices.push_back(count == 1 ? 0 : (batch-1)/(count-1)*i + ((batch-1)%(count-1)*i)/(count-1));
    return indices;
}
