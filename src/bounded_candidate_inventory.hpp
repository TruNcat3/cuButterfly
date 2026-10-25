#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <type_traits>
#include <vector>

namespace cuntt::detail {

// Preserve candidate identity and order while sharing a finite budget across
// execution families. Budgets below the family count cover only the first
// encountered families; zero retains exhaustive search.
template <typename Candidate, typename FamilyKey>
std::vector<std::size_t> bounded_candidate_indices(
    const std::vector<Candidate>& inventory, std::size_t budget, FamilyKey family_key) {
    if (!budget || inventory.size() <= budget) {
        std::vector<std::size_t> indices(inventory.size());
        for (std::size_t i = 0; i < inventory.size(); ++i) indices[i] = i;
        return indices;
    }
    if (budget == 1) return {0};

    using Key = std::decay_t<decltype(family_key(inventory.front()))>;
    std::map<Key, std::size_t> family_ids;
    std::vector<std::vector<std::size_t>> families;
    for (std::size_t i = 0; i < inventory.size(); ++i) {
        const auto inserted = family_ids.emplace(family_key(inventory[i]), families.size());
        if (inserted.second) families.emplace_back();
        families[inserted.first->second].push_back(i);
    }

    // Capped round-robin allocation lets small families retain all candidates,
    // then redistributes their unused shares to the remaining families.
    std::vector<std::size_t> quotas(families.size(), 0);
    std::size_t remaining = budget;
    while (remaining) {
        for (std::size_t i = 0; i < families.size() && remaining; ++i) {
            if (quotas[i] == families[i].size()) continue;
            ++quotas[i];
            --remaining;
        }
    }

    std::vector<std::size_t> selected;
    selected.reserve(budget);
    for (std::size_t i = 0; i < families.size(); ++i) {
        const auto count = quotas[i];
        for (std::size_t j = 0; j < count; ++j) {
            const auto position = count == 1 ? 0 : j * (families[i].size() - 1) / (count - 1);
            selected.push_back(families[i][position]);
        }
    }
    // The incumbent is the first member of the first family, so it is always
    // included. Sorting restores inventory order without copying configurations.
    std::sort(selected.begin(), selected.end());
    return selected;
}

} // namespace cuntt::detail
