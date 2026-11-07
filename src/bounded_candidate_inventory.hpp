#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <type_traits>
#include <vector>

namespace cuntt::detail {

// Preserve candidate identity and order while sharing a finite budget across
// execution families. Prioritize one family per execution organization before
// spending a small budget on additional core/exchange families. Zero is exhaustive.
template <typename Candidate, typename FamilyKey, typename OrganizationKey>
std::vector<std::size_t> bounded_candidate_indices(
    const std::vector<Candidate>& inventory, std::size_t budget, FamilyKey family_key, OrganizationKey organization_key) {
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

    // The first family contains the incumbent. Discover organizations in input
    // order so late backends/overlap modes are not hidden by early core variants.
    using Organization = std::decay_t<decltype(organization_key(inventory.front()))>;
    std::map<Organization, bool> organizations;
    std::vector<std::size_t> family_order, deferred;
    for (std::size_t i = 0; i < families.size(); ++i) {
        if (organizations.emplace(organization_key(inventory[families[i].front()]), true).second)
            family_order.push_back(i);
        else
            deferred.push_back(i);
    }
    family_order.insert(family_order.end(), deferred.begin(), deferred.end());

    // Capped round-robin allocation lets small families retain all candidates,
    // then redistributes their unused shares to the remaining families.
    std::vector<std::size_t> quotas(families.size(), 0);
    std::size_t remaining = budget;
    while (remaining) {
        for (const auto i : family_order) {
            if (!remaining) break;
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

// Compatibility: without a coarser key each family is its own organization,
// retaining the original first-occurrence allocation policy.
template <typename Candidate, typename FamilyKey>
std::vector<std::size_t> bounded_candidate_indices(
    const std::vector<Candidate>& inventory, std::size_t budget, FamilyKey family_key) {
    return bounded_candidate_indices(inventory, budget, family_key, family_key);
}

} // namespace cuntt::detail
