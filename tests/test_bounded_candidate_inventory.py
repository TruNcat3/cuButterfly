import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r"""
#include "bounded_candidate_inventory.hpp"
#include <cassert>
#include <numeric>
#include <string>
#include <tuple>

struct Candidate {
    int backend, core, exchange;
    bool overlap;
    std::size_t identity;
};

auto key(const Candidate& c) {
    return std::make_tuple(c.backend, c.core, c.exchange, c.overlap);
}

auto select(const std::vector<Candidate>& candidates, std::size_t budget) {
    return cuntt::detail::bounded_candidate_indices(candidates, budget, key);
}

void check_indices(const std::vector<std::size_t>& selected,
                   std::size_t inventory_size, std::size_t expected_size) {
    assert(selected.size() == expected_size);
    if (selected.empty()) return;
    assert(selected.front() == 0);
    assert(selected.back() < inventory_size);
    assert(std::adjacent_find(selected.begin(), selected.end(),
                             [](auto a, auto b) { return a >= b; }) == selected.end());
}

std::vector<Candidate> regression_inventory() {
    std::vector<Candidate> candidates;
    candidates.push_back({0, 0, 0, false, 0});
    for (std::size_t i = 1; i <= 30; ++i)
        candidates.push_back({0, i <= 15 ? 0 : 1, 0, false, i});
    for (std::size_t i = 31; i < 20758; ++i)
        candidates.push_back({1, 0, 0, i % 2 != 0, i});
    return candidates;
}

void regression() {
    const auto candidates = regression_inventory();
    const auto selected = select(candidates, 512);
    check_indices(selected, candidates.size(), 512);
    for (std::size_t i = 0; i <= 30; ++i)
        assert(std::binary_search(selected.begin(), selected.end(), i));
    for (const bool overlap : {false, true}) {
        std::vector<std::size_t> family, sampled;
        for (std::size_t i = 31; i < candidates.size(); ++i)
            if (candidates[i].overlap == overlap) family.push_back(i);
        for (const auto i : selected)
            if (candidates[i].backend == 1 && candidates[i].overlap == overlap)
                sampled.push_back(i);
        assert(sampled.size() >= 240);
        assert(sampled.front() == family.front());
        assert(sampled.back() == family.back());
        for (std::size_t i = 0; i < sampled.size(); ++i)
            assert(sampled[i] == family[i * (family.size() - 1) / (sampled.size() - 1)]);
    }
    // The former global sampler skipped every non-incumbent hybrid candidate.
    for (std::size_t i = 1; i < 512; ++i)
        assert(i * (candidates.size() - 1) / 511 > 30);
}

void boundaries() {
    for (std::size_t size = 0; size <= 23; ++size) {
        std::vector<Candidate> candidates;
        for (std::size_t i = 0; i < size; ++i)
            candidates.push_back({static_cast<int>(i % 5), 0, 0, false, i});
        for (std::size_t budget = 0; budget <= 25; ++budget) {
            const auto selected = select(candidates, budget);
            check_indices(selected, size, budget == 0 ? size : std::min(size, budget));
            if (budget == 0 || budget >= size)
                for (std::size_t i = 0; i < selected.size(); ++i) assert(selected[i] == i);
            if (size && budget == 1) assert(selected == std::vector<std::size_t>{0});
        }
    }
}

void families() {
    const std::vector<Candidate> candidates{
        {9, 2, 3, false, 0}, {9, 2, 3, false, 1},
        {8, 2, 3, false, 2}, {9, 1, 3, false, 3},
        {9, 2, 1, false, 4}, {9, 2, 3, true, 5}};
    // Each key field separates a family; first-occurrence order wins shortages.
    assert((select(candidates, 3) == std::vector<std::size_t>{0, 2, 3}));
    assert((select(candidates, 5) == std::vector<std::size_t>{0, 2, 3, 4, 5}));
    assert((select(candidates, 6) == std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
}

void fairness() {
    std::vector<Candidate> candidates;
    for (int family = 0; family < 4; ++family)
        for (int i = 0; i < 20; ++i)
            candidates.push_back({family, 0, 0, false, candidates.size()});
    for (std::size_t budget = 2; budget < candidates.size(); ++budget) {
        const auto selected = select(candidates, budget);
        std::vector<std::size_t> counts(4, 0);
        for (auto i : selected) ++counts[candidates[i].backend];
        assert(*std::max_element(counts.begin(), counts.end()) -
               *std::min_element(counts.begin(), counts.end()) <= 1);
    }
}

void organization_coverage() {
    // Early backends have many core variants; the last two organizations were
    // absent under first-family-only budget allocation (BF16 regression shape).
    const std::vector<Candidate> candidates{
        {0,0,0,false,0}, {0,1,0,false,1}, {0,2,0,false,2},
        {1,0,0,false,3}, {2,0,0,false,4}, {1,1,0,false,5},
        {2,1,0,false,6}, {1,2,0,false,7}, {2,2,0,false,8},
        {3,0,0,false,9}, {3,0,0,true,10}, {3,0,0,true,11}};
    const auto organization = [](const Candidate& c) {
        return std::make_tuple(c.backend, c.overlap);
    };
    const auto run = [&](std::size_t budget) {
        return cuntt::detail::bounded_candidate_indices(candidates, budget, key, organization);
    };
    const auto selected = run(8);
    check_indices(selected, candidates.size(), 8);
    for (const auto i : {0U, 3U, 4U, 9U, 10U})
        assert(std::binary_search(selected.begin(), selected.end(), i));
    // A budget smaller than the five organizations remains strict and only
    // covers their first occurrences; it cannot promise complete coverage.
    assert((run(3) == std::vector<std::size_t>{0,3,4}));
    for (std::size_t budget=0; budget<=14; ++budget) {
        const auto result=run(budget);
        check_indices(result,candidates.size(),budget?std::min(budget,candidates.size()):candidates.size());
        assert(result==run(budget));
    }
    // Exhaustive and empty cases also work with the coarser key.
    assert(cuntt::detail::bounded_candidate_indices(
        std::vector<Candidate>{}, 8, key, organization).empty());
}

void deterministic() {
    const auto candidates = regression_inventory();
    const auto before = candidates;
    assert(select(candidates, 512) == select(candidates, 512));
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        assert(candidates[i].identity == before[i].identity);
        assert(key(candidates[i]) == key(before[i]));
    }
    std::vector<Candidate> one_family(100, Candidate{1, 2, 3, true, 0});
    assert((select(one_family, 5) == std::vector<std::size_t>{0, 24, 49, 74, 99}));
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    if (mode == "regression") regression();
    else if (mode == "boundaries") boundaries();
    else if (mode == "families") families();
    else if (mode == "fairness") fairness();
    else if (mode == "deterministic") deterministic();
    else if (mode == "organizations") organization_coverage();
    else return 1;
}
"""


@pytest.fixture(scope="module")
def inventory_fixture(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required for the CPU inventory fixture")
    directory = tmp_path_factory.mktemp("bounded-candidate-inventory")
    source = directory / "fixture.cpp"
    source.write_text(FIXTURE)
    binary = directory / "fixture"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "src"),
         str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    return binary


@pytest.mark.parametrize("mode", ["regression", "boundaries", "families", "fairness", "deterministic", "organizations"])
def test_bounded_candidate_inventory(inventory_fixture, mode):
    subprocess.run([str(inventory_fixture), mode], check=True, capture_output=True, text=True)
