#include "fft_tile_layout.cuh"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

void require(bool condition) {
    if (!condition) throw std::runtime_error("FFT tile placement contract failed");
}

template <unsigned Slots>
void check_tile() {
    using Layout = cuntt::detail::FftTileLayout<Slots, true>;
    constexpr unsigned rows = 1024;
    std::vector<unsigned> tile(rows * Slots, ~0U);
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned slot = 0; slot < Slots; ++slot) {
            const auto index = Layout::index(row, slot);
            require(index < tile.size() && tile[index] == ~0U);
            tile[index] = row * Slots + slot;
        }
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned slot = 0; slot < Slots; ++slot) {
            require(tile[Layout::index(row, slot)] == row * Slots + slot);
            // Placement is an involution within each row: no element or
            // neighbouring transform may cross the tile boundary.
            require(Layout::index(row, Layout::index(row, slot) % Slots) == row * Slots + slot);
        }
    if constexpr (Slots >= 2) {
        for (unsigned row = 0; row < rows; ++row)
            for (unsigned slot = 0; slot < Slots; slot += 2) {
                const auto first = Layout::index(row, slot);
                const auto second = Layout::index(row, slot + 1);
                require((first ^ second) == 1);
                std::array<unsigned, 2> pair{tile[first / 2 * 2], tile[first / 2 * 2 + 1]};
                if (first & 1) std::swap(pair[0], pair[1]);
                require(pair[0] == row * Slots + slot && pair[1] == row * Slots + slot + 1);
            }
    }
}

template <bool Swizzled>
unsigned column_conflicts() {
    // A 16-lane portion of the 16x16 CTA reads 16 distinct complex values.
    // Count four-byte bank accesses; the ideal 128-byte request uses all 32.
    using Layout = cuntt::detail::FftTileLayout<16, Swizzled>;
    std::array<unsigned, 32> banks{};
    for (unsigned lane = 0; lane < 16; ++lane) {
        const auto address = Layout::index(lane, 0) * 2;
        ++banks[address % 32];
        ++banks[(address + 1) % 32];
    }
    return *std::max_element(banks.begin(), banks.end());
}

int main() {
    check_tile<1>(); check_tile<2>(); check_tile<4>(); check_tile<8>();
    check_tile<16>(); check_tile<32>(); check_tile<64>(); check_tile<128>();
    check_tile<256>(); check_tile<512>(); check_tile<1024>();
    require(column_conflicts<false>() == 16 && column_conflicts<true>() == 1);
    std::cout << "PASS tile bijection, row isolation, vector-pair order, column bank distribution\n";
}
