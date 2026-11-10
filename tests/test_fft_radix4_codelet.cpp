#include "../src/fft_radix4_codelet.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

namespace {
template <typename T>
struct Complex2 {
    T x;
    T y;
};
using Float2 = Complex2<float>;

// Independent same-precision matrix-vector DFT. Fourth roots are exactly representable;
// using the full matrix avoids reproducing the codelet's sum/difference graph.
template <typename T>
void reference_dft(const Complex2<T> (&input)[4], Complex2<T> (&output)[4], bool inverse) {
    constexpr T root_real[4] = {1, 0, -1, 0};
    constexpr T root_imag[4] = {0, 1, 0, -1};
    for (unsigned k = 0; k < 4; ++k) {
        output[k] = {0.0f, 0.0f};
        for (unsigned n = 0; n < 4; ++n) {
            const unsigned phase = (n * k) % 4;
            const T wr = root_real[phase];
            const T wi = inverse ? root_imag[phase] : -root_imag[phase];
            output[k].x += input[n].x * wr - input[n].y * wi;
            output[k].y += input[n].x * wi + input[n].y * wr;
        }
    }
}

template <typename T>
void check_equal(const Complex2<T> (&actual)[4], const Complex2<T> (&expected)[4],
                 T tolerance, const char* label) {
    for (unsigned k = 0; k < 4; ++k) {
        const T error = std::max(std::abs(actual[k].x - expected[k].x),
                                     std::abs(actual[k].y - expected[k].y));
        if (!std::isfinite(error) || error > tolerance) {
            std::cerr << label << " bin " << k << ": got (" << actual[k].x
                      << ", " << actual[k].y << "), expected (" << expected[k].x
                      << ", " << expected[k].y << "), error=" << error
                      << ", tolerance=" << tolerance << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
}

template <bool Inverse, typename T>
void check_direction(const Complex2<T> (&input)[4], T tolerance) {
    Complex2<T> expected[4];
    reference_dft(input, expected, Inverse);
    Complex2<T> actual[4] = {input[0], input[1], input[2], input[3]};
    cuntt::detail::fft_radix4<Inverse>(actual);
    check_equal(actual, expected, tolerance, Inverse ? "inverse" : "forward");

    // Exercise the four-reference interface separately from the array overload.
    Complex2<T> a = input[0], b = input[1], c = input[2], d = input[3];
    cuntt::detail::fft_radix4<Inverse>(a, b, c, d);
    const Complex2<T> separate[4] = {a, b, c, d};
    check_equal(separate, expected, tolerance, "four-reference interface");
}

template <typename T>
void check_case(const Complex2<T> (&input)[4], T tolerance) {
    check_direction<false>(input, tolerance);
    check_direction<true>(input, tolerance);
    Complex2<T> roundtrip[4] = {input[0], input[1], input[2], input[3]};
    cuntt::detail::fft_radix4<false>(roundtrip);
    cuntt::detail::fft_radix4<true>(roundtrip);
    Complex2<T> scaled[4];
    for (unsigned n = 0; n < 4; ++n) {
        scaled[n] = {T(4) * input[n].x, T(4) * input[n].y};
    }
    check_equal(roundtrip, scaled, T(4) * tolerance, "unscaled roundtrip");
}

void check_fp64() {
    using Double2 = Complex2<double>;
    const Double2 zero[4] = {};
    check_case(zero, 0.0);
    for (unsigned basis = 0; basis < 8; ++basis) {
        Double2 input[4] = {};
        if (basis < 4) input[basis].x = 1.0;
        else input[basis - 4].y = 1.0;
        check_case(input, 0.0);
    }
    // Retain bits beyond FP32 to detect narrowing anywhere in the codelet.
    const Double2 precise[4] = {{1.0 + 0x1p-30, 0x1p-40}, {0, 0}, {0, 0}, {0, 0}};
    check_case(precise, 0.0);
    std::mt19937 random(0x44424c34u);
    std::uniform_real_distribution<double> sample(-8.0, 8.0);
    for (unsigned trial = 0; trial < 4096; ++trial) {
        Double2 input[4];
        for (auto& value : input) value = {sample(random), sample(random)};
        check_case(input, 2.0e-14);
    }
}
} // namespace

int main() {
    check_fp64();
    const Float2 zero[4] = {};
    check_case(zero, 0.0f);
    for (unsigned basis = 0; basis < 8; ++basis) {
        Float2 input[4] = {};
        if (basis < 4) input[basis].x = 1.0f;
        else input[basis - 4].y = 1.0f;
        check_case(input, 0.0f);
    }

    // An explicit natural-order/sign check: DFT([1,2,3,4]) = [10,-2+2i,-2,-2-2i].
    Float2 ordered[4] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    const Float2 expected[4] = {{10, 0}, {-2, 2}, {-2, 0}, {-2, -2}};
    cuntt::detail::fft_radix4<false>(ordered);
    check_equal(ordered, expected, 0.0f, "natural order and forward sign");

    std::mt19937 random(0x52414434u);
    std::uniform_real_distribution<float> sample(-8.0f, 8.0f);
    for (unsigned trial = 0; trial < 4096; ++trial) {
        Float2 input[4];
        for (auto& value : input) value = {sample(random), sample(random)};
        // Fixed tolerance for this bounded FP32 input domain; not adjusted on failure.
        check_case(input, 8.0e-6f);
    }
    std::cout << "radix4 codelet: FP32/FP64 zero, bases, and 4096 random cases "
                 "passed in both directions; unscaled roundtrip and FP64 precision passed\n";
}
