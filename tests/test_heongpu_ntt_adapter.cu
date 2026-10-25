// Optional integration regression, built by run_heongpu_ckks_e2e.py.
// CPU arithmetic is independent of both GPU implementations. Covers primitive
// root conversion, RNS/subset ordering, inverse scaling, and nondefault streams.
#include "ntt_adapter.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using U64 = std::uint64_t;
using U128 = unsigned __int128;
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
U64 mul(U64 a, U64 b, U64 q) { return U128(a) * b % q; }
U64 powmod(U64 a, U64 e, U64 q) {
    U64 x = 1;
    for (; e; e >>= 1, a = mul(a, a, q)) if (e & 1) x = mul(x, a, q);
    return x;
}
unsigned rev(unsigned x, int bits) {
    unsigned r = 0;
    for (int i = 0; i < bits; ++i, x >>= 1) r = (r << 1) | (x & 1);
    return r;
}
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(std::size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
};
template<class T> void upload(Device<T>& d, const std::vector<T>& h, cudaStream_t s) {
    check(cudaMemcpyAsync(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice, s));
}
std::vector<U64> cpu(std::vector<U64> v, U64 q, U64 psi, int bits, bool inverse) {
    const unsigned n = v.size();
    if (inverse) {
        auto ordered = v;
        for (unsigned j = 0; j < n; ++j) v[j] = ordered[rev(j, bits)];
    } else {
        U64 t = 1;
        for (unsigned j = 0; j < n; ++j, t = mul(t, psi, q)) v[j] = mul(v[j], t, q);
    }
    for (unsigned j = 0; j < n; ++j) if (j < rev(j, bits)) std::swap(v[j], v[rev(j, bits)]);
    U64 root = mul(psi, psi, q);
    if (inverse) root = powmod(root, q - 2, q);
    for (unsigned len = 2; len <= n; len <<= 1) {
        const U64 step = powmod(root, n / len, q);
        for (unsigned base = 0; base < n; base += len) {
            U64 w = 1;
            for (unsigned j = 0; j < len / 2; ++j, w = mul(w, step, q)) {
                U64 a = v[base + j], b = mul(v[base + j + len / 2], w, q);
                v[base + j] = (a + b) % q;
                v[base + j + len / 2] = (a + q - b) % q;
            }
        }
    }
    if (inverse) {
        U64 t = powmod(n, q - 2, q), step = powmod(psi, q - 2, q);
        for (unsigned j = 0; j < n; ++j, t = mul(t, step, q)) v[j] = mul(v[j], t, q);
    } else {
        auto ordered = v;
        for (unsigned j = 0; j < n; ++j) v[rev(j, bits)] = ordered[j];
    }
    return v;
}
}

int main() {
    try {
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::mt19937_64 random(20261018);
        int cases = 0;
        int placement_executions = 0, placement_misses = 0;
        for (int bits : {12, 14}) {
            const int n = 1 << bits, physical_polynomials = 6, batch = 4;
            std::vector<Modulus64> moduli{Modulus64(1152921504606584833ULL),
                                        Modulus64(576460756061519873ULL)};
            std::vector<U64> psi, inverse_n;
            for (const auto& m : moduli) {
                U64 candidate = 0;
                for (U64 g = 2; !candidate; ++g) {
                    U64 t = powmod(g, (m.value - 1) / (2 * n), m.value);
                    if (powmod(t, n, m.value) == m.value - 1) candidate = t;
                }
                psi.push_back(candidate);
                inverse_n.push_back(powmod(n, m.value - 2, m.value));
            }
            Device<Modulus64> device_moduli(2);
            Device<U64> device_inverse(2), tables(2 * n), input(physical_polynomials * n), native(physical_polynomials * n);
            Device<int> device_order(4);
            upload(device_moduli, moduli, stream);
            upload(device_inverse, inverse_n, stream);
            for (auto ordering : {cubutterfly_heongpu::Ordering::Contiguous,
                                  cubutterfly_heongpu::Ordering::Modulus,
                                  cubutterfly_heongpu::Ordering::Polynomial}) {
                std::vector<int> order = ordering == cubutterfly_heongpu::Ordering::Modulus
                    ? std::vector<int>{1, 0} : std::vector<int>{5, 1, 4, 2};
                upload(device_order, order, stream);
                for (bool inverse : {false, true}) {
                    std::vector<U64> table(2 * n), values(physical_polynomials * n, 777);
                    for (int m = 0; m < 2; ++m) {
                        U64 t = 1, step = inverse ? powmod(psi[m], moduli[m].value - 2, moduli[m].value) : psi[m];
                        for (int j = 0; j < n; ++j, t = mul(t, step, moduli[m].value))
                            table[m * n + rev(j, bits)] = t;
                    }
                    for (int b = 0; b < batch; ++b) {
                        const int m = ordering == cubutterfly_heongpu::Ordering::Modulus ? order[b % 2] : b % 2;
                        const int p = ordering == cubutterfly_heongpu::Ordering::Polynomial ? order[b] : b;
                        for (int j = 0; j < n; ++j) values[p * n + j] = random() % moduli[m].value;
                        values[p * n] = 0;
                        values[p * n + 1] = moduli[m].value - 1;
                    }
                    auto expected = values;
                    for (int b = 0; b < batch; ++b) {
                        const int m = ordering == cubutterfly_heongpu::Ordering::Modulus ? order[b % 2] : b % 2;
                        const int p = ordering == cubutterfly_heongpu::Ordering::Polynomial ? order[b] : b;
                        std::vector<U64> poly(values.begin() + p * n, values.begin() + (p + 1) * n);
                        const auto result = cpu(poly, moduli[m].value, psi[m], bits, inverse);
                        std::copy(result.begin(), result.end(), expected.begin() + p * n);
                    }
                    upload(tables, table, stream);
                    upload(input, values, stream);
                    upload(native, values, stream);
                    gpuntt::ntt_rns_configuration<Data64> cfg{};
                    cfg.n_power = bits;
                    cfg.ntt_type = inverse ? gpuntt::INVERSE : gpuntt::FORWARD;
                    cfg.ntt_layout = gpuntt::PerPolynomial;
                    cfg.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
                    cfg.mod_inverse = device_inverse.p;
                    cfg.stream = stream;
                    if (ordering == cubutterfly_heongpu::Ordering::Contiguous) {
                        if (inverse) gpuntt::GPU_INTT_Inplace(native.p, tables.p, device_moduli.p, cfg, batch, 2);
                        else gpuntt::GPU_NTT_Inplace(native.p, tables.p, device_moduli.p, cfg, batch, 2);
                    } else if (ordering == cubutterfly_heongpu::Ordering::Modulus)
                        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(native.p, tables.p, device_moduli.p, cfg, batch, 2, device_order.p);
                    else gpuntt::GPU_NTT_Poly_Ordered_Inplace(native.p, tables.p, device_moduli.p, cfg, batch, 2, device_order.p);
                    cubutterfly_heongpu::transform(input.p, input.p, tables.p, device_moduli.p, cfg, batch, 2, ordering,
                        ordering == cubutterfly_heongpu::Ordering::Contiguous ? nullptr : device_order.p);
                    std::vector<U64> actual(values.size()), reference(values.size());
                    check(cudaMemcpyAsync(actual.data(), input.p, actual.size() * sizeof(U64), cudaMemcpyDeviceToHost, stream));
                    check(cudaMemcpyAsync(reference.data(), native.p, reference.size() * sizeof(U64), cudaMemcpyDeviceToHost, stream));
                    check(cudaStreamSynchronize(stream));
                    if (actual != expected || reference != expected) {
                        std::size_t ca = 0, na = 0;
                        for (std::size_t j = 0; j < expected.size(); ++j) { ca += actual[j] != expected[j]; na += reference[j] != expected[j]; }
                        throw std::runtime_error("adapter/CPU mismatches=" + std::to_string(ca) + " native/CPU=" + std::to_string(na) +
                                                 " ordering=" + std::to_string(int(ordering)) + " inverse=" + std::to_string(inverse));
                    }
                    if (bits == 12 && !inverse &&
                        ordering == cubutterfly_heongpu::Ordering::Contiguous) {
                        Device<U64> separate(values.size());
                        const std::vector<U64> sentinel(values.size(), 0xfedcba9876543210ULL);
                        auto separate_expected = sentinel;
                        std::copy(expected.begin(), expected.begin() + batch * n,
                                  separate_expected.begin());
                        const auto initial_calls = cubutterfly_heongpu::replacement_calls();
                        upload(input, values, stream);
                        upload(separate, sentinel, stream);
                        cubutterfly_heongpu::freeze_plans(true);
                        bool missed = false;
                        try {
                            cubutterfly_heongpu::transform(input.p, separate.p, tables.p,
                                device_moduli.p, cfg, batch, 2, ordering);
                        } catch (const std::runtime_error& error) {
                            if (std::string(error.what()).find("plan miss") == std::string::npos)
                                throw;
                            missed = true;
                        }
                        if (!missed || cubutterfly_heongpu::replacement_calls() != initial_calls)
                            throw std::runtime_error("frozen adapter reused in-place plan for separate output");
                        ++placement_misses;

                        auto execute_and_check = [&](bool in_place) {
                            upload(input, values, stream);
                            upload(separate, sentinel, stream);
                            U64* destination = in_place ? input.p : separate.p;
                            cubutterfly_heongpu::transform(input.p, destination, tables.p,
                                device_moduli.p, cfg, batch, 2, ordering);
                            std::vector<U64> result(values.size()), preserved(values.size());
                            check(cudaMemcpyAsync(result.data(), destination, result.size() * sizeof(U64),
                                                  cudaMemcpyDeviceToHost, stream));
                            if (!in_place)
                                check(cudaMemcpyAsync(preserved.data(), input.p, preserved.size() * sizeof(U64),
                                                      cudaMemcpyDeviceToHost, stream));
                            check(cudaStreamSynchronize(stream));
                            if (result != (in_place ? expected : separate_expected) ||
                                (!in_place && preserved != values))
                                throw std::runtime_error("adapter placement cache CPU/input/gap mismatch");
                            ++placement_executions;
                        };
                        cubutterfly_heongpu::freeze_plans(false);
                        execute_and_check(false);
                        cubutterfly_heongpu::freeze_plans(true);
                        execute_and_check(true);
                        execute_and_check(false);
                        if (cubutterfly_heongpu::replacement_calls() != initial_calls + 3)
                            throw std::runtime_error("adapter placement cache call count mismatch");
                    }
                    // The tables intentionally change on next case: invalidate.
                    cubutterfly_heongpu::clear_cache();
                    ++cases;
                }
            }
        }
        check(cudaStreamDestroy(stream));
        if (placement_executions != 3 || placement_misses != 1)
            throw std::runtime_error("adapter placement cache coverage changed");
        std::cout << "PASS " << cases << " NTT adapter cases, exact CPU and GPU-NTT agreement\n";
        std::cout << "PASS " << placement_executions
                  << " extra exact placement-cache executions and " << placement_misses
                  << " frozen placement miss\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
