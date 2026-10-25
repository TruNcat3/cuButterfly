#pragma once

#include <cstddef>
#include <gpuntt/ntt_merge/ntt.cuh>

// Scoped integration for pinned HEonGPU's word64, per-polynomial RNS NTT ABI.
// An enabled adapter never silently falls back to GPU-NTT. Prepare with real
// calls, then freeze before timing or CUDA Graph capture. Contexts, coefficient
// tables and streams must outlive cached plans; clear after synchronization.
namespace cubutterfly_heongpu {
void set_enabled(bool enabled);
bool enabled();
void freeze_plans(bool freeze);
void reset_counters();
std::size_t replacement_calls();
std::size_t replacement_polynomials();
void clear_cache();

enum class Execution { NativeRns, Grouped };
void set_execution(Execution execution);
Execution get_execution();
const char* execution_name(Execution execution);

enum class Ordering { Contiguous, Modulus, Polynomial };
void transform(Data64* input, Data64* output, Root64* roots, Modulus64* moduli,
               gpuntt::ntt_rns_configuration<Data64> config, int batch,
               int modulus_count, Ordering ordering = Ordering::Contiguous,
               int* order = nullptr);
}  // namespace cubutterfly_heongpu

namespace gpuntt {
inline void cubutterfly_GPU_NTT(Data64* input, Data64* output, Root64* roots,
    Modulus64* moduli, ntt_rns_configuration<Data64> cfg, int batch, int count) {
    if (!cubutterfly_heongpu::enabled())
        return GPU_NTT(input, output, roots, moduli, cfg, batch, count);
    cfg.ntt_type = FORWARD; // Ordinary GPU-NTT APIs choose direction by name.
    cubutterfly_heongpu::transform(input, output, roots, moduli, cfg, batch, count);
}
inline void cubutterfly_GPU_INTT(Data64* input, Data64* output, Root64* roots,
    Modulus64* moduli, ntt_rns_configuration<Data64> cfg, int batch, int count) {
    if (!cubutterfly_heongpu::enabled())
        return GPU_INTT(input, output, roots, moduli, cfg, batch, count);
    cfg.ntt_type = INVERSE;
    cubutterfly_heongpu::transform(input, output, roots, moduli, cfg, batch, count);
}
inline void cubutterfly_GPU_NTT_Inplace(Data64* input, Root64* roots,
    Modulus64* moduli, ntt_rns_configuration<Data64> cfg, int batch, int count) {
    if (!cubutterfly_heongpu::enabled())
        return GPU_NTT_Inplace(input, roots, moduli, cfg, batch, count);
    cfg.ntt_type = FORWARD;
    cubutterfly_heongpu::transform(input, input, roots, moduli, cfg, batch, count);
}
inline void cubutterfly_GPU_INTT_Inplace(Data64* input, Root64* roots,
    Modulus64* moduli, ntt_rns_configuration<Data64> cfg, int batch, int count) {
    if (!cubutterfly_heongpu::enabled())
        return GPU_INTT_Inplace(input, roots, moduli, cfg, batch, count);
    cfg.ntt_type = INVERSE;
    cubutterfly_heongpu::transform(input, input, roots, moduli, cfg, batch, count);
}
inline void cubutterfly_GPU_NTT_Modulus_Ordered_Inplace(Data64* input,
    Root64* roots, Modulus64* moduli, ntt_rns_configuration<Data64> cfg,
    int batch, int count, int* order) {
    if (!cubutterfly_heongpu::enabled())
        return GPU_NTT_Modulus_Ordered_Inplace(input, roots, moduli, cfg, batch, count, order);
    cubutterfly_heongpu::transform(input, input, roots, moduli, cfg, batch, count,
                                 cubutterfly_heongpu::Ordering::Modulus, order);
}
inline void cubutterfly_GPU_NTT_Poly_Ordered_Inplace(Data64* input,
    Root64* roots, Modulus64* moduli, ntt_rns_configuration<Data64> cfg,
    int batch, int count, int* order) {
    if (!cubutterfly_heongpu::enabled())
        return GPU_NTT_Poly_Ordered_Inplace(input, roots, moduli, cfg, batch, count, order);
    cubutterfly_heongpu::transform(input, input, roots, moduli, cfg, batch, count,
                                 cubutterfly_heongpu::Ordering::Polynomial, order);
}
}  // namespace gpuntt
