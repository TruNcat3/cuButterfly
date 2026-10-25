// Recompile the pinned operator with a narrow NTT backend seam. The upstream
// source and non-transform kernels are unchanged. Resolving these definitions
// first prevents extraction of the corresponding object from libheongpu.a.
#include <NTL/ZZ.h>
#include <NTL/RR.h>
#include <heongpu/host/ckks/operator.cuh>
#include <heongpu/host/ckks/cosine_approx.cuh>
#include "ntt_adapter.hpp"

#define GPU_NTT cubutterfly_GPU_NTT
#define GPU_INTT cubutterfly_GPU_INTT
#define GPU_NTT_Inplace cubutterfly_GPU_NTT_Inplace
#define GPU_INTT_Inplace cubutterfly_GPU_INTT_Inplace
#define GPU_NTT_Modulus_Ordered_Inplace cubutterfly_GPU_NTT_Modulus_Ordered_Inplace
#define GPU_NTT_Poly_Ordered_Inplace cubutterfly_GPU_NTT_Poly_Ordered_Inplace
#include CUBUTTERFLY_HEONGPU_CKKS_SOURCE
