// Independent GPL FXT measurement adapter; not linked into cuButterfly.
#include <cstdint>
#include <cstring>
#include <limits>
#include "walsh/arithtransform.h"
#include "walsh/walshwakloc2.h"

template <typename T>
static void execute(const T* source, T* output, std::uint64_t log_n,
                    std::uint64_t batch, int op, int inverse, int normalize) {
  const std::uint64_t n = std::uint64_t(1) << log_n;
  for (std::uint64_t b = 0; b < batch; ++b) {
    T* x = output + b * n;
    std::memcpy(x, source + b * n, n * sizeof(T));
    if (op == 2) {
      walsh_wak_loc_dif2<T>(x, log_n);
      if (inverse && normalize) {
        const T scale = T(1) / T(n);
        for (std::uint64_t i = 0; i < n; ++i) x[i] *= scale;
      }
    } else if (op == 3) {
      if (inverse) arith_transform_minus<T>(x, log_n);
      else arith_transform_plus<T>(x, log_n);
    } else {
      if (inverse) rev_arith_transform_minus<T>(x, log_n);
      else rev_arith_transform_plus<T>(x, log_n);
    }
  }
}

extern "C" int fxt_execute_batch(const void* source, void* output,
    std::uint64_t log_n, std::uint64_t batch, int storage, int op,
    int inverse, int normalize) {
  if (!source || !output || source == output || batch == 0 ||
      log_n >= std::numeric_limits<ulong>::digits || log_n >= 64 ||
      (op != 2 && op != 3 && op != 4)) return 1;
  if (op == 2 && storage != 2 && storage != 3) return 2;
  if (op != 2 && storage != 8 && storage != 9) return 2;
  switch (storage) {
    case 2: execute(static_cast<const float*>(source), static_cast<float*>(output), log_n, batch, op, inverse, normalize); break;
    case 3: execute(static_cast<const double*>(source), static_cast<double*>(output), log_n, batch, op, inverse, normalize); break;
    case 8: execute(static_cast<const std::uint32_t*>(source), static_cast<std::uint32_t*>(output), log_n, batch, op, inverse, normalize); break;
    case 9: execute(static_cast<const std::uint64_t*>(source), static_cast<std::uint64_t*>(output), log_n, batch, op, inverse, normalize); break;
    default: return 2;
  }
  return 0;
}
