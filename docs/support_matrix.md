# Capability And Compatibility Matrix

This page is the application-facing support contract. The runtime capability
query and plan-construction validation remain authoritative for a particular
build because optional processing units change the compiled matrix.

## Toolchain And Packaging

| Item | Current contract |
|:--|:--|
| operating system | Linux |
| language | C++17 |
| build system | CMake 3.20 or newer |
| required CUDA components | CUDA Runtime and cuFFT development packages |
| validated artifact toolchain | CUDA 11.8, GCC 11, `sm_70` |
| default CUDA architecture | `70`; applications should set `CMAKE_CUDA_ARCHITECTURES` explicitly |
| CMake package | `find_package(cuButterfly CONFIG REQUIRED)` |
| imported target | `cuButterfly::cuButterfly` |
| version compatibility | generated package file uses same-major compatibility |

Compiling for another CUDA architecture is supported at build-system level,
but automatic mapping calibration is currently V100-only. A successful build
on another GPU is not a performance-portability claim.

## Public Semantic Coverage

| Operator | Types | Direction | Placement/layout | Length |
|:--|:--|:--|:--|:--|
| FFT | complex FP16/BF16 storage with native-width or FP32 accumulation; complex FP32/FP64; legacy mixed FP16/FP32 WMMA core | forward/inverse, optional inverse normalization | in/out of place, positive element stride and valid batch stride | power of two, common runtime `logN=1..20` subject to backend |
| FWHT | FP16/BF16 storage with native-width or FP32 accumulation; FP32/FP64 | forward/inverse, optional inverse normalization | in/out of place, positive element stride and valid batch stride | power of two, common runtime `logN=1..20` subject to backend |
| subset/superset zeta/Mobius | uint32 | forward/inverse | in/out of place, positive element stride and valid batch stride | power of two, common runtime `logN=1..20` subject to backend |
| structured 2x2 | FP16/BF16 storage with native-width or FP32 accumulation; FP32/FP64 | forward/inverse for nonsingular matrices | in/out of place, positive element stride and valid batch stride | power of two, common runtime `logN=1..20`; generated FP32 warp-register core at `logN=3..15` |
| legacy `xor-zeta` name | uint32 | established subset-zeta behavior | same as subset zeta | retained for API/CLI/result compatibility |
| NTT | uint32 or uint64 physical words | forward/inverse subject to backend | contiguous batches, device API out of place | power of two, `logN=1..30` with stricter backend ranges |

For the detailed processing-unit ranges, use
`cubutterfly_bench --list-capabilities` and see [Feature
Coverage](cubutterfly_feature_coverage.md). Important backend-specific NTT
restrictions are rejected during `Plan` construction:

- Hybrid2D requires `logN=12..20`;
- CompactStage currently supports its documented forward 64-bit `logN=20`
  path, with output order controlling workspace behavior;
- StagePipeline currently supports forward, natural-order 64-bit `logN=8`;
- 32-bit words currently require Hybrid2D and a modulus below `2^31`.

FP16/BF16 results are narrowed after every logical butterfly output. V100 has
no native BF16 arithmetic path, so BF16 `Native` measurements on `sm_70` are
explicitly labeled `emulated_native=1`; this is an arithmetic-contract control,
not a native-BF16 throughput claim. The legacy `Fp16Fp32` precision remains a
separate FP32-storage WMMA processing-unit contract.

## Optional Build Features

| Option | Default | Effect |
|:--|:--:|:--|
| `CUBUTTERFLY_ENABLE_CUFFTDX` | off | compile cuFFTDx block/direct/resident processing-unit adapters from the configured MathDx package |
| `CUBUTTERFLY_ENABLE_TURBOFFT` | off | compile available TurboFFT-generated kernels |
| `CUBUTTERFLY_ENABLE_VKFFT` | off | build the comparison benchmark; does not change the public plan backend |
| `CUBUTTERFLY_BUILD_EXAMPLES` | on | build task-oriented API examples |

Selecting a processing unit omitted at build time fails explicitly. There is no
silent substitution with another core.

## Runtime Features

| Feature | Status |
|:--|:--|
| caller-owned CUDA stream | supported |
| caller-owned device buffers | supported |
| queryable/caller-owned workspace | supported, 16-byte minimum alignment |
| allocation-free steady-state device submission | supported after plan construction and workspace binding |
| multiple independent plans/streams | supported when resources do not overlap |
| concurrent use of one plan | not supported |
| CUDA Graph capture | not currently claimed/tested |
| one transform across multiple GPUs | not supported |
| non-power-of-two transforms | not supported |
| multidimensional FFT/NTT | not supported |
| fused user epilogues | not supported |

## Automatic Selector Coverage

The checked-in selector recognizes V100 `sm_70` only and accepts a calibrated
subset of forward contiguous workloads. Its exact FFT, FWHT, legacy XOR-name and NTT shape
table is documented in [Runtime Mapping Selector](runtime_selector.md).
Unsupported hardware or semantics throw `std::invalid_argument`; explicit
backend configuration remains available outside selector coverage. The new
canonical subset/superset and structured-2x2 names require explicit mapping
until their sweeps are recorded.

## Versioning And ABI

The installed CMake version file promises package compatibility within the same
major version. The project does not yet promise a stable binary ABI across minor
releases. Recompile consumers when upgrading the library. Public source-level
changes and removals are recorded in [`CHANGELOG.md`](../CHANGELOG.md); no
deprecated API is currently maintained as a compatibility layer.

Generated design-point JSON, selector tables, benchmark CSV schemas, and
command-line options are research/artifact interfaces and do not carry the same
compatibility expectation as the installed C++ headers.
