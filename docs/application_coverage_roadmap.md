# Non-cryptographic application coverage roadmap

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

Date: 2026-09-20

This document records the remaining high-value coverage outside the
cryptographic application extension. It is an experiment boundary document,
not a promise to add every transform family or every possible tensor shape.
The current A100 matrices and their queues remain frozen; this roadmap does
not change their inputs or start GPU work.

## Current conclusion

The public C API is broader than the current comparison matrix. It has six
operator contracts, rank-one/rank-two shapes, positive strides, placement,
direction, inverse normalization where meaningful, exact arbitrary-length
FFT/NTT paths, and zero-extended embedding. The evidence is not equally broad:
the main research workload is still dominated by rank-one power-of-two cases,
and several semantic combinations have correctness coverage without a matched
performance comparison.

This distinction matters for the paper and for V100 migration:

* a supported API path is an implementation claim;
* a measured point is evidence for one hardware/semantic/shape contract;
* a finite representative screen does not establish an application-wide
  optimum or an end-to-end application speedup.

The current 72-cell workload is visible in
[`config/research_comprehensive_workloads.json`](../config/research_comprehensive_workloads.json).
Its non-cryptographic matrix contains many rank-one forward power-of-two
points. The separate batch/precision extension already covers floating-point
precision contracts, batch scaling, and inverse FFT/FWHT anchors; it should not
be duplicated by another broad numeric cross-product
(extension scope (local artifact: `../results/batch_precision_20260920/README.md`; not included in this source release)).

## Evidence inventory

### Shape and layout

The C plan accepts rank one or two and positive per-axis and batch strides
([`cubutterflySetShape`](../src/c_api.cpp#L1194),
[`cubutterflySetStrides`](../src/c_api.cpp#L1211)). Rank two is lowered as two
axis plans with transpose boundaries, while exact arbitrary FFT uses direct
cuFFT or Bluestein and embedding pads each axis
([C API shape lowering](../src/c_api.cpp#L1296),
[general-shape mapping](general_shapes.md#L22)).

There is already a bounded rank-two preparation path for two square FFT
shapes with four axis-core combinations. Its test explicitly labels the run as
an eight-point screen rather than an exhaustive measure
([preparation contract](../tests/test_prepare_large_fft_extension.py#L132)).
The archived general-shape results also contain exact FFT examples at `N=1000`
and `30x50`, and embedded FWHT/zeta/structured examples
([general-shape results](general_shape_results.md#L11)). These are useful evidence,
but they are not a replacement for a current, same-build acceptance matrix.

### Operator semantics

The public operator set is FFT, NTT, FWHT, subset zeta, superset zeta, and
structured 2x2
([public enum](../include/cubutterfly/cubutterfly.h#L37)). FFT requires complex
storage; FWHT and structured 2x2 use real floating storage; zeta currently
requires `uint32`; this is enforced by the runtime rather than inferred from a
benchmark flag
([storage validation](../src/c_api.cpp#L271)).

Forward/inverse and normalized inverse are implemented for FFT/FWHT. Zeta
inverse is Mobius subtraction modulo `2^32`, with no scale. Structured 2x2
supports a broadcast matrix or one matrix per stage and checks nonsingularity
for inverse plans ([operator catalog](operator_catalog.md#L8),
[structured reference](../src/butterfly_reference.cpp#L498)). The comprehensive
performance workload has few inverse zeta and structured-inverse points, even
though correctness tests exercise those contracts.

### Non-power-of-two semantics

Standard arbitrary-length FFT is currently restricted to complex FP32/FP64 and
uses direct cuFFT or a Bluestein composition. Standard arbitrary-length NTT is
restricted to the uint64/root-compatible contract. Other operators use the
explicit zero-extended embedding mode, not an invented native non-power-of-two
butterfly
([length rules](../src/c_api.cpp#L1296),
[public length semantics](c_api.md#L35)). Therefore an embedded `N=2^k-1`
measurement must be reported as a logical-shape boundary experiment; it must
not be described as a native non-power-of-two FWHT or zeta core.

## Bounded P0 extensions

These are high-value checks that use existing public semantics and lowerings.
They should be added as small, separately identified screens after the frozen
matrices, with correctness before timing and with no cross-product explosion.

| Area | Finite proposed screen | Why it matters | Claim boundary |
|:--|:--|:--|:--|
| Rank-two FFT | `2048x2048` and `4096x4096`, batch 1; the four existing direct/block axis combinations | Tests whether each axis may select a different processing unit and whether transpose/output boundaries dominate | A composition screen, not proof of rank-two optimality |
| Rank-two non-FFT | One `64x256` or `256x64` case for FWHT and structured 2x2, batch 1/16 | Checks that the same axis/transpose contract is not FFT-only | No new rank-two core; report transpose and core time separately if available |
| Exact arbitrary FFT | `N=1000` and one rectangular `30x50`, FP32 batch 64; inverse only for the packed case | Validates the direct-versus-Bluestein decision and preserves the distinction between wrapper cost and butterfly-core cost | Compare only with the matching cuFFT contract; do not claim a cuButterfly core win when direct cuFFT is selected |
| Embedded shapes | `N=255` or `32767` for FWHT, subset/superset zeta, and structured 2x2, with one small and one throughput batch | Exposes zero-fill, direct-output, and tail behavior used by real irregular workloads | Label as zero-extended embedding; retain physical extent and boundary kernels |
| Inverse semantic anchors | One `logN=12` and one `logN=20` anchor for subset/superset Mobius; one broadcast and one per-stage structured matrix with inverse | Closes the largest non-FFT semantic gap in the performance matrix | Zeta has modular subtraction, structured inverse has matrix inversion; they are not normalized FFTs |
| Layout pair | One strided out-of-place and one padded in-place case for a medium FFT/FWHT | Separates address-generation and boundary costs from packed throughput | Existing correctness coverage is broader than timing coverage; retain stride and batch distance in the group key |
| Coefficient policy | Structured 2x2 broadcast versus per-stage matrices at the same `N`, precision, and batch | Tests whether coefficient supply changes the best mapping, rather than treating matrices as a constant benchmark detail | Use finite matrices with recorded condition/norm; no claim for all coefficient distributions |

The first rank-two row is already prepared by the bounded large-FFT tooling;
it should be reused rather than regenerated. The remaining rows are small
semantic anchors, not a second comprehensive matrix. The exact arbitrary and
embedded cases can also reuse the existing plan-bench and verification
contracts. No new kernel is required for this P0 list.

### Numerical caveats for P0

The batch/precision extension has already separated FP16/BF16 storage,
accumulation policy, FP32, and FP64. On V100, BF16 is an emulated contract and
cuFFT BF16 is unavailable for SM70
([V100 numeric regimes](../config/v100_numeric_regime_space.json#L37)). A V100
comparison must therefore keep BF16 in a separate group and must not compare
it to a missing cuFFT row. Similarly, zeta values near `UINT32_MAX` should be
used in correctness checks to make modular wraparound explicit; they are not
a new precision mode.

## P1 extensions that require a new contract or lowering

These should not be presented as missing samples in the current study.

| Requested family | Why it is P1 |
|:--|:--|
| Real FFT (`R2C`, `C2R`, packed real spectrum) | Requires new storage, Hermitian-spectrum, output-size, and normalization contracts; current FFT validation requires complex storage. |
| Native mixed-radix/non-power-of-two butterfly cores | Current exact arbitrary FFT uses direct cuFFT or Bluestein; accepting mixed-radix stages changes the stage graph, factor legality, cost model, and code generation. |
| Rank greater than two or general batched tensor axes | The C API intentionally accepts only rank 1 or 2. A rank-3 API needs axis order, transpose planning, workspace, and selector-key semantics. |
| Complex structured 2x2 coefficients | Current structured operator is real-valued. Complex coefficients require complex value traits, coefficient storage, inverse checks, and new baselines. |
| uint64/modular zeta and named modular XOR convolution | Current zeta storage is uint32 and the historical `xor-zeta` name is subset-zeta compatibility, not modular XOR convolution. New arithmetic traits and exact references are required. |
| DCT/DST, Haar, convolution, quantization epilogues | These add pre/post permutations or application semantics beyond the common pair update. They need explicit composition contracts and end-to-end baselines. |

The existing feature document records these boundaries, including rank greater
than two, mixed-radix native cores, and fused epilogues
([deliberately unsupported](cubutterfly_feature_coverage.md#L164)). They should
not be approximated by padding or by renaming an FFT measurement.

## Recommended order while A100 is occupied

1. Keep the running A100 original, batch/precision, and cryptographic queues
   immutable.
2. Finish the V100 hardware identity, calibration, and comprehensive baseline
   as a separate hardware cohort. Do not reuse A100 selector timings; the V100
   manifest records SM70 resource limits and baseline availability separately
   ([V100 regime manifest](../config/v100_numeric_regime_space.json#L3)).
3. Reuse the existing rank-two preparation and run the small P0 semantic
   anchors only after the target hardware has a valid calibration identity.
4. Only after P0 results expose a real missing lowering should a P1 contract be
   designed. Do not add real FFT, rank-3, mixed-radix, or complex-structured
   work to the current performance queue.

## Reporting rules

Every comparison group must include operator, numeric contract, logical shape,
physical shape, rank, batch, direction, normalization, placement, strides,
coefficient policy, hardware UUID/memory, build fingerprint, and timing scope.
Kernel-only numbers exclude compilation, plan construction, host transfers, and
CPU checking. Composite plans additionally report pack, transpose, chirp, or
direct-output boundaries where those are present.

External baselines are required only when their semantics match. A missing
baseline is a coverage limitation, not a zero or a win. Results from one
representative size support that size and contract; they do not establish
uniform superiority across all butterfly applications.
