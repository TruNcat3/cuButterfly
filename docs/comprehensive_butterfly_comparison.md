# Comprehensive Butterfly Comparison

This document defines the current single-GPU comparison protocol for the
supported butterfly applications. The comparison is intentionally layered:
architecture candidates, v0.6 compatibility fallbacks, runtime-selected
points, and external libraries are reported separately.

The library is a configuration space, not a single fixed implementation. A
valid point is keyed by hardware, operator, precision, scale (`logN` and
batch), direction, placement, and output layout. The selected result is a
complete architecture mapping plus a processing-unit choice and lowering
contract. Therefore different rows are allowed, and expected, to use
different stage/data unfolding, residence, layouts, or arithmetic cores. The
purpose of the matrix is to measure this conditional selection behavior, not
to claim that one kernel is optimal for every operator or GPU.

For descriptions and source links for the external libraries, see [Research
Positioning](cubutterfly_positioning.md), [FFT Library Comparison](fft_library_comparison.md),
and [V100 External Baselines](v100_external_baselines.md). This report only
joins an external row when the semantic and timing contracts match.

## Current V100 Screening Matrix

The current build (`build-cuda118-cufftdx2`) has a fresh screening matrix with
one event-timed trial per point and correctness enabled on the first execution.

| Layer | Operators | Lengths | Batches | Points | Result |
|:--|:--|--:|--:|--:|:--|
| Native operator matrix | FFT, FWHT, Structured 2x2, Subset Zeta, Superset Zeta, XOR Zeta | logN 10/12/14 | 1/16/64 | 102 | complete |
| Long NTT matrix | NTT, uint32, natural order | logN 20 | 1/16/64 | 6 | complete |
| Long FFT protocol | FFT, FP32, cuFFTDx/direct/online/cuFFT | logN 18/20 | 2/8/16/64 | existing three-way protocol | complete for listed shapes |

The fresh outputs are:

- `results/cross_operator_v08_wide_current/comparison.md`
- `results/cross_operator_v08_wide_current/raw.csv`
- `results/cross_operator_v08_ntt_current/comparison.md`
- `results/cross_operator_v08_ntt_current/raw.csv`

The external-library coverage is consolidated in
`results/cross_operator_v08_external_baselines_current_v2/comparison.md`. It
contains cuFFT rows from the broad matrix, a current-build exact-contract
Dao-AILab-FHT run, and the archived exact-contract GPU-NTT rows.

The v0.8 candidate set includes the mature v0.6 physical points. In
particular, FWHT comparison includes both the legacy hierarchical point and
the faster warp-register/radix-2 point; the summarizer selects the fastest
measured incumbent rather than the first manifest entry. The long FFT rows
must be read from the separate cuFFTDx/online three-way matrix, not from the
generic short-length resident envelope.

## Screening Results

Ratios above one mean that the v0.8 candidate has lower kernel time than the
v0.6 incumbent. The v0.8 search includes the measured v0.6 point as an
explicit fallback, so a ratio below one is not selected in the comparison
closure.

| Operator | Cells | Geometric mean v0.8/v0.6 | Cells faster than v0.6 | External baseline |
|:--|--:|--:|--:|:--|
| FFT FP32 | 6 | 1.306x | 5/6 | cuFFT, v0.8 remains below cuFFT in these short native cells |
| FWHT FP32 | 9 | 1.494x | 7/9 | Dao FHT at separate exact logN15 matrix |
| Structured 2x2 FP32 | 6 | 1.598x | 6/6 | none in this run |
| Subset Zeta uint32 | 9 | 1.491x | 7/9 | none in this run |
| Superset Zeta uint32 | 9 | 1.504x | 7/9 | none in this run |
| XOR Zeta uint32 | 9 | 1.472x | 7/9 | none in this run |
| NTT uint32, logN20 | 3 | 1.046x | 1/3 | GPU-NTT protocol differs; archived uint64 matches reported separately |

The ratios are screening evidence, not final performance claims: one trial is
enough to find candidate crossovers but not to establish variance or clock
stability. The logN14 points that select the v0.6 fallback are useful evidence
that the closure works; they should be re-run with three or five trials before
publication.

## What Is Comparable

Every row is matched by operator, precision, direction, normalization, output
order, placement, length, and batch. FFT rows use the same in-place and
normalization contract. NTT rows use the same modulus and natural output
order. Zeta and Structured rows are compared only against the internal v0.6
incumbent because there is no equivalent external CUDA library in the current
environment.

The v0.8 column is a searched candidate from the registered profile set plus
the v0.6 fallback. It is not an exhaustive launch of every generated point;
the physical candidate sweeps and NCU attribution remain separate artifacts.

## Remaining Coverage

The following are required for a publication-grade “broad” table:

1. Repeat the stable points with three to five trials and retain variance,
   outlier, and clock-state metadata.
2. Add FP64, FP16/BF16, inverse transforms, strided layouts, and out-of-place
   contracts where the operator supports them.
3. Add logN 8, 16, 18, and 20 for the operators whose resource envelope
   permits them; reject unsupported points explicitly rather than padding the
   table with failures.
4. Keep matching Dao FHT and GPU-NTT measurements in the external coverage
   report using their native output and modulus contracts. No external result
   is fabricated for Structured or Zeta.
5. Keep FFT/cuFFT, v0.6, and v0.8 resident/online/direct paths in one protocol
   for long lengths. The existing logN18/20 three-way report is the authority
   for the saturated FFT cases.

This separation makes the comparison broad without turning internal
compatibility wins into unsupported claims about external libraries.
