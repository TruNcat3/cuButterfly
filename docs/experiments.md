# Experimental Results

## 1. Measurement Policy

The checked-in results use resident kernel time unless explicitly labelled
end-to-end. Comparisons pin GPU, CUDA version, workload shape, precision or
modulus, direction, normalization, placement, output order, warmups, repeats,
and external-library revision. Raw and summarized CSV files are retained.

The reference system is a Tesla V100-SXM2-16GB (`sm_70`, 80 SMs, 6 MiB L2)
with CUDA 11.8. Cross-GPU rows are placeholders until measured.

## 2. What The Results Establish

The controlled full-suite cross-section and its evidence boundaries are
reported in [V100 Comprehensive Results](comprehensive_v100_results.md). It is
the current authority for claims spanning multiple lengths, precisions,
semantics, and runnable FFT libraries; the focused experiments below retain
their original protocols.

The follow-up [V100 Length/Batch Scaling](v100_scaling_results.md) experiment
holds transform length fixed while varying batch, exposing saturation points
and mapping crossovers hidden by constant-total-point comparisons.

The [V100 Mapping Selector](v100_mapping_selector.md) performs
leave-one-complete-shape-out validation over the internal multi-candidate rows.
The [matching-protocol external refresh](v100_external_baselines.md) is the
current authority for Dao FHT and GPU-NTT comparisons with matching semantics.
Targeted [V100 counter attribution](v100_ncu_attribution.md) explains the
length/batch saturation and FWHT mapping crossover in hardware-service terms.

### Common Mapping Across Operators

At `N=256`, a controlled eight-warp stage pipeline selects the interior
`Us=4` point for NTT, FFT, FWHT, and XOR-zeta, with 1.90x-2.40x speedup over
`Us=1` inside that family. Its absolute performance is below specialized
temporal kernels, so the result validates the unfolding trade-off rather than
the queue-based implementation.

An expanded search selects different local realizations:

| Operator | Best local realization | Median | Reference |
|:--|:--|--:|:--|
| FWHT FP32 | radix-4 temporal, 128 threads | 0.051292 ms | 327.09 Gbutterfly/s |
| XOR-zeta uint32 | radix-4 temporal, 128 threads | 0.050156 ms | 334.50 Gbutterfly/s |
| FFT FP32 | radix-4 temporal, 128 threads | 0.110346 ms | 75.9% cuFFT throughput |

This operator dependence is expected: the graph is common, while arithmetic,
coefficient traffic, and local transport change the hardware balance.

### Processing-Unit Reuse

The Dao-derived warp-register FWHT unit reaches 92%-100% of the same-machine
Dao FP32 baseline from `logN=8..15`, while remaining behind the common
cuButterfly stride, placement, inverse, and batching interface.

The generated FP32 CTA DFT8 reaches 92.4% of cuFFT throughput at `N=8`. Its
selected spatial factor changes with length, and it crosses below the existing
shared radix-8 schedule after `N=64`. This identifies layout synchronization as
a composition cost rather than presenting the codelet as a universal winner.

### Online Reordering

At `logN=20`, fusing the boundary permutation and retaining the suffix in one
CTA improves the prior hierarchy by 1.52x for FWHT/XOR-zeta and 1.87x for FFT.
Nsight Systems shows the FFT suffix shrinking from ten kernels to one. The
permuted prefix itself becomes about 25% slower, which confirms that online
reordering is beneficial only when the eliminated downstream work is larger
than its transaction cost.

### NTT Against GPU-NTT

With the same V100, 60-bit prime, resident timing, and output semantics:

| Workload | cuButterfly | GPU-NTT | Throughput ratio |
|:--|--:|--:|--:|
| `logN=16`, natural Hybrid2D | 0.274104 ms | 0.291133 ms | 1.062x |
| `logN=18`, natural Hybrid2D | 0.342482 ms | 0.355820 ms | 1.039x |
| `logN=20`, native bit-reversed compact stage | 0.341320 ms | 0.396186 ms | 1.161x |

NCU localizes the original `logN=20` Hybrid2D gap to expanded fused-root
traffic and long-scoreboard stalls. Compact roots remove the native-layout gap;
a required natural-order permutation changes the ranking and is reported
separately.

## 3. Remaining Gaps

| Area | Current boundary | Needed evidence or implementation |
|:--|:--|:--|
| FFT | two-pass `logN=16..20` reaches 79.3%-88.8%; direct units at `logN=11,12,14` reach 1.006x-1.029x cuFFT while `logN=13` reaches 0.965x | compose larger units into long FFTs, NCU validation, automatic selection |
| cross GPU | only V100 fully measured | predict and validate mappings on another GPU generation |
| automatic selection | scripts rank measured candidates | calibrated cost model selecting before exhaustive sweep |
| XOR-zeta baseline | internal comparisons only | pinned same-machine external implementation |
| public API | host-vector plan interface | device-pointer and stream-aware integration contract |
| numeric coverage | current listed precisions/moduli | additional FFT mixed precision and NTT reduction contracts |

## 4. Result Locations

| Topic | Report | Primary records |
|:--|:--|:--|
| comprehensive V100 suite | `comprehensive_v100_results.md` | `comprehensive_v100_full_raw.csv`, `comprehensive_v100_full_summary.csv` |
| orthogonal length/batch scaling | `v100_scaling_results.md` | `v100_scaling_full_raw.csv`, `v100_scaling_full_summary.csv` |
| calibrated V100 selector | `v100_mapping_selector.md` | `v100_mapping_selector_evaluation.csv`, `v100_mapping_selector_metrics.json` |
| scaling-crossover counters | `v100_ncu_attribution.md` | `ncu_scaling_crossovers/summary.csv`, `ncu_scaling_crossovers/attribution.csv` |
| refreshed external baselines | `v100_external_baselines.md` | `v100_external_baselines_raw.csv`, `v100_external_baselines_summary.csv` |
| V100 NTT baseline | `v100_initial_results.md` | `hybrid2d_matrix*.csv` |
| NTT vs GPU-NTT | `gpu_ntt_gap_analysis.md` | `fused_vs_gpuntt.csv`, `gpu_ntt_gap_same_modulus.csv` |
| processing units | `processing_unit_design_space.md` | `processing_units_v100_*.csv` |
| cross operator | `cubutterfly_cross_operator_results.md` | `cubutterfly_design_sweep_v100_*.csv` |
| large transforms | `cubutterfly_large_results.md` | `cubutterfly_large_v100_*.csv`, `cubutterfly_online_*.csv` |
| CTA DFT8 | `fft_cta_space_time_results.md` | `fft_cta_*_v100_raw.csv` |
| hardware model | `hardware_mapping_methodology.md` | `hardware_capabilities_v100.json`, `v100_compact_mapping_model.csv` |
| NCU | `ncu_profiling.md` | `ncu_summary_logN20.csv`, `ncu_fft_units_analysis.md` |

The detailed reports contain commands, revisions, caveats, and negative
results omitted from this overview.
