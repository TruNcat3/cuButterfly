# V100 Comprehensive Results

This report is the current single-GPU comparison baseline. All fresh rows were
measured on one Tesla V100-SXM2-16GB with CUDA 11.8. The full protocol uses
`2^22` total points per comparison group, 1000 unmeasured warmups, 100 measured
iterations, five independently launched trials, CUDA-event resident time, and
a correctness preflight for every implementation.

The raw samples, machine-readable summary, and generated complete table are:

- [`comprehensive_v100_full_raw.csv`](../results/comprehensive_v100_full_raw.csv)
- [`comprehensive_v100_full_summary.csv`](../results/comprehensive_v100_full_summary.csv)
- [`comprehensive_v100_full_report.md`](../results/comprehensive_v100_full_report.md)

The long-FFT processing-unit optimization was subsequently rerun under the
same full protocol. Its focused records are
`results/comprehensive_v100_vectorized_fft_raw.csv`,
`results/comprehensive_v100_vectorized_fft_summary.csv`, and
`results/comprehensive_v100_vectorized_fft_report.md`. Those rows supersede the
`logN=18/20` cuButterfly rows below:

| Workload | cuButterfly | cuFFT | VkFFT | cuButterfly/cuFFT |
|:--|--:|--:|--:|--:|
| FP32 `logN=18`, batch 16 | 0.195942 ms | 0.211098 ms | 0.201103 ms | 1.077x |
| FP32 `logN=20`, batch 4 | 0.214733 ms | 0.210330 ms | 0.206520 ms | 0.979x |

The FP64 row was also followed by a focused physical-unit study. The original
table below intentionally remains the scalar comprehensive baseline; it is not
the current FP64 ceiling. A generated double-precision cuFFTDx unit, recurrence
twiddles, XOR-swizzled prefix staging, and strength-reduced addressing reach
0.339364 ms versus cuFFT's 0.343040 ms at the same `logN=16`, batch-64 shape
under the focused interleaved five-trial protocol. The expanded `logN=14..18`
matrix has a higher cuButterfly median on 20 of 25 stable shapes. See [FP64
robustness](../results/fp64_robustness_batch_analysis.md); focused and
comprehensive rows must not be mixed as one protocol.

## FFT Against Runnable Libraries

Each group below has identical precision, transform length, batch, direction,
normalization, placement, and stride. A ratio above one means higher throughput
than cuFFT.

| Workload | N | Batch | cuButterfly | cuFFT | VkFFT | cuButterfly/cuFFT |
|:--|--:|--:|--:|--:|--:|--:|
| FP32 forward in-place, `logN=8` | 256 | 16,384 | 0.084234 ms | 0.083825 ms | 0.083548 ms | 0.995x |
| FP32 inverse normalized, `logN=12` | 4,096 | 1,024 | 0.088064 ms | 0.172892 ms | - | 1.963x |
| FP32 forward in-place, `logN=14` | 16,384 | 256 | 0.121129 ms | 0.121610 ms | 0.171766 ms | 1.004x |
| FP64 forward, `logN=16` | 65,536 | 64 | 0.534313 ms | 0.342804 ms | - | 0.642x |
| FP32 forward in-place, `logN=18` | 262,144 | 16 | 0.205087 ms | 0.211343 ms | 0.201083 ms | 1.031x |
| FP32 forward in-place, `logN=20` | 1,048,576 | 4 | 0.235438 ms | 0.210063 ms | 0.206275 ms | 0.892x |
| FP32 forward stride-2 in-place, `logN=8` | 256 | 16,384 | 0.172636 ms | 0.171428 ms | - | 0.993x |

The original FP32 forward result is therefore at library parity for `logN=8`
and `logN=14`, ahead of cuFFT by 3.1% at `logN=18`, and behind by 10.8% at
`logN=20`; the superseding rows above narrow the latter gap to 2.1%. The scalar
FP64 implementation is a clear processing-unit gap, while the focused
double-precision unit closes it at the selected shape. The inverse result is
semantically fair, but its 1.963x ratio mainly demonstrates fused
normalization: the cuButterfly path folds scaling into the transform while the
measured cuFFT path launches a separate normalization kernel. It must not be
presented as a 1.963x raw FFT core advantage.

## Internal Design-Space Evidence

These rows compare alternative mappings inside this repository. They establish
that the architecture parameters matter; without a fresh external reference
they do not establish superiority over another library.

| Operator and workload | N | Batch | Selected point | Alternative | Selected advantage |
|:--|--:|--:|:--|:--|--:|
| FWHT FP32, `logN=8` | 256 | 16,384 | warp-register 0.042742 ms | shared radix-4 0.048466 ms | 1.134x |
| FWHT FP32, `logN=15` | 32,768 | 128 | warp-register 0.064707 ms | online radix-4 0.316396 ms | 4.890x |
| FWHT FP32, `logN=20` | 1,048,576 | 4 | online radix-4 0.319560 ms | hierarchical radix-4 0.488079 ms | 1.527x |
| NTT 30-bit natural, `logN=12` | 4,096 | 1,024 | radix-4 0.140012 ms | radix-2 0.164291 ms | 1.173x |
| NTT 60-bit natural, `logN=16` | 65,536 | 64 | radix-4 0.277780 ms | radix-2 0.304835 ms | 1.097x |
| XOR-zeta uint32, `logN=20` | 1,048,576 | 4 | online radix-4 0.319304 ms | hierarchical radix-4 0.486994 ms | 1.525x |

The choices are not monotonic across operators or lengths. Register exchange
wins for small and medium FWHT, online reordering wins at long length, and NTT
radix selection changes the local arithmetic/resource balance. This is the
intended evidence for treating processing unit, stage unfolding, data
unfolding, residence, and layout as independent hardware-selected axes.

## Stability And Evidence Boundary

All 37 cases passed correctness. Of 37 median results, 36 are stable over their
full five-trial range. The short `logN=14` cuFFT case contains one repeatable
process-level timing outlier and is labelled `stable-with-outlier`; its central
three-trial range passes the 3% gate. The largest central relative range over
the suite is 0.82%.

Short per-process warmups exposed two V100 clock modes near the configured
1312 MHz application clock and the 1530 MHz boost clock. The full protocol uses
1000 warmups to reach the boosted steady state. The full range remains in the
CSV even when the robust central range is used for the stability decision.

Dao-AILab FHT and GPU-NTT were subsequently refreshed under the matching
five-trial protocol. They remain in a separate result table because their
supported batches and semantic contracts do not coincide with every
comprehensive-suite row. See [V100 External Baselines](v100_external_baselines.md).

## Current Conclusion

The defensible statement is: **cuButterfly reaches top-library performance on
selected V100 FP32 and FP64 FFT shapes and on matching-protocol FWHT/NTT
results.** FP32 `logN=20` at saturated batch and five FP64 crossover shapes
remain measurable gaps. The evidence does not support a blanket claim that
every operator, length, precision, and semantic mode is already faster than
the best specialized library.
