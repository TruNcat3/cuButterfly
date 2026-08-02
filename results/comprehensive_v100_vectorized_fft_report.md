# Single-GPU Comprehensive Summary

Every row uses the listed transform length and batch. `Mtransform/s` and
`Gpoint/s` are derived from the median resident kernel time. Ratios above one
mean higher throughput than `Basis`; groups without a declared reference use
their fastest internal implementation as the basis.

## FFT Results

| Numeric | Semantics | logN | N | Batch | Layout | Implementation | Design point | Median ms | Mtransform/s | Gpoint/s | vs basis | Basis | Stability |
|:--|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|:--|
| fp32 | forward | 18 | 262,144 | 16 | in-place, es=1, bs=262144 | cuButterfly-auto-vectorized | online-reorder / cufftdx-block | 0.195942 | 0.082 | 21.406 | 1.077x | cuFFT | stable |
| fp32 | forward | 18 | 262,144 | 16 | in-place, es=1, bs=262144 | VkFFT | library | 0.201103 | 0.080 | 20.856 | 1.050x | cuFFT | stable |
| fp32 | forward | 18 | 262,144 | 16 | in-place, es=1, bs=262144 | cuFFT | library | 0.211098 | 0.076 | 19.869 | 1.000x | cuFFT | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | in-place, es=1, bs=1048576 | VkFFT | library | 0.206520 | 0.019 | 20.309 | 1.018x | cuFFT | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | in-place, es=1, bs=1048576 | cuFFT | library | 0.210330 | 0.019 | 19.942 | 1.000x | cuFFT | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | in-place, es=1, bs=1048576 | cuButterfly-auto-vectorized | online-reorder / cufftdx-block | 0.214733 | 0.019 | 19.533 | 0.979x | cuFFT | stable |

## External Evidence

- **fwht / Dao-AILab FHT**: measured-matching-protocol; five-trial correctness-checked refresh is archived separately. Existing evidence: `results/v100_external_baselines_summary.csv`.
- **ntt / GPU-NTT**: measured-matching-protocol; natural and native bit-reversed contracts are archived separately. Existing evidence: `results/v100_external_baselines_summary.csv`.
