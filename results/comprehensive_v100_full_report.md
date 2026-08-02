# Single-GPU Comprehensive Summary

Every row uses the listed transform length and batch. `Mtransform/s` and
`Gpoint/s` are derived from the median resident kernel time. Ratios above one
mean higher throughput than `Basis`; groups without a declared reference use
their fastest internal implementation as the basis.

## FFT Results

| Numeric | Semantics | logN | N | Batch | Layout | Implementation | Design point | Median ms | Mtransform/s | Gpoint/s | vs basis | Basis | Stability |
|:--|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|:--|
| fp16-fp32 | forward | 3 | 8 | 524,288 | out-of-place, es=1, bs=8 | cuButterfly-WMMA | temporal-tile / wmma-dft8 / radix8 | 0.087020 | 6024.914 | 48.199 | 1.000x | cuButterfly-WMMA | stable |
| fp32 | forward | 8 | 256 | 16,384 | in-place, es=1, bs=256 | VkFFT | library | 0.083548 | 196.102 | 50.202 | 1.003x | cuFFT | stable |
| fp32 | forward | 8 | 256 | 16,384 | in-place, es=1, bs=256 | cuFFT | library | 0.083825 | 195.456 | 50.037 | 1.000x | cuFFT | stable |
| fp32 | forward | 8 | 256 | 16,384 | in-place, es=1, bs=256 | cuButterfly-cuFFTDx | temporal-tile / cufftdx-block | 0.084234 | 194.506 | 49.793 | 0.995x | cuFFT | stable |
| fp32 | forward | 8 | 256 | 16,384 | in-place, es=2, bs=518 | cuFFT | cufft | 0.171428 | 95.574 | 24.467 | 1.000x | cuFFT | stable |
| fp32 | forward | 8 | 256 | 16,384 | in-place, es=2, bs=518 | cuButterfly-scalar | temporal-tile / radix4 | 0.172636 | 94.905 | 24.296 | 0.993x | cuFFT | stable |
| fp32 | inverse, norm=inverse | 12 | 4,096 | 1,024 | out-of-place, es=1, bs=4096 | cuButterfly-cuFFTDx-direct | temporal-tile / cufftdx-direct | 0.088064 | 11.628 | 47.628 | 1.963x | cuFFT | stable |
| fp32 | inverse, norm=inverse | 12 | 4,096 | 1,024 | out-of-place, es=1, bs=4096 | cuFFT | cufft | 0.172892 | 5.923 | 24.260 | 1.000x | cuFFT | stable |
| fp32 | forward | 14 | 16,384 | 256 | in-place, es=1, bs=16384 | cuButterfly-cuFFTDx-direct | temporal-tile / cufftdx-direct | 0.121129 | 2.113 | 34.627 | 1.004x | cuFFT | stable |
| fp32 | forward | 14 | 16,384 | 256 | in-place, es=1, bs=16384 | cuFFT | library | 0.121610 | 2.105 | 34.490 | 1.000x | cuFFT | stable-with-outlier |
| fp32 | forward | 14 | 16,384 | 256 | in-place, es=1, bs=16384 | VkFFT | library | 0.171766 | 1.490 | 24.419 | 0.708x | cuFFT | stable |
| fp64 | forward | 16 | 65,536 | 64 | out-of-place, es=1, bs=65536 | cuFFT | cufft | 0.342804 | 0.187 | 12.235 | 1.000x | cuFFT | stable |
| fp64 | forward | 16 | 65,536 | 64 | out-of-place, es=1, bs=65536 | cuButterfly-scalar-online | online-reorder / radix4 | 0.534313 | 0.120 | 7.850 | 0.642x | cuFFT | stable |
| fp32 | forward | 18 | 262,144 | 16 | in-place, es=1, bs=262144 | VkFFT | library | 0.201083 | 0.080 | 20.859 | 1.051x | cuFFT | stable |
| fp32 | forward | 18 | 262,144 | 16 | in-place, es=1, bs=262144 | cuButterfly-cuFFTDx-online | online-reorder / cufftdx-block | 0.205087 | 0.078 | 20.451 | 1.031x | cuFFT | stable |
| fp32 | forward | 18 | 262,144 | 16 | in-place, es=1, bs=262144 | cuFFT | library | 0.211343 | 0.076 | 19.846 | 1.000x | cuFFT | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | in-place, es=1, bs=1048576 | VkFFT | library | 0.206275 | 0.019 | 20.334 | 1.018x | cuFFT | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | in-place, es=1, bs=1048576 | cuFFT | library | 0.210063 | 0.019 | 19.967 | 1.000x | cuFFT | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | in-place, es=1, bs=1048576 | cuButterfly-cuFFTDx-online | online-reorder / cufftdx-block | 0.235438 | 0.017 | 17.815 | 0.892x | cuFFT | stable |

## NTT Results

| Numeric | Semantics | logN | N | Batch | Layout | Implementation | Design point | Median ms | Mtransform/s | Gpoint/s | vs basis | Basis | Stability |
|:--|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|:--|
| uint32/30-bit | forward | 12 | 4,096 | 1,024 | n/a, es=1, bs=4096, natural | cuNTT-Hybrid2D-radix4 | hybrid2d / radix4 | 0.140012 | 7.314 | 29.957 | 1.173x | cuNTT-Hybrid2D-radix2 | stable |
| uint32/30-bit | forward | 12 | 4,096 | 1,024 | n/a, es=1, bs=4096, natural | cuNTT-Hybrid2D-radix2 | hybrid2d / radix2 | 0.164291 | 6.233 | 25.530 | 1.000x | cuNTT-Hybrid2D-radix2 | stable |
| uint64/60-bit | inverse | 16 | 65,536 | 64 | n/a, es=1, bs=65536, natural | cuNTT-Hybrid2D-radix2 | hybrid2d / radix2 | 0.307671 | 0.208 | 13.632 | 1.000x | cuNTT-Hybrid2D-radix2 | stable |
| uint64/60-bit | forward | 16 | 65,536 | 64 | n/a, es=1, bs=65536, natural | cuNTT-Hybrid2D-radix4 | hybrid2d / radix4 | 0.277780 | 0.230 | 15.099 | 1.000x | cuNTT-Hybrid2D-radix4 | stable |
| uint64/60-bit | forward | 16 | 65,536 | 64 | n/a, es=1, bs=65536, natural | cuNTT-Hybrid2D-radix2 | hybrid2d / radix2 | 0.304835 | 0.210 | 13.759 | 0.911x | cuNTT-Hybrid2D-radix4 | stable |
| uint64/60-bit | forward | 20 | 1,048,576 | 4 | n/a, es=1, bs=1048576, bit-reversed | cuNTT-compact-stage | compact-stage / radix2 | 0.341924 | 0.012 | 12.267 | 1.000x | cuNTT-compact-stage | stable |

## FWHT Results

| Numeric | Semantics | logN | N | Batch | Layout | Implementation | Design point | Median ms | Mtransform/s | Gpoint/s | vs basis | Basis | Stability |
|:--|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|:--|
| fp32 | forward | 8 | 256 | 16,384 | out-of-place, es=1, bs=256 | cuButterfly-warp-register | temporal-tile / warp-register / radix2 | 0.042742 | 383.323 | 98.131 | 1.000x | cuButterfly-warp-register | stable |
| fp32 | forward | 8 | 256 | 16,384 | out-of-place, es=1, bs=256 | cuButterfly-shared-radix4 | temporal-tile / radix4 | 0.048466 | 338.051 | 86.541 | 0.882x | cuButterfly-warp-register | stable |
| fp64 | forward | 12 | 4,096 | 1,024 | out-of-place, es=1, bs=4096 | cuButterfly-online-radix4 | online-reorder / radix4 | 0.373197 | 2.744 | 11.239 | 1.000x | cuButterfly-online-radix4 | stable |
| fp32 | forward | 15 | 32,768 | 128 | out-of-place, es=1, bs=32768 | cuButterfly-warp-register | temporal-tile / warp-register / radix2 | 0.064707 | 1.978 | 64.820 | 1.000x | cuButterfly-warp-register | stable |
| fp32 | forward | 15 | 32,768 | 128 | out-of-place, es=1, bs=32768 | cuButterfly-online-radix4 | online-reorder / radix4 | 0.316396 | 0.405 | 13.257 | 0.205x | cuButterfly-warp-register | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | out-of-place, es=1, bs=1048576 | cuButterfly-online-radix4 | online-reorder / radix4 | 0.319560 | 0.013 | 13.125 | 1.000x | cuButterfly-online-radix4 | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | out-of-place, es=1, bs=1048576 | cuButterfly-hierarchical-radix4 | hierarchical / radix4 | 0.488079 | 0.008 | 8.593 | 0.655x | cuButterfly-online-radix4 | stable |

## XOR-ZETA Results

| Numeric | Semantics | logN | N | Batch | Layout | Implementation | Design point | Median ms | Mtransform/s | Gpoint/s | vs basis | Basis | Stability |
|:--|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|:--|
| uint32 | forward | 8 | 256 | 16,384 | out-of-place, es=1, bs=256 | cuButterfly-radix4 | temporal-tile / radix4 | 0.047759 | 343.056 | 87.822 | 1.000x | cuButterfly-radix4 | stable |
| uint32 | forward | 8 | 256 | 16,384 | out-of-place, es=1, bs=256 | cuButterfly-radix2 | temporal-tile / radix2 | 0.049766 | 329.221 | 84.281 | 0.960x | cuButterfly-radix4 | stable |
| uint32 | inverse | 12 | 4,096 | 1,024 | out-of-place, es=1, bs=4096 | cuButterfly-online-radix4 | online-reorder / radix4 | 0.354243 | 2.891 | 11.840 | 1.000x | cuButterfly-online-radix4 | stable |
| uint32 | forward | 20 | 1,048,576 | 4 | out-of-place, es=1, bs=1048576 | cuButterfly-online-radix4 | online-reorder / radix4 | 0.319304 | 0.013 | 13.136 | 1.000x | cuButterfly-online-radix4 | stable |
| uint32 | forward | 20 | 1,048,576 | 4 | out-of-place, es=1, bs=1048576 | cuButterfly-hierarchical-radix4 | hierarchical / radix4 | 0.486994 | 0.008 | 8.613 | 0.656x | cuButterfly-online-radix4 | stable |

## External Evidence

- **fwht / Dao-AILab FHT**: measured-matching-protocol; five-trial correctness-checked refresh is archived separately. Existing evidence: `results/v100_external_baselines_summary.csv`.
- **ntt / GPU-NTT**: measured-matching-protocol; natural and native bit-reversed contracts are archived separately. Existing evidence: `results/v100_external_baselines_summary.csv`.
