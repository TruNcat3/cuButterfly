# Single-GPU Comprehensive Summary

Every row uses the listed transform length and batch. `Mtransform/s` and
`Gpoint/s` are derived from the median resident kernel time. Ratios above one
mean higher throughput than `Basis`; groups without a declared reference use
their fastest internal implementation as the basis.

## FFT Results

| Numeric | Semantics | logN | N | Batch | Layout | Implementation | Design point | Median ms | Mtransform/s | Gpoint/s | vs basis | Basis | Stability |
|:--|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|:--|
| fp32 | forward | 8 | 256 | 16,384 | in-place, es=1, bs=256 | cuFFT | library | 0.084224 | 194.529 | 49.799 | 1.000x | cuFFT | stable |
| fp32 | forward | 8 | 256 | 16,384 | in-place, es=1, bs=256 | VkFFT | library | 0.084275 | 194.411 | 49.769 | 0.999x | cuFFT | stable |
| fp32 | forward | 8 | 256 | 16,384 | in-place, es=1, bs=256 | cuButterfly-cuFFTDx | temporal-tile / cufftdx-block | 0.084838 | 193.121 | 49.439 | 0.993x | cuFFT | stable |
| fp32 | forward | 14 | 16,384 | 256 | in-place, es=1, bs=16384 | cuButterfly-cuFFTDx-direct | temporal-tile / cufftdx-direct | 0.133683 | 1.915 | 31.375 | 1.015x | cuFFT | stable |
| fp32 | forward | 14 | 16,384 | 256 | in-place, es=1, bs=16384 | cuFFT | library | 0.135731 | 1.886 | 30.902 | 1.000x | cuFFT | stable |
| fp32 | forward | 14 | 16,384 | 256 | in-place, es=1, bs=16384 | VkFFT | library | 0.173722 | 1.474 | 24.144 | 0.781x | cuFFT | stable |
| fp32 | forward | 18 | 262,144 | 16 | in-place, es=1, bs=262144 | cuButterfly-cuFFTDx-online | online-reorder / cufftdx-block | 0.212122 | 0.075 | 19.773 | 1.067x | cuFFT | stable |
| fp32 | forward | 18 | 262,144 | 16 | in-place, es=1, bs=262144 | VkFFT | library | 0.213299 | 0.075 | 19.664 | 1.061x | cuFFT | stable |
| fp32 | forward | 18 | 262,144 | 16 | in-place, es=1, bs=262144 | cuFFT | library | 0.226254 | 0.071 | 18.538 | 1.000x | cuFFT | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | in-place, es=1, bs=1048576 | VkFFT | library | 0.217856 | 0.018 | 19.253 | 1.031x | cuFFT | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | in-place, es=1, bs=1048576 | cuFFT | library | 0.224512 | 0.018 | 18.682 | 1.000x | cuFFT | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | in-place, es=1, bs=1048576 | cuButterfly-cuFFTDx-online | online-reorder / cufftdx-block | 0.247245 | 0.016 | 16.964 | 0.908x | cuFFT | stable |

## NTT Results

| Numeric | Semantics | logN | N | Batch | Layout | Implementation | Design point | Median ms | Mtransform/s | Gpoint/s | vs basis | Basis | Stability |
|:--|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|:--|
| uint32/30-bit | forward | 12 | 4,096 | 1,024 | n/a, es=1, bs=4096, natural | cuNTT-Hybrid2D-radix4 | hybrid2d / radix4 | 0.157901 | 6.485 | 26.563 | 1.169x | cuNTT-Hybrid2D-radix2 | stable |
| uint32/30-bit | forward | 12 | 4,096 | 1,024 | n/a, es=1, bs=4096, natural | cuNTT-Hybrid2D-radix2 | hybrid2d / radix2 | 0.184627 | 5.546 | 22.718 | 1.000x | cuNTT-Hybrid2D-radix2 | stable |
| uint64/60-bit | forward | 16 | 65,536 | 64 | n/a, es=1, bs=65536, natural | cuNTT-Hybrid2D-radix4 | hybrid2d / radix4 | 0.306074 | 0.209 | 13.704 | 1.000x | cuNTT-Hybrid2D-radix4 | stable |
| uint64/60-bit | forward | 16 | 65,536 | 64 | n/a, es=1, bs=65536, natural | cuNTT-Hybrid2D-radix2 | hybrid2d / radix2 | 0.332288 | 0.193 | 12.622 | 0.921x | cuNTT-Hybrid2D-radix4 | stable |
| uint64/60-bit | forward | 20 | 1,048,576 | 4 | n/a, es=1, bs=1048576, bit-reversed | cuNTT-compact-stage | compact-stage / radix2 | 0.349901 | 0.011 | 11.987 | 1.000x | cuNTT-compact-stage | stable |

## FWHT Results

| Numeric | Semantics | logN | N | Batch | Layout | Implementation | Design point | Median ms | Mtransform/s | Gpoint/s | vs basis | Basis | Stability |
|:--|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|:--|
| fp32 | forward | 8 | 256 | 16,384 | out-of-place, es=1, bs=256 | cuButterfly-warp-register | temporal-tile / warp-register / radix2 | 0.043264 | 378.698 | 96.947 | 1.000x | cuButterfly-warp-register | stable |
| fp32 | forward | 8 | 256 | 16,384 | out-of-place, es=1, bs=256 | cuButterfly-shared-radix4 | temporal-tile / radix4 | 0.051456 | 318.408 | 81.512 | 0.841x | cuButterfly-warp-register | stable |
| fp32 | forward | 15 | 32,768 | 128 | out-of-place, es=1, bs=32768 | cuButterfly-warp-register | temporal-tile / warp-register / radix2 | 0.069478 | 1.842 | 60.369 | 1.000x | cuButterfly-warp-register | stable |
| fp32 | forward | 15 | 32,768 | 128 | out-of-place, es=1, bs=32768 | cuButterfly-online-radix4 | online-reorder / radix4 | 0.369920 | 0.346 | 11.338 | 0.188x | cuButterfly-warp-register | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | out-of-place, es=1, bs=1048576 | cuButterfly-online-radix4 | online-reorder / radix4 | 0.372890 | 0.011 | 11.248 | 1.000x | cuButterfly-online-radix4 | stable |
| fp32 | forward | 20 | 1,048,576 | 4 | out-of-place, es=1, bs=1048576 | cuButterfly-hierarchical-radix4 | hierarchical / radix4 | 0.495002 | 0.008 | 8.473 | 0.753x | cuButterfly-online-radix4 | stable |

## XOR-ZETA Results

| Numeric | Semantics | logN | N | Batch | Layout | Implementation | Design point | Median ms | Mtransform/s | Gpoint/s | vs basis | Basis | Stability |
|:--|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|:--|
| uint32 | forward | 8 | 256 | 16,384 | out-of-place, es=1, bs=256 | cuButterfly-radix4 | temporal-tile / radix4 | 0.050227 | 326.199 | 83.507 | 1.000x | cuButterfly-radix4 | stable |
| uint32 | forward | 8 | 256 | 16,384 | out-of-place, es=1, bs=256 | cuButterfly-radix2 | temporal-tile / radix2 | 0.052531 | 311.892 | 79.844 | 0.956x | cuButterfly-radix4 | stable |
| uint32 | forward | 20 | 1,048,576 | 4 | out-of-place, es=1, bs=1048576 | cuButterfly-online-radix4 | online-reorder / radix4 | 0.372480 | 0.011 | 11.260 | 1.000x | cuButterfly-online-radix4 | stable |
| uint32 | forward | 20 | 1,048,576 | 4 | out-of-place, es=1, bs=1048576 | cuButterfly-hierarchical-radix4 | hierarchical / radix4 | 0.493875 | 0.008 | 8.493 | 0.754x | cuButterfly-online-radix4 | stable |

## External Coverage Gaps

- **fwht / Dao-AILab FHT**: not-runnable; PyTorch is not installed in the active Python environment. Existing evidence: `results/external_dao_fht_v100_summary.csv`.
- **ntt / GPU-NTT**: not-runnable; the external comparator harness binary is not present. Existing evidence: `results/compact_stage_logN20.csv`.
