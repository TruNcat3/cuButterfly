# FFT Vectorized-Boundary NCU Attribution

All cuButterfly rows use FP32 forward `logN=20`, batch 16. The fixed
rows use the same 512/512-thread, EPT 8/8 mapping, isolating the code change.
NCU replay time is attribution-only; CUDA-event scans remain authoritative.

| Configuration | CUDA-event ms | NCU time us | Kernels | DRAM read MiB | DRAM write MiB | Warp inst. | Bank conflicts | DRAM peak | Barrier stall | Scoreboard stall |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pre-vector fixed | 0.926618 | 917.600 | 2 | 274.49 | 256.63 | 138739712 | 19765589 | 67.6% | 11.9% | 32.8% |
| post-vector fixed | 0.829174 | 832.352 | 2 | 268.85 | 256.66 | 111607808 | 19873481 | 73.7% | 13.3% | 27.9% |
| post-vector selected | 0.764334 | 855.552 | 2 | 263.10 | 259.74 | 94928896 | 18111223 | 71.3% | 10.6% | 25.9% |
| cuFFT | 0.729416 | 773.568 | 2 | 256.01 | 257.27 | 69468160 | 2097152 | 77.4% | 4.7% | 21.3% |

## Controlled Changes

- Vectorizing the same fixed mapping changes replay time by -9.3%, warp instructions by -19.6%, and DRAM writes by +0.0%. The matching CUDA-event change is -10.5%.
- CUDA-event timing selects the 256/128 mapping: it is -7.8% versus the post-vector fixed point. NCU replay reverses that ordering by +2.8%; replay time must not be used as the mapping-performance result.
