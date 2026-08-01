# FP64 FFT NCU Attribution

All rows are FP64 forward `logN=16`, batch 64 on V100. CUDA-event medians use
five process trials, 1000 warmups, and 100 repetitions. NCU replay time is
attribution-only and is not used to rank the implementations.

| Implementation | CUDA-event ms | NCU us | DRAM MiB | Warp inst. | FP64 inst. | Shared conflicts | DRAM peak | Barrier stall | Scoreboard stall |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| scalar | 0.521492 | 595.360 | 259.98 | 38076416 | 285212672 | 9243776 | 51.0% | 34.9% | 24.3% |
| cufftdx | 0.383949 | 409.536 | 257.26 | 33783808 | 268656384 | 10073373 | 73.3% | 15.4% | 34.1% |
| cufft | 0.342927 | 358.336 | 255.56 | 16252928 | 273678336 | 74358 | 83.2% | 4.4% | 39.4% |

## Attribution

- Replacing the scalar unit with cuFFTDx reduces CUDA-event time by 26.4% and NCU replay time by 31.2%. It also reduces FP64 instructions by 5.8%.
- Against cuFFT, cuFFTDx transfers 1.007x the DRAM bytes and executes 0.982x the FP64 instructions. The remaining gap is therefore not explained by external traffic volume or double-precision arithmetic work.
- cuFFTDx executes 2.08x the warp instructions and incurs 135.5x the shared-memory bank conflicts of cuFFT. Its weighted barrier and long-scoreboard stalls are 15.4% and 34.1%.

## Prefix/Suffix Split

| cuFFTDx pass | NCU us | Warp inst. | Shared conflicts | DRAM peak | Barrier stall | Scoreboard stall |
|:--|--:|--:|--:|--:|--:|--:|
| prefix + twiddle/reorder | 236.000 | 20152320 | 5790521 | 63.7% | 14.4% | 46.7% |
| suffix + natural-order store | 173.536 | 13631488 | 4282852 | 86.4% | 16.8% | 17.0% |

The prefix consumes 57.6% of replay time and is 1.36x slower than the suffix. The suffix already runs in 173.536 us, below either cuFFT pass (about 179 us). The next optimization target is therefore the prefix boundary: coalesced online input layout, cross-twiddle epilogue, shared-memory indexing, and synchronization. Increasing occupancy or changing FP64 arithmetic alone is not supported by these counters.
