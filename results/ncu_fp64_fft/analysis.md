# FP64 FFT NCU Attribution

All rows are FP64 forward `logN=16`, batch 64 on V100. CUDA-event medians use
five process trials, 1000 warmups, and 100 repetitions. NCU replay time is
attribution-only and is not used to rank the implementations.

| Implementation | CUDA-event ms | NCU us | DRAM MiB | Warp inst. | FP64 inst. | Shared conflicts | DRAM peak | Barrier stall | Scoreboard stall |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| scalar | 0.521492 | 597.312 | 259.91 | 38076416 | 285212672 | 9245248 | 50.8% | 34.8% | 24.4% |
| cufftdx | 0.383949 | 421.920 | 257.30 | 35127296 | 268656384 | 10284505 | 71.2% | 16.2% | 30.1% |
| cufft | 0.342927 | 361.120 | 255.51 | 16252928 | 273678336 | 70824 | 82.6% | 4.0% | 39.0% |
| recurrence | 0.361708 | 379.584 | 257.21 | 33652736 | 279969792 | 10291466 | 79.1% | 16.7% | 28.0% |
| xor_swizzle | 0.353526 | 367.648 | 257.16 | 35651584 | 279969792 | 7076984 | 81.6% | 15.5% | 31.7% |

## Attribution

- Replacing the scalar unit with cuFFTDx reduces CUDA-event time by 26.4% and NCU replay time by 29.4%. It also reduces FP64 instructions by 5.8%.
- Against cuFFT, cuFFTDx transfers 1.007x the DRAM bytes and executes 0.982x the FP64 instructions. The remaining gap is therefore not explained by external traffic volume or double-precision arithmetic work.
- cuFFTDx executes 2.16x the warp instructions and incurs 145.2x the shared-memory bank conflicts of cuFFT. Its weighted barrier and long-scoreboard stalls are 16.2% and 30.1%.
- Twiddle recurrence changes CUDA-event time by -5.8%, replay time by -10.0%, and FP64 instructions by +4.2%.
- XOR swizzle changes CUDA-event time by -2.3% and replay time by -3.1% versus linear recurrence. It finishes within 1.8% of cuFFT replay while still executing 2.19x its warp instructions and incurring 99.9x its shared conflicts.

## Prefix/Suffix Split

| cuFFTDx pass | NCU us | Warp inst. | FP64 inst. | Shared conflicts | DRAM peak | Barrier stall | Scoreboard stall |
|:--|--:|--:|--:|--:|--:|--:|--:|
| table prefix + twiddle/reorder | 249.824 | 21495808 | 135487232 | 6009354 | 60.2% | 15.5% | 39.2% |
| table suffix + natural-order store | 172.096 | 13631488 | 133169152 | 4275151 | 87.2% | 17.2% | 17.0% |
| recurrence prefix + twiddle/reorder | 206.016 | 20021248 | 146800640 | 6012796 | 72.9% | 16.5% | 37.3% |
| recurrence suffix + natural-order store | 173.568 | 13631488 | 133169152 | 4278670 | 86.4% | 16.9% | 17.0% |
| XOR-swizzle prefix + twiddle/reorder | 195.008 | 22020096 | 146800640 | 2793737 | 77.0% | 13.9% | 44.9% |
| XOR-swizzle suffix + natural-order store | 172.640 | 13631488 | 133169152 | 4283247 | 86.9% | 17.2% | 16.8% |

Recurrence reduces prefix replay time by 17.5%, warp instructions by 6.9%, and raises DRAM peak from 60.2% to 72.9%. It spends 8.4% more FP64 instructions while registers, shared allocation, and waves/SM remain unchanged. The suffix changes by only +0.9%.

The recurrence result confirms a useful architecture trade: idle FP64 capacity replaces dependent twiddle-table service in the prefix. Shared conflicts remain essentially unchanged, so the next target is the local exchange/address path rather than another twiddle change.

XOR swizzle changes prefix replay time by -5.3%, shared conflicts by -53.5%, and warp instructions by +10.0%. The suffix changes by -0.5%.
