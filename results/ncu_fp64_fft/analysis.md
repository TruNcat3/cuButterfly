# FP64 FFT NCU Attribution

All rows are FP64 forward `logN=16`, batch 64 on V100. CUDA-event medians use
five process trials, 1000 warmups, and 100 repetitions. NCU replay time is
attribution-only and is not used to rank the implementations.

| Implementation | CUDA-event ms | NCU us | DRAM MiB | Warp inst. | FP64 inst. | Shared conflicts | DRAM peak | Barrier stall | Scoreboard stall |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| scalar | 0.521492 | 595.136 | 259.77 | 38076416 | 285212672 | 9251555 | 50.9% | 34.9% | 24.2% |
| cufftdx | 0.383949 | 419.936 | 257.33 | 34996224 | 268656384 | 10289107 | 71.5% | 16.0% | 30.1% |
| cufft | 0.342927 | 358.240 | 255.45 | 16252928 | 273678336 | 57157 | 83.2% | 4.1% | 39.3% |
| recurrence | 0.361708 | 380.832 | 257.02 | 33521664 | 279969792 | 10298406 | 78.8% | 16.8% | 27.9% |

## Attribution

- Replacing the scalar unit with cuFFTDx reduces CUDA-event time by 26.4% and NCU replay time by 29.4%. It also reduces FP64 instructions by 5.8%.
- Against cuFFT, cuFFTDx transfers 1.007x the DRAM bytes and executes 0.982x the FP64 instructions. The remaining gap is therefore not explained by external traffic volume or double-precision arithmetic work.
- cuFFTDx executes 2.15x the warp instructions and incurs 180.0x the shared-memory bank conflicts of cuFFT. Its weighted barrier and long-scoreboard stalls are 16.0% and 30.1%.
- Twiddle recurrence changes CUDA-event time by -5.8%, replay time by -9.3%, and FP64 instructions by +4.2%.

## Prefix/Suffix Split

| cuFFTDx pass | NCU us | Warp inst. | FP64 inst. | Shared conflicts | DRAM peak | Barrier stall | Scoreboard stall |
|:--|--:|--:|--:|--:|--:|--:|--:|
| table prefix + twiddle/reorder | 247.136 | 21364736 | 135487232 | 6007490 | 60.8% | 15.5% | 39.0% |
| table suffix + natural-order store | 172.800 | 13631488 | 133169152 | 4281617 | 86.8% | 16.7% | 17.4% |
| recurrence prefix + twiddle/reorder | 207.776 | 19890176 | 146800640 | 6020286 | 72.2% | 16.5% | 37.2% |
| recurrence suffix + natural-order store | 173.056 | 13631488 | 133169152 | 4278120 | 86.6% | 17.1% | 16.7% |

Recurrence reduces prefix replay time by 15.9%, warp instructions by 6.9%, and raises DRAM peak from 60.8% to 72.2%. It spends 8.4% more FP64 instructions while registers, shared allocation, and waves/SM remain unchanged. The suffix changes by only +0.1%.

The recurrence result confirms a useful architecture trade: idle FP64 capacity replaces dependent twiddle-table service in the prefix. Shared conflicts remain essentially unchanged, so the next target is the local exchange/address path rather than another twiddle change.
