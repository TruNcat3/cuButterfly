# FP64 FFT NCU Attribution

All rows are FP64 forward `logN=16`, batch 64 on V100. CUDA-event medians use
five process trials, 1000 warmups, and 100 repetitions. NCU replay time is
attribution-only and is not used to rank the implementations.

| Implementation | CUDA-event ms | NCU us | DRAM MiB | Warp inst. | Integer inst. | FP64 inst. | Shared conflicts | DRAM peak | Barrier stall | Scoreboard stall |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| scalar | 0.521226 | 597.632 | 260.11 | 38076416 | 512769792 | 285212672 | 9245706 | 50.8% | 34.9% | 24.3% |
| cufftdx | 0.369459 | 387.616 | 257.33 | 30507008 | 409993216 | 268656384 | 9886634 | 77.5% | 18.4% | 23.8% |
| cufft | 0.342856 | 359.744 | 255.40 | 16252928 | 108536656 | 273678336 | 60026 | 82.9% | 4.0% | 39.8% |
| recurrence | 0.345027 | 362.656 | 257.00 | 29392896 | 381681664 | 279969792 | 10114620 | 82.7% | 19.1% | 19.7% |
| xor_swizzle | 0.342098 | 350.880 | 257.09 | 29753344 | 393216000 | 279969792 | 6826525 | 85.5% | 17.7% | 24.2% |

## Attribution

- Replacing the scalar unit with cuFFTDx reduces CUDA-event time by 29.1% and NCU replay time by 35.1%. It also reduces FP64 instructions by 5.8%.
- Against cuFFT, cuFFTDx transfers 1.008x the DRAM bytes and executes 0.982x the FP64 instructions. The remaining gap is therefore not explained by external traffic volume or double-precision arithmetic work.
- cuFFTDx executes 1.88x the warp instructions and incurs 164.7x the shared-memory bank conflicts of cuFFT. Its weighted barrier and long-scoreboard stalls are 18.4% and 23.8%.
- Twiddle recurrence changes CUDA-event time by -6.6%, replay time by -6.4%, and FP64 instructions by +4.2%.
- XOR swizzle changes CUDA-event time by -0.8% and replay time by -3.2% versus linear recurrence. It finishes 2.5% below cuFFT replay while still executing 1.83x its warp instructions, 3.62x its integer instructions, and incurring 113.7x its shared conflicts.

## Prefix/Suffix Split

| cuFFTDx pass | NCU us | Warp inst. | Integer inst. | FP64 inst. | Shared conflicts | DRAM peak | Barrier stall | Scoreboard stall |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| table prefix + twiddle/reorder | 214.464 | 16875520 | 207618048 | 135487232 | 5607036 | 70.1% | 19.6% | 29.5% |
| table suffix + natural-order store | 173.152 | 13631488 | 202375168 | 133169152 | 4279598 | 86.6% | 17.0% | 16.7% |
| recurrence prefix + twiddle/reorder | 189.344 | 15761408 | 179306496 | 146800640 | 5828681 | 79.2% | 21.1% | 22.1% |
| recurrence suffix + natural-order store | 173.312 | 13631488 | 202375168 | 133169152 | 4285939 | 86.5% | 16.9% | 17.1% |
| XOR-swizzle prefix + twiddle/reorder | 178.912 | 16121856 | 190840832 | 146800640 | 2552475 | 83.9% | 18.4% | 31.2% |
| XOR-swizzle suffix + natural-order store | 171.968 | 13631488 | 202375168 | 133169152 | 4274050 | 87.2% | 17.0% | 16.9% |

Recurrence reduces prefix replay time by 11.7%, warp instructions by 6.6%, and raises DRAM peak from 70.1% to 79.2%. It spends 8.4% more FP64 instructions while registers, shared allocation, and waves/SM remain unchanged. The suffix changes by only +0.1%.

The recurrence result confirms a useful architecture trade: idle FP64 capacity replaces dependent twiddle-table service in the prefix. Shared conflicts remain essentially unchanged, so the next target is the local exchange/address path rather than another twiddle change.

XOR swizzle changes prefix replay time by -5.5%, shared conflicts by -56.2%, and warp instructions by +2.3%. Integer instructions change by +6.4%. The suffix changes by -0.8%.
