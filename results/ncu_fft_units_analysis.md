# FFT Processing-Unit NCU Analysis

## Controlled Run

The administrator NCU run uses a Tesla V100-SXM2-16GB, `logN=3`, batch
524,288, forward resident-transform semantics, and one profiled kernel. The
WMMA row has mixed FP16-input/FP32-accumulation semantics; the other rows are
FP32.

| Core | NCU us | DRAM peak | L2 hit | Warp instructions | FP32 thread instructions | Tensor warp instructions | Active warps | Barrier stall | Long scoreboard | Registers | Shared |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| scalar shared radix-8 | 862.944 | 8.69% | 49.98% | 172.49M | 60.29M | 0 | 45.65% | 61.56% | 9.24% | 48 | 64 B |
| thread-register DFT8 | 154.208 | 51.62% | 89.86% | 3.26M | 58.72M | 0 | 65.57% | 0% | 5.73% | 34 | 0 |
| WMMA DFT8 mixed | 89.152 | 82.66% | 49.97% | 11.08M | 16.78M | 524,288 | 83.45% | 7.50% | 59.41% | 32 | 12,800 B |
| cuFFT | 91.808 | 83.53% | 83.29% | 3.67M | 62.91M | 0 | 88.84% | 6.52% | 19.06% | 30 | 2,048 B |

## Attribution

The scalar control is synchronization dominated. Its 61.56% barrier stall,
45.65% active warps, and 172.49M warp instructions explain why adding more
floating-point variants did not close the gap.

The thread-register core removes that failure mode: barrier stall becomes zero
and warp instructions fall by 52.9x. Its arithmetic instruction count is
already slightly below cuFFT. The remaining loss is the warp-level transaction
shape: every lane owns one contiguous transform, so corresponding element
loads across a warp are separated by 64 bytes. This produces only 51.62% DRAM
peak and 65.57% active warps despite a small register footprint.

WMMA is executing real Tensor Core work: 524,288 Tensor warp instructions were
recorded. Tensor-pipe activity is only 2.92%, while DRAM reaches 82.66% and
long-scoreboard stall reaches 59.41%. Therefore Tensor arithmetic is not the
limiter at this design point; operand movement is. Its memory and active-warp
levels already resemble cuFFT.

## NCU-Driven Design Point

The new generated `cta-dft8` retains one FP32 DFT8 per thread but cooperatively
loads and stores a `128 x 8` transform tile in contiguous order. The CTA uses
34 registers/thread and 8 KiB dynamic shared memory. CUDA-event medians over
five trials are:

| Core | Precision | Median ms | Throughput relative to cuFFT |
|:--|:--|--:|--:|
| thread-register DFT8 | FP32 | 0.131000 | 65.9% |
| CTA-staged register DFT8 | FP32 | 0.096645 | 89.3% |
| WMMA DFT8 | mixed FP16/FP32 | 0.089590 | not an FP32 ranking |
| cuFFT | FP32 | 0.086323 | 100% |

CTA staging improves the FP32 register core by 1.36x and leaves only a 12.0%
latency gap to cuFFT at DFT8. This selects `cta-dft8`, not the uncoalesced
thread core, as the FP32 local codelet for long-transform generation.

The NCU table predates the CTA design point and the small WMMA input-load
deduplication. `scripts/profile_fft_units_ncu.sh` now includes CTA DFT8, so a
later paper-quality refresh can measure the new transaction and stall profile
without changing commands.
