# FFT Resident Mapping NCU Analysis

## Protocol

- Tesla V100-SXM2-16GB, FP32 forward C2C, `logN=12`, batch 1024.
- One profiled transform invocation, base profiling clocks, cache flush disabled.
- All implementations read and write the same `2^22` complex values.
- NCU reports 93.248 us for cuFFT; the independent repeated benchmark reports
  88.750 us. Counter runs are used for relative attribution, not replacement
  of the five-trial timing table.

## Kernel Counters

| Mapping/kernel | Time us | DRAM R+W MiB | Warp inst. M | FP32 thread inst. M | Shared LD conflicts M | Shared ST conflicts M | Active warps | Barrier stall | Long scoreboard |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| two-pass first | 129.024 | 63.882 | 15.663 | 116.392 | 2.238 | 2.638 | 70.31% | 11.15% | 38.09% |
| two-pass second | 97.536 | 63.808 | 11.338 | 93.323 | 0.390 | 2.621 | 48.25% | 12.86% | 22.57% |
| resident 64x64 | 115.936 | 63.868 | 16.122 | 223.347 | 0.208 | 2.144 | 95.55% | 15.37% | 30.13% |
| direct 4096 | 94.304 | 63.576 | 11.010 | 206.045 | 0.017 | 1.049 | 47.39% | 11.03% | 32.10% |
| cuFFT | 93.248 | 63.794 | 8.290 | 207.356 | 0.034 | 0.271 | 57.20% | 8.43% | 24.60% |

## Findings

1. The two-pass boundary is physical, not merely logical. Summed over its two
   kernels, it moves 127.69 MiB through DRAM, almost exactly twice the resident,
   direct, and cuFFT paths. Removing that boundary explains the large first
   recovery.
2. The resident 64x64 kernel has already reached the minimum external traffic.
   Its 95.55% active-warp rate is the highest of all candidates, so occupancy is
   not the limiting resource.
3. Resident versus direct isolates local organization. Both move about 64 MiB
   externally and execute a similar number of FP32 thread instructions, but
   resident executes 46.4% more warp instructions, has 2.0x the shared-store
   conflicts, and spends 15.37% rather than 11.03% of eligible issue cycles on
   barrier stalls. It is 22.9% slower in the counter run.
4. Long-scoreboard stall is not worse in resident than direct. The remaining
   loss is therefore local exchange/control overhead, not an uncovered global
   memory dependency.
5. Direct cuFFTDx and cuFFT have nearly identical FP32 instruction counts,
   external traffic, DRAM utilization, and time. The independent timing sweep
   places them within 0.3%, validating the direct 4096-point unit as the V100
   `logN=12` selected point.

## Architecture Consequence

The schedule should first choose the largest efficient resident equivalent
unit, then apply two-dimensional spatial/temporal composition only beyond that
unit's hardware boundary. For V100 FP32 this experiment selects a direct
4096-point unit at `logN=12`; forcing a 64x64 composition adds exchange and
barrier work even though it avoids global scratch. At `logN=14`, the measured
persistent one-CTA form also loses to the spatial two-pass form, so maximum
temporal expansion is not a universal objective.

The next composed-kernel optimization target is shared exchange count and
layout, not occupancy: fuse cross-twiddle with the minimum required transpose,
use warp-register exchange where ownership permits, and avoid materializing
both cuFFTDx local-unit boundaries in shared memory.
