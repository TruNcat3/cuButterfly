# Numeric Boundary NCU Attribution

NCU replay/base-clock time is mechanism evidence; CUDA-event timing remains the winner authority.

| Event | Workload | A | B | NCU winner | Ratio | Candidate mechanisms |
|:--|:--|:--|:--|:--|--:|:--|
| crossover-009 | fft fp16 logN=15 batch=5 | online-r4-t256 | online-r2-t256 | online-r2-t256 | 1.005x | register-footprint+instruction-work |
| crossover-026 | structured-2x2 fp32 logN=15 batch=8 | hier-r4-t256 | online-r4-t256 | hier-r4-t256 | 1.814x | register-footprint+grid-wave+synchronization+dependency-memory+instruction-work+shared-conflict |
| crossover-044 | subset-zeta uint32 logN=8 batch=1281 | temporal-r2-t128 | temporal-r4-t128 | temporal-r4-t128 | 1.239x | register-footprint+synchronization+dependency-memory+instruction-work |
| crossover-053 | fft fp16 logN=8 batch=1281 | temporal-r4-t128 | temporal-r2-t128 | temporal-r4-t128 | 1.084x | register-footprint+synchronization+instruction-work |
| crossover-064 | ntt uint64 logN=15 batch=241 | ntt-hybrid-r4-t256 | ntt-hybrid-r2-t256 | ntt-hybrid-r4-t256 | 1.083x | register-footprint+synchronization+instruction-work |
| crossover-069 | fft fp64 logN=8 batch=961 | temporal-r2-t128 | temporal-r4-t128 | temporal-r4-t128 | 1.018x | register-footprint+resource-capacity+grid-wave+synchronization+dependency-memory+instruction-work+shared-conflict |

## Counter Differences

| Event | Warp inst B/A | Registers A/B | Waves/SM A/B | Barrier A/B | Scoreboard A/B |
|:--|--:|--:|--:|--:|--:|
| crossover-009 | 1.292x | 32/24 | 3.00/3.00 | 40.4%/41.1% | 7.1%/9.2% |
| crossover-026 | 1.745x | 27/30 | 4.00/4.80 | 13.5%/34.9% | 20.7%/7.7% |
| crossover-044 | 0.553x | 16/17 | 1.00/1.00 | 9.6%/24.3% | 18.2%/25.9% |
| crossover-053 | 1.390x | 29/18 | 1.00/1.00 | 38.1%/10.6% | 17.5%/18.1% |
| crossover-064 | 1.191x | 32/28 | 36.15/36.15 | 22.1%/13.3% | 27.2%/26.6% |
| crossover-069 | 0.720x | 28/48 | 0.75/1.20 | 15.9%/30.1% | 42.9%/35.6% |

## Interpretation

- The FP16 `logN=15,batch=5` paths differ by only 0.5% under NCU. Radix-4 executes fewer warp instructions but uses more registers; this is a genuine flat tradeoff, consistent with overlapping CUDA-event ranges.
- Structured 2x2 `logN=15,batch=8` is a composition boundary rather than a local arithmetic limit. The online path executes substantially more warp work and barrier stall than hierarchical, so additional online handoff work reverses the quick winner.
- For uint32 zeta, FP16 FFT at `logN=8`, and uint64 NTT, radix-4 reduces instruction work enough to offset its larger register footprint and, in two cases, higher synchronization stall.
- FP64 `logN=8,batch=961` is the only pair where the tighter register footprint changes the resident-CTA bound. Radix-4 gains instruction efficiency but loses residency and raises barrier pressure, leaving only a 1.8% NCU difference and overlapping event-time ranges.

Mechanism labels are screening rules. CUDA-event timing remains the winner authority, and paper-facing attribution must retain the per-kernel controls.
