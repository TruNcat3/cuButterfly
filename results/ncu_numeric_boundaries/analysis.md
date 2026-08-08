# Numeric Boundary NCU Attribution

NCU replay/base-clock time is mechanism evidence; CUDA-event timing remains the winner authority.

| Event | Workload | A | B | NCU winner | Ratio | Candidate mechanisms |
|:--|:--|:--|:--|:--|--:|:--|
| crossover-009 | fft fp16 logN=15 batch=5 | online-r4-t256 | online-r2-t256 | online-r2-t256 | 1.005x | resource-capacity+instruction-work |
| crossover-026 | structured-2x2 fp32 logN=15 batch=8 | hier-r4-t256 | online-r4-t256 | hier-r4-t256 | 1.814x | resource-capacity+grid-wave+synchronization+dependency-memory+instruction-work+memory-traffic+shared-conflict |
| crossover-044 | subset-zeta uint32 logN=8 batch=1281 | temporal-r2-t128 | temporal-r4-t128 | temporal-r4-t128 | 1.239x | resource-capacity+synchronization+dependency-memory+instruction-work+memory-traffic |
| crossover-053 | fft fp16 logN=8 batch=1281 | temporal-r4-t128 | temporal-r2-t128 | temporal-r4-t128 | 1.084x | resource-capacity+synchronization+instruction-work |
| crossover-064 | ntt uint64 logN=15 batch=241 | ntt-hybrid-r4-t256 | ntt-hybrid-r2-t256 | ntt-hybrid-r4-t256 | 1.083x | resource-capacity+synchronization+instruction-work |
| crossover-069 | fft fp64 logN=8 batch=961 | temporal-r2-t128 | temporal-r4-t128 | temporal-r4-t128 | 1.018x | resource-capacity+grid-wave+synchronization+dependency-memory+instruction-work+shared-conflict |

Mechanism labels are screening rules. The paper-facing explanation must inspect per-kernel rows and matched controls.
