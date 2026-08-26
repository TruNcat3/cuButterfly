# APPT Pipeline NCU Attribution

Matched Tesla V100 captures, uint64 `logN=20`. NCU reports one cooperative kernel for every implementation; the APPT backend does not use the CUDA Graph API.

## Fixed-Cost Fit

The two-point model is `T(batch)=T_fixed+batch*T_steady`. It includes cooperative admission and pipeline fill/drain in the fixed term; it does not misclassify repeated in-kernel handoffs as launch cost.

| implementation | batch 1 us | batch 4 us | fixed us | steady us/transform | fixed share b1 | fixed share b4 |
|:--|---:|---:|---:|---:|---:|---:|
| barrier | 153.9 | 546.9 | 22.9 | 131.0 | 14.9% | 4.2% |
| resident-10+10 | 229.9 | 609.4 | 103.3 | 126.5 | 45.0% | 17.0% |
| APPT Us8 role1 | 2227.2 | 8432.6 | 158.8 | 2068.4 | 7.1% | 1.9% |
| APPT Us8 role2 | 3001.9 | 11484.2 | 174.5 | 2827.4 | 5.8% | 1.5% |

## Batch-1 Hardware Attribution

Ratios use the v0.6 barrier kernel as 1.0x.

| implementation | load sectors | store sectors | warp inst | integer inst | DRAM peak | active warps | barrier stall | wait stall | long scoreboard | regs | shared B | waves/SM |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| barrier | 1.00x | 1.00x | 1.00x | 1.00x | 36.0% | 22.6% | 14.3% | 14.5% | 38.8% | 48 | 32800 | 1.00 |
| resident-10+10 | 1.00x | 1.00x | 1.26x | 1.04x | 24.1% | 23.6% | 2.7% | 15.2% | 21.6% | 48 | 32816 | 1.00 |
| APPT Us8 role1 | 4.17x | 4.00x | 13.50x | 5.28x | 10.5% | 25.0% | 39.2% | 16.7% | 9.2% | 127 | 28728 | 1.00 |
| APPT Us8 role2 | 4.18x | 4.00x | 9.18x | 3.68x | 4.8% | 12.5% | 39.7% | 17.1% | 14.3% | 132 | 12312 | 0.67 |

## Conclusion

The APPT fixed term is `158.8 us`, only `7.1%` of batch-1 time. Its fitted steady cost is `2068.4 us/transform`, versus `131.0 us/transform` for v0.6. Therefore even infinite batching cannot close the present gap.

The dominant costs are sustained inside the kernel: token handoff polling and barriers, 4x global sectors from fold layout traffic, 13.5x warp instructions, and 127 registers/thread. DRAM reaches only 10.5% of peak. Fusing two stages reduces instructions but halves active warps to 12.5%, so it is slower. CUDA Graph capture could reduce host submission latency for a multi-operation application, but it cannot improve these NCU kernel counters.
