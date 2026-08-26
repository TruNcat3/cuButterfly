# APPT Pipeline NCU Attribution

Matched Tesla V100 captures, uint64 `logN=20`. NCU reports one cooperative kernel for every implementation; the APPT backend does not use the CUDA Graph API.

## Fixed-Cost Fit

The two-point model is `T(batch)=T_fixed+batch*T_steady`. It includes cooperative admission and pipeline fill/drain in the fixed term; it does not misclassify repeated in-kernel handoffs as launch cost.

| implementation | batch 1 us | batch 4 us | fixed us | steady us/transform | fixed share b1 | fixed share b4 |
|:--|---:|---:|---:|---:|---:|---:|
| barrier | 155.4 | 530.1 | 30.4 | 124.9 | 19.6% | 5.7% |
| resident-10+10 | 227.8 | 602.5 | 102.9 | 124.9 | 45.2% | 17.1% |
| APPT Us8 role1 | 2143.3 | 8055.8 | 172.5 | 1970.8 | 8.0% | 2.1% |
| APPT Us8 role2 | 2992.1 | 11388.4 | 193.4 | 2798.8 | 6.5% | 1.7% |
| APPT Us7 role2 rep2 group2 | 659.4 | 2282.2 | 118.5 | 540.9 | 18.0% | 5.2% |

## Batch-1 Hardware Attribution

Ratios use the v0.6 barrier kernel as 1.0x.

| implementation | load sectors | store sectors | warp inst | integer inst | DRAM peak | active warps | barrier stall | wait stall | long scoreboard | regs | shared B | waves/SM |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| barrier | 1.00x | 1.00x | 1.00x | 1.00x | 35.2% | 22.9% | 15.1% | 14.6% | 37.5% | 48 | 32800 | 1.00 |
| resident-10+10 | 1.02x | 1.00x | 1.26x | 1.04x | 24.3% | 23.5% | 2.8% | 15.3% | 21.8% | 48 | 32816 | 1.00 |
| APPT Us8 role1 | 3.67x | 4.00x | 13.53x | 5.26x | 10.9% | 25.0% | 39.1% | 17.3% | 8.9% | 125 | 28728 | 1.00 |
| APPT Us8 role2 | 3.69x | 4.00x | 9.29x | 3.66x | 4.8% | 12.5% | 39.7% | 17.6% | 13.6% | 142 | 12312 | 0.67 |
| APPT Us7 role2 rep2 group2 | 3.56x | 4.00x | 3.86x | 2.06x | 24.0% | 25.0% | 24.2% | 18.4% | 30.8% | 116 | 24600 | 1.00 |

## Conclusion

The APPT fixed term is `118.5 us`, only `18.0%` of batch-1 time. Its fitted steady cost is `540.9 us/transform`, versus `124.9 us/transform` for v0.6. Therefore even infinite batching cannot close the present gap.

The initial Us8 capture attributes its sustained cost to token handoff polling, fold-layout traffic, excessive integer/index instructions, and register-limited latency hiding. The optional Us7 generated row measures the replicated/grouped-handoff optimization using the same counters. CUDA Graph capture could reduce application-side submission latency, but it cannot improve these in-kernel counters.
