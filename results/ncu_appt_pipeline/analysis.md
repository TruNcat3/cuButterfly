# APPT Pipeline NCU Attribution

Matched Tesla V100 captures, uint64 `logN=20`. NCU reports one cooperative kernel for every implementation; the APPT backend does not use the CUDA Graph API.

## Fixed-Cost Fit

The two-point model is `T(batch)=T_fixed+batch*T_steady`. It includes cooperative admission and pipeline fill/drain in the fixed term; it does not misclassify repeated in-kernel handoffs as launch cost.

| implementation | batch 1 us | batch 4 us | fixed us | steady us/transform | fixed share b1 | fixed share b4 |
|:--|---:|---:|---:|---:|---:|---:|
| barrier | 158.0 | 541.2 | 30.2 | 127.8 | 19.1% | 5.6% |
| resident-10+10 | 230.3 | 614.9 | 102.1 | 128.2 | 44.3% | 16.6% |
| APPT Us8 role1 | 2158.3 | 8053.0 | 193.4 | 1964.9 | 9.0% | 2.4% |
| APPT Us8 role2 | 2990.3 | 11310.7 | 216.8 | 2773.5 | 7.3% | 1.9% |
| APPT Us7 role2 rep2 group2 | 654.2 | 2252.9 | 121.3 | 532.9 | 18.5% | 5.4% |
| APPT online folds | 674.0 | 2023.9 | 224.1 | 450.0 | 33.2% | 11.1% |

## Batch-1 Hardware Attribution

Ratios use the v0.6 barrier kernel as 1.0x.

| implementation | load sectors | store sectors | warp inst | integer inst | DRAM peak | active warps | barrier stall | wait stall | long scoreboard | regs | shared B | waves/SM |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| barrier | 1.00x | 1.00x | 1.00x | 1.00x | 34.6% | 22.9% | 15.4% | 14.4% | 38.0% | 48 | 32800 | 1.00 |
| resident-10+10 | 1.00x | 1.00x | 1.26x | 1.04x | 24.1% | 23.3% | 2.8% | 15.2% | 21.6% | 48 | 32816 | 1.00 |
| APPT Us8 role1 | 3.65x | 4.00x | 13.56x | 5.27x | 10.9% | 25.0% | 39.4% | 17.3% | 9.0% | 125 | 28728 | 1.00 |
| APPT Us8 role2 | 3.67x | 4.00x | 9.29x | 3.66x | 4.9% | 12.5% | 39.8% | 17.5% | 13.7% | 142 | 12312 | 0.67 |
| APPT Us7 role2 rep2 group2 | 3.54x | 4.01x | 3.88x | 2.06x | 24.4% | 25.0% | 23.3% | 18.2% | 30.9% | 116 | 24600 | 1.00 |
| APPT online folds | 3.14x | 4.50x | 1.73x | 0.91x | 11.9% | 18.3% | 30.4% | 10.1% | 33.9% | 80 | 33024 | 1.00 |

## Conclusion

The APPT fixed term is `224.1 us`, only `33.2%` of batch-1 time. Its fitted steady cost is `450.0 us/transform`, versus `127.8 us/transform` for v0.6. Therefore even infinite batching cannot close the present gap.

The initial Us8 capture attributes its sustained cost to token handoff polling, fold-layout traffic, excessive integer/index instructions, and register-limited latency hiding. The optional Us7 generated row measures the replicated/grouped-handoff optimization using the same counters. CUDA Graph capture could reduce application-side submission latency, but it cannot improve these in-kernel counters.
