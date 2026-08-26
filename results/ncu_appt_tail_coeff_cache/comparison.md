# APPT Coefficient-Tree Cache Attribution

Matched V100 base-clock NCU replay. The baseline is the selected readiness-policy capture; the after capture adds CTA-local coefficient reuse for the producer and low tail and uses the corresponding role weights.

| bits | batch | before us | cached us | speedup | load sectors change | store sectors change | warp instructions change | integer instructions change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 244.5 | 235.3 | 1.039x | -20.53% | +0.00% | -1.96% | -4.23% |
| 32 | 4 | 792.1 | 730.0 | 1.085x | -20.55% | +0.00% | -3.34% | -4.41% |
| 64 | 1 | 349.5 | 336.2 | 1.040x | -23.06% | +0.00% | -1.29% | -2.28% |
| 64 | 4 | 1113.1 | 1050.0 | 1.060x | -23.21% | +0.00% | -1.34% | -2.16% |

The cache removes repeated coefficient requests rather than a materialized state: global-load sectors fall by 20.5%--23.2% and replay time falls by 3.8%--7.8%, while store sectors are unchanged.

The L1 hit-rate percentage falls from 27.6%--30.9% to 14.2%--14.6% because the removed requests were the repeatedly reused, L1-friendly part of the stream. The remaining state, readiness, and final coefficient accesses have poorer L1 locality and remain mostly L2-served.

The residual register-tail traffic is 1.81x--2.85x the v0.6 load sectors and 2.25x--2.50x the store sectors. For uint32 the store count is approximately N+N/8 sectors; for uint64 it is N+N/4. The state1 write is coalesced, but one fixed-c tail task writes natural-order values at 128-element strides. Removing that sector term requires a multi-c tail microgroup, not another readiness change.

The current source additionally preserves the same coefficient tree across the low and high tail halves. That later change is validated by CUDA-event timing and correctness but is not included in this NCU capture.
