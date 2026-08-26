# APPT Writer-Final Screen

Median CUDA-event time. The grouped baseline uses the corrected group-32 layout with the prior selected mapping.

## Best Overall

| bits | batch | fragment | writer tiles | roles (P/T/W) | kernel ms | throughput/v0.6 | throughput/grouped |
|---:|---:|---:|---:|---|---:|---:|---:|
| 32 | 1 | 8 | 1 | `6-12-4` | 0.174353 | 0.850x | 1.111x |
| 32 | 4 | 8 | 2 | `6-12-4` | 0.510054 | 0.685x | 1.190x |
| 32 | 16 | 16 | 2 | `6-12-4` | 1.737694 | 0.593x | 1.158x |
| 64 | 1 | 8 | 2 | `8-8-4` | 0.286583 | 0.710x | 0.996x |
| 64 | 4 | 8 | 2 | `8-8-4` | 0.936687 | 0.651x | 1.028x |
| 64 | 16 | 16 | 2 | `8-8-4` | 3.067631 | 0.601x | 0.994x |

## Best Per Fragment

| bits | batch | fragment | writer tiles | roles (P/T/W) | kernel ms | throughput/grouped |
|---:|---:|---:|---:|---|---:|---:|
| 32 | 1 | 8 | 1 | `6-12-4` | 0.174353 | 1.111x |
| 32 | 4 | 8 | 2 | `6-12-4` | 0.510054 | 1.190x |
| 32 | 16 | 16 | 2 | `6-12-4` | 1.737694 | 1.158x |
| 64 | 1 | 8 | 2 | `8-8-4` | 0.286583 | 0.996x |
| 64 | 4 | 8 | 2 | `8-8-4` | 0.936687 | 1.028x |
| 64 | 16 | 16 | 2 | `8-8-4` | 3.067631 | 0.994x |

Writer-final preserves the `7+7+6` logical graph. It moves stage 19 into the existing static-to-natural writer so the final coefficient axis is lane-coalesced.
