# APPT Online Per-Role Breakdown

Intrusive device-globaltimer diagnostics; normal performance kernels do not include these timers.

| bits | batch | role | wait us/task | compute us/task | boundary us/task | role span us | start delay us |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 0 | 0.00 | 59.10 | 3.90 | 63.5 | 0.0 |
| 32 | 1 | 1 | 64.11 | 102.11 | 42.48 | 193.5 | 63.5 |
| 32 | 1 | 2 | 141.30 | 42.57 | 1.95 | 197.6 | 98.3 |
| 32 | 4 | 0 | 0.00 | 122.15 | 7.69 | 512.0 | 0.0 |
| 32 | 4 | 1 | 23.09 | 148.84 | 47.20 | 757.8 | 68.6 |
| 32 | 4 | 2 | 65.49 | 50.86 | 2.82 | 736.3 | 132.1 |
| 64 | 1 | 0 | 0.00 | 155.33 | 5.90 | 162.8 | 0.0 |
| 64 | 1 | 1 | 162.14 | 199.76 | 44.05 | 328.7 | 159.7 |
| 64 | 1 | 2 | 59.82 | 93.82 | 1.76 | 430.1 | 234.5 |
| 64 | 4 | 0 | 0.00 | 216.32 | 6.85 | 1598.5 | 0.0 |
| 64 | 4 | 1 | 106.03 | 218.90 | 37.80 | 1668.1 | 156.7 |
| 64 | 4 | 2 | 128.43 | 104.00 | 2.58 | 1458.2 | 470.0 |

## Pipeline Lower Bound

The lower bound divides summed measured active CTA time by resident grid size. It removes all readiness waiting and assumes perfect work-conserving role balance.

| bits | batch | measured ms | v0.6 ms | current/v0.6 | active lower bound ms | efficiency | lower-bound/v0.6 | wait share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 0.3033 | 0.1481 | 0.488x | 0.0791 | 26.1% | 1.872x | 53.9% |
| 32 | 4 | 0.8815 | 0.3525 | 0.400x | 0.4621 | 52.4% | 0.763x | 26.2% |
| 64 | 1 | 0.6724 | 0.2296 | 0.341x | 0.2385 | 35.5% | 0.963x | 32.1% |
| 64 | 4 | 1.9349 | 0.6136 | 0.317x | 1.1089 | 57.3% | 0.553x | 34.4% |

## Uint64 Batch-4 Attribution

Compute time normalized by task points is `13.20`, `13.36`, and `12.70 ns/point` for folds 0/1/2. The butterfly work is balanced; there is no isolated slow fold formula.

Relative to v0.6, `62.5%` of the measured excess is above the work-conserving active-work lower bound, while `37.5%` remains below that bound. The first part is the target for ready-task stealing and role migration. The residual requires reducing the extra global fold and using the resident radix-4/twiddle core.
