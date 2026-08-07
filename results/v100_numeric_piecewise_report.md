# V100 Numeric Piecewise Selector

The model auto-selects only inside batch intervals bracketed by two stable anchors with the same winner.
Unknown numeric contracts, unmeasured lengths, extrapolation, and boundary intervals require measurement.

| Metric | Result |
|:--|--:|
| Evaluated shapes | 310 |
| Auto-selected leave-one-batch-out shapes | 86 |
| Auto-selection coverage | 27.74% |
| Top-1 accuracy when selected | 98.84% |
| Geometric-mean regret when selected | 1.0001x |
| Worst regret when selected | 1.0111x |
| Calibrated-region status | validated |
| Global runtime status | measurement-required |

Coverage is reported explicitly and is not treated as a global selector claim.
