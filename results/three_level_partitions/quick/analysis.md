# Parameterized Three-Level Partition Screen

All points use the same uint32 arithmetic, 32-row resident radix-4 unit, two target CTAs/SM, and stage-proportional service weights. Only the position of the 6/7/8-stage subgraph changes.

| batch | partition | kernel (ms) | relative to batch winner | correct |
|---:|:--:|---:|---:|:--:|
| 1 | `6+6+8` | 0.476365 | 1.000x | yes |
| 1 | `7+6+7` | 0.528794 | 1.110x | yes |
| 1 | `8+6+6` | 0.538419 | 1.130x | yes |
| 1 | `6+7+7` | 0.572416 | 1.202x | yes |
| 1 | `7+7+6` | 0.600678 | 1.261x | yes |
| 1 | `6+8+6` | 0.650445 | 1.365x | yes |
| 4 | `6+6+8` | 1.993728 | 1.000x | yes |
| 4 | `8+6+6` | 2.137088 | 1.072x | yes |
| 4 | `7+6+7` | 2.290483 | 1.149x | yes |
| 4 | `6+7+7` | 2.534810 | 1.271x | yes |
| 4 | `7+7+6` | 2.575974 | 1.292x | yes |
| 4 | `6+8+6` | 2.659738 | 1.334x | yes |

The winner is a partition-order result, not yet a fully searched design point; CTA weights and row granularity remain fixed in this screen.
