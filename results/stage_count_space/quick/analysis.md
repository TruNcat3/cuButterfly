# Stage-Count Design-Space Screen

The generic radix-4 core, full-scratch boundaries, thread count, and CTA residency are held fixed. Each row is the static shortlist winner within one value of M; CUDA-event timing remains the ranking authority.

| batch | M | partition | kernel (ms) | relative to batch winner |
|---:|---:|:--:|---:|---:|
| 1 | 2 | `10+10` | 0.391987 | 1.000x |
| 1 | 3 | `6+6+8` | 0.573645 | 1.463x |
| 1 | 4 | `5+5+5+5` | 1.108787 | 2.829x |
| 1 | 5 | `4+4+4+4+4` | 2.189107 | 5.585x |
| 1 | 6 | `3+3+3+3+4+4` | 4.889805 | 12.474x |
| 1 | 7 | `2+3+3+3+3+3+3` | 8.332289 | 21.257x |
| 1 | 8 | `2+2+2+2+3+3+3+3` | 12.722177 | 32.456x |

This table isolates the number of materialized large stages. Physical-core replacement is a subsequent search layer, not folded into this comparison.
