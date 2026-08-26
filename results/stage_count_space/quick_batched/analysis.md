# Stage-Count Design-Space Screen

The generic radix-4 core, full-scratch boundaries, thread count, and CTA residency are held fixed. Each row is the static shortlist winner within one value of M; CUDA-event timing remains the ranking authority.

| batch | M | partition | kernel (ms) | relative to batch winner |
|---:|---:|:--:|---:|---:|
| 4 | 2 | `10+10` | 1.017242 | 1.000x |
| 4 | 3 | `6+6+8` | 1.783603 | 1.753x |
| 4 | 4 | `5+5+5+5` | 2.948301 | 2.898x |
| 4 | 5 | `4+4+4+4+4` | 5.741773 | 5.644x |
| 4 | 6 | `3+3+3+3+4+4` | 13.005824 | 12.785x |
| 4 | 7 | `2+3+3+3+3+3+3` | 23.475201 | 23.077x |
| 4 | 8 | `2+2+2+2+3+3+3+3` | 28.848537 | 28.360x |
| 16 | 2 | `10+10` | 3.704422 | 1.000x |
| 16 | 3 | `6+6+8` | 11.519385 | 3.110x |
| 16 | 4 | `5+5+5+5` | 9.560473 | 2.581x |
| 16 | 5 | `4+4+4+4+4` | 17.431553 | 4.706x |
| 16 | 6 | `3+3+3+3+4+4` | 39.270401 | 10.601x |
| 16 | 7 | `2+3+3+3+3+3+3` | 84.023087 | 22.682x |
| 16 | 8 | `2+2+2+2+3+3+3+3` | 101.388901 | 27.370x |

This table isolates the number of materialized large stages. Physical-core replacement is a subsequent search layer, not folded into this comparison.
