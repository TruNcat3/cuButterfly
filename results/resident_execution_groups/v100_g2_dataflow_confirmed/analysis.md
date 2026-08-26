# Resident Execution-Group Screen

M is the logical homogeneous-subgraph count. G is the physical execution-group count after resident boundary lowering. The arithmetic core is fixed. When present, a resident row has a matched G=M control with the same logical partition; different (M,G) cells may select different logical partitions.

| batch | M | G | logical | execution | boundaries | kernel (ms) | vs G=M |
|---:|---:|---:|:--:|:--:|---:|---:|---:|
| 1 | 2 | 2 | `10+10` | `10+10` | 1 | 0.148480 | 1.000x |
| 1 | 3 | 2 | `5+5+10` | `10+10` | 1 | 0.148924 | n/a |
| 1 | 4 | 2 | `5+5+5+5` | `10+10` | 1 | 0.149197 | n/a |
| 1 | 5 | 2 | `3+3+4+5+5` | `10+10` | 1 | 0.148480 | n/a |
| 1 | 6 | 2 | `3+3+4+3+3+4` | `10+10` | 1 | 0.148548 | n/a |
| 1 | 7 | 2 | `2+2+3+3+3+3+4` | `10+10` | 1 | 0.148617 | n/a |
| 1 | 8 | 2 | `2+2+3+3+2+2+3+3` | `10+10` | 1 | 0.149026 | n/a |
| 4 | 2 | 2 | `10+10` | `10+10` | 1 | 0.365158 | 1.000x |
| 4 | 3 | 2 | `5+5+10` | `10+10` | 1 | 0.367172 | n/a |
| 4 | 4 | 2 | `5+5+5+5` | `10+10` | 1 | 0.366933 | n/a |
| 4 | 5 | 2 | `3+3+4+5+5` | `10+10` | 1 | 0.366046 | n/a |
| 4 | 6 | 2 | `3+3+4+3+3+4` | `10+10` | 1 | 0.365636 | n/a |
| 4 | 7 | 2 | `2+2+3+3+3+3+4` | `10+10` | 1 | 0.366114 | n/a |
| 4 | 8 | 2 | `2+2+3+3+2+2+3+3` | `10+10` | 1 | 0.365466 | n/a |
| 16 | 2 | 2 | `10+10` | `10+10` | 1 | 1.154935 | 1.000x |
| 16 | 3 | 2 | `5+5+10` | `10+10` | 1 | 1.154048 | n/a |
| 16 | 4 | 2 | `5+5+5+5` | `10+10` | 1 | 1.145583 | n/a |
| 16 | 5 | 2 | `3+3+4+5+5` | `10+10` | 1 | 1.159987 | n/a |
| 16 | 6 | 2 | `3+3+4+3+3+4` | `10+10` | 1 | 1.158212 | n/a |
| 16 | 7 | 2 | `2+2+3+3+3+3+4` | `10+10` | 1 | 1.156096 | n/a |
| 16 | 8 | 2 | `2+2+3+3+2+2+3+3` | `10+10` | 1 | 1.154423 | n/a |

## Physical-Equivalence Check

Rows below lower different logical M values to the same physical schedule.

| batch | G | execution | logical variants | min (ms) | max (ms) | spread |
|---:|---:|:--:|---:|---:|---:|---:|
| 1 | 2 | `10+10` | 7 | 0.148480 | 0.149197 | 0.48% |
| 4 | 2 | `10+10` | 7 | 0.365158 | 0.367172 | 0.55% |
| 16 | 2 | `10+10` | 7 | 1.145583 | 1.159987 | 1.26% |

This screen measures boundary realization. It does not rank logical M independently of the physical core.
