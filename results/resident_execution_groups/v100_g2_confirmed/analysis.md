# Resident Execution-Group Screen

M is the logical homogeneous-subgraph count. G is the physical execution-group count after resident boundary lowering. The arithmetic core is fixed. When present, a resident row has a matched G=M control with the same logical partition; different (M,G) cells may select different logical partitions.

| batch | M | G | logical | execution | boundaries | kernel (ms) | vs G=M |
|---:|---:|---:|:--:|:--:|---:|---:|---:|
| 1 | 2 | 2 | `10+10` | `10+10` | 1 | 0.427008 | 1.000x |
| 1 | 3 | 2 | `5+5+10` | `10+10` | 1 | 0.427315 | n/a |
| 1 | 4 | 2 | `5+5+5+5` | `10+10` | 1 | 0.426633 | n/a |
| 1 | 5 | 2 | `3+3+4+5+5` | `10+10` | 1 | 0.427110 | n/a |
| 1 | 6 | 2 | `3+3+4+3+3+4` | `10+10` | 1 | 0.427315 | n/a |
| 1 | 7 | 2 | `2+2+3+3+3+3+4` | `10+10` | 1 | 0.427486 | n/a |
| 1 | 8 | 2 | `2+2+3+3+2+2+3+3` | `10+10` | 1 | 0.427349 | n/a |
| 4 | 2 | 2 | `10+10` | `10+10` | 1 | 1.112508 | 1.000x |
| 4 | 3 | 2 | `5+5+10` | `10+10` | 1 | 1.111586 | n/a |
| 4 | 4 | 2 | `5+5+5+5` | `10+10` | 1 | 1.124659 | n/a |
| 4 | 5 | 2 | `3+3+4+5+5` | `10+10` | 1 | 1.109231 | n/a |
| 4 | 6 | 2 | `3+3+4+3+3+4` | `10+10` | 1 | 1.110494 | n/a |
| 4 | 7 | 2 | `2+2+3+3+3+3+4` | `10+10` | 1 | 1.114692 | n/a |
| 4 | 8 | 2 | `2+2+3+3+2+2+3+3` | `10+10` | 1 | 1.112644 | n/a |
| 16 | 2 | 2 | `10+10` | `10+10` | 1 | 3.759753 | 1.000x |
| 16 | 3 | 2 | `5+5+10` | `10+10` | 1 | 3.795934 | n/a |
| 16 | 4 | 2 | `5+5+5+5` | `10+10` | 1 | 3.759104 | n/a |
| 16 | 5 | 2 | `3+3+4+5+5` | `10+10` | 1 | 3.740979 | n/a |
| 16 | 6 | 2 | `3+3+4+3+3+4` | `10+10` | 1 | 3.722001 | n/a |
| 16 | 7 | 2 | `2+2+3+3+3+3+4` | `10+10` | 1 | 3.741867 | n/a |
| 16 | 8 | 2 | `2+2+3+3+2+2+3+3` | `10+10` | 1 | 3.756715 | n/a |

## Physical-Equivalence Check

Rows below lower different logical M values to the same physical schedule.

| batch | G | execution | logical variants | min (ms) | max (ms) | spread |
|---:|---:|:--:|---:|---:|---:|---:|
| 1 | 2 | `10+10` | 7 | 0.426633 | 0.427486 | 0.20% |
| 4 | 2 | `10+10` | 7 | 1.109231 | 1.124659 | 1.39% |
| 16 | 2 | `10+10` | 7 | 3.722001 | 3.795934 | 1.99% |

This screen measures boundary realization. It does not rank logical M independently of the physical core.
