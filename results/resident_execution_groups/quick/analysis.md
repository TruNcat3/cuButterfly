# Resident Execution-Group Screen

M is the logical homogeneous-subgraph count. G is the physical execution-group count after resident boundary lowering. The arithmetic core and logical partition are held fixed within each M comparison.

| batch | M | G | logical | execution | boundaries | kernel (ms) | vs G=M |
|---:|---:|---:|:--:|:--:|---:|---:|---:|
| 1 | 4 | 2 | `5+5+5+5` | `10+10` | 1 | 0.433835 | 2.575x |
| 1 | 4 | 3 | `5+5+5+5` | `5+5+10` | 2 | 1.081685 | 1.033x |
| 1 | 4 | 4 | `5+5+5+5` | `5+5+5+5` | 3 | 1.117184 | 1.000x |

This screen measures boundary realization. It does not rank logical M independently of the physical core.
