# HierarchicalDataflow comparison

| bits | logN | batch | Hybrid2D ms | best hierarchical ms | speedup | core | rows | Td | threads | Rb | correct |
|---:|---:|---:|---:|---:|---:|:--|---:|---:|---:|---:|:---:|
| 32 | 20 | 4 | n/a | 0.270848 | n/a | dataflow-radix4 | 4 | 1 | 256 | 5 | yes |
| 64 | 20 | 4 | n/a | 0.518144 | n/a | dataflow-radix4 | 2 | 1 | 512 | 2 | yes |
