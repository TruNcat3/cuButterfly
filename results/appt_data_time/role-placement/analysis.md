# APPT Batch Data-Time Screen

Median CUDA-event time. Each row selects the best P/T/W mapping at fixed `T_d`; `T_d=1` is the controlled schedule-equivalence point.

Role mask bits are producer=1, tail=2, writer=4.

| bits | batch | T_d | role mask | best weights | kernel ms | throughput/writer-final | throughput/v0.6 |
|---:|---:|---:|---:|---|---:|---:|---:|
| 32 | 4 | 1 | 4 | `6-12-4` | 0.518076 | 0.993x | 0.674x |
| 32 | 4 | 1 | 6 | `6-12-4` | 0.514628 | 1.000x | 0.679x |
| 32 | 4 | 1 | 7 | `6-12-4` | 0.512375 | 1.004x | 0.682x |
| 32 | 4 | 2 | 4 | `6-12-4` | 0.531695 | 0.968x | 0.657x |
| 32 | 4 | 2 | 6 | `6-12-4` | 0.532753 | 0.966x | 0.656x |
| 32 | 4 | 2 | 7 | `6-12-4` | 0.582007 | 0.884x | 0.600x |
| 32 | 4 | 4 | 4 | `6-12-4` | 0.569481 | 0.903x | 0.613x |
| 32 | 4 | 4 | 6 | `6-12-4` | 0.573474 | 0.897x | 0.609x |
| 32 | 4 | 4 | 7 | `6-12-4` | 0.650206 | 0.791x | 0.537x |
| 32 | 16 | 1 | 4 | `6-12-4` | 1.738820 | 0.995x | 0.589x |
| 32 | 16 | 1 | 6 | `6-12-4` | 1.736021 | 0.997x | 0.590x |
| 32 | 16 | 1 | 7 | `6-12-4` | 1.735714 | 0.997x | 0.590x |
| 32 | 16 | 2 | 4 | `6-12-4` | 1.744725 | 0.992x | 0.587x |
| 32 | 16 | 2 | 6 | `6-12-4` | 1.720047 | 1.006x | 0.596x |
| 32 | 16 | 2 | 7 | `6-12-4` | 1.774114 | 0.976x | 0.577x |
| 32 | 16 | 4 | 4 | `6-12-4` | 1.778893 | 0.973x | 0.576x |
| 32 | 16 | 4 | 6 | `6-12-4` | 1.766366 | 0.980x | 0.580x |
| 32 | 16 | 4 | 7 | `6-12-4` | 1.860779 | 0.930x | 0.551x |
| 64 | 4 | 1 | 4 | `8-8-4` | 0.912759 | 0.993x | 0.637x |
| 64 | 4 | 1 | 6 | `8-8-4` | 0.913988 | 0.992x | 0.636x |
| 64 | 4 | 1 | 7 | `8-8-4` | 0.935083 | 0.969x | 0.622x |
| 64 | 4 | 2 | 4 | `8-8-4` | 0.961911 | 0.942x | 0.605x |
| 64 | 4 | 2 | 6 | `8-8-4` | 0.967714 | 0.936x | 0.601x |
| 64 | 4 | 2 | 7 | `8-8-4` | 0.999799 | 0.906x | 0.582x |
| 64 | 4 | 4 | 4 | `8-8-4` | 1.020484 | 0.888x | 0.570x |
| 64 | 4 | 4 | 6 | `8-8-4` | 1.041135 | 0.870x | 0.559x |
| 64 | 4 | 4 | 7 | `8-8-4` | 1.153911 | 0.785x | 0.504x |
| 64 | 16 | 1 | 4 | `8-8-4` | 3.109308 | 0.996x | 0.593x |
| 64 | 16 | 1 | 6 | `8-8-4` | 3.119718 | 0.993x | 0.591x |
| 64 | 16 | 1 | 7 | `8-8-4` | 3.111902 | 0.995x | 0.593x |
| 64 | 16 | 2 | 4 | `8-8-4` | 3.206280 | 0.966x | 0.575x |
| 64 | 16 | 2 | 6 | `8-8-4` | 3.157641 | 0.981x | 0.584x |
| 64 | 16 | 2 | 7 | `8-8-4` | 3.204472 | 0.966x | 0.575x |
| 64 | 16 | 4 | 4 | `8-8-4` | 3.358993 | 0.922x | 0.549x |
| 64 | 16 | 4 | 6 | `8-8-4` | 3.300147 | 0.938x | 0.559x |
| 64 | 16 | 4 | 7 | `8-8-4` | 3.399646 | 0.911x | 0.542x |

## Interpretation

The best temporal placement is mask `6` at 32-bit batch 16 and `T_d=2`: 1.006x writer-final and 0.596x v0.6 throughput.

The best all-role (`7`) temporal point reaches only 0.976x writer-final. Keeping producer in spatial wavefront order therefore avoids a measurable downstream-readiness delay, but role placement alone does not close the physical-core gap.

This data-time core reuses coefficient/layout state inside each physical role. Producer, tail, and writer still exchange transform state through global memory, so this experiment is a role-local temporal traversal rather than end-to-end on-chip subgraph residence.

The `7+7+6` DAG, grouped state format, writer-final codelet, and global boundary count are fixed.
