# APPT Physical-Core Comparison

Median CUDA-event time. `radix4-matched` and `radix8-matched` use identical role and layout mappings.

| bits | batch | variant | kernel ms | throughput/v0.6 | radix8/radix4 |
|---:|---:|---|---:|---:|---:|
| 32 | 1 | `radix4-default` | 0.217651 | 0.678x | - |
| 32 | 1 | `radix4-matched` | 0.213658 | 0.690x | - |
| 32 | 1 | `radix8-matched` | 0.213914 | 0.690x | 0.999x |
| 32 | 1 | `v06` | 0.147507 | 1.000x | - |
| 32 | 4 | `radix4-default` | 0.707533 | 0.494x | - |
| 32 | 4 | `radix4-matched` | 0.710246 | 0.492x | - |
| 32 | 4 | `radix8-matched` | 0.703027 | 0.497x | 1.010x |
| 32 | 4 | `v06` | 0.349542 | 1.000x | - |
| 32 | 16 | `radix4-default` | 3.658752 | 0.277x | - |
| 32 | 16 | `radix4-matched` | 3.145779 | 0.322x | - |
| 32 | 16 | `radix8-matched` | 3.064883 | 0.331x | 1.026x |
| 32 | 16 | `v06` | 1.013965 | 1.000x | - |
| 64 | 1 | `radix4-default` | 0.318515 | 0.639x | - |
| 64 | 1 | `radix4-matched` | 0.298803 | 0.681x | - |
| 64 | 1 | `radix8-matched` | 0.294093 | 0.692x | 1.016x |
| 64 | 1 | `v06` | 0.203571 | 1.000x | - |
| 64 | 4 | `radix4-default` | 1.134797 | 0.538x | - |
| 64 | 4 | `radix4-matched` | 1.033370 | 0.590x | - |
| 64 | 4 | `radix8-matched` | 1.048730 | 0.582x | 0.985x |
| 64 | 4 | `v06` | 0.610202 | 1.000x | - |
| 64 | 16 | `radix4-default` | 5.824256 | 0.316x | - |
| 64 | 16 | `radix4-matched` | 5.172122 | 0.356x | - |
| 64 | 16 | `radix8-matched` | 5.461299 | 0.338x | 0.947x |
| 64 | 16 | `v06` | 1.843302 | 1.000x | - |
