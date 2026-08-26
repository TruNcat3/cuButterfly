# Packet-Shared Radix-4 Batch Crossover

CUDA-event medians after warmup. Packet role weights are independently selected at each batch.

| batch | v0.6 (ms) | lane32 (ms) | packet128 (ms) | weights | packet/v0.6 | packet/lane32 | packet/best old |
|--:|--:|--:|--:|:--|--:|--:|--:|
| 16 | 1.448254 | 1.168241 | 1.088235 | `7:13` | 1.331x | 1.074x | 1.074x |
| 32 | 4.109404 | 2.327491 | 2.455829 | `5:15` | 1.673x | 0.948x | 0.948x |
| 64 | 9.206579 | 5.312614 | 5.420585 | `5:15` | 1.698x | 0.980x | 0.980x |

## Interpretation

Packet128 beats v0.6 at 3/3 points and the best legacy core at 1/3 points. Its best-control-relative throughput ranges from 0.948x to 1.074x.
The core consistently replaces v0.6, but the lane32 crossover remains batch-dependent; retain per-batch selector entries.
