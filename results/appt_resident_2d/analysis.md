# APPT Resident 2D Screen

Median CUDA-event time. The resident core fixes `a`, executes the first two 7-stage dimensions in one 128x128 subgraph, then streams stage-13 state to the existing register tail and natural writer.

| bits | batch | best P/T/W | resident ms | throughput/writer-final | throughput/v0.6 |
|---:|---:|---|---:|---:|---:|
| 32 | 1 | `10-8-4` | 0.531354 | 0.330x | 0.279x |
| 32 | 4 | `12-8-2` | 1.354214 | 0.380x | 0.258x |
| 32 | 16 | `12-8-2` | 3.827328 | 0.472x | 0.277x |
| 64 | 1 | `8-8-4` | 0.698035 | 0.415x | 0.291x |
| 64 | 4 | `12-6-2` | 2.019200 | 0.432x | 0.275x |
| 64 | 16 | `12-6-2` | 6.198989 | 0.498x | 0.296x |

## Interpretation

This is the first mathematically dependency-closed resident 2D point. A fixed producer `c` tile is not closed because the online `[b][c][a] -> [c][d][a]` reorder makes one tail tile consume columns from 128 producer tiles.

The implementation uses 96 KiB dynamic shared memory, so every physical role in the unified cooperative kernel is limited to one CTA/SM. Event timing determines whether subgraph locality compensates for that loss of role concurrency.
