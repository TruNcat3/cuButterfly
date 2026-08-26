# Packet Warp-Row Service-Weight Screen

CUDA-event medians from independent processes.

| batch | mode | producer:consumer | kernel (ms) |
|---:|:--|:--:|---:|
| 16 | aggregate | - | 1.086566 |
| 16 | interleaved | 7:13 | 1.146286 |
| 16 | warp_rows | 4:16 | 1.590907 |
| 16 | warp_rows | 5:15 | 1.236275 |
| 16 | warp_rows | 6:14 | 1.233961 |
| 16 | warp_rows | 7:13 | 1.152225 |
| 16 | warp_rows | 8:12 | 1.126359 |
| 16 | warp_rows | 9:11 | 1.427333 |
| 16 | warp_rows | 10:10 | 1.380168 |
| 16 | warp_rows | 11:9 | 1.720709 |
| 16 | warp_rows | 12:8 | 1.578435 |
| 32 | aggregate | - | 2.420593 |
| 32 | interleaved | 5:15 | 2.447749 |
| 32 | warp_rows | 4:16 | 3.214500 |
| 32 | warp_rows | 5:15 | 2.453094 |
| 32 | warp_rows | 6:14 | 2.623283 |
| 32 | warp_rows | 7:13 | 2.548388 |
| 32 | warp_rows | 8:12 | 2.804900 |
| 32 | warp_rows | 9:11 | 3.462759 |
| 32 | warp_rows | 10:10 | 3.552358 |
| 32 | warp_rows | 11:9 | 4.558766 |
| 32 | warp_rows | 12:8 | 4.565627 |
| 64 | aggregate | - | 4.957225 |
| 64 | interleaved | 5:15 | 5.038490 |
| 64 | warp_rows | 4:16 | 6.473666 |
| 64 | warp_rows | 5:15 | 5.045965 |
| 64 | warp_rows | 6:14 | 6.054564 |
| 64 | warp_rows | 7:13 | 6.288651 |
| 64 | warp_rows | 8:12 | 6.836163 |
| 64 | warp_rows | 9:11 | 7.977533 |
| 64 | warp_rows | 10:10 | 7.845867 |
| 64 | warp_rows | 11:9 | 10.489959 |
| 64 | warp_rows | 12:8 | 11.116400 |

## Selected Points

Batch 16: best=8:12; warp-row/interleaved throughput=1.018x; warp-row/aggregate=0.965x.
Batch 32: best=5:15; warp-row/interleaved throughput=0.998x; warp-row/aggregate=0.987x.
Batch 64: best=5:15; warp-row/interleaved throughput=0.999x; warp-row/aggregate=0.982x.
