# Packet Streaming versus v0.6

CUDA-event medians from three order-rotated independent processes per implementation. Forward and inverse correctness are checked before timing.

| batch | v0.6 (ms) | v0.7 aggregate (ms) | v0.7 online (ms) | aggregate/v0.6 | online/v0.6 | online/aggregate |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.520865 | 1.083413 | 1.144320 | 1.404x | 1.329x | 0.947x |
| 32 | 4.094136 | 2.436465 | 2.524795 | 1.680x | 1.622x | 0.965x |
| 64 | 9.227468 | 4.967137 | 5.128664 | 1.858x | 1.799x | 0.969x |

## Result

v0.7 aggregate beats v0.6 at 3/3 points with 1.636x geomean throughput. v0.7 online beats v0.6 at 3/3 points with 1.571x geomean throughput. Online reaches 0.960x of the aggregate path on this batch set.
