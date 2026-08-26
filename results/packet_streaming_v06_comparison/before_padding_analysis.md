# Packet Streaming versus v0.6

CUDA-event medians from three order-rotated independent processes per implementation. Forward and inverse correctness are checked before timing.

| batch | v0.6 (ms) | v0.7 aggregate (ms) | v0.7 online (ms) | aggregate/v0.6 | online/v0.6 | online/aggregate |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.460367 | 1.081549 | 1.129964 | 1.350x | 1.292x | 0.957x |
| 32 | 4.113428 | 2.417971 | 2.502431 | 1.701x | 1.644x | 0.966x |
| 64 | 9.128469 | 4.930581 | 5.137776 | 1.851x | 1.777x | 0.960x |

## Result

v0.7 aggregate beats v0.6 at 3/3 points with 1.620x geomean throughput. v0.7 online beats v0.6 at 3/3 points with 1.557x geomean throughput. Online reaches 0.961x of the aggregate path on this batch set.
