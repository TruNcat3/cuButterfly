# Warp-Granular 10+10 Physical-Core Screen

Median CUDA-event time across repeated trials; trial count, warmup, and repetitions are controlled by the benchmark environment.

| bits | batch | v0.6 ms | best full packet | full ms | best half packet | half ms | selected mapping | selected ms | selected/v0.6 |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 32 | 1 | 0.148531 | static_io/t128 | 0.227687 | static_io_half/t128/r4/w11:9 | 0.176691 | static_io_half/t128/r4/w11:9 | 0.176691 | 0.841x |
| 32 | 4 | 0.350515 | static_io/t128 | 0.593203 | static_io_half/t128/r4/w8:7 | 0.419687 | static_io_half/t128/r4/w8:7 | 0.419687 | 0.835x |
| 32 | 16 | 1.094196 | static_io/t256 | 1.885440 | static_io_half/t128/r4/w17:13 | 1.361562 | static_io_half/t128/r4/w17:13 | 1.361562 | 0.804x |
| 64 | 1 | 0.219495 | static_io/t128 | 0.379648 | static_io_half/t128/r2 | 0.421273 | static_io/t128 | 0.379648 | 0.578x |
| 64 | 4 | 0.592026 | static_io/t128 | 0.954368 | static_io_half/t128/r2 | 1.030861 | static_io/t128 | 0.954368 | 0.620x |
| 64 | 16 | 1.907354 | static_io/t128 | 2.916403 | static_io_half/t128/r2 | 3.194419 | static_io/t128 | 2.916403 | 0.654x |

## Interpretation

Full packets fill 32-byte sectors; half packets trade transaction packing for one additional resident 128-thread CTA. The selected mapping jointly searches IO mode, shared layout, packet width, CTA width, and producer/consumer role balance.
