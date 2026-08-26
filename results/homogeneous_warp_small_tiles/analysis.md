# Warp-Granular 10+10 Physical-Core Screen

Median CUDA-event time across repeated trials; trial count, warmup, and repetitions are controlled by the benchmark environment.

| bits | batch | v0.6 ms | best full packet | full ms | best half packet | half ms | selected mapping | selected ms | selected/v0.6 |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 32 | 16 | 1.065472 | static_io/t256 | 1.806541 | static_io_half/t128/r4/w17:13 | 1.306726 | static_io_half/t128/r4/w17:13 | 1.306726 | 0.815x |
| 64 | 16 | 1.939456 | static_io/t128 | 2.915635 | static_io_half/t128/r2 | 3.175629 | static_io/t128 | 2.915635 | 0.665x |

## Interpretation

Full packets fill 32-byte sectors; half packets trade transaction packing for one additional resident 128-thread CTA. The selected mapping jointly searches IO mode, shared layout, packet width, CTA width, and producer/consumer role balance.
