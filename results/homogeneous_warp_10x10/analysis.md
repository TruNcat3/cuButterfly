# Warp-Granular 10+10 Physical-Core Screen

Median CUDA-event time across repeated trials; trial count, warmup, and repetitions are controlled by the benchmark environment.

| bits | batch | v0.6 ms | best full packet | full ms | best half packet | half ms | selected mapping | selected ms | selected/v0.6 |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 32 | 1 | 0.147712 | static_io/t128 | 0.227277 | static_io_half/t128/r4/w11:9 | 0.176384 | static_io_half/t128/r4/w11:9 | 0.176384 | 0.837x |
| 32 | 4 | 0.348723 | static_io/t128 | 0.593101 | static_io_half/t128/r4/w8:7 | 0.419430 | static_io_half/t128/r4/w8:7 | 0.419430 | 0.831x |
| 32 | 16 | 1.014477 | static_io/t128 | 1.683968 | static_io_half/t128/r4/w17:13 | 1.233510 | static_io_half/t128/r4/w17:13 | 1.233510 | 0.822x |
| 64 | 1 | 0.228454 | static_io/t128 | 0.398387 | static_io_half/t128/r2 | 0.442829 | static_io/t128 | 0.398387 | 0.573x |
| 64 | 4 | 0.551475 | static_io/t128 | 0.877158 | static_io_half/t128/r2 | 0.935578 | static_io/t128 | 0.877158 | 0.629x |
| 64 | 16 | 1.840640 | static_io/t128 | 2.917939 | static_io_half/t128/r2 | 3.188071 | static_io/t128 | 2.917939 | 0.631x |

## Interpretation

Full packets fill 32-byte sectors; half packets trade transaction packing for one additional resident 128-thread CTA. The selected mapping jointly searches IO mode, shared layout, packet width, CTA width, and producer/consumer role balance.
