# Packet Readiness Mode Screen

CUDA-event medians from independent processes.

| batch | mode | ready window | sleep cycles | kernel (ms) |
|---:|:--|---:|---:|---:|
| 16 | aggregate | 0 | 0 | 1.086300 |
| 16 | per_packet | 4 | 128 | 1.132974 |
| 16 | wave_bitmap | 1 | 32 | 1.137029 |
| 16 | wave_bitmap | 2 | 64 | 1.139528 |
| 16 | wave_bitmap | 4 | 128 | 1.143173 |
| 16 | wave_bitmap | 8 | 256 | 1.144504 |
| 16 | wave_bitmap | 16 | 512 | 1.136128 |
| 32 | aggregate | 0 | 0 | 2.424771 |
| 32 | per_packet | 2 | 64 | 2.522071 |
| 32 | wave_bitmap | 1 | 32 | 2.455142 |
| 32 | wave_bitmap | 2 | 64 | 2.452357 |
| 32 | wave_bitmap | 4 | 128 | 2.454098 |
| 32 | wave_bitmap | 8 | 256 | 2.444616 |
| 32 | wave_bitmap | 16 | 512 | 2.463437 |
| 64 | aggregate | 0 | 0 | 4.942787 |
| 64 | per_packet | 16 | 512 | 5.152133 |
| 64 | wave_bitmap | 1 | 32 | 5.001871 |
| 64 | wave_bitmap | 2 | 64 | 4.998267 |
| 64 | wave_bitmap | 4 | 128 | 5.010493 |
| 64 | wave_bitmap | 8 | 256 | 4.990546 |
| 64 | wave_bitmap | 16 | 512 | 5.005885 |

## Selected Bitmap Points

Batch 16: bitmap ready_window=16, bitmap/per-packet throughput=0.997x, bitmap/aggregate=0.956x.
Batch 32: bitmap ready_window=8, bitmap/per-packet throughput=1.032x, bitmap/aggregate=0.992x.
Batch 64: bitmap ready_window=8, bitmap/per-packet throughput=1.032x, bitmap/aggregate=0.990x.

Bitmap/per-packet geomean throughput: 1.020x.
