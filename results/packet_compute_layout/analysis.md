# Packet Compute Layout Screen

CUDA-event medians from independent processes.

| batch | mode | kernel (ms) |
|---:|:--|---:|
| 16 | aggregate | 1.090007 |
| 16 | per_packet_interleaved_rows | 1.155400 |
| 16 | per_packet_warp_rows | 1.133978 |
| 16 | wave_bitmap_interleaved_rows | 1.140920 |
| 16 | wave_bitmap_warp_rows | 1.149317 |
| 32 | aggregate | 2.427331 |
| 32 | per_packet_interleaved_rows | 2.523197 |
| 32 | per_packet_warp_rows | 2.526208 |
| 32 | wave_bitmap_interleaved_rows | 2.456597 |
| 32 | wave_bitmap_warp_rows | 2.461737 |
| 64 | aggregate | 4.960625 |
| 64 | per_packet_interleaved_rows | 5.164626 |
| 64 | per_packet_warp_rows | 5.174538 |
| 64 | wave_bitmap_interleaved_rows | 5.021655 |
| 64 | wave_bitmap_warp_rows | 5.045084 |

## Deltas

Batch 16: warp-row per-packet=1.019x; warp-row bitmap=0.993x; best=aggregate; best/aggregate throughput=1.000x.
Batch 32: warp-row per-packet=0.999x; warp-row bitmap=0.998x; best=aggregate; best/aggregate throughput=1.000x.
Batch 64: warp-row per-packet=0.998x; warp-row bitmap=0.995x; best=aggregate; best/aggregate throughput=1.000x.
