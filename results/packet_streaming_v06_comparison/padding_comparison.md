# Packet Staging Row-Padding Screen

The candidate changes the online staging row stride from 256 to 257 words.
The staging allocation grows from 1024 to 1028 words (16 bytes) and preserves
four resident CTAs per SM. Butterfly arithmetic, packet readiness, global
layout, and role weights are unchanged.

Both tables use three order-rotated independent processes, 10 warmups, and 50
timed iterations per process.

| batch | aggregate before (ms) | aggregate padded (ms) | online before (ms) | online padded (ms) | padded online / before |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.081549 | 1.083413 | 1.129964 | 1.144320 | 1.013x |
| 32 | 2.417971 | 2.436465 | 2.502431 | 2.524795 | 1.009x |
| 64 | 4.930581 | 4.967137 | 5.137776 | 5.128664 | 0.998x |

CUDA-event timing is neutral to slightly negative. Padding is not promoted on
this evidence alone. The fixed-clock NCU comparison must show a material
shared-load-conflict reduction without moving the cost to extra instructions
or scoreboard stalls; otherwise the candidate is rejected.

## Fixed-Clock Decision

The candidate is rejected. Relative to the unpadded capture, online
shared-load conflicts increase by 0.7% at batch 16 and 0.6% at batch 32.
Registers rise from 64 to 71 per thread. Online fixed-clock time regresses by
2.4% and 0.4%, respectively. The row phase was therefore not the source of the
measured conflicts, and the implementation has been reverted.
