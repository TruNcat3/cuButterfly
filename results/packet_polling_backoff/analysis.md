# Packet Readiness Polling Backoff

CUDA-event medians from independent processes. `ready_window` is the number of 32-cycle sleep quanta between acquire loads.

| batch | ready window | sleep cycles | aggregate (ms) | online (ms) | online/aggregate |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 32 | 1.080382 | 1.142129 | 0.946x |
| 16 | 2 | 64 | 1.080382 | 1.137910 | 0.949x |
| 16 | 4 | 128 | 1.080382 | 1.132913 | 0.954x |
| 16 | 8 | 256 | 1.080382 | 1.136046 | 0.951x |
| 16 | 16 | 512 | 1.080382 | 1.134264 | 0.952x |
| 32 | 1 | 32 | 2.420347 | 2.519429 | 0.961x |
| 32 | 2 | 64 | 2.420347 | 2.507797 | 0.965x |
| 32 | 4 | 128 | 2.420347 | 2.529055 | 0.957x |
| 32 | 8 | 256 | 2.420347 | 2.521190 | 0.960x |
| 32 | 16 | 512 | 2.420347 | 2.513838 | 0.963x |
| 64 | 1 | 32 | 4.938015 | 5.115843 | 0.965x |
| 64 | 2 | 64 | 4.938015 | 5.124465 | 0.964x |
| 64 | 4 | 128 | 4.938015 | 5.116785 | 0.965x |
| 64 | 8 | 256 | 4.938015 | 5.112381 | 0.966x |
| 64 | 16 | 512 | 4.938015 | 5.108982 | 0.967x |

## Selected Points

Batch 16: ready_window=4 (128 cycles), online/aggregate=0.954x.
Batch 32: ready_window=2 (64 cycles), online/aggregate=0.965x.
Batch 64: ready_window=16 (512 cycles), online/aggregate=0.967x.
