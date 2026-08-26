# APPT Resident Quarter Screen

Median CUDA-event time. The quarter core executes four 32x128 tiles, joins quarter pairs at stage 12, and joins the two halves at stage 13.

| bits | batch | best P/T/W | quarter ms | throughput/96KiB | throughput/writer-final | throughput/v0.6 |
|---:|---:|---|---:|---:|---:|---:|
| 32 | 1 | `10-8-4` | 0.364621 | 1.457x | 0.481x | 0.404x |
| 32 | 4 | `8-10-4` | 0.974182 | 1.389x | 0.525x | 0.357x |
| 32 | 16 | `8-10-4` | 3.755520 | 1.070x | 0.481x | 0.280x |
| 64 | 1 | `8-8-4` | 0.474317 | 1.473x | 0.610x | 0.428x |
| 64 | 4 | `8-8-4` | 1.317427 | 1.655x | 0.717x | 0.464x |
| 64 | 16 | `8-8-4` | 5.587430 | 1.110x | 0.562x | 0.346x |

## Resource Interpretation

The 48 KiB specialization restores two CTA/SM, but launch-bounds cap it at 128 registers/thread. The current compiler allocation spills retained state to the thread stack, so event timing must be read together with cuobjdump resource usage.
