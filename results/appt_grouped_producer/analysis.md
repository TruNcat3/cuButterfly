# APPT Grouped-Producer Screen

> **Superseded correctness-negative capture.** The grouped producer published
> `[c][b][a]` while the tail consumed `[b][c][a]`. A delta input did not expose
> the transposition. These timings are retained for diagnosis only and must
> not be reported as valid NTT performance.

Median CUDA-event time. Each grouped row is the best role mapping for that `a`-space group.

| bits | batch | a group | best roles (P/T/W) | kernel ms | throughput/v0.6 | throughput/radix4 |
|---:|---:|---:|---|---:|---:|---:|
| 32 | 1 | 8 | `7-13-3` | 0.200832 | 0.735x | 1.056x |
| 32 | 1 | 16 | `7-13-3` | 0.196659 | 0.751x | 1.079x |
| 32 | 1 | 32 | `7-13-3` | 0.193792 | 0.762x | 1.095x |
| 32 | 4 | 8 | `6-14-2` | 0.674151 | 0.518x | 1.041x |
| 32 | 4 | 16 | `6-14-2` | 0.639129 | 0.547x | 1.098x |
| 32 | 4 | 32 | `6-14-2` | 0.604288 | 0.578x | 1.161x |
| 32 | 16 | 8 | `6-16-2` | 2.375680 | 0.450x | 1.344x |
| 32 | 16 | 16 | `6-16-2` | 2.131353 | 0.501x | 1.498x |
| 32 | 16 | 32 | `6-16-2` | 2.025088 | 0.528x | 1.577x |
| 64 | 1 | 8 | `10-9-3` | 0.329651 | 0.656x | 0.951x |
| 64 | 1 | 16 | `10-9-3` | 0.324326 | 0.667x | 0.967x |
| 64 | 1 | 32 | `10-9-3` | 0.315879 | 0.684x | 0.992x |
| 64 | 4 | 8 | `8-10-2` | 1.065651 | 0.571x | 0.974x |
| 64 | 4 | 16 | `8-10-2` | 1.007539 | 0.604x | 1.030x |
| 64 | 4 | 32 | `8-10-2` | 0.970829 | 0.627x | 1.069x |
| 64 | 16 | 8 | `9-11-2` | 4.138368 | 0.469x | 1.258x |
| 64 | 16 | 16 | `9-11-2` | 3.253120 | 0.596x | 1.600x |
| 64 | 16 | 32 | `8-10-2` | 3.049907 | 0.636x | 1.707x |

The grouped core changes the physical request layout only: the logical `7+7+6` APPT graph and butterfly arithmetic remain fixed.
