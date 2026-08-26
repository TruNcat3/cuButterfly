# APPT Physical-Core Design Space

The APPT schedule is fixed at 7+7+6. Each physical core is searched with the same role-weight simplex; timings are medians across trials.

| bits | batch | physical core | best weights | kernel ms | vs v0.6 | vs warp core |
|---:|---:|---|---:|---:|---:|---:|
| 64 | 1 | `appt-online-radix4` | `4-8-8` | 0.442368 | 0.521x | 1.000x |

The role weights are part of a core's physical mapping, not a transferable scheduler constant. A core comparison is valid only after each core has been independently balanced.
