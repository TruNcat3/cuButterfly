# APPT Physical-Core Design Space

The APPT schedule is fixed at 7+7+6. Each physical core is searched with the same role-weight simplex; timings are medians across trials.

| bits | batch | physical core | best weights | kernel ms | vs v0.6 | vs warp core |
|---:|---:|---|---:|---:|---:|---:|
| 32 | 1 | `appt-online` | `6-6-8` | 0.295322 | 0.506x | 1.000x |
| 32 | 1 | `appt-online-radix4` | `7-6-7` | 0.307405 | 0.486x | 0.961x |
| 32 | 4 | `appt-online` | `7-7-6` | 0.884122 | 0.396x | 1.000x |
| 32 | 4 | `appt-online-radix4` | `7-6-7` | 0.968294 | 0.362x | 0.913x |
| 64 | 1 | `appt-online` | `4-8-8` | 0.671539 | 0.342x | 1.000x |
| 64 | 1 | `appt-online-radix4` | `4-8-8` | 0.442368 | 0.519x | 1.518x |
| 64 | 4 | `appt-online` | `6-8-6` | 1.926144 | 0.317x | 1.000x |
| 64 | 4 | `appt-online-radix4` | `5-7-8` | 1.374413 | 0.445x | 1.401x |

The role weights are part of a core's physical mapping, not a transferable scheduler constant. A core comparison is valid only after each core has been independently balanced.
