# APPT Physical-Core Design Space

The APPT schedule is fixed at 7+7+6. Each physical core is searched with the same role-weight simplex; timings are medians across trials.

| bits | batch | physical core | best weights | kernel ms | vs v0.6 | vs warp core |
|---:|---:|---|---:|---:|---:|---:|
| 32 | 1 | `appt-online` | `6-6-8` | 0.295253 | 0.501x | 1.000x |
| 32 | 1 | `appt-online-radix4` | `7-6-7` | 0.306859 | 0.482x | 0.962x |
| 32 | 4 | `appt-online` | `7-7-6` | 0.888627 | 0.394x | 1.000x |
| 32 | 4 | `appt-online-radix4` | `7-6-7` | 0.962253 | 0.364x | 0.923x |
| 64 | 1 | `appt-online` | `8-8-4` | 0.671266 | 0.341x | 1.000x |
| 64 | 1 | `appt-online-radix4` | `4-8-8` | 0.441276 | 0.518x | 1.521x |
| 64 | 4 | `appt-online` | `6-8-6` | 1.771793 | 0.314x | 1.000x |
| 64 | 4 | `appt-online-radix4` | `5-7-8` | 1.271364 | 0.437x | 1.394x |

The role weights are part of a core's physical mapping, not a transferable scheduler constant. A core comparison is valid only after each core has been independently balanced.
