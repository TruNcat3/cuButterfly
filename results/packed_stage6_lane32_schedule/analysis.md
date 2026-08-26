# Packed Stage-6 Lane32 Schedule Search

v0.6 bracket median: 1.483537 ms (range 11.3%); d7 control: 1.229551 ms (range 5.3%).

| rank | D_t producer:consumer | producer:consumer CTAs | kernel ms | vs v0.6 | vs d7 |
|---:|:---:|:---:|---:|---:|---:|
| 1 | `1:1` | `86:74` | 1.173521 | 1.264x | 1.048x |
| 2 | `1:1` | `84:76` | 1.180604 | 1.257x | 1.041x |
| 3 | `1:1` | `85:75` | 1.193745 | 1.243x | 1.030x |
| 4 | `1:1` | `82:78` | 1.195008 | 1.241x | 1.029x |
| 5 | `1:1` | `80:80` | 1.196817 | 1.240x | 1.027x |
| 6 | `1:1` | `88:72` | 1.205692 | 1.230x | 1.020x |
| 7 | `1:2` | `86:74` | 1.215880 | 1.220x | 1.011x |
| 8 | `2:1` | `86:74` | 1.230046 | 1.206x | 1.000x |
| 9 | `2:1` | `82:78` | 1.233699 | 1.203x | 0.997x |
| 10 | `1:1` | `78:82` | 1.235780 | 1.200x | 0.995x |
| 11 | `1:2` | `88:72` | 1.239962 | 1.196x | 0.992x |
| 12 | `2:1` | `84:76` | 1.241191 | 1.195x | 0.991x |

## Decision

Cross-core event controls exceed the 3% stability limit. Use this screen only to select the lane32-local schedule; fixed-clock NCU is required for v0.6/d7 claims.
Best point: D_t=1:1, CTAs=86:74, time=1.173521 ms.
