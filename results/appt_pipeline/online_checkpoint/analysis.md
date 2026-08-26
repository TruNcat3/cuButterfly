# APPT Four-Axis Pipeline Scan

Each APPT row is one fixed physical subgraph reused across stage-time folds. The reported state bandwidth counts token reads/writes and online fold reorders, but excludes twiddle traffic.

| bits | logN | batch | best staged APPT point | staged ms | online ms | v0.6 ms | online/staged | online/v0.6 |
|---:|---:|---:|:--|---:|---:|---:|---:|---:|
| 32 | 20 | 1 | (7,2,2,4,2,2,2) | 0.3360 | 0.2574 | 0.1290 | 1.305x | 0.501x |
| 32 | 20 | 4 | (7,2,2,8,2,2,2) | 1.2080 | 0.8616 | 0.3103 | 1.402x | 0.360x |
| 64 | 20 | 1 | (7,2,2,4,2,2,2) | 0.5401 | 0.6082 | 0.2034 | 0.888x | 0.334x |
| 64 | 20 | 4 | (7,2,2,4,2,2,2) | 1.9801 | 1.8293 | 0.5541 | 1.082x | 0.303x |
