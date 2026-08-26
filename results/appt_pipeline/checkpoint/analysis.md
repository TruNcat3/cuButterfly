# APPT Four-Axis Pipeline Scan

Each APPT row is one fixed physical subgraph reused across stage-time folds. The reported state bandwidth counts token reads/writes and online fold reorders, but excludes twiddle traffic.

| bits | logN | batch | best (Us,role-stages,Td,buffers,CTA/SM) | APPT ms | state GB/s | v0.6 ms | barrier ms | speedup vs v0.6 |
|---:|---:|---:|:--|---:|---:|---:|---:|---:|
| 32 | 20 | 1 | (7,1,4,2,2) | 0.6696 | 62.6 | 0.1284 | 0.0796 | 0.192x |
| 32 | 20 | 4 | (7,1,8,2,2) | 2.5930 | 64.7 | 0.3102 | 0.2423 | 0.120x |
| 64 | 20 | 1 | (7,1,4,2,2) | 0.9222 | 91.0 | 0.2040 | 0.1421 | 0.221x |
| 64 | 20 | 4 | (7,1,8,2,2) | 3.4876 | 96.2 | 0.5545 | 0.5044 | 0.159x |
