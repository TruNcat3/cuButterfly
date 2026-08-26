# APPT Four-Axis Pipeline Scan

Each APPT row is one fixed physical subgraph reused across stage-time folds. The reported state bandwidth counts token reads/writes and online fold reorders, but excludes twiddle traffic.

| bits | logN | batch | best (Us,role-stages,replicas,Td,token-group,buffers,CTA/SM) | APPT ms | state GB/s | v0.6 ms | barrier ms | speedup vs v0.6 |
|---:|---:|---:|:--|---:|---:|---:|---:|---:|
| 32 | 20 | 1 | (7,2,2,4,2,2,2) | 0.3515 | 119.3 | 0.1364 | 0.0832 | 0.388x |
| 32 | 20 | 4 | (7,2,2,8,2,2,2) | 1.2081 | 138.9 | 0.3194 | 0.2525 | 0.264x |
| 64 | 20 | 1 | (7,2,2,4,1,2,2) | 0.5377 | 156.0 | 0.2033 | 0.1414 | 0.378x |
| 64 | 20 | 4 | (7,2,2,4,2,2,2) | 1.9723 | 170.1 | 0.5551 | 0.5047 | 0.281x |
