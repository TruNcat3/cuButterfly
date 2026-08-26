# APPT Static Layout Search

Median V100 logN=20 timings. Natural includes the online writer; static is the chaining ceiling.

| bits | batch | contract | best point | kernel ms | throughput/v0.6 |
|---:|---:|---|---|---:|---:|
| 32 | 1 | natural | `natural-fw32-wt1-r8-12-2` | 0.285184 | 0.519x |
| 32 | 1 | static | `static-fw32` | 0.177280 | 0.835x |
| 32 | 4 | natural | `natural-fw32-wt4-r8-12-2` | 0.822272 | 0.423x |
| 32 | 4 | static | `static-fw32` | 0.574208 | 0.606x |
| 64 | 1 | natural | `natural-fw32-wt4-r11-9-2` | 0.440704 | 0.519x |
| 64 | 1 | static | `static-fw16` | 0.332032 | 0.689x |
| 64 | 4 | natural | `natural-fw32-wt1-r11-9-2` | 1.246336 | 0.491x |
| 64 | 4 | static | `static-fw32` | 1.020800 | 0.600x |
