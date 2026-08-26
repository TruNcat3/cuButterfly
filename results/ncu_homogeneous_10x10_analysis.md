# Homogeneous 10+10 NCU Analysis

V100, 64-bit NTT, `logN=20`, batch 16. The three rows use the same full-scratch
boundary and identical arithmetic core unless stated otherwise.

| point | time (us) | DRAM read/write (MiB) | DRAM peak | active warps | barrier stall | wait stall | long scoreboard | registers | shared (B) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.6 | 1998.272 | 512.868 / 257.290 | 44.53% | 24.80% | 3.80% | 37.63% | 16.07% | 48 | 32816 |
| homogeneous `D_t=1` | 1999.968 | 512.897 / 257.308 | 44.82% | 24.79% | 3.83% | 37.70% | 16.17% | 48 | 32816 |
| homogeneous `D_t=(2,1)` | 2364.256 | 512.661 / 256.968 | 37.90% | 24.51% | 3.16% | 45.71% | 17.78% | 48 | 32816 |

The `D_t=1` template is an exact control. Non-degenerate data-time traversal
does not add global traffic, registers, shared memory, or occupancy pressure.
Its 18.3% latency increase instead coincides with an 8.08 percentage-point
increase in wait stalls and a 6.63-point decrease in sustained DRAM rate. The
outer CTA traversal removes ready-warp interleaving already supplied by the
V100 warp scheduler; it does not create a load/compute/store overlap inside the
physical subgraph. This result motivates a warp-granular physical core.
