# Homogeneous 10+10 Subgraph Screen

Median CUDA-event time after warmup. v0.6 and the generated template use the same resident radix-4 arithmetic core, full-scratch boundary, and cooperative launch.

| bits | batch | v0.6 ms | template D_t=1 ms | equivalence | best non-degenerate D_t | weights | temporal ms | temporal/v0.6 |
|---:|---:|---:|---:|---:|---|---|---:|---:|
| 32 | 1 | 0.148377 | 0.148583 | 0.999x | `4:1` | `10-10` | 0.148327 | 1.000x |
| 32 | 4 | 0.349235 | 0.346828 | 1.007x | `1:2` | `9-11` | 0.367411 | 0.951x |
| 32 | 16 | 1.085850 | 1.088871 | 0.997x | `2:1` | `9-11` | 1.143808 | 0.949x |
| 64 | 1 | 0.229325 | 0.229479 | 0.999x | `2:1` | `9-11` | 0.258816 | 0.886x |
| 64 | 4 | 0.612864 | 0.613120 | 1.000x | `2:1` | `8-12` | 0.761139 | 0.805x |
| 64 | 16 | 1.846067 | 1.899980 | 0.972x | `2:1` | `8-12` | 2.198733 | 0.840x |

## Interpretation

- 32-bit best/v0.6 by batch: batch 1: 1.000x, batch 4: 0.951x, batch 16: 0.949x.
- 64-bit best/v0.6 by batch: batch 1: 0.886x, batch 4: 0.805x, batch 16: 0.840x.

`D_t=1:1` isolates template/selector overhead. Non-degenerate rows test role-local temporal traversal; the GPU warp scheduler selects among ready resident warps but does not overlap load/compute/store inside one CTA subgraph.
