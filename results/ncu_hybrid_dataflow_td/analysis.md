# HybridDataflow V100 NCU attribution

This capture profiles one launch per point on a Tesla V100-SXM2-16GB at base
profiling clocks. The source table is [summary.csv](summary.csv). NCU replay
time is used only for controlled comparisons within this capture; CUDA-event
timings remain the performance reference.

## Controlled logN=12 results

| Comparison | Time (us) | Relative time | Warp instructions | Integer thread instructions | Active warps | Barrier stall |
|---|---:|---:|---:|---:|---:|---:|
| balanced `Us=6, Td=1` | 187.552 | 1.00x | 4.395 M | 54.416 M | 9.37% | 13.26% |
| balanced `Us=6, Td=4` | 150.272 | 0.80x | 3.290 M | 39.391 M | 9.37% | 25.55% |
| unbalanced `Us=8, Td=1` | 331.232 | 1.77x | 6.066 M | 68.667 M | 12.50% | 50.55% |
| fine-grained `Us=2, Td=1` | 5392.480 | 28.75x | 46.757 M | 472.569 M | 3.12% | 10.94% |

`Td=4` is 1.25x faster than `Td=1`. It removes 25.1% of warp instructions
and 27.6% of integer thread instructions while active-warp utilization remains
9.37%. Barrier stall rises from 13.26% to 25.55%: token packing amortizes
routing, address, and loop-control work, but exposes inter-role synchronization
as a larger fraction of the shorter execution.

Balanced `6+6` decomposition is 1.77x faster than unbalanced `8+4`. It removes
27.5% of warp instructions and 20.8% of integer thread instructions, and cuts
barrier stalls from 50.55% to 13.26%. This establishes stage decomposition as
a first-order architecture parameter.

`Us=2` is not merely synchronization-bound. It executes 10.64x as many warp
instructions and 8.68x as many integer thread instructions as balanced
`Us=6`, while active-warp utilization falls to 3.12%. The 28.75x slowdown is
therefore caused by a long sequence of undersized roles and channel handoffs.

## Secondary ablations

At unbalanced `8+4`, atomic handoff is within 1.1% of named-barrier replay
time, despite executing 22.6% more integer instructions. This capture does not
establish a handoff winner. Linear layout is 3.1% faster in replay time but has
3.93x as many shared-load bank conflicts as `hermes-xor`; steady-state timing
must decide whether its apparent lead is reproducible.

The `logN=13, Us=7, Td=2` point consumes 90160 B of shared memory and is
limited to one CTA per SM. `Td=4` would require 114736 B and is illegal on the
V100 98304 B per-CTA limit. This is a concrete resource boundary that the
search model must encode rather than discover by a failed launch.

DRAM byte counters are not used to claim zero traffic: NCU kernel replay keeps
most input state in cache. Global load/store sectors are retained in the CSV
as the traffic evidence for future controlled comparisons.
