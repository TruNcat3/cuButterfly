# v0.7 hierarchical streaming NCU diagnosis

Device: Tesla V100-SXM2-16GB (`sm_70`). The raw reports were collected on
2026-08-17 with CUDA 11.8 NCU. These reports measure the correctness-first
readiness implementation, not a generated specialized streaming kernel.

## Observed gap

| case | v0.6 barrier | v0.7 streaming | streaming / barrier |
|:--|--:|--:|--:|
| `logN=15,batch=2` time | 20.96 us | 270.40 us | 12.90x |
| `logN=20,batch=4` time | 537.06 us | 7268.03 us | 13.53x |
| `logN=20,batch=4` DRAM read | 128.41 MiB | 609.18 MiB | 4.74x |
| `logN=20,batch=4` DRAM write | 64.70 MiB | 238.05 MiB | 3.68x |
| `logN=20,batch=4` warp instructions | 75.17 M | 423.39 M | 5.63x |
| `logN=20,batch=4` integer thread instructions | 1.90 B | 8.07 B | 4.24x |
| `logN=20,batch=4` active warps | 24.36% | 20.69% | 0.85x |
| `logN=20,batch=4` long-scoreboard stall | 44.19% | 11.71% | 0.27x |

The lower scoreboard and MIO-throttle stall percentages rule out raw DRAM
latency or shared-memory pressure as the primary cause. The streaming kernel
does much more protocol and index work while sustaining much less useful work
per cycle.

## Root causes

1. Each consumer element polls a producer-task token with an atomic operation
   and then executes a device fence. For the `7+7+6` decomposition there are
   at least `2*N*batch = 8,388,608` consumer token reads and fences before any
   retries are counted.
2. Three segments materialize two complete global-memory boundaries. Readiness
   tokens, twiddle tables, and fence-induced transactions raise measured DRAM
   traffic far beyond the data payload.
3. A generic runtime descriptor performs division, remainder, digit reversal,
   producer-task reconstruction, and 64-bit address arithmetic in inner loops.
   The 4.24x integer-instruction increase is direct evidence of this cost.
4. The `7+7+6` tiles contain only 128, 128, and 64 values, but the profiled CTA
   has 256 threads. Half to three quarters of its lanes are inactive in tile
   load/store loops, while every CTA still executes all block barriers.
5. Static producer/consumer CTA roles remain resident while polling. With 76
   registers per thread, register pressure permits three CTAs per SM, but the
   default launches only two (`waves_per_sm=0.67`) and the profile reaches only
   20.69% active warps. Pipeline overlap exists, but waiting roles consume the
   resources needed by ready roles. A local event scan at three CTAs per SM
   improves v0.7 by about 24%, but still leaves a large gap.
6. `ready_window` is currently configuration metadata only; no task-stealing or
   alternate-ready-task window is implemented. Per-segment thread counts also
   collapse to one kernel-wide block size.

The first NCU sweep accidentally selected the generic radix-2 subgraph because
the CLI partially initialized mappings when only CTA weights were specified.
The profiling script and CLI defaults have been corrected. CUDA-event checks
with an explicit dataflow radix-4 core remain roughly 9--12x slower at
`logN=20,batch=4` (depending on CTA residency), so this mistake amplified but
did not cause the regression.

## Required implementation direction

The architecture should keep readiness at a coarse subgraph packet or warp
tile granularity, publish one token per packet, and use generated compile-time
indexing. Ready work must be pulled dynamically (or assigned in dependency-safe
waves) so consumer CTAs do not spin while upstream work is runnable. Specialized
resident radix cores should then replace the generic physical core. Until that
path wins measured selection, v0.6 remains the release fallback.

## Matched resident-core capture

A second capture on 2026-08-17 compares the same mature resident radix-4 unit
in v0.6 and the generated v0.7 `10+10` path. It was collected before the
release/acquire handoff optimization and used a symmetric `10:10` role split.

| 64-bit `logN=20,batch=4` metric | v0.6 barrier | v0.7 resident stream | ratio |
|:--|--:|--:|--:|
| kernel time | 539.74 us | 617.44 us | 1.144x |
| DRAM read | 128.42 MiB | 129.01 MiB | 1.005x |
| DRAM write | 64.75 MiB | 64.87 MiB | 1.002x |
| warp instructions | 75.17 M | 75.54 M | 1.005x |
| integer thread instructions | 1.904 B | 1.898 B | 0.997x |
| active warps | 24.27% | 24.67% | 1.016x |
| barrier stall | 6.11% | 15.79% | 2.584x |
| long-scoreboard stall | 44.35% | 32.80% | 0.740x |
| registers / shared memory / waves | 48 / 32,800 B / 1.0 | 48 / 32,800 B / 1.0 | equal |

The work, traffic, and residency are effectively identical. The remaining
14.4% in this capture is therefore not a physical butterfly-core problem: it
comes from readiness handoff and fixed producer/consumer scheduling. For
32-bit, v0.7 also launches four CTAs/SM versus five in the barrier kernel, but
event scans show that forcing five streaming CTAs/SM is slower; residency alone
does not explain the gap.

## Implemented remediation

The next implementation replaces per-coefficient consumer tokens with one
counter per `(outer prefix, later remainder)` dependency wave. A producer
subgraph publishes once; the counter releases all consumer subgraphs that vary
the preceding output digit. The generic physical mapping now assigns one
subgraph to each warp, so a 256-thread CTA executes eight independent tiles and
uses warp rather than block barriers.

For V100 `logN=20`, a generated `10+10` kernel reuses the same four-row resident
radix-4 unit and fused coefficient table as v0.6, but assigns fixed producer and
consumer CTA roles and publishes one readiness counter per transform. Exact
forward/inverse verification passes. Representative 64-bit event times are:

| implementation | partition | batch | kernel ms |
|:--|:--|--:|--:|
| original readiness core | `7+7+6` | 4 | about 6.0 |
| wave + warp generic core | `7+7+6` | 4 | about 1.49 |
| generated resident core | `10+10` | 4 | 0.61--0.66 |
| v0.6 barrier resident core | `10+10` | 4 | 0.54--0.55 |

Thus the original NCU diagnosis led to a roughly 9--10x end-to-end kernel
improvement at the key point. The matched capture then isolates readiness and
role balance as the remaining costs rather than arithmetic-core quality.

After that capture, the consumer wait was changed from a CTA barrier behind a
single polling lane to a shared flag with warp-level wakeup. Producer packets
now use one CTA barrier followed by a device-scope release RMW instead of three
barriers plus an all-thread device fence. Exact 32/64-bit forward and inverse
verification passes. A measured `9:11` producer/consumer allocation reflects
the second subgraph's fused twiddle and finalization work. CUDA-event tests show
the expected fill/drain boundary: v0.7 is within roughly 4--6% at batch 8 and
meets or exceeds v0.6 at batch 16 and 32. The optimized NCU capture below
quantifies why the small-batch boundary remains.

## Optimized-handoff capture

The follow-up capture uses the optimized release/acquire protocol and the
measured `9:11` role balance:

| `logN=20,batch=4` metric | 64-bit v0.6 | 64-bit v0.7 | 32-bit v0.6 | 32-bit v0.7 |
|:--|--:|--:|--:|--:|
| kernel time | 540.32 us | 613.22 us | 272.70 us | 344.38 us |
| warp instructions | 75.17 M | 82.37 M | 44.22 M | 50.46 M |
| integer thread instructions | 1.904 B | 1.935 B | 1.028 B | 1.048 B |
| active warps | 24.23% | 24.31% | 56.67% | 46.61% |
| barrier stall | 6.35% | 3.48% | 15.68% | 5.02% |
| wait stall | 15.19% | 15.83% | 8.57% | 10.12% |
| long-scoreboard stall | 43.65% | 32.00% | 34.89% | 32.64% |

Relative to the preceding v0.7 capture, barrier stall falls by 12.31 percentage
points for 64-bit and 15.57 points for 32-bit. Kernel time improves only 0.7%
and 3.7%, respectively, because the wait moved into explicit shared-flag
polling: warp instructions increase by 9.0% and 13.3%. This falsifies further
CTA-barrier tuning as the primary optimization direction.

The `10+10` row/column factorization has a full-transform dependency closure:
each second-layer row consumes values produced by all 256 first-layer packets.
It can pipeline independent batch transforms, but it cannot form an
intra-transform wavefront. A work-conserving prototype with one global atomic
task head was also evaluated; packet-level CAS contention erased its small
local scheduling gain, while coarser two- and four-packet claims reduced load
balance. It was therefore rejected rather than retained as a default.

The next scheduler must be distributed or hierarchical. More importantly, a
small-batch path needs a generated three-or-more-layer decomposition or online
reordering whose packets have local dependency closure. That is the mechanism
required to expose the homogeneous subgraph stream within one long transform.
