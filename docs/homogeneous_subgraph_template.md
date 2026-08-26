# Homogeneous Two-Level Subgraph Template

## Purpose

This experiment separates the architecture schedule from the arithmetic
codelet. A two-level transform is described by:

- stage partition `(K0, K1)`;
- producer and consumer stage-space roles;
- row-group data-space ownership;
- independent producer/consumer data-time factors `(Dtp, Dtc)`;
- a selectable physical butterfly core and CTA service weights.

`10+10` is the first generated instance, not the definition of the method.
The CUDA template accepts `FirstLogN`, `SecondLogN`, and `RowsPerBlock`; the
current build emits the V100 `10+10/rows4/threads256/full-scratch` point.

## GPU Mapping

Each resident CTA has a fixed producer or consumer role. A spatial task owns
one four-row group. For `D_t > 1`, that CTA keeps the spatial group fixed and
walks a group of batch transforms before taking another spatial task. Producer
completion is published with a release operation; the consumer waits with an
acquire operation. Both roles reuse the same resident radix-4 codelet shape.

The warp scheduler automatically selects among ready warps from resident CTAs.
It therefore already overlaps independent producer and consumer CTAs in the
v0.6 kernel. It does not create the graph partition, data layout, readiness
protocol, or a load/compute/store pipeline inside one CTA.

## Static Physical Selection

The logical `D_t=(1,1)` point is exactly the mature v0.6 physical schedule and
is dispatched to that codelet. Non-degenerate points use the generated
homogeneous traversal kernel. This is deliberate: a logical mapping can select
an existing best physical core without changing the architecture definition.

The generated kernels use the following resources on V100:

| word | v0.6 registers/thread | homogeneous registers/thread | occupancy limit |
|---:|---:|---:|---|
| 32-bit | 40 | 45 | 4 CTA/SM; shared memory |
| 64-bit | 48 | 48 | 2 CTA/SM; shared memory |

Neither kernel spills local state. The 64-bit temporal gap is consequently a
control/address scheduling cost, not an occupancy or arithmetic-core change.

## V100 Result

Two bracketed trials and ten timed repetitions were collected for 32/64-bit
NTT, batch 1/4/16, six data-time pairs, and three CTA weight pairs. The full
records are in
[`results/homogeneous_10x10/`](../results/homogeneous_10x10/).

| bits | batch | v0.6 ms | best non-degenerate `(Dtp,Dtc)` | temporal ms | temporal/v0.6 |
|---:|---:|---:|---|---:|---:|
| 32 | 1 | 0.1484 | `(4,1)` | 0.1483 | 1.000x |
| 32 | 4 | 0.3492 | `(1,2)` | 0.3674 | 0.951x |
| 32 | 16 | 1.0859 | `(2,1)` | 1.1438 | 0.949x |
| 64 | 1 | 0.2293 | `(2,1)` | 0.2588 | 0.886x |
| 64 | 4 | 0.6129 | `(2,1)` | 0.7611 | 0.805x |
| 64 | 16 | 1.8461 | `(2,1)` | 2.1987 | 0.840x |

The static `D_t=(1,1)` control is within measurement noise of v0.6. Merely
traversing complete CTA subgraphs in data-time order does not improve the
steady-state pipeline.

NCU confirms that this is a scheduling effect rather than extra traffic. For
64-bit batch 16, `D_t=(2,1)` takes 2364.256 us versus 1998.272 us for v0.6,
while both move approximately 513 MiB read and 257 MiB written and use 48
registers/thread plus 32816 bytes shared memory. Wait stalls increase from
37.63% to 45.71%, and DRAM utilization falls from 44.53% to 37.90%. The full
counter table is in
[`results/ncu_homogeneous_10x10_analysis.md`](../results/ncu_homogeneous_10x10_analysis.md).

## Required Next Core

The current resident radix-4 core is a full-CTA codelet: all 256 threads share
one tile and synchronize after each radix round. An outer `D_t` loop therefore
serializes complete subgraphs. A true mixed-dataflow implementation needs a
physical core with all of these properties:

1. warp-granular dependency-closed subgraphs;
2. disjoint or bank-safe state partitions for concurrent tokens;
3. explicit prologue, compute, and epilogue phases;
4. at least two in-flight token buffers without reducing occupancy below the
   latency-hiding requirement;
5. static output remapping before publication;
6. fine readiness that does not wait for a whole transform.

On V100, `cp.async` is unavailable and a second 64-bit full CTA tile would
reduce occupancy. The next generated physical candidate should therefore use
smaller warp-register tiles or multiple independently addressable shared
partitions, not a double-buffered copy of the current 32 KiB tile.

## Warp-Granular Physical-Core Screen

Two generated cores now test that requirement:

- `homogeneous-warp-radix2` assigns one complete 1024-point row to each warp;
- `homogeneous-warp256-radix2` assigns a 256-point register prefix to each
  warp, then uses one four-warp named-barrier exchange for the final two stages.

The producer writes the second coordinate in static bit-reversed order, so the
consumer reads its row contiguously. The 256-point variant supports 128-thread
and 256-thread CTAs and therefore exposes the physical subgraph size as a
search parameter. Both are correct for 32-bit and 64-bit NTT.

| bits | batch | v0.6 ms | 1024/warp relative throughput | best 256/warp threads | 256/warp relative throughput |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 0.147763 | 0.334x | 128 | 0.490x |
| 32 | 4 | 0.350004 | 0.278x | 128 | 0.421x |
| 32 | 16 | 1.080781 | 0.235x | 256 | 0.321x |
| 64 | 1 | 0.203879 | 0.267x | 128 | 0.571x |
| 64 | 4 | 0.582451 | 0.228x | 256 | 0.432x |
| 64 | 16 | 1.935258 | 0.235x | 256 | 0.305x |

Reducing the warp tile is clearly beneficial, but these radix-2 prototypes do
not yet replace v0.6. The full-warp core uses 159/238 registers per thread for
32/64-bit words and permits only one CTA per SM. The 256-point core reduces the
64-bit count to 167 registers and allows three 128-thread CTAs per SM, but it
still executes a radix-2 instruction stream and retains strided producer and
final-writer accesses. Since the relative throughput decreases with batch,
launch/prologue amortization is not the dominant remaining loss. The next
physical core should combine the 256-point ownership boundary with radix-4 or
radix-8 register butterflies and a coalesced static writer.

The first coalesced writer is now implemented as
`homogeneous-warp256-static-radix2`. It groups eight uint32 rows or four uint64
rows, fills complete 32-byte sectors, and uses an XOR row swizzle in shared
memory. It applies the same mapping to the producer boundary and final natural
output while leaving the radix-2 computation unchanged.

| bits | batch | best plain 256/warp ms | static ms | static/plain | static/v0.6 |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 0.301721 | 0.233472 | 1.292x | 0.635x |
| 32 | 4 | 0.831897 | 0.700877 | 1.187x | 0.498x |
| 32 | 16 | 3.301991 | 2.077747 | 1.589x | 0.514x |
| 64 | 1 | 0.454503 | 0.405709 | 1.120x | 0.533x |
| 64 | 4 | 1.360896 | 1.133107 | 1.201x | 0.540x |
| 64 | 16 | 6.410649 | 3.536333 | 1.813x | 0.547x |

This is direct evidence for the APPT static-output rule: the benefit grows to
1.59--1.81x at batch 16, where launch overhead is already amortized. The new
core still trails v0.6 because the physical prefix remains radix-2 and the
static packet consumes 32 KiB shared memory per warp group. A matched NCU pass
now profiles both static variants to separate reduced sectors from added
shared-memory barriers and occupancy loss.

NCU shows that the writer does exactly what the layout model predicts: global
store sectors fall from 33.554M to 4.194M, an 8x reduction, and below v0.6's
8.389M. Warp instructions are only 1.039x v0.6 and integer instructions are
0.617x, so radix-2 instruction count is not the next dominant limit. Load
sectors remain 1.774x v0.6, active warps fall to 12.35%, and barrier stalls
remain high.

`homogeneous-warp256-static-io-radix2` therefore reuses the same staging memory
for coalesced producer row-packet loads, keeps the boundary in natural order,
and applies the consumer bit reversal while loading shared memory. It adds no
shared capacity but introduces an extra group barrier on the read path.

The first static-IO NCU pass reduces load sectors to 38.074M and DRAM reads
below v0.6, but exposes 17.980M/19.416M shared load/store bank conflicts. The
packet-wide `P(i)=i xor (i>>5)` revision cuts those counters to
4.813M/6.359M and reduces matched NCU time from about 2.29 ms to 2.07 ms.
However, it uses 172 registers per thread. Consequently the half packet is
also limited to two CTAs/SM despite requesting three and cannot turn its lower
shared allocation into occupancy.

The next physical instance stores uint32 four-row packets in row-major shared
memory with a seven-word bank skew while retaining `P(i)` within each row.
This reduces the address expression, compiles to 164 registers per thread with
no local memory, and restores three resident CTAs/SM. Other word/packet
combinations retain the faster XOR-AoS layout; the layout choice is therefore
part of the numerical design space rather than a universal constant.

It also exposes the row packet through each role's `data_space`. Full packets
fill 32-byte sectors; half packets halve shared staging and permit a third
128-thread CTA on V100.

| bits | batch | selected IO | rows | threads | selected ms | selected/v0.6 |
|---:|---:|---|---:|---:|---:|---:|
| 32 | 1 | skewed static-IO, weight 11:9 | 4 | 128 | 0.176384 | 0.837x |
| 32 | 4 | skewed static-IO, weight 8:7 | 4 | 128 | 0.419430 | 0.831x |
| 32 | 16 | skewed static-IO, weight 17:13 | 4 | 128 | 1.233510 | 0.822x |
| 64 | 1 | XOR-AoS static-IO | 4 | 128 | 0.398387 | 0.573x |
| 64 | 4 | XOR-AoS static-IO | 4 | 128 | 0.877158 | 0.629x |
| 64 | 16 | XOR-AoS static-IO | 4 | 128 | 2.917939 | 0.631x |

This is a multi-dimensional workload crossover rather than one universal core:
precision and batch jointly select layout, packet width, and producer/consumer
CTA balance. Coefficient caching remains deferred because static-IO DRAM reads
are already close to v0.6.

The final matched NCU capture confirms that the skewed half packet launches all
240 requested blocks and is 1.50x faster than the previous half packet. Shared
store conflicts fall to zero, load conflicts to 2.482M, and barrier stall to
7.44%. The residual limit is instead 164 registers and only 12 resident warps
per SM-equivalent scheduling set, versus v0.6's 32, together with 18.46%
fixed-latency wait stall. This selects smaller 128/64-point-per-warp physical
subgraphs and radix-4/8 codelets as the next screen; another global-layout
variant is not justified by these counters.

The 128/64-point screen is now implemented for uint32 as
`homogeneous-warp128-static-io-radix2` and
`homogeneous-warp64-static-io-radix2`. Both reuse the four-row skewed packet
and fused per-row coefficient tree. An occupancy-4 wrapper lowers their
compiled register counts to 101 and 113 registers/thread, respectively,
without local memory, so four 128-thread CTAs can reside on V100. This is a
useful resource-threshold result, but not a performance win: at `logN=20`,
batch 16, the two-trial event medians are 1.609574 and 1.659750 ms, versus
1.361562 ms for the 256-point half-packet core and 1.094196 ms for v0.6.

Thus reducing lane-owned state is necessary but insufficient. Four warps
still synchronize to complete every 1024-point row, while the smaller prefix
adds shared completion work and gives up instruction-level parallelism. The
next physical instance must expose separate prefix and merge warp counts and
overlap those roles through a shared ring; it must not merely replace a
256-point synchronous prefix by a smaller synchronous prefix.

Focused NCU confirms that attribution. The 128/64 points launch 320 CTAs and
raise active warps from 18.42% to about 24.3%; barrier stall falls from 7.98%
to about 5.6%. They nevertheless execute about 12.5% more warp instructions
and raise long-scoreboard stall from 26.08% to 35.6--36.7%, with global load
sectors unchanged within 1%. The first double-buffered row pipeline is also a
negative result: even after correcting its physical role ratio to three
prefix warps and one register-merge warp, batch-16 time is 4.257 ms. A
single-slot tile-online successor preserves four-CTA residency and overlaps
early merge stages with later prefix tiles, yet reaches 4.376 ms. This rules
out buffer capacity and coarse publication granularity. The remaining
physical requirement is continuation parallelism: tail tasks must be split or
stealable so prefix warps remain useful after publication and can cross row
boundaries without a full-CTA reuse barrier. Matched NCU reports 21.08%
barrier stall and 2.904M local-load sectors for that point, versus 4.86% and
0.080M for synchronous warp128. Wait and long-scoreboard stalls both fall,
which rejects token polling and DRAM latency as the primary explanation.

The follow-up `4 -> 2x2 -> 4` continuation removes the permanent merge role
and improves the fixed 3+1 point by 2.13x--2.30x. Its selected occupancy-3
form reaches 0.234/0.590/2.009 ms at batch 1/4/16 and compiles to 157
registers with a 16-byte stack frame. This validates continuation parallelism,
but it remains 1.23x--1.29x slower than synchronous warp128: the explicit
stage-7/8 hierarchy adds shared-memory materialization and pair barriers where
the synchronous core performs one barrier followed by a register-resident
three-stage continuation. The next pipeline must overlap independent rows
without breaking that register continuation.

That successor is now instantiated as two independent synchronous warp128
row groups in a 256-thread CTA. It retains 101 registers/thread, uses separate
named barriers, and targets two CTAs/SM, preserving 16 resident warps. It is a
workload crossover: batch 1 is unchanged, batch 4 is 9.4% slower, while batch
16 improves single-group throughput by 15.4% and reaches 0.847x v0.6. The
generator must select row-group replication jointly with batch and CTA
residency rather than treating 128 or 256 threads as universal.

Matched batch-16 NCU attributes the dual-group benefit to subgraph issue
interleaving. Kernel time improves from 1614.944 us to 1495.808 us and
long-scoreboard stall falls from 35.59% to 28.58%, while active warps,
global-load sectors, and warp instructions remain effectively constant. This
validates row-group replication as a real architecture mapping axis rather
than a launch-shape artifact. The residual 1.191x time gap to v0.6 belongs to
the physical core: dual warp128 has 33.5% more global loads, 30.6% more warp
instructions, 101 versus 40 registers/thread, and 24.74% versus 47.77% active
warps. The next template point therefore combines the validated dual-group
schedule with a radix-4/8 or vectorized codelet.

The initial coefficient-only codelet substitution is implemented as
`homogeneous-warp128-coefficient-reuse-static-io-radix2`. Matched NCU shows
that explicit stage-0--5 reuse removes 31.9% of global-load sectors and 15.5%
of long-scoreboard stall, but improves throughput by only 1.1%. It raises the
dual kernel from 101 to 104 registers/thread without changing active-warp
percentage. The result narrows the required codelet change: coefficient reuse
is valid but insufficient; the replacement must also reduce instruction count
and/or register state.

Coefficient reuse is consequently a codelet-local parameter, not a universal
policy. The generated warp256 reuse point reduces registers from 164 to 158
and matched NCU removes 37.4% of global-load sectors, but it is 6.4% slower
than ordinary warp256. Wait stall rises 21.6% while active warps are unchanged,
confirming dependency/ILP loss rather than residency. The generator must
select coefficient-reuse depth jointly with prefix size instead of attaching
one global cache policy to every physical core.

`coefficient_reuse_stages` materializes this axis as compile-time depths. A
matched fixed-clock NCU pass at uint32 `logN=20`, batch 16 confirms different
preferences: warp256 depth 2 reaches 1517.888 us but remains 0.58% slower than
its 1509.120 us no-reuse control; dual warp128 depth 4 reaches 1430.176 us and
improves no-reuse throughput by 4.8%. Their compiled register counts are also
non-monotonic: warp256 uses 165/161/158 registers at depths 2/4/6, while dual
warp128 uses 106/112/104. Depth 4 additionally cuts dual-warp128 executed warp
instructions by 4.6%. This is the intended distinction between the
architecture schedule and a physical codelet: prefix width changes the
ILP/code-generation/resource response of the same reuse transformation.
The complete depths 1--6 pass correctness and fixed-clock NCU. Warp256 selects
depth 1 at 1437.568 us, a 3.3% throughput gain over no reuse. Dual warp128
selects depth 4 at 1405.984 us, a 6.0% gain over no reuse and 89.0% of v0.6
throughput. Depth 5 is slower despite using fewer registers because executed
warp instructions rebound. The calibrated search must therefore track the
residency threshold and generated instruction count, not minimize register
count or coefficient traffic independently.

## Vector Radix-4 Physical Codelet

`homogeneous-warp128-vector-radix4-static-io` keeps the selected dual-row-group
schedule but changes each 128-point warp prefix. A lane owns four consecutive
values, stages 0 and 1 are fused into a register-local radix-4 prefix, and only
stages 2--6 use warp shuffle. The existing APPT XOR static layout remains
bank-bijective for each of the four lane slots, so the substitution needs no
new shared-memory permutation or barrier.

On V100 the core compiles to 106 registers/thread with no local allocation,
versus 112 registers/thread for the selected coefficient-reuse depth-4 core.
In the current-build uint32 `logN=20`, batch-16 event screen, its three-trial
median is 1.238446 ms, compared with 1.275003 ms for reuse depth 4 and
1.514537 ms for the v0.6 control. The 2.95% throughput gain over depth 4 is the
first direct positive result from replacing the arithmetic codelet while
holding the validated dual-subgraph schedule fixed.

The focused fixed-clock pass rejects promotion: vector radix-4 takes
1448.032 us versus 1409.856 us for reuse depth 4 and 1270.944 us for v0.6.
Active warps and DRAM bytes are unchanged from reuse d4, but global-load
sectors rise 68.8%, integer instructions rise 2.8%, and barrier stall rises
from 6.38% to 9.72%. Four-consecutive-value ownership reduces register state
but causes repeated cache-resident coefficient requests across lane groups.
The follow-up keeps the vector arithmetic and exposes coefficient-broadcast
depth 3--6 as a compile-time axis. Full event records and the exact protocol
are in
[`results/homogeneous_vector_radix4/`](../results/homogeneous_vector_radix4/).

All combined depths pass correctness without local allocation. The event
medians for vector d0/d3/d4/d5/d6 are 1.239/1.327/1.330/1.284/1.292 ms;
radix-2 reuse d4 reaches 1.274 ms. Early broadcast alone is counterproductive,
while d5 nearly closes the event gap at the cost of increasing register count
from 106 to 128. The expanded fixed-clock pass therefore retains all depths to
measure the sector/instruction tradeoff instead of selecting from unlocked
events.

That NCU pass rejects the first broadcast realization: d3--d6 retain roughly
53.2M load sectors and take 1477--1507 us. Masking lanes within four slot-wise
loads cannot reduce sector requests because the hardware coalescer already
merges equal addresses. The corrected physical unit performs one distributed
coefficient load across consecutive lanes and lets all four register slots
shuffle from those lanes. This distinction is now represented in the codelet,
while the depth axis and outer architecture schedule remain unchanged.

The distributed-load correction passes full correctness and changes the
resource trend: d3/d4/d5/d6 use 103/105/103/106 registers per thread with no
local allocation. Five-process event medians are 1.317/1.330/1.239/1.201 ms,
versus 1.273 ms for radix-2 reuse d4 and 1.239 ms for vector d0. Depth 6 is
therefore 5.9% faster than the selected radix-2 control without spending more
registers than plain vector. This is a candidate, not yet a new default: its
load-sector reduction and extra shuffle count must reproduce under base-clock
NCU.

The refreshed capture reproduces it. Distributed d6 reaches 1345.696 us,
5.1% faster than radix-2 reuse d4 and 5.5% faster than vector d0. It cuts d0
load sectors from 53.403M to 39.989M and integer instructions by 4.4%, while
keeping 106 registers, two CTA/SM residency, and no local allocation. It is
now the selected dual-warp128 physical unit. The remaining boundary is equally
important: v0.6 reaches 1268.832 us, still 6.1% faster under fixed clocks,
with 28.3% fewer warp instructions and roughly twice the active-warp
percentage.

## Reproduction

```bash
scripts/benchmark_homogeneous_10x10.sh
scripts/benchmark_homogeneous_warp_10x10.sh

OUTPUT_DIR=results/homogeneous_warp_small_tiles_occupancy4 \
BATCHES=1,4,16 TRIALS=2 WARMUP=2 REPEAT=10 \
scripts/benchmark_homogeneous_warp_10x10.sh

BITS=64 BATCH=16 BEST_DT=2,1 BEST_WEIGHTS=8,12 \
sudo -E scripts/profile_homogeneous_10x10_ncu.sh
sudo -E scripts/profile_homogeneous_warp_10x10_ncu.sh
sudo -E scripts/profile_homogeneous_small_tiles_ncu.sh
sudo -E scripts/profile_homogeneous_vector_radix4_ncu.sh
```

The small-tile NCU script writes to
`results/ncu_homogeneous_online_tiles/`, compares v0.6, the static controls,
and the single-slot online point, and validates each raw CSV before replacing
an existing result. It requires permission to access GPU performance counters.
