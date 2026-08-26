# Tile-Online NCU Attribution

V100, uint32, `logN=20`, batch 16, matched 10+10 factorization and four-row
static IO.

| point | NCU us | regs | active warps | barrier | wait | long scoreboard | local load/store sectors | warp inst. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.6 | 1264.672 | 40 | 47.71% | 5.91% | 10.46% | 43.37% | 0.000M / 0.016M | 181.981M |
| warp256 | 1535.328 | 164 | 18.41% | 9.32% | 17.92% | 26.42% | 0.000M / 0.008M | 211.430M |
| warp128 | 1603.360 | 101 | 24.27% | 4.86% | 18.20% | 37.49% | 0.080M / 0.016M | 237.509M |
| warp64 | 1733.152 | 113 | 24.31% | 6.16% | 20.10% | 34.58% | 0.085M / 0.016M | 238.206M |
| warp128 tile-online | 4471.008 | 128 | 24.05% | 21.08% | 10.53% | 22.26% | 2.904M / 0.461M | 275.206M |

The tile-online kernel achieves its requested four-CTA residency; active warp
percentage is essentially identical to synchronous warp128. Its 2.79x runtime
regression therefore is not an occupancy failure. Two costs are exposed:

1. The forced 128-register boundary creates substantial stack traffic.
2. Fixed prefix/merge roles raise barrier stall 4.3x even though tile tokens
   are published early.

The second cost is architectural. Wait stall falls 42% and long-scoreboard
stall falls 41%, so neither spinning for tokens nor global-memory dependency
latency explains the runtime. Three prefix warps reach the row-reuse barrier
while one merge warp owns the continuation, shrinking the runnable DAG front.
The same loss repeats for every row and therefore does not amortize with batch.

The next candidate must keep continuation tasks warp-parallel or stealable.
Increasing register budget may remove the local traffic, but cannot address
the barrier-dominated critical path and is not selected as a standalone
optimization.

That candidate is now implemented as a `4 -> 2x2 -> 4` cooperative
continuation. It improves the fixed-role kernel by 2.13x--2.30x across batch
1/4/16, confirming the runnable-front diagnosis. Its occupancy-3 form uses
157 registers and a 16-byte stack frame and outperforms the 128-register,
104-byte-stack occupancy-4 form by up to 13%. It nevertheless remains slower
than synchronous warp128 because explicit stage-7/8 materialization adds
shared round trips and pair barriers. The NCU script includes both residency
forms for the next privileged counter pass.

The privileged pass is complete. Occupancy-4 executes 5.694M local-load and
0.507M local-store sectors and raises long-scoreboard stall to 48.17%.
Occupancy-3 reduces those values to 0.056M/0.016M and lowers runtime from
2.260 ms to 2.083 ms. Its barrier stall is only 4.58%, below synchronous
warp128's 5.15%, so barriers are no longer the residual limit after removing
the fixed merge role.

The occupancy-3 point instead exposes only 17.95% active warps, versus 24.29%
for synchronous warp128, together with 23.96% wait stall and 16% more global
load sectors. This rejects further tuning of the explicit pair hierarchy. The
next point preserves warp128's one-barrier register continuation and places
two independent four-warp row subgraphs in a 256-thread CTA so the hardware
warp schedulers can interleave their dependency windows.

The dual-row-group counter pass validates that mechanism. Relative to the
single-group warp128 point, NCU time falls from 1614.944 us to 1495.808 us
(1.080x throughput). Active warps remain effectively unchanged
(24.29% versus 24.74%), and global-load sectors and warp instructions change
by less than 0.2%. The gain therefore does not come from occupancy or less
work. Instead, long-scoreboard stall falls from 35.59% to 28.58% while the two
independent named-barrier groups give the schedulers another ready subgraph.
The small increases in barrier and wait stall do not offset that latency
hiding. This is direct hardware evidence that row-subgraph replication is a
useful mapping parameter once batch supplies sustained work.

The same pass also isolates the remaining gap. Dual warp128 is 1.191x slower
than v0.6 (1495.808 us versus 1255.808 us). It executes 39.819M global-load
sectors versus 29.819M (+33.5%) and 237.821M warp instructions versus
182.169M (+30.6%), uses 101 versus 40 registers/thread, and reaches only
24.74% versus 47.77% active warps. The scheduling hierarchy is no longer the
first-order defect. The next experiment must retain dual-group replication
while replacing the radix-2 physical subgraph with a lower-instruction,
lower-state radix-4/8 or vectorized core; further barrier-only tuning cannot
close this gap.

An explicit coefficient-reuse codelet is now available as a controlled
physical-core ablation. The matched pass shows that the optimization works:
global-load sectors fall from 39.759M to 27.058M (-31.9%), long-scoreboard
stall falls from 28.86% to 24.39%, and time improves from 1484.672 us to
1468.992 us (1.011x throughput). Registers rise from 101 to 104/thread and
active warps stay nearly fixed at 24.6%, explaining why the large request
reduction has only a small elapsed-time effect.

Coefficient traffic is no longer the residual gap. The cached point already
issues 9.0% fewer global-load sectors than v0.6, yet it is 14.9% slower,
executes 28.7% more warp instructions, and exposes only 51.7% of v0.6's active
warps. The next controlled point combines coefficient reuse with the lower
warp-instruction warp256 codelet; subsequent work must reduce state if that
combination remains residency limited.

The next generated point applies the same reuse policy to the lower-warp-
instruction warp256 prefix. It compiles to 158 registers/thread versus 164
for ordinary warp256 and retains the three-CTA residency target. Matched NCU
confirms that request reduction and elapsed time move in opposite directions:
global-load sectors fall from 40.065M to 25.066M (-37.4%), while time rises
from 1543.680 us to 1643.104 us (+6.4%). Active warps remain effectively
fixed (18.44%/18.06%), so this is not an occupancy regression.

The counter signature identifies lost instruction-level overlap. Wait stall
rises from 18.41% to 22.38% (+21.6%) even as long-scoreboard stall falls from
26.56% to 23.55%; warp instructions also rise 1.7%. Reusing one coefficient
across all same-stage register-slot butterflies shortens the load stream but
creates a longer producer fan-out before those independent modular
multiplications can retire. Coefficient-reuse depth must therefore be searched
per physical codelet. The complete fixed-clock depth sweep is:

| core | none | d1 | d2 | d3 | d4 | d5 | d6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| warp256 us | 1484.672 | **1437.568** | 1514.144 | 1528.352 | 1622.976 | 1628.960 | 1627.008 |
| dual warp128 us | 1489.728 | 1515.232 | 1441.056 | 1411.744 | **1405.984** | 1465.728 | 1459.200 |

Warp256 depth 1 is a real 1.033x throughput improvement over no reuse. It
removes 4.8% of load sectors, reduces warp instructions slightly, and uses
163 rather than 164 registers; deeper reuse raises wait stall faster than it
removes requests. Dual warp128 depth 4 gives 1.060x throughput, removes 21.1%
of load sectors and 4.7% of warp instructions, and reduces long-scoreboard
stall from 29.95% to 26.92%. Its 112-register state does not cross the
two-CTA residency boundary, so the extra registers are effectively free.

Depth 5 is the opposite side of that code-generation cliff: registers fall
to 104, but warp instructions rebound 3.7% relative to depth 4 and elapsed
time grows 4.2%. This proves that minimum registers or maximum coefficient
reuse are both incomplete objectives. The calibrated depths are therefore
warp256=1 and dual-warp128=4. The latter reaches 89.0% of v0.6 throughput;
the remaining 11.0% gap coincides with 24.5% more warp instructions and about
half the active-warp percentage, while load sectors are now only 5.5% higher.
