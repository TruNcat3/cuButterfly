# Research Status: V100 Evidence Closure

## Scope

cuButterfly is a single-GPU research artifact for hardware-mapped space-time
parallelism across regular butterfly computations. V100 is the only fully
measured hardware target. Cross-GPU transfer is explicitly future work and is
not used to support any current portability claim.
Power-of-two and numeric-regime evidence below is the v0.5 authority. The
general-shape/API extension and its separate protocols are summarized in
[v0.6 Implementation Status](v0.6_implementation_status.md); those results do
not retroactively change older controlled tables.

The V100 uint32 `10+10` study now has aggregate and packet-granular online
modes. The aggregate packet core closes the physical-codelet gap and beats
v0.6 by 1.187x under fixed clocks at batch 16. The online mode is functionally
complete, preserves single-read/single-write boundary traffic, and beats v0.6
across batch 16/32/64, while the latest bitmap/warp-row points reach
0.967x/0.989x/0.992x aggregate throughput. Polling backoff, readiness packing,
row mapping, CTA service weights, and wave-barrier folding have now been
screened independently.

The matched fixed-clock captures separate the remaining mechanisms. Bitmap
readiness primarily removes control work: relative to per-packet, batch-16/32
CBU instructions fall by 6.4%/11.0%, while load sectors fall only 0.7% and
shared-load conflicts are unchanged. Warp-row stage-0/1 mapping then removes
8.7% of online load sectors and about 19% of shared-load conflicts, but warm
event time changes by only -0.4% to +0.1% at batch 32/64. Its faster consumer
does not move the optimal `5:15` producer/consumer allocation, and eliminating
one barrier per q-wave changes time by less than 0.6%. The remaining gap is
therefore fine-grained two-segment protocol work that is hidden or absent in
the aggregate schedule, not an unresolved coalescing, bank-conflict, service
balance, or redundant-barrier defect.

The multi-segment checkpoint now makes the large-stage count itself a design
dimension. One runtime descriptor executes `M=2..8`; each generic segment has
`2..10` stages and the ordered partition sums to `logN`. Random forward/inverse
checks pass through the eight-role limit. The generator enumerates all 3950
legal ordered `logN=20` partitions, then retains a bounded shortlist within
each `M` before expanding physical cores and boundary choices. The compiled
V100 specializations still cover every permutation of `6+7+7` and `6+6+8`.
Within that controlled three-level family, `6+6+8` beats `7+7+6` by 1.26x at
batch 1 and 1.29x at batch 4.

This is not yet a competitive physical point. `6+6+8` takes
0.478/1.966/10.272 ms at batch 1/4/16 after its batch-16 service weights change
from `6:6:8` to `4:4:12`; packet128 `10+10` takes
0.140/0.325/1.144 ms. A 128-thread/four-row three-level ablation is slower at
0.482/2.524/14.244 ms. Task granularity alone is therefore not the
transferable packet128 feature: the next codelet must fuse the four-row
butterfly computation itself inside each variable-size subgraph.

A first `M` screen deliberately holds the generic radix-4 core, 256 threads,
two CTAs/SM, and full-scratch boundaries fixed. At batch 1, the shortlisted
`M=2..8` points take 0.392/0.574/1.109/2.189/4.890/8.332/12.722 ms. This is
not evidence that two logical subgraphs are universally optimal: in this
control every added subgraph materializes another complete global boundary.
At batch 16, `M=4` (9.560 ms) overtakes `M=3` (11.519 ms) despite both losing
to `M=2` (3.704 ms), confirming that service balance and homogeneous load can
change the local ordering. The next search layer must therefore cross `M`
with full/ring/resident-fused boundary realization before NCU attribution.

That boundary-realization layer is now executable. `ResidentFused` keeps the
logical `stage_partition` unchanged and lowers adjacent segments into `G`
physical execution groups, with independent physical mappings and only `G-1`
materialized boundaries. A first controlled V100 check holds logical
`5+5+5+5` and generic radix-4 fixed: `G=4/3/2` takes
0.957/0.918/0.370 ms at batch 1, so `G=2` is 2.58x faster than the former
full-scratch `G=M` control. This identifies the earlier M-scaling loss as a
boundary-realization artifact, not a rejection of additional logical
homogeneous subgraphs.

The three-trial equivalence check then lowers every `M=2..8` to the same
physical `G=2,10+10` schedule. With the generic radix-4 group core, the spread
across logical M is only 0.20%/1.39%/1.99% at batch 1/4/16. Replacing only the
execution-group mapping with the mature `dataflow-radix4` core reduces the
median from 0.427/1.113/3.757 ms to 0.149/0.366/1.155 ms, a
2.88x/3.04x/3.25x gain; the corresponding logical-M spread remains below
1.3%. This is the intended separation: logical factorization no longer forces
traffic, while physical-core quality remains an independent design-space
axis. The results and reproducer are in
`results/resident_execution_groups/v100_initial/`,
`results/resident_execution_groups/v100_g2_confirmed/`,
`results/resident_execution_groups/v100_g2_dataflow_confirmed/`, and
`scripts/benchmark_resident_execution_groups.sh`.

The matched NCU capture confirms both mechanisms. Lowering `M4/G4` to
`M4/G2` reduces global-load sectors by 2.73x/2.56x and warp instructions by
5.01x/4.91x at batch 1/16, even though the combined group increases shared
state from 1 to 32 KiB. Replacing the generic G2 core with the mature
dataflow core then reduces load sectors by another 2.64x/2.62x and warp
instructions by 1.44x/1.43x, drops registers from 72 to 40, halves shared
state to 16 KiB, and raises active warps by 1.93x/1.99x. Finally, physically
equivalent M2/M4 dataflow points differ by at most 1.27% across sector and
instruction counters. Full attribution is in
`results/ncu_resident_execution_groups/analysis.md`.

The subsequent 30-point lowering ablation separates that architecture result
from physical-core coverage. Resident M-to-G lowering improves the
full-scratch path by 3.598x geometric mean, but the measured M/G candidates
reach only 0.394x v0.6-best and win 1/30 points. This is not a version-level
v0.7 comparison because earlier HybridDataflow and packet-shared v0.7 cores
are intentionally excluded. At `logN=20`, matched M4/G2 v0.7
and M2/G2 v0.6 physical schedules agree within 0.02%--0.66%; at shorter
lengths, however, 6--9-stage groups still use the generic descriptor core
because only 10+10 has a generated dataflow specialization. The architecture
lowering is therefore validated, while the next implementation task is
compile-time physical-core generation across stage logs 6--9. See
[`resident_v06_comparison.md`](resident_v06_comparison.md) and
`results/resident_v06_comparison/v100_confirmed/`.

This does not revise the matched 18-shape release result: HybridDataflow
v0.7-search remains 1.028x v0.6-search overall, with 8 wins, 1 parity point,
and 9 losses. That table includes the uint64 `logN=10` region where v0.7 is
2.127x faster; the M/G ablation starts at `logN=12` and intentionally omits
that physical core.

The focused crossover audit identifies why a version-level crossing remains.
For uint32 `logN=12`, explicit `Td=6/12` controls differ by less than 1%, so
the generated default-table threshold is not causal. The resident radix-4
kernel instead produces one CTA per transform and admits two CTAs/SM: batch
160 fills the 80-SM physical wave and reaches 1.132x Hybrid2D, while batch 256
requires a second wave whose tail fills only 60% of resident slots and falls
to 0.859x. Hybrid2D supplies finer intra-transform CTA work. At uint32
`logN=20,batch=16`, refreshing the v0.6 envelope from the historical 17:13
control to Hybrid2D/9:11 changes packet128's comparison to 0.935x. Thus the
crossovers come from non-nested physical spaces, CTA-wave quantization, and
baseline evolution, not from the butterfly dependency graph. The release
selector must retain the lower envelope, while the generator must import
Hybrid2D multi-CTA mappings before claiming a nested v0.7 space. Full data is
in `results/v06_v07_crossover/analysis.md`.

## Research Claims And Evidence

| Claim | Evidence | Current result |
|:--|:--|:--|
| Logical factorization is independent of physical execution | controlled `D2/D3/D4 -> P2` FFT lowering | the same physical schedule agrees within 1.8% |
| Processing units are replaceable mapping parameters | scalar, register, warp, cuFFTDx, TurboFFT, WMMA, radix and reduction alternatives | winners change with operator, length, batch and precision |
| Boundary realization affects the optimum without changing the paradigm | scalar versus paired 128-bit online cuFFTDx boundaries | best-point gain is 0.6%-13.5% across six long-FFT shapes |
| The physical-unit gain has a counter-level mechanism | fixed-mapping pre/post NCU capture at `logN=20`, batch 16 | CUDA-event -10.5%, replay -9.3%, warp instructions -19.6%, unchanged DRAM writes |
| Static hardware constraints are useful but insufficient for final ranking | 48-candidate FFT static-score evaluation | top-1 geomean regret 1.0762x; worst 1.4220x |
| A small hardware calibration set can replace exhaustive timing | leave-one-batch-out FFT interpolation | top-3 geomean regret 1.0007x, worst 1.0020x using 6.2% of candidates |
| The selection method extends beyond FFT | leave-one-complete-shape-out selector over NTT/FWHT/XOR-zeta | 93.88% top-1, 100% top-3, 1.0049x geomean regret |
| Numeric and batch preference is piecewise rather than globally smooth | 736-case screen, 29 full follow-ups, 12 paired NCU profiles, 99 adaptive anchors | all 9 length crossovers reproduce; 6 batch events are unstable; bounded selector coverage is 27.74% with 1.01112x worst regret |
| General-shape composition must select both physical units and boundaries | exact FFT/NTT, embedding, rank-two, direct-output and fused-input follow-ups | direct exact FFT reaches vendor parity by abstention; embedded `32767 -> 32768` FWHT reaches 0.11759 ms after selected boundary fusion |

The static-model failure is part of the result, not a row to hide. Resource
legality, occupancy, waves, and issued work provide a useful shortlist, but do
not capture launch-limited low-batch behavior accurately enough. The supported
method is therefore hierarchical: static legality and service constraints,
then a small measured calibration set, then dispatch.

## Current Performance Boundary

The latest matching-protocol FP32 long-FFT comparison uses five process trials,
1000 warmups, 100 timed repetitions, correctness preflight, and resident CUDA
event time.

| Shape | cuButterfly | cuFFT | VkFFT | cuButterfly/cuFFT |
|:--|--:|--:|--:|--:|
| `logN=18`, batch 16 | 0.195942 ms | 0.211098 ms | 0.201103 ms | 1.077x |
| `logN=20`, batch 4 | 0.214733 ms | 0.210330 ms | 0.206520 ms | 0.979x |
| FP64 `logN=14`, batch 256 | 0.340173 ms | 0.344433 ms | not measured | 1.013x |
| FP64 `logN=16`, batch 64 | 0.339364 ms | 0.343040 ms | not measured | 1.011x |
| FP64 `logN=17`, batch 32 | 0.346307 ms | 0.395213 ms | not measured | 1.141x |
| FP64 `logN=18`, batch 16 | 0.354877 ms | 0.421970 ms | not measured | 1.189x |

The broader design scan reaches 1.085x cuFFT at `logN=18`, batch 2 and 1.030x
at `logN=20`, batch 2. At saturated `logN=20` batch 8/16 it reaches
0.962x/0.954x. These focused results use a different batch protocol and must
not be mixed into the comprehensive rows above.

The previous ten-shape snapshot reported 1.015x FFT, 1.054x FWHT, and 1.313x
NTT throughput against the matching libraries. The final full-protocol refresh
reports 1.019x FFT, 1.052x FWHT, and 1.314x NTT. The temporary FWHT reversal
was traced to general-shape bounds and independent-stride predicates entering
the exact power-of-two register kernel; compile-time exact-shape specialization
restores the historical performance. GPU-NTT rows remain archived same-machine matching-
protocol measurements rather than interleaved rows from this refresh. XOR-zeta
remains an internal architecture ablation because no maintained matching
external baseline has been identified.

## Semantic Coverage

| Operator | Numeric coverage | Measured lengths | Additional contracts |
|:--|:--|:--|:--|
| FFT | FP16/BF16 storage with native-width or FP32 accumulation, FP32, FP64 | 3, 8, 12, 14, 16, 18, 20 | forward/inverse, normalized inverse, in/out-of-place, stride 2 |
| NTT | 30-bit/32-bit and 60-bit/64-bit | 12, 16, 20 | forward/inverse, natural and native bit-reversed order |
| FWHT | FP16/BF16 storage with native-width or FP32 accumulation, FP32, FP64 | 8, 12, 15, 20 | forward, multiple exchange and residency choices |
| XOR-zeta | uint32 | 8, 12, 20 | forward/inverse, multiple mapping families |
| Structured 2x2 | FP16/BF16 storage with native-width or FP32 accumulation, FP32, FP64 | 8, 12, 15, 20 | broadcast/per-stage matrices, forward/inverse, shared/register cores |

Coverage establishes generality of the abstraction, not universal superiority.
The original FP64 scalar row remains 0.642x cuFFT in the comprehensive suite.
A controlled follow-up first scans 264 scalar mappings and confirms the winner
at 0.658x, then replaces the local unit with double-precision cuFFTDx and reaches
0.893x with table twiddles. Register recurrence reduces the selected prefix's
table lookups and raises the final result to 0.948x. This isolates most of the
former deficit in the physical unit and its precision-specific CTA/EPT mapping.
The controlled NCU capture shows recurrence reduces prefix replay time by
17.5% and warp instructions by 6.9%, while spending 8.4% more FP64 instructions
with unchanged registers, shared allocation, and waves/SM. Prefix DRAM peak
rises from 60.2% to 72.9%; suffix time is effectively unchanged. The remaining 5.5%
event-time gap is now localized to shared exchange and address work: total
shared-bank conflicts remain unchanged and warp instructions are still 2.06x
cuFFT.
An XOR-swizzled prefix then redistributes the same-capacity shared slot layout
and reaches 0.353526 ms, or 0.970x cuFFT. The complete 36-point remap retains
the same `256/4 + 128/8` winner, so this is a boundary-layout gain rather than
a mapping-selection artifact. NCU confirms that it cuts prefix shared conflicts
by 53.5% and prefix replay by 5.3%, raising peak DRAM utilization from 72.9% to
77.0%; the suffix remains unchanged. Its 10.0% increase in prefix warp
instructions identifies swizzled address generation as the next optimization
target. The implemented strength reduction computes each swizzled pointer once,
advances it by a compile-time stride, and hoists invariant transform/global
address work out of the staging loops. It lowers linear recurrence from
0.361708 ms to 0.345027 ms and XOR swizzle from 0.353526 ms to 0.342098 ms.
The latter matches the 0.342856 ms cuFFT measurement at 1.002x throughput; five
trial ranges are 0.341893-0.342170 ms and 0.342610-0.343040 ms, respectively.
This is parity evidence rather than a claim of a robust lead. Updated NCU
counters confirm the predicted mechanism: versus the pre-optimization XOR
kernel, prefix warp instructions fall 26.8%, prefix replay falls 8.3%, and
prefix shared conflicts fall 8.6%; suffix replay changes by less than 0.4%.
The full XOR path now replays 2.5% below cuFFT while executing 1.83x its warp
instructions, 3.62x its integer instructions, and about 114x its shared
conflicts. Prefix registers rise from 48 to 64 and the register block limit
falls from 5 to 4, so strength reduction exchanges occupancy for substantially
less dynamic address work.

The FP64 adapter now instantiates 128/256/512-point segments and covers every
two-segment composition whose dimensions are both `logN=7..9`. A 648-point
coarse scan selects distinct mappings for `logN=14..18`; a subsequent
five-trial, execution-order-interleaved matrix covers batch 1..64. Applying the
existing 0.020-ms timing floor leaves 25 stable shapes: 20 have higher median
throughput than cuFFT and 19 have completely separated faster trial ranges.
All 13 stable `logN=17/18` shapes are faster. The five remaining median deficits
are confined to `logN=14..16` crossover batches and range from 0.1% to 3.2%.

## Remaining Work

The v0.7 research branch introduces a strict-resident NTT `HybridDataflow`
backend. It maps a fixed-size subgraph to persistent warp roles, streams
independent data tokens through bounded shared channels, and feeds online
reordered results back into one CTA-resident state. Forward/inverse uint32 and
uint64 paths, batch execution, layout/state/handoff ablations, and generated
`T_s/U_s/T_d/U_d` points now pass the complete V100 correctness suite.

The controlled NCU attribution separates the architectural effects. At
`logN=12`, balanced `6+6` stage decomposition reduces the old `8+4` replay
time from 331.2 us to 187.6 us, barrier stall from 50.55% to 13.26%, warp
instructions by 27.5%, and integer instructions by 20.8%. `U_s=2` executes
10.64x as many warp instructions and 8.68x as many integer instructions as
balanced `U_s=6`, reaches only 3.12% active warps, and takes 5.39 ms. Generated
compile-time data unfolding selects `T_d=4`: it reduces the balanced point to
150.3 us under NCU by removing another 25.1% of warp instructions and 27.6% of
integer instructions. CUDA-event scans independently place it at 0.0526 ms for
uint64 `logN=10`, batch 16 and 0.1454 ms at `logN=12`; `T_d=8` crosses the
channel-resource/pipeline optimum and regresses.

The first concrete capacity boundary occurs at uint64 `logN=13`: balanced
`U_s=7,T_d=2` occupies 90160 B, while `T_d=4` would require 114736 B and is
illegal on V100. The selector must reject such points analytically before
benchmarking. Full counters and secondary handoff/layout ablations are in
[`results/ncu_hybrid_dataflow_td/analysis.md`](../results/ncu_hybrid_dataflow_td/analysis.md).

The next implementation turns the streamed stage sequence into a true fused
subgraph. `role_stages` (`U_r`) controls how many consecutive stages remain in
lane-owned registers inside one persistent role; warp shuffle performs local
reordering, and shared channels now occur only between fused roles. Released
stage warps are reassigned to independent tokens, converting stage-space
parallelism into data-space parallelism instead of reducing CTA width. All
forward/inverse uint32/uint64 fusion tests pass exact reference equality.

The V100 screen shows the expected piecewise preference. Full fusion improves
`logN=10` by 1.36x--1.55x for uint32 and 1.04x--1.29x for uint64. At uint64
`logN=12`, `U_r=6,T_d=8` loses at batch 1/16 and is 1.46x faster at batch
256, but uint32 `logN=12` still prefers `U_r=1`. The best batch-256 uint64
points are now 1.58x behind Tile256 at `logN=10` and 2.00x behind it at
`logN=12`, substantially narrower than the original 4.0x/6.6x gaps. See
[`results/hybrid_dataflow_roles/analysis.md`](../results/hybrid_dataflow_roles/analysis.md).

The pre-launch-bounds NCU capture validates the physical mechanism rather than only the timing outcome. Full
fusion removes 25%--33% of integer instructions and roughly 90% of shared-load
bank conflicts. At `logN=12`, it reduces shared allocation from 53288 B to
32768 B, moving V100 residency from one CTA/SM to two at saturated batch. The
tradeoff in that capture is 142 registers per thread and long-scoreboard stall
rising from 3.52% to 15.63%. The generated launch bound now reduces the current
`R_b=1` cubin to 103 registers and permits three CTAs/SM without local
allocation. An `R_b=3` candidate reaches 89 registers but creates no additional
residency and loses performance, so the default remains `R_b=1`. See
[`results/ncu_hybrid_dataflow_roles/analysis.md`](../results/ncu_hybrid_dataflow_roles/analysis.md).

This remains an explicit research backend. The remaining gap is now attributed
to the physical register ownership and role-channel units, not to a forced
global-memory boundary and not to the configurable graph-scheduling
abstraction.

The confirmed preference regions are now emitted from the V100 generator as a
runtime default table. `logN=10` selects `U_r=5,T_d=10,T_i=1` for
uint32/uint64, and uint64 `logN=12` selects `U_r=6,T_d=12,T_i=1` for all
batches. Deeper packets amortize packet-boundary control enough to remove the
previous uint64 `logN=12` batch threshold. Other shapes explicitly report
`resource-fallback` and retain `U_r=1`. User-provided `U_r/T_d/T_i/R_b`
values are never overwritten. A generated
`R_b=3` launch-bounds candidate reduces the full-fusion cubin from 103 to 89
registers per thread but adds no residency beyond the 32 KiB shared-memory
ceiling and is 2%--26% slower over the primary batches, so `R_b=1` remains the
default.

The register-level token-interleave experiment separates packet depth from
instruction scheduling. Against the previous `T_d=4/8` points, the new
`T_d=10/12,T_i=2` candidates improve every screened shape by 1.08x--1.37x.
However, at matched packet depth `T_i=2` is normally 3%--8% slower than
`T_i=1`; the gain belongs to packet amortization rather than long-scoreboard
hiding. The `T_i=2` kernels use no local memory, so this is a register live-range
and scheduling tradeoff rather than a spill artifact. The option remains in the
generated design space as an NCU-backed physical-unit ablation.

The matched NCU capture makes that attribution concrete. Deep `T_i=1` is
1.189x/1.197x faster than the old packet at uint64 `logN=10/12`. At `logN=12`,
register count, shared-memory block limit, and active warps do not change while
active-cycle IPC rises from 0.27 to 0.32, ruling out occupancy as the gain.
`T_i=2` lowers `logN=10` long-scoreboard stall by 2.92 percentage points, but
adds 13.9% integer instructions and loses 5.8%; at `logN=12` it adds 8.3%
integer instructions and loses 4.4%. The compiler-level interleave mechanism
works narrowly, but its bookkeeping/live-range cost dominates on V100.

The subsequent fine packet-depth study covers every `T_d=1..16`, repeats the
candidate plateaus, and scans batch in steps of 16 through 640 (1,452 checked
runs in total). It exposes a structural sawtooth: with `R=U_r` replicated
warps, packet critical length is
`ceil(K/T_d) * ceil(T_d/R)`, so `T_d=kR+1` makes only part of the warps execute
an additional token slot. Useful candidates cluster at or just below `kR`.
Register-limited CTA wave capacity supplies the second breakpoint variable.

The strongest stable new region is uint32 `logN=12`: `T_d=12` wins through
`2*SM_count`, while `T_d=6` wins every sampled batch from `2*SM_count+1`
through 640, by up to 1.50x. This hardware-relative boundary is now in the
generated default table. uint64 `logN=10` alternates between `T_d=5` and
`T_d=9/10` as different four/five-CTA wave capacities are crossed; the robust
minimax `T_d=10` remains the default until a wave-aware cost selector replaces
range overfitting.

The first wave-aware predictor has also been evaluated rather than assumed.
Training uses the six representative batches and testing uses 140 unseen
batch/shape scenarios from the dense scan. Features include cubin register
usage, derived CTA residency, packet count, critical slots, discrete CTA
density, and partial waves. Histogram-gradient regression reaches 1.0188
geometric regret but 1.151 worst regret, failing the runtime admission gate.
The guarded static strategy reaches 1.0052 geometric, 1.0357 p95, and 1.0694
worst regret with 0.986 top-2 recall. The library therefore keeps the measured
table and exposes the learned model only as an offline top-2 search hint. This
is a negative result for calibration-free selection, not a reason to obscure
the unstable cases with a larger opaque model.

The v0.5 quick screen now covers FP16/BF16 accumulation contracts, FP32/FP64,
uint32 zeta, and uint32/uint64 NTT. It finds 29 confirmed and 67 ambiguous
mapping crossovers. Complete-regime top-3 recall is 1.0, but geometric-mean and
worst regret are 1.047 and 2.078, so global transfer remains
`measurement-required`. A conservative piecewise model now auto-selects only
between same-winner stable batch anchors. Its leave-one-batch-out coverage is
27.74% after adaptive full timing, with 1.00013x geometric-mean and 1.01112x
worst regret. Of 99 weak non-boundary anchors, 66 become stable and 33 remain
near-ties; none reverse direction. This validates a bounded region with
explicit abstention; it does not promote the model to a global runtime selector.

1. **Conditional cliff model:** expand the current stable-interval model and determine how numeric width, coefficient
   policy, modular reduction state, local-unit instruction mix, length, batch,
   stride, direction, and normalization move residency cliffs and mapping
   crossovers on the fixed V100 hardware target.
2. **Boundary-focused selector validation:** hold out complete numeric and
workload regimes, not just individual shapes, and report top-k recall and
   regret around launch, occupancy, register, shared-memory, and bandwidth
boundaries.
3. **Remaining performance margins:** explain and reduce the FP32 `logN=20`
   saturated-batch gap, the five FP64 `logN=14..16` crossover deficits, and the
   Structured register cliff at `logN=13..15` without conflating physical-core
   improvements with the scheduling contribution.
4. **Cross-GPU transfer:** only after the conditional single-GPU model is
   established, capture a newer GPU profile and test whether the learned
   descriptors and predicted boundaries transfer before and after calibration.

The repository does not currently claim a production cuFFT replacement,
universal library superiority, or architecture portability beyond V100.
Non-power-of-two and rank-two execution are implemented through the v0.6 C
plan composition layer, but only the documented exact/embedding semantics and
single-V100 evidence are claimed.

The v0.7 matched NTT refresh makes the new backend boundary explicit. Across
18 uint32/uint64 `logN=10/12` and batch shapes, the resident radix-4
HybridDataflow point improves the original role-pipeline point by 4.760x and
reaches 1.028x of v0.6 searched throughput overall (8 wins, 1 parity, 9 losses).
It is 2.127x faster for uint64 `logN=10`, while uint32 and uint64 `logN=12`
reach 0.785x and 0.651x. The streaming semantics are now performance-credible;
larger resident states still need a hierarchical multi-CTA composition.

That composition is now implemented as `HierarchicalDataflow`. It executes two
resident radix-4 graph layers in one cooperative kernel, uses persistent CTAs,
and materializes exactly one online-transposed global boundary. Cross-layer
twiddles are fused into the second layer. Forward/inverse uint32/uint64 tests
cover balanced and asymmetric splits through `logN=20` with exact reference
equality.

At the V100 `logN=20,batch=4` crossover, three-run medians put the selected
uint32 point at 0.2421 ms versus 0.2806 ms for Hybrid2D (1.159x), while uint64
is 0.4572 ms versus 0.4280 ms (0.936x). The selected occupancy changes from
five to two CTAs/SM, identifying numeric state width as the next physical-unit
boundary. The complete mapping and reproduction commands are in
[Hierarchical Dataflow NTT](hierarchical_dataflow_ntt.md).

The corrected two-pass NCU comparison rules out global traffic as the cause:
hierarchical/Hybrid2D read ratios are 0.966x for uint32 and 1.000x for uint64.
uint32 reduces warp/integer instructions to 0.934x/0.985x and reaches 1.215x
NCU speedup. uint64 raises them to 1.054x/1.071x, uses 48 versus 32
registers/thread, and reaches 0.932x. Forced register caps spill and regress;
the remaining target is 64-bit persistent-control work, not another boundary
or a modified NTT formula.

The fixed-graph local-core ablation sharpens that target. Importing the mature
Hybrid2D radix-4 schedule reduces warp instructions by 5.7%/4.6%, integer
instructions by 8.3%/6.0%, and registers from 38/48 to 32/44 for uint32/uint64.
It nevertheless loses 14.7%/2.9% under NCU because effective residency stays
at 5/2 CTAs per SM and long-scoreboard stall rises by 12.45/3.43 percentage
points. Physical-unit selection must therefore model both discrete residency
cliffs and dependency-latency hiding; register or instruction minimization
alone is not a sufficient objective.

The next v0.7 checkpoint adds a generated `7+7+6` resident kernel for
`logN=20`. Unlike `10+10`, its dependency graph lets segment 1 start before
segment 0 finishes inside one transform. Exact forward/inverse tests pass for
uint32 and uint64, and the physical scan selects 32 units per CTA with
`6:6:8` role weights. At batch 1 it remains 6.5%/20.4% slower than the generic
three-layer kernel for uint32/uint64 because the second global boundary closes
late and fixed roles spend capacity waiting. This separates a successful
streaming-semantics result from the still-open distributed-scheduler problem.

The matched follow-up NCU capture confirms that attribution. At uint64
`logN=20,batch=4`, resident `7+7+6` reduces warp/integer instructions from the
generic core's 2.182x/1.534x of v0.6 to 1.218x/1.131x, but measured DRAM read
rises to 6.282x and barrier stall to 37.02%. A three-process batch sweep shows
resident `10+10` crossing v0.6 near batch 16, whereas resident `7+7+6` becomes
relatively slower. The architecture exposes a valid wave; fixed CTA ownership
and the second broad dependency closure prevent the physical implementation
from converting more homogeneous work into throughput. The current report is
[`results/hierarchical_streaming/v07_current_comprehensive_comparison.md`](../results/hierarchical_streaming/v07_current_comprehensive_comparison.md).

The subsequent `appt-pipeline` checkpoint replaces fixed CTA segment ownership
with the intended `A=(Us/Ts,Ud/Td)` macro-tile. Persistent warp roles pipeline
register tokens, stage-time folds reuse those roles, and one-kernel online
reordering produces the next token-major layout. Exact 32/64-bit tests pass at
`logN=12` and `logN=20`, including inverse execution. A matched V100 screen
selects `Us=7`, one stage per warp, two channel buffers, and `Td=4` at batch 1
or `Td=8` at batch 4. State-path throughput rises with batch from 62.6 to
64.7 GB/s for uint32 and 91.0 to 96.2 GB/s for uint64, confirming the expected
amortization trend. Absolute latency is still only 0.12x--0.22x of v0.6.
`Us=8` crosses a register/channel resource cliff and two-stage fused roles lose
modular-multiply issue capacity. The next optimization target is a coalesced
tiled online transpose plus a lower-live-range physical unit, not a return to
fixed role scheduling. See
[`results/appt_pipeline/checkpoint/analysis.md`](../results/appt_pipeline/checkpoint/analysis.md).

Matched NCU further shows that graph admission is not the dominant deficit.
The APPT `Us=8,role=1` fixed term is only 158.8 us (7.1% at batch 1), while its
fitted steady cost is 2068.4 us/transform versus 131.0 us for v0.6. The kernel
executes 13.50x the warp instructions, generates roughly 4x the global sectors,
stalls 39.2% at barriers, and uses 127 registers/thread while reaching only
10.5% of peak DRAM throughput. CUDA Graph capture may reduce application-side
submission latency, but cannot amortize these repeated in-kernel costs. See
[`results/appt_pipeline/ncu/analysis.md`](../results/appt_pipeline/ncu/analysis.md).

The generated physical-subgraph follow-up makes a substantial but not yet
decisive improvement. A coalesced tiled fold transpose, grouped token
handshakes, two complete pipeline replicas per CTA, and a compile-time
`logN=20,Us=7` schedule reduce the four 32/64-bit batch-1/4 checkpoints by
41.7%--53.4%. Current medians are 0.3515/1.2081 ms for uint32 and
0.5377/1.9723 ms for uint64. They reach 0.264x--0.388x of the matched v0.6
resident baseline. This confirms that generated physical-unit selection was a
real missing axis, while also ruling out single-kernel launch overhead as the
remaining explanation. The matched counter pass reduces warp/integer
instructions from `13.53x/5.26x` to `3.86x/2.06x` of the barrier kernel and
raises DRAM utilization from 10.9% to 24.0%. However, global load/store sectors
remain `3.56x/4.00x`, barrier stall is still 24.2%, and long-scoreboard stall
rises to 30.8%. The residual target is now the materialized fold boundary and
its memory dependency chain, followed by handoff cost and modular-multiply
latency hiding. See
[`results/appt_pipeline/generated_checkpoint/analysis.md`](../results/appt_pipeline/generated_checkpoint/analysis.md)
and
[`results/appt_pipeline/generated_ncu/analysis.md`](../results/appt_pipeline/generated_ncu/analysis.md).

The benchmark now reports both CUDA-event `kernel_ms` and host
`timed_region_wall_ms`. On the uint64 batch-1 generated point the median
one-shot difference is 8.10 us, only about 1.3% of kernel time, and it falls to
0.27 us/invocation over a 30-launch region. This is a measured upper bound that
also includes event/synchronization overhead; CUDA Graph submission cannot
account for the residual APPT gap.

The next `appt-online` checkpoint now executes the three `7+7+6` folds as
concurrent persistent CTA roles. Producers publish directly in the next fold's
token-major layout, device counters release dependency groups online, and no
fold-wide grid barrier remains. Exact forward/inverse uint32/uint64 validation
passes. Against staged APPT, uint32 improves by 1.305x at batch 1 and 1.402x at
batch 4; uint64 changes from 0.888x at batch 1 to 1.082x at batch 4. The
load-dependent crossover is the first direct evidence that more homogeneous
subgraphs improve the intended cross-fold pipeline. Absolute throughput is
still only 0.303x--0.501x of v0.6, and the implementation pays two N-sized
online boundary buffers. See
[`results/appt_pipeline/online_checkpoint/analysis.md`](../results/appt_pipeline/online_checkpoint/analysis.md).

The subsequent matched NCU capture confirms that this is a real steady-state
improvement rather than a launch artifact. Online folds improve staged APPT
steady throughput by 1.184x and reach 1.113x staged performance at batch 4,
while reducing warp/integer instructions by about 56%. The current limit is
now physical: `4.50x` store sectors, 11.9% DRAM utilization, 18.3% active warps,
and a 3.51x steady-cost gap to the v0.6 resident core. An independent 3x3 scan
of fold-1 publication and fold-2 output microgroups finds a small 1.043x win
only for uint32 batch 1; that result was subsequently invalidated by crossed
template dispatch conditions. The corrected scan finds 1.055x/1.022x
publication-only gains for uint32 batch 1/4, while uint64 still prefers
fine-grained readiness. This
identifies non-blocking ready-task selection, rather than wider grouping alone,
as the next pipeline-scheduling target. A follow-up role-weight screen moves
uint64 batch 4 from `7/7/6` to `5/7/8` for a further 1.4% gain; this confirms
fold 2 as the bottleneck but is too small to replace work-conserving role
stealing. See
[`results/ncu_appt_pipeline_analysis/analysis.md`](../results/ncu_appt_pipeline_analysis/analysis.md)
and
[`results/appt_pipeline/online_fold_microgroup_matrix.md`](../results/appt_pipeline/online_fold_microgroup_matrix.md).

The new intrusive per-role timer separates the remaining uint64 batch-4 cost.
Fold 0/1/2 active work is balanced after task-size normalization, while fold 1
pays the dominant boundary publication cost and fold 2 the dominant readiness
wait. A perfect work-conserving schedule has a measured active-work lower bound
of 1.109 ms versus the 1.935 ms diagnostic runtime and 0.614 ms v0.6 runtime.
Therefore scheduling and pipeline holes are the first target; they explain
roughly 63% of the excess over v0.6. The residual 37% requires reducing the
extra `7+7+6` global pass and replacing the warp-radix2/twiddle physical core
with the mature resident radix-4 unit. See
[`results/appt_role_breakdown/analysis.md`](../results/appt_role_breakdown/analysis.md).

The physical subgraph core is now an explicit online-schedule design axis.
`appt-online-radix4` retains the exact three-role DAG and online layouts but
replaces the warp radix-2 executor with a 32-row CTA-resident radix-4 codelet;
direct twiddle addressing avoids the historical expanded-root-table traffic.
After independent role-weight searches, the new core improves uint64 batch 1/4
by 1.521x/1.394x, reaching 0.4413/1.2714 ms. It is 3.8%/7.7% slower for uint32,
where native 32-bit shuffle is preferable. The resulting selector rule is
therefore numeric-type -> physical core -> measured service weights, rather
than a fixed core and a separately tuned scheduler. A batch-1 active-work
lower bound of about 0.165 ms is already faster than v0.6, but the batch-4
bound rises to about 0.852 ms as per-task service time degrades under load.
The next NCU pass targets this load-dependent core slowdown before dynamic role
migration is evaluated. See
[`results/appt_core_space/v100_confirmed/analysis.md`](../results/appt_core_space/v100_confirmed/analysis.md).

That matched NCU pass is now complete. The uint64 CTA core executes only
0.83x/0.93x the v0.6 warp instructions and 0.95x/0.97x the integer thread
instructions at batch 1/4, so the butterfly arithmetic core no longer explains
the remaining 1.94x/2.29x time ratio. The online graph instead performs
1.56x--1.87x DRAM writes, 2.10x--3.13x global-load sectors, and
4.25x--4.50x global-store sectors. Its CTA barrier stall is 25.7%--46.3%
versus 2.8%--5.1% for v0.6, while achieved DRAM throughput remains only
13.0%--31.9%. The batch-dependent active-work increase is therefore caused by
materialized boundary traffic, poor L1 locality, memory-dependency latency,
and static-role synchronization rather than extra modular arithmetic. The
next physical point should fuse a dependency-closed pair of folds into one
resident tile to remove one global boundary; work-conserving role migration is
then applied to the remaining inter-tile edge. See
[`results/ncu_appt_core_space/analysis.md`](../results/ncu_appt_core_space/analysis.md).

That dependency-closed tail is now implemented as three controlled physical
points. Full-tail removes state2 but its 64x128 tile restricts uint64 to one
CTA/SM. Split-tail restores residency but adds an N/2 global handoff.
Register-tail instead reuses a 32x129 padded tile and retains the closed low
half in the remaining shared-memory budget plus registers. The current cubin
uses 47 registers and 32 KiB shared for uint32 (3 CTA/SM), and 128 registers
and 48 KiB shared for uint64 (2 CTA/SM), with zero stack/local spill.

After role-weight/readiness calibration and CTA-local coefficient-tree reuse,
register-tail reaches 0.2223/0.7449 ms for uint32 batch 1/4 and
0.3129/0.9783 ms for uint64. It is the fastest v0.7 online
core in every measured case and improves the prior best online core by up to
1.32x, but remains at 0.470x--0.732x v0.6 throughput. Batch-1 traces put fold0
at about 67/88 us and the fused tail at about 176/250 us for uint32/uint64;
the next attribution target is therefore the physical tail executor rather
than role dispatch. See
[`results/appt_tail_cores/analysis.md`](../results/appt_tail_cores/analysis.md).

The matched NCU pass confirms that the remaining gap is not modular
arithmetic or register spill. Register-tail executes 0.67x--0.85x the v0.6
warp instructions and 0.81x--0.89x the integer-thread instructions; local
load/store sectors are zero. However, online state and readiness addressing
generate 2.33x--3.58x global-load sectors and 2.25x--2.50x store sectors.
L1 hit rate falls to 27.5%--31.0%, while L2 hit rate rises to 83.1%--93.5%
and long-scoreboard stall reaches 37.6%--49.9%. The kernel reaches only
13.0%--31.1% peak DRAM throughput, so it is request-latency limited rather
than bandwidth saturated. A readiness ablation selects aggregate transform
counters for uint32. The matched before/after NCU replay improves by
1.4%--6.8%, but global load/store sectors change by at most 0.03%; readiness
latency is real but does not cause the excess request traffic. Uint64 retains
distributed flags because counter contention offsets reduced polling. These
counters predate the coefficient-tree cache. The refreshed capture confirms
that caching the producer and low-tail trees removes 20.5%--23.2% of global
load sectors and improves replay time by 3.8%--7.8%, with unchanged store
sectors. The residual stores are approximately `N+N/8` sectors for uint32 and
`N+N/4` for uint64: state1 is coalesced, but a fixed-`c` tail task scatters the
natural-order output across 128-element strides. The current source also
preserves the same tree through the high tail, adding another 2.1%--4.5% event
gain without changing residency. Detailed counters are in
[`results/ncu_appt_tail_coeff_cache/comparison.md`](../results/ncu_appt_tail_coeff_cache/comparison.md).

The current library refresh reports 1.019x cuFFT, 1.052x Dao FHT, and 1.314x
archived GPU-NTT throughput. Detailed scope and per-length results are in
`v06_v07_comprehensive_comparison.md`.

### Grouped APPT producer checkpoint

**Status correction:** the event and NCU captures below used a grouped
producer state ordered as `[c][b][a]`, while the tail consumed `[b][c][a]`.
The delta-only regression did not expose the swap. Those numbers are retained
as diagnostic history, not valid NTT performance; corrected random smoke now
passes and both event timing and NCU must be recollected.

The first physical data-space implementation groups 8/16/32 low-bit `a`
coordinates in the producer, transposes the tile on chip, and stores the first
boundary as `[c][b][a]`. This preserves the logical `7+7+6` graph and radix-4
arithmetic while making the APPT data-space parameter affect actual memory
requests. The V100 event screen selects group 32 at every numeric point. It
improves matched register-tail radix-4 by 1.095x/1.161x/1.577x for uint32 batch
1/4/16 and 0.992x/1.069x/1.707x for uint64. The best large-batch points reach
0.528x and 0.636x v0.6 throughput, materially reducing but not closing the
remaining gap. The next counter pass must verify that the gain comes from
fewer global sectors and restored cache locality; only then should group 64 or
tail-side request aggregation be added. See
[`results/appt_grouped_producer/analysis.md`](../results/appt_grouped_producer/analysis.md).

The superseded counter pass reported batch-16 DRAM-read reductions of
53.4%/45.9% versus matched radix-4 for uint32/uint64 and L2-hit increases of
4.8/12.7 percentage points, while L1 remained zero. Because its layout was
incorrect, those ratios are hypotheses about request organization rather than
valid NTT evidence and require recapture. They motivated the controlled move
of stage 19 from the fixed-`c` tail into the already-required natural-order
writer without adding a memory boundary. See
[`results/ncu_appt_grouped_producer/analysis.md`](../results/ncu_appt_grouped_producer/analysis.md).

The writer-final physical core is implemented as a separate design-space
point. It preserves the grouped `7+7+6` graph and existing global value
boundaries: the tail publishes pre-final low/high halves, and the natural
writer performs stage 19 while transposing fragments. Correctness covers
uint32/uint64 producer groups 8/16/32 with random-vector verification. The
V100 event scan jointly selected fragment width, writer task coarsening, and
P/T/W weights before the next NCU capture.

That event scan and a three-trial confirmation are complete. Writer-final
improves corrected grouped throughput by 1.111x/1.190x/1.158x for uint32 batch
1/4/16, reaching 0.850x/0.685x/0.593x v0.6. For uint64 the ratios are
0.996x/1.028x/0.994x, so the core is neutral rather than a replacement. The
numeric split is now the main attribution question: NCU must determine whether
uint64 loses the request reduction or whether fewer active writer warps and
64-bit modular arithmetic consume it. Confirmed timings are in
[`results/appt_writer_final/confirmed/analysis.md`](../results/appt_writer_final/confirmed/analysis.md),
and the selected capture command is
`sudo -E "$PWD/scripts/profile_appt_writer_final_ncu.sh"`.

The selected NCU capture is complete. Writer-final reduces load sectors by
21.6%--23.4% for uint32 and 16.4%--17.3% for uint64 with essentially unchanged
warp/integer instructions. Uint64 also keeps the same two-CTA/SM shared-memory
residency and 127 versus 128 registers/thread, rejecting arithmetic work,
spill, and occupancy as the cause of its neutral timing. The distinction is
cache behavior: uint32 DRAM reads fall 6.6%--7.6%, while uint64 loses 6.1--8.6
L2-hit points and DRAM reads rise 0.7%--6.4%. Relative to v0.6, uint64 also has
an exact 1.50x store-sector floor from three materialized value writes versus
two. The next mapping target is batch data-time traversal inside each fixed
spatial subgraph, so transforms reuse stage coefficients before advancing the
spatial coordinate. Full counters are in
[`results/ncu_appt_writer_final/analysis.md`](../results/ncu_appt_writer_final/analysis.md).

Role-local data-time unfolding is now implemented as a separate physical core.
The kernel fixes one spatial producer/tail/writer task in a CTA or warp and
traverses `T_d` batch transforms locally. A three-bit role mask exposes where
that traversal is placed (producer=1, tail=2, writer=4), independently of the
logical `7+7+6` graph and P/T/W service weights. Twelve random uint32/uint64
checks cover `T_d={2,4}` and masks `{4,6,7}`.

The V100 event screen shows that producer temporal traversal delays the
downstream readiness wavefront. At uint32 batch 16 and `T_d=2`, mask 6
(tail+writer) is 1.006x writer-final, while mask 7 is 0.976x. The best point is
still 0.596x v0.6; uint64 remains below writer-final for every `T_d>1` point.
Thus role-local coefficient reuse is valid but insufficient. The physical
roles still exchange full transform state through global memory, so this is
not yet the intended end-to-end resident subgraph. The next implementation
target is a composite physical role that retains a dependency-closed tile
across producer and tail work; another role-weight or radix-only sweep cannot
remove the measured traffic floor. Results are in
[`results/appt_data_time/role-placement/analysis.md`](../results/appt_data_time/role-placement/analysis.md).

That composite point is now implemented as
`appt-online-register-tail-grouped-writer-final-resident`. Dependency tracing
showed that a fixed producer `c` is not closed: the grouped online transpose
makes one tail tile consume columns from 128 producer tiles. The corrected
kernel fixes `a`, keeps the first two seven-stage dimensions in a 128x128
resident subgraph, publishes stage-13 state, then reuses the existing
stages-14--18 register tail and stage-19 natural writer. Random uint32/uint64
verification passes at batch 1 and 4.

The first V100 screen is a resource negative control. The 96 KiB/CTA shared
state and 255-register specialization force every role in the unified kernel
to one CTA/SM. Resident-2D reaches only 0.279x/0.258x/0.277x v0.6 throughput
for uint32 batch 1/4/16 and 0.291x/0.275x/0.296x for uint64. Its throughput
relative to writer-final improves from 0.330x to 0.472x (uint32) and 0.415x to
0.498x (uint64) as batch grows, supporting workload amortization but rejecting
the current physical storage point. The next gated step is a four-quarter
resident implementation capped at 48 KiB and at least two CTA/SM; role and
batch scans resume only after that resource gate passes. Full timings are in
[`results/appt_resident_2d/analysis.md`](../results/appt_resident_2d/analysis.md).

The four-quarter follow-up is also complete. It executes stages 0--11 in four
32x128 tiles, performs the stage-12 pair joins and stage-13 half join without
changing the grouped boundary contract, and reduces shared memory to 48 KiB.
This restores two CTA/SM and improves the 96 KiB core by 1.070x--1.655x over
the uint32/uint64 batch 1/4/16 matrix. Absolute throughput remains
0.280x--0.464x v0.6 and 0.481x--0.717x writer-final.

The remaining loss is now localized to the physical retained-state context.
The two-CTA launch cap produces 128 registers/thread plus 528-byte uint32 or
672-byte uint64 stacks. A 128-thread experiment raises the register allocation
to 255 but increases per-thread state and is slower. The next implementation
is therefore role-specialized or warp-register quarter execution, not another
role-weight sweep. Timings and selected service mappings are in
[`results/appt_resident_quarter/analysis.md`](../results/appt_resident_quarter/analysis.md).

The homogeneous `10+10` branch has now tested that warp-granular boundary
directly. The original CTA-radix4 `D_t=(2,1)` point moves the same DRAM bytes
and uses the same 48 registers/thread and 32816 shared bytes as v0.6, but wait
stall rises from 37.63% to 45.71% and runtime rises 18.3%. A complete
1024-point-per-warp core is register-limited to one CTA/SM. Splitting it into
256-point warp prefixes plus one four-warp exchange improves every tested
shape, but supplies only 0.305x--0.571x v0.6 throughput. The loss grows rather
than shrinks with batch, ruling out graph launch amortization as the primary
cause. The remaining search is therefore a warp-owned radix-4/8 codelet and
coalesced static edge writer, not another outer data-time sweep. See
[`docs/homogeneous_subgraph_template.md`](homogeneous_subgraph_template.md).

That static-writer point is now implemented. It forms eight-row uint32 or
four-row uint64 packets, uses an XOR shared-memory swizzle, and writes complete
32-byte sectors at the producer boundary and final output. Holding radix-2
arithmetic fixed, it improves warp256 by 1.12x--1.81x across batch 1/4/16 and
reaches 0.514x/0.547x v0.6 at uint32/uint64 batch 16. The generated kernels use
162/169 registers per thread without local spills; the 32 KiB packet staging
now makes shared-memory residency and radix instruction count the explicit
next tradeoff.

Matched NCU sharpens that conclusion: static output reduces store sectors 8x
to 4.194M, warp instructions are only 1.039x v0.6, and integer instructions are
0.617x. Load sectors remain 1.774x and barrier stalls remain 13.5%. The new
static-IO candidate coalesces producer row packets and performs consumer bit
reversal in shared memory. Its first counter pass reduces load sectors to
38.074M but exposes roughly 18M shared bank conflicts in both directions. The
integrated mapping now uses `P(i)=i xor (i>>5)` for the complete packet and one
barrier per packet, reducing those conflicts to 4.813M/6.359M. The resulting
172-register kernel still limits half packets to two CTAs/SM. A uint32
four-row row-major specialization with a seven-word bank skew reduces that to
164 registers without local memory and restores three CTAs/SM. Fine role
balance selects 11:9, 8:7, and 17:13 producer/consumer weights at batch 1/4/16,
reaching 0.837x/0.831x/0.822x v0.6. Uint64 retains four-row XOR-AoS static IO
and reaches 0.573x--0.631x. This rejects a single universal physical point
while preserving one architecture template.

The confirmation NCU pass closes the layout attribution. At uint32 batch 16,
the skewed half packet launches 240 rather than 160 blocks and improves the
previous half packet by 1.50x. Shared store conflicts fall to zero, load
conflicts to 2.482M, and barrier stall to 7.44%. The remaining loss is the
physical warp core: 164 registers permit only 12 resident warps, compared with
32 for v0.6, and fixed-latency wait stall remains 18.46% versus 10.44%.
Accordingly the next screen reduces the per-warp subgraph from 256 points to
128/64 and treats radix-4/8 as an internal codelet choice. Additional boundary
layouts or coefficient caches are not selected by the current counters.

That screen is complete for uint32. Occupancy-specific 128/64-point kernels
compile to 101/113 registers without local memory and admit four 128-thread
CTAs/SM, so the intended residency increase is real. Nevertheless, at
`logN=20`, batch 16 they reach only 0.680x/0.659x v0.6 throughput and trail the
256-point half-packet point. Across batch 1/4/16, warp128 reaches
0.785x/0.750x/0.680x v0.6, warp64 reaches 0.749x/0.721x/0.659x, and warp256
reaches 0.841x/0.835x/0.804x. The experiment separates two notions
that the next schedule must not conflate: smaller per-warp state raises
resident parallelism, but a synchronous four-warp row still serializes prefix
and merge work. The selected follow-up is therefore a generated prefix/merge
warp pipeline with explicit role counts and buffer depth, using 128/64 as
prefix codelet choices rather than complete row executors.

The first explicit prefix/merge screen is now closed. A single-slot online
core publishes 128-point tiles, consumes tile pairs at stage 7, and restores
four-CTA V100 residency at exactly 128 registers/thread. It is numerically
correct but reaches only 0.497/1.354/4.376 ms at uint32 batch 1/4/16, versus
0.149/0.351/1.069 ms for v0.6. Its deficit is stable under larger batch and is
slightly worse than the prior double-slot prototype. Therefore graph launch,
shared buffer capacity, and all-tile publication are not the controlling
factors. The selected next physical abstraction is a continuation-capable
subgraph DAG in which stage-7/8/9 work is split or stealable; a permanently
assigned merge warp is rejected because it collapses runnable warp count at
every row boundary. Matched NCU confirms 21.08% barrier stall, 2.904M local
load sectors, and 0.461M local store sectors, while active warps remain 24.05%
and wait/long-scoreboard stalls fall. Thus register spill is a secondary
penalty, but the dominant architectural defect is the non-stealable
continuation and its row-reuse barrier.

A cooperative continuation now closes that attribution experimentally. Four
warps retain work through stage 7, two warp pairs execute stage 8, and all
four execute stage 9. It is correct and improves fixed 3+1 by 2.13x--2.30x.
The 157-register occupancy-3 instance reaches 0.234/0.590/2.009 ms for batch
1/4/16 and is up to 13% faster than the 128-register occupancy-4 instance.
It still trails synchronous warp128 by 1.23x--1.29x because its explicit
hierarchy materializes stage-7/8 state through shared memory. The next
architecture point must preserve the one-barrier register continuation while
overlapping independent row subgraphs; further intra-row barriers are not
selected.

The matched counter pass confirms that choice. The cooperative occupancy-4
instance spills 5.694M local-load sectors; occupancy-3 removes the spill and
reduces barrier stall to 4.58%, but exposes only 17.95% active warps and
23.96% wait stall. The next generated instance therefore packs two independent
synchronous warp128 groups into one 256-thread CTA. This retains the
register-resident tail while delegating cross-row issue interleaving to the
V100 warp schedulers.

The dual-row-group instance is implemented and correct. Two independent
synchronous warp128 groups share a 256-thread CTA, use separate named
barriers, and preserve the 101-register physical core. Three-trial medians
show no change at batch 1, a 9.4% loss at batch 4, and a 15.4% gain at batch
16, where it reaches 0.847x v0.6 throughput. This establishes a second mapping
crossover: row-subgraph replication should be enabled only when sustained
work is sufficient to benefit from intra-CTA warp interleaving.

The matched NCU pass confirms the mechanism rather than merely the timing.
Dual versus single-group warp128 lowers kernel time from 1614.944 us to
1495.808 us and long-scoreboard stall from 35.59% to 28.58%, with essentially
unchanged active warps, global-load sectors, and warp instructions. Independent
row subgraphs therefore hide dependency latency inside a CTA as intended.
The remaining gap is now localized to the radix-2 physical core: compared
with v0.6, dual warp128 executes 33.5% more global-load sectors and 30.6% more
warp instructions, consumes 101 versus 40 registers/thread, and exposes only
24.74% versus 47.77% active warps. The next design-space implementation keeps
dual row-group scheduling and substitutes radix-4/8 or vectorized butterfly
codelets; it does not add another synchronization hierarchy.

The first such ablation makes low-stage coefficient reuse explicit without
changing the schedule. It is a separate generated core and passes full NTT
correctness. Matched NCU measures 1468.992 us versus 1484.672 us for the
unchanged dual core. Global-load sectors fall 31.9% and long-scoreboard stall
falls 15.5%, proving that the reuse is effective, while 104 versus 101
registers/thread leaves active warps unchanged. The point reaches only
1.011x throughput gain because it still executes 28.7% more warp instructions
than v0.6 and exposes about half as many active warps. Subsequent physical-core
work must therefore combine coefficient reuse with lower instruction count
and lower state.

A second generated ablation combines coefficient reuse with the warp256
prefix. It passes full correctness, retains three CTAs/SM, and reduces
registers from 164 to 158/thread. Its initial event median is nevertheless
1.455 ms versus 1.234 ms for ordinary warp256. Matched NCU confirms the
mechanism: global-load sectors fall 37.4%, but time rises 6.4%, wait stall
rises 21.6%, and active warps stay fixed near 18%. The fully reused warp256
point is rejected because coefficient fan-out reduces instruction-level
overlap. Reuse depth now becomes a physical-codelet search parameter rather
than a global rule.

That parameter is exposed through the CLI, CSV, generator, and architecture
manifests. The complete fixed-clock sweep selects depth 1 for warp256 and
depth 4 for dual warp128. They improve their no-reuse controls by 3.3% and
6.0% in throughput, respectively. The selected dual-warp128 point reaches
89.0% of v0.6; its remaining 11.0% gap is now dominated by 24.5% more executed
warp instructions and roughly half the active-warp percentage, while global
load sectors are only 5.5% higher. The response is non-monotonic in loads,
instructions, registers, and dependency stalls. All depths 1--6 compile,
pass full NTT correctness, and are retained as generated physical-core axes;
V100 defaults are calibrated to warp256=1 and dual-warp128=4.

The next physical substitution preserves that dual-row-group schedule and
replaces its prefix arithmetic with four consecutive values per lane and a
register-local radix-4 first pair. It passes the full cuNTT correctness suite,
uses 106 rather than 112 registers/thread, and allocates no local memory. The
current-build uint32 `logN=20`, batch-16 event median is 1.238446 ms, versus
1.275003 ms for coefficient-reuse depth 4 and 1.514537 ms for the v0.6
control. However, fixed-clock NCU reverses the ranking: vector radix-4 takes
1448.032 us versus 1409.856 us for reuse d4 and 1270.944 us for v0.6. It issues
68.8% more global-load sectors at unchanged DRAM bytes and active warps, adds
2.8% integer instructions, and raises barrier stall by 3.34 percentage points.
The unlocked-clock event lead is therefore not used to change the V100
default. The next combined codelet retains vector arithmetic while making
warp coefficient-broadcast depth 3--6 a generated physical-unit axis. That
axis is now implemented and passes the full correctness suite. Event medians
for vector d0/d3/d4/d5/d6 are 1.239/1.327/1.330/1.284/1.292 ms, versus
1.274 ms for radix-2 reuse d4. Depth 5 is the only broadcast point near the
selected control, but raises register count from 106 to 128; the expanded NCU
pass will determine whether its lower coefficient requests compensate for the
shuffle/instruction cost under fixed clocks.

The first broadcast pass does not: masking lanes leaves every slot-wise load
instruction intact and NCU sectors remain near 53.2M. The corrected
distributed-load codelet issues one consecutive coefficient load across lanes
and shuffles it to four register slots. Its d6 event median reaches 1.201336 ms,
5.9% faster than radix-2 reuse d4 and 3.1% faster than vector d0, while using
106 registers with no local allocation. This now qualifies as the next
fixed-clock candidate; it is deliberately not installed as the V100 default
until refreshed counters confirm the intended load-instruction reduction.

The refreshed fixed-clock pass confirms that reduction. Distributed d6 lowers
vector d0 load sectors by 25.1%, integer instructions by 4.4%, and
long-scoreboard stall from 27.98% to 19.38%; replay improves from 1424.224 to
1345.696 us without changing its 106-register footprint or active warps. It is
5.1% faster than radix-2 reuse d4 and becomes the calibrated dual-warp128
physical unit. V0.6 remains 6.1% faster at 1268.832 us, so this promotion is
inside the v0.7 design space and does not override the mature global dispatch
for this measured shape.

### v0.7 sector root-cause audit

The external-memory hypothesis is now separated from the request-organization
hypothesis. Relative to v0.6, vector d6 changes DRAM reads by only 3.1% and
global-store sectors by 0.6%, but raises global-load sectors from 29.85M to
39.45M. The additional sectors are therefore cache requests rather than an
extra N-sized state.

A warp-address model attributes the main delta to the vector codelet's stage-6
coefficient loads. Depth 6 covers `stage < 6`; stage 6 still issues four
half-warp, 16-byte-stride loads per tile. Across 32,768 producer/consumer rows,
the complete model predicts an 11.01M-sector v0.7/v0.6 gap versus the refreshed
9.60M measurement, explaining 87.2%. SourceCounters then measures exactly the
same 6.29M external-data sectors in both kernels and attributes the remaining
v0.7 traffic primarily to coefficient PCs. Source classification covers 100%
of v0.6 and 95.27% of v0.7 theoretical sectors; the 4.73% remainder stays
attached to explicit SASS addresses instead of being assigned speculatively.

The audit also narrows the scheduling claim. Current v0.7 performs packet load,
four complete row NTTs, and packet store in sequence within each warp group.
Two groups interleave through hardware scheduling, but there is no row-token
load/prefix/tail/store software pipeline. Its 106-register, 41,184-byte CTA
permits 16 resident warps/SM versus v0.6's 32. Source/SASS capture commands and
the completed PC-level closure is documented in
`docs/v07_sector_root_cause.md`; no kernel or selector is changed by the audit.

The follow-up depth-7 experiment closes the stage-6 address model directly.
Its even/odd distributed bank reduces d6's 39,999,693 global-load sectors to
31,432,704. The measured 8,566,989-sector reduction is 1.021x the static
8,388,608 prediction and removes 83.7% of d6's excess over v0.6. Time still
rises by 0.8% because warp instructions increase by 1.1% and MIO throttle by
13.7%, despite a 12.3% reduction in long-scoreboard stall. Thus stage-6 request
fragmentation is proven, but the shuffle-based repair is not the final core.
The next candidate is an aligned, prepacked stage-6 coefficient table consumed
with `LDG.128`; d6 remains the calibrated v0.7 point until that candidate wins.

That candidate is now a separate generated core. A plan-owned aligned side
table adds no execution launch, and SM70 SASS confirms `LDG.E.128.SYS` for both
coefficient and Shoup vectors. The physical point reduces register use from
d6's 106 to 98/thread without local allocation. An unlocked-clock 100-repeat
screen measures d6/d7/packed at 1.360/1.240/1.273 ms, so packed beats d6 but not
d7 in that screen. The controlled capture explains why: vector16 reduces
global-load sectors from d7's 31.34M to 26.73M, yet increases time by 7.1% and
long-scoreboard stall by 27.3%. Sixteen wide-load lanes lose memory-level
parallelism even though alignment and register pressure improve.

A second generated core now distributes two aligned stage-6 coefficients to
each of all 32 lanes. It produces `LDG.E.64.SYS`, uses 109 registers/thread with
no local allocation, and retains two CTAs/SM. The first same-session screen is
1.191 ms versus d7's 1.245 ms and vector16's 1.274 ms, making it the fastest
v0.7 physical core in that event screen. Global selection remains unchanged
until the updated five-way, cache-controlled NCU comparison in
`scripts/profile_packed_stage6_ncu.sh` is captured.

The five-way fixed-clock result measures lane32 at 1336.3 us: 0.5% ahead of d7
and 6.65% behind v0.6. Its 26.36M load sectors and 16.47% long-scoreboard stall
close the coefficient-access problem; the remaining v0.6 advantage correlates
with 181.2M versus 237.7M warp instructions and 47.9% versus 24.7% active
warps. A lane32-local event scan selects `D_t=1:1`, `86:74` CTAs, but rejects
cross-core claims because the event controls vary by more than 3%. The focused
fixed-clock confirmation command is
`scripts/profile_packed_stage6_lane32_schedule_ncu.sh`.

### v0.7 instruction closure and packet-shared radix-4

Dynamic SourceCounters attribution identifies the remaining lane32 deficit as
physical-codelet overhead rather than external traffic: d6 adds 21.49M `SHFL`
instructions and 23.06M control instructions over v0.6. The new packet-shared
core fuses four rows into five 128-thread radix-4 stage pairs and keeps one or
two packets independently schedulable. Forward/inverse checks pass for batch 1
and 16. The first same-session screen places packet128 at 1.073 ms versus
lane32 at 1.341 ms and v0.6 at 1.583 ms; this remains an event result pending
the fixed-clock instruction-pipe capture.

## Reproduction Map

- FFT mapping and vectorization: `results/v100_fft_pipeline_vectorized_*`
- FFT prediction evaluation: `results/v100_fft_pipeline_model_*`
- latest comprehensive FFT: `results/comprehensive_v100_vectorized_fft_*`
- cross-operator semantics: `results/comprehensive_v100_full_*`
- calibrated selector: `results/v100_mapping_selector_*`
- numeric quick/follow-up/coverage: `results/v100_numeric_*`
- numeric boundary attribution: `results/ncu_numeric_boundaries/`
- library/base/search matrix: `results/v100_three_way_*`
- v0.6/v0.7 matched NTT matrix: `results/v100_v06_v07_*`
- v0.6/v0.7 crossover attribution: `results/v06_v07_crossover/`
- matching external baselines: `results/v100_external_baselines_*`
- counter attribution: `results/ncu_scaling_crossovers/`
- APPT physical-core mapping: `results/appt_core_space/v100_confirmed/`
- APPT physical-core counter command: `scripts/profile_appt_core_space_ncu.sh`
- APPT physical-core counter analysis: `results/ncu_appt_core_space/analysis.md`
- APPT resident-2D event screen: `results/appt_resident_2d/analysis.md`
- APPT resident-quarter event screen: `results/appt_resident_quarter/analysis.md`
- homogeneous warp-core screen: `results/homogeneous_warp_10x10/analysis.md`
- homogeneous schedule NCU: `results/ncu_homogeneous_10x10_analysis.md`
- dual-warp vector radix-4 event screen:
  `results/homogeneous_vector_radix4/analysis.md`
- vectorized FFT attribution: `results/ncu_fft_vectorized/`
- FP64 scalar/core/mapping validation: `results/fp64_*logN16*`
- FP64 length/batch robustness: `results/fp64_robustness_*`
- FP64 privileged counter command: `scripts/profile_fp64_fft_ncu.sh`
- general-shape composition: `results/general_shape_*`,
  `docs/general_shape_results.md`

Run `./scripts/reproduce_v100_analysis.sh` to regenerate derived artifacts from
the checked-in raw measurements. It does not recollect timings or counters.
