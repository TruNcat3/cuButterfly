# Hardware-Mapped Space-Time NTT Methodology

This document derives the original two graph-axis model in detail. The
[operator-independent design-space contract](butterfly_design_space.md) adds
ordered multi-dimensional graph factorization, the orthogonal batch unfolding,
and explicit lane-to-node hierarchy mapping without changing this `2 x 2`
graph invariant.

## 1. Two-dimensional iteration domain

An `N=2^K` radix-2 NTT is a two-dimensional iteration domain:

```text
for stage k in [0, K):
    for butterfly b in [0, N/2):
        B(k, b)
```

The architecture decision is how each dimension is unfolded in space and
folded in time. Define the four primary factors:

```text
Us = stage-space unfolding: physically concurrent stage positions
Ts = stage-time folding:    stage groups executed by reusing those positions
Ud = data-space unfolding:  physically concurrent butterflies per stage
Td = data-time folding:     butterfly groups streamed through the same lanes

Us * Ts >= K
Ud * Td >= N/2
```

The useful-work utilization before pipeline and memory effects is:

```text
eta_stage = K     / (Us * ceil(K / Us))
eta_data  = (N/2) / (Ud * ceil((N/2) / Ud))
eta       = eta_stage * eta_data
```

This `2 x 2` unfolding matrix is the architecture invariant. `Npart`, tile
size, CUDA block shape, radix, and modular reduction are mappings or processing
unit choices beneath it.

The APPT/Hermes notation is one concrete parameterization:

```text
Us = Spart = log2(Npart)
Ud = p                       parallel butterflies per stage
Ts = ceil(K / Spart)
Td = ceil(N / (2p))
```

With `Npart=256` and `p=16`, Hermes selects `Us=8`, `Ud=16`; these values are
not part of the invariant.

## 2. Independent hardware demands of the four factors

| Factor | What is replicated or reused | First-order hardware demand | Scaling benefit | Principal limit |
|:--|:--|:--|:--|:--|
| Stage-space `Us` | `Us` stage positions are simultaneously resident | `Us*Ud` butterfly capacity, `Us` stage-local twiddle streams, `(Us-1)*2Ud` inter-stage coefficient links, pipeline state | raises work per state-memory access and reduces stage folds | arithmetic area/issue slots, inter-stage communication, twiddle delivery |
| Stage-time `Ts` | the `Us` positions process later stage groups | feedback path of `2Ud` coefficients/cycle, state storage at the feedback level, stage/mode controller | supports arbitrary `K` without replicating all stages | feedback bandwidth, synchronization, partial-group underutilization |
| Data-space `Ud` | `Ud` butterflies execute per stage per cycle | `2Ud` coefficient read and write bandwidth at pipeline boundaries, `Ud` twiddles/stage/cycle, enough banks/lanes | increases throughput within one stage pipeline | memory ports/banks, execution lanes, routing width |
| Data-time `Td` | later butterfly groups reuse those lanes | capacity for the full resident working set, address generation for changing strides, sustained pipeline II | amortizes the spatial hardware over large `N` | storage capacity, bank conflicts, pipeline fill/drain and tails |

The first-order arithmetic capacity is the product `Us*Ud`, not either factor
alone. Boundary coefficient bandwidth scales primarily with `Ud`, because
stage-space positions forward results internally. Internal link and twiddle
bandwidth scale with `Us*Ud`. This separation is the reason the two dimensions
must not be collapsed into thread count or occupancy.

For a fully sustained pipeline, the ideal body cycles are:

```text
Cbody = ceil(K / Us) * ceil((N/2) / Ud)
```

The implementation adds pipeline fill/drain, feedback, synchronization, bank
conflicts, and memory stalls. Hermes is the point `Us>1`, `Ts>1`, `Ud>1`, and
`Td>1`; stage-based and pipeline-based designs are boundary cases of the same
space.

For `K=16` and `Ud=16`, the architectural trade-off is visible before choosing
FPGA or GPU primitives:

| Point | `Us` | `Ts` | `Td` | Spatial cells `Us*Ud` | Ideal cycles | Inter-stage coefficient links/cycle |
|:--|--:|--:|--:|--:|--:|--:|
| stage-based boundary | 1 | 16 | 2,048 | 16 | 32,768 | 0 |
| Hermes hybrid | 8 | 2 | 2,048 | 128 | 4,096 | 224 |
| pipeline-based boundary | 16 | 1 | 2,048 | 256 | 2,048 | 480 |

All three demand the same 32 coefficient words/cycle at the external pipeline
boundary because `Ud` is fixed. What changes is arithmetic replication,
inter-stage transport, twiddle concurrency, stage feedback count, and latency.
This is the architectural comparison that an occupancy table cannot express.

## 3. Fixed-compute-budget trade-off

The most useful architecture comparison fixes the spatial butterfly budget:

```text
C = Us * Ud
Ud = C / Us
```

Ignoring ceilings, every factorization of the same `C` has the same ideal body
cycles:

```text
Cbody ~= (K / Us) * ((N/2) / Ud)
       = K*N / (2*C)
```

The factorization therefore redistributes hardware pressure rather than
creating free arithmetic throughput:

```text
boundary coefficient words/cycle = 2*Ud = 2*C/Us
inter-stage words/cycle           = 2*Ud*(Us-1)
                                  = 2*C*(1 - 1/Us)
twiddle words/cycle upper bound   = Us*Ud = C
stage feedback folds              = ceil(K/Us)
data stream folds                 = ceil(N*Us/(2*C))
```

Increasing `Us` at fixed `C` reduces boundary and feedback pressure, raises
compute intensity at the state-memory boundary, and increases internal
inter-stage transport and pipeline depth. Increasing `Ud` does the opposite.
The hybrid optimum is where the target hardware can sustain both sides with
acceptable utilization and state residency.

This yields a disciplined experiment: compare multiple `(Us,Ud)` pairs at
equal `C`, coefficient width, root representation, state level, and output
semantics. A speed difference can then be attributed to the space-time
factorization rather than to more arithmetic hardware.

For `K=16` and `C=128`, the ideal cycles remain exactly 4,096 for every legal
factorization in this example:

| `Us x Ud` | Stage folds `Ts` | Data folds `Td` | Boundary words/cycle | Inter-stage words/cycle | State-level bytes across folds |
|:--|--:|--:|--:|--:|--:|
| `1 x 128` | 16 | 256 | 256 | 0 | 16 MiB |
| `2 x 64` | 8 | 512 | 128 | 128 | 8 MiB |
| `4 x 32` | 4 | 1,024 | 64 | 192 | 4 MiB |
| `8 x 16` | 2 | 2,048 | 32 | 224 | 2 MiB |
| `16 x 8` | 1 | 4,096 | 16 | 240 | 1 MiB |

This table is the architecture-level design space. GPU blocks and FPGA module
counts are implementations of a selected row.

## 4. The residency hierarchy couples the factors

The four factors determine traffic only after choosing the memory level `L`
that retains intermediate state. Let `Cap(L)` and `BW(L)` be its capacity and
bandwidth:

```text
state_capacity(L) >= resident_coefficients * word_bytes
feedback_BW(L)    >= 2 * Ud * word_bytes / cycle
```

If all `Ts` stage folds remain at `L`, higher levels see only compulsory input
and output traffic. If state spills between folds, traffic at the next level is:

```text
coefficient_bytes(next level) = 2 * N * word_bytes * spilled_stage_folds
```

Thus stage-time folding itself is not inefficient. It becomes expensive when
the selected hardware level cannot hold or feed its state. FPGA Hermes keeps
the feedback state in banked URAM. A CUDA kernel can retain a CTA tile in
shared memory, but a boundary between kernels normally spills the full array to
device memory. L2 may cache part of that traffic but is not explicit state
storage.

The same hierarchy applies to roots. Stage-space replication raises concurrent
root bandwidth; stage-time folding expands the union of roots reused over time.
The choice between replication, caching, generation, Shoup, and Barrett is a
response to these two different demands.

## 5. Processing unit independence

Define the architecture, processing unit, and hardware realization separately:

```text
Architecture A = (K, Us, Ts, Ud, Td, Ub, Tb, Hs, Rs, Rd, Rb)
Processing P   = (radix, reduction, root_representation, coefficient_width)
Layout L       = (bank_mapping, global_order, output_order)
Realization F  = (kernel_form, tile_threads, warp_stages, pipeline_warps)
```

Here `Hs` is the physical handoff for a spatial stage edge, while `Rs` and
`Rd` are the residence levels across stage-time and data-time folds. The
cross-operator cuButterfly experiment shows why they must be explicit:
temporal-tile and multi-kernel `Us=1` have the same four logical factors but
place `Rs` in shared and global memory, respectively, producing very different
traffic and performance.

Changing the equivalent butterfly does not change `A`, but changes the cost of
one spatial cell and therefore the feasible `Us*Ud`. Layout does not change the
unfolding either, but determines whether the theoretical `2Ud` boundary access
can actually be conflict-free and coalesced.

`F` is deliberately separate from `A`: CUDA thread count and warp count are
mechanisms used to realize logical data and stage parallelism. The current
runtime exposes three forms. `temporal-tile` retains all state in shared memory;
`warp-hybrid` uses shuffle links for an initial prefix and shared memory after
the partner distance crosses a warp; `stage-pipeline` assigns persistent stage
roles to warp groups and explicitly hands off tokens. A portable tuner first
prunes candidates with `A`, `P`, and hardware capabilities, then searches the
legal `F` parameters.

Operator traits guide that pruning:

| Trait | Hardware pressure | Candidate response |
|:--|:--|:--|
| low arithmetic per butterfly | communication and barriers dominate | maximize warp-local exchange; avoid explicit stage queues unless reuse amortizes them |
| expensive or wide arithmetic | execution and registers dominate | preserve independent warps; reduce spatial replication if occupancy collapses |
| asymmetric update | one output may bypass arithmetic | use lane-selective processing and measure shuffle overhead against shared exchange |
| stage coefficients | coefficient bandwidth and cache capacity | fuse or generate coefficients; choose the warp/shared boundary with the working set |
| changing partner distance | transport scope grows by stage | map lane-local, warp-local, CTA-local, then global boundaries explicitly |

This is the reusable method: the graph defines the four unfolding factors;
operator traits define per-cell resource demand; hardware calibration defines
legal transport and residency; measured candidates select the realization.

## 6. Mapping the architectural demands to a GPU

| Architectural demand | GPU mechanisms | Counters or constraints |
|:--|:--|:--|
| `Us*Ud` butterfly capacity | integer execution lanes occupied by resident warps | integer instructions, issue rate, registers, active warps |
| stage-space pipeline | stage-specialized warp groups plus shared-memory queues/barriers | barrier stalls, eligible warps, shared-memory traffic |
| `2Ud` boundary bandwidth | coalesced global/shared loads and stores | sectors/request, bank conflicts, DRAM/L1/L2 throughput |
| stage-time feedback | shared-memory feedback within a CTA; global/L2 feedback across kernels | global passes, long scoreboard, synchronization |
| data-time state | shared memory, registers, or global working set | capacity-derived occupancy, cache hit rate |
| twiddle supply | broadcast/cache/shared staging/generation | root working set, L2 hit, long scoreboard, integer cost |

A block shape is not one of the four architecture factors. It is a mechanism
for realizing `Ud`, `Td`, feedback, and layout on a particular GPU.

Resident CUDA threads are execution contexts, not physical `Ud`. A rigorous GPU
mapping distinguishes:

```text
Ud_physical = calibrated equivalent-butterfly issue capacity of the SMs
Ud_logical  = butterflies exposed by threads/warps/CTAs
Td_sched    = scheduler waves that time-multiplex Ud_logical on Ud_physical
```

Likewise, unrolled stage instructions or CTAs at different program counters do
not by themselves establish `Us`. `Us>1` requires intentionally distinct
resident stage roles and an observable inter-stage transport. The CUDA resource
model below is therefore a feasibility model beneath the unfolding model, not a
replacement for it.

## 7. GPU resource feasibility model

For a candidate with `T` threads, `r` 32-bit registers per thread, `S` shared
bytes per CTA, and hardware vector `H`, estimate resident CTAs per SM as:

```text
Breg  = floor(H.registers_per_sm / allocated_registers(T, r))
Bsmem = floor(H.shared_bytes_per_sm / allocated_shared_bytes(S))
Bthr  = floor(H.threads_per_sm / T)
Bwarp = floor(H.warps_per_sm / ceil(T / H.warp_size))
Bres  = min(Breg, Bsmem, Bthr, Bwarp, H.max_ctas_per_sm)
occupancy_est = Bres * ceil(T / warp_size) / H.warps_per_sm
waves = total_ctas / (H.sm_count * Bres)
```

Candidates are rejected when they exceed per-CTA limits, produce too few waves
for the target batch, or leave no occupancy to hide a long-latency root load.
Maximum occupancy is not the objective: a lower-occupancy candidate can win if
it removes a global pass or substantially improves locality.

The memory footprint model must separate coefficient and root traffic:

```text
coefficient_bytes_min = 2 * N * word_bytes * number_of_global_passes
root_working_set(j)   = unique_roots(j) * bytes_per_root_representation
```

For a standard radix-2 table, a stage group ending after stage `e` needs at
most `2^(e-1)` roots. This identifies which stage group crosses L1, L2, or HBM
capacity. Expanded per-row/per-`k2` roots must use their actual expanded size.

## 8. What the current V100 kernels do and do not prove

For `logN=20`, the current V100 candidate uses a 512-coefficient tile, 256
threads, two coefficients per thread, and temporal stage groups `5+6+9`:

| Kernel | Stages | CUDA block | Grid per transform | Root working set | Physical interpretation |
|:--|:--|:--|:--|:--|:--|
| 0 | `[0,5)` | `(16,16)`, 8 warps | `(2048,1)` | 16 root/Shoup pairs, 256 B | broad CTA parallelism; early roots are cache-resident |
| 1 | `[5,11)` | `(8,32)`, 8 warps | `(64,32)` | 1,024 pairs, 16 KiB | more independent groups in `y`; still a small root set |
| 2 | `[11,20)` | `(256,1)`, 8 warps | `(1,2048)` | 524,288 pairs, 8 MiB | fully contiguous lanes; L2/HBM behavior dominates roots |

At batch 4 every kernel launches 8,192 CTAs. The V100 has 80 SMs, so all three
kernels expose the same spatial work and avoid a short final tail. Each CTA
uses 4 KiB shared memory. The compiled kernels use 24, 24, and 32 registers per
thread. With 65,536 registers, 96 KiB shared memory, 2,048 threads, and 64
warps per V100 SM, the 2,048-thread limit permits eight such CTAs per SM; neither
registers nor shared memory lowers that ceiling.

This table explains the controls without making them universal. A GPU with a
larger L2 may profit from a larger final stage group. A GPU with weaker 64-bit
integer throughput may prefer Shoup over Barrett even when Barrett halves table
bytes. A GPU with more shared memory may increase `tile_points`; one with fewer
CTA slots may prefer a different block shape or more work per CTA.

These kernels explicitly implement data-space unfolding through threads/CTAs,
data-time folding through waves and lane reuse, and stage-time folding through
the sequential stage loops. They do **not** instantiate an explicit
stage-specialized GPU pipeline. Different resident warps may happen to occupy
different loop positions, but that scheduler effect is not a controlled `Us`
and must not be counted as proof of stage-space unfolding.

Consequently, compact-stage is a calibrated GPU baseline for processing-unit,
root, residency, and layout costs. A full GPU realization of the APPT/Hermes
point requires a second kernel family with `Us>1`: warp groups specialized to
different stage positions, communicating tiles through shared-memory queues or
another explicitly measured inter-stage mechanism. The comparison between
`Us=1` and `Us>1` is the central architecture experiment.

### Controlled GPU experiment at fixed logical budget

The next kernel family should hold `C=Us*Ud=256` logical butterfly contexts and
the CTA size constant while changing their assignment:

| Candidate | Stage roles `Us` | Butterfly lanes per role `Ud` | Total contexts | Intended realization |
|:--|--:|--:|--:|:--|
| D256-S1 | 1 | 256 | 256 | current temporal-stage baseline |
| D128-S2 | 2 | 128 | 256 | two stage-specialized warp groups |
| D64-S4 | 4 | 64 | 256 | four-stage shared-memory pipeline |
| D32-S8 | 8 | 32 | 256 | one warp per stage position |

Each candidate must use the same modular unit, coefficient/root format, state
scope, and output layout. Stage roles exchange tiles through explicit
shared-memory queues with measured synchronization; they must not be inferred
from program-counter overlap. For `K` not divisible by `Us`, report
`eta_stage` and the inactive/swap work. This experiment isolates the
architecture factorization under the GPU's fixed physical execution resources.

The first V100 microarchitecture prototype uses eight warps, two 256-point
tokens per independent pipeline, 32 KiB shared state, and explicit
producer/consumer flags between stage-role warps. Five steady-state process
trials (100 warmups and 100 repeats each) give:

| `Us` | Aggregate logical `Ud` contexts | Median Gbutterfly/s | Relative to `Us=1` |
|--:|--:|--:|--:|
| 1 | 256 | 23.800 | 1.000x |
| 2 | 128 | 34.336 | 1.443x |
| 4 | 64 | 28.055 | 1.179x |
| 8 | 32 | 19.821 | 0.833x |

All four execute the same eight dependent modular-butterfly stages and produce
identical outputs. Compute Sanitizer reports zero errors. `Us=2` is the measured
optimum for this queue implementation: moving from 1 to 2 stage roles removes
enough global feedback traffic to dominate the new handoff cost; deeper
pipelines lose to shared-memory state, atomic polling, and reduced independent
pipeline count. This is evidence for a hybrid point, but it is a local 256-point
pipeline microarchitecture rather than a complete large-`N` NTT result.

### Correct NTT256 architecture experiment

The `stage-pipeline` backend turns the synthetic pipeline into a complete
forward `N=256` NTT. It performs the initial bit-reversed read, uses the real
twiddle and Shoup pair for every stage, and returns natural-order output. All
four `Us` values match the CPU reference coefficient-for-coefficient for a
17-transform non-multiple batch; the full cuNTT regression suite also passes.

The experiment fixes the CTA at eight warps, uses two tokens per independent
pipeline, and changes only the role allocation. Thus `Us` stage-role warps and
`8/Us` independent data pipelines coexist in one CTA, while `Ts=8/Us` kernel
groups cover the eight stages. On a V100, five trials at batch 16,384 with 20
warmups and 100 timed executions give:

| Handoff | `Us` | data pipelines/CTA | `Ts` launches | Median ms | Gbutterfly/s | Speedup vs same-handoff `Us=1` |
|:--|--:|--:|--:|--:|--:|--:|
| atomic polling | 1 | 8 | 8 | 0.786 | 21.35 | 1.000x |
| atomic polling | 2 | 4 | 4 | 0.486 | 34.52 | 1.617x |
| atomic polling | 4 | 2 | 2 | 0.613 | 27.38 | 1.283x |
| atomic polling | 8 | 1 | 1 | 0.865 | 19.40 | 0.909x |
| named barrier | 1 | 8 | 8 | 0.778 | 21.58 | 1.000x |
| named barrier | 2 | 4 | 4 | 0.426 | 39.40 | 1.828x |
| named barrier | 4 | 2 | 2 | 0.403 | 41.61 | 1.930x |
| named barrier | 8 | 1 | 1 | 0.429 | 39.07 | 1.813x |
| tile256 reference | - | - | 1 | 0.277 | 60.48 | - |

This is the first end-to-end evidence for the two-dimensional argument. More
stage-space unfolding reduces temporal stage feedback and launch count, but it
also reduces independent data pipelines and increases shared handoff pressure.
The optimum is therefore interior rather than either endpoint. More
importantly, it moves from `Us=2` with atomic polling to `Us=4` with named
barriers. The hardware mapping is inseparable from the cost of the available
inter-stage service.

This comparison holds both paths at 256 threads and 32,832 bytes of total
shared memory per CTA. The named-barrier launch includes 64 bytes of dynamic
shared padding because otherwise removing the atomic flags crosses the V100's
32-KiB boundary and changes residency from two to three CTAs per SM. Compiled
atomic kernels use 32 registers/thread; named-barrier kernels use 37 for
`Us>1`. Shared memory remains the occupancy limiter in both cases.

The absolute comparison is equally important. The best explicit pipeline
(`named-barrier, Us=4`) is still 1.454x slower than the conventional
single-kernel `tile256` implementation. The experiment establishes the
architecture tradeoff and hardware-dependent optimum, but it does not yet
establish that explicit stage-space execution wins on V100. Register/shuffle
handoff for early stages and buffer-layout changes are the next mechanisms to
compare before extending this hierarchy to large `N`.

Compute Sanitizer memcheck reports zero errors for both handoffs. Racecheck
reports zero hazards for the named-barrier path, while it reports shared-memory
hazards for the atomic polling path. Whether this is an analyzer limitation or
an insufficient synchronization contract, that path cannot support the
correctness claim. Atomic polling is therefore retained only as an empirical
service-cost baseline; named barrier is the default and the only current
stage-pipeline result used as a correctness claim.

Raw and reduced data are stored in
`results/stage_pipeline_ntt256_v100_raw.csv` and
`results/stage_pipeline_ntt256_v100_summary.csv`. Reproduce them with
`scripts/benchmark_stage_pipeline.sh` followed by
`scripts/summarize_stage_pipeline.py`.

The analytical rank based on independent copy/shared/barrier peaks does not
predict the exact optimum, because an atomic producer/consumer queue and a
64-thread named barrier are different composed services. The mapping model must
therefore consume a handoff-specific service curve, not a single ideal shared
link or full-CTA barrier peak.

## 9. Counter-driven bottleneck classification

Use CUDA-event time for ranking and NCU for explanation. NCU replay time is not
the cross-implementation performance number. Classify each kernel using a
counter vector rather than a single occupancy metric:

| Signature | Evidence | Mapping response |
|:--|:--|:--|
| HBM/L2 latency | high `long_scoreboard`, low L2 hit, high DRAM bytes | compact/generate roots, change stage boundary, increase independent warps |
| bandwidth | DRAM throughput near peak with high bytes | remove a pass, fuse operations, reduce root/coefficient representation |
| synchronization | high barrier stall, sufficient active warps | reduce fused stages, use warp-local final levels, change block shape |
| register limited | register occupancy limit below thread/warp limit | shorten live ranges, change reduction/radix, reduce per-thread reuse |
| shared-memory limited | shared occupancy limit is minimum | smaller tile/rows, compact shared layout |
| instruction limited | high integer instructions with modest memory stalls | choose a cheaper equivalent butterfly/reduction |
| insufficient parallelism | few waves or low active warps without resource limit | increase batch/grid decomposition or reduce work per CTA |
| layout limited | natural-order scatter adds large time/write inefficiency | retain native layout or fuse permutation with the consumer |

For each kernel `j`, retain a calibrated lower-bound feature model:

```text
Tj >= max(integer_instructions / effective_integer_rate,
          dram_bytes / effective_dram_bandwidth,
          l2_bytes / effective_l2_bandwidth,
          barrier_count * effective_barrier_cost)
```

This is a pruning and interpretation model, not a cycle-accurate predictor.
Final selection uses measured steady-state CUDA-event time with correctness and
output semantics held constant.

The targeted V100 length/batch experiment validates that the categories above
must remain separate:

| Case | Coverage/residency | Useful-work evidence | Diagnosis |
|:--|:--|:--|:--|
| FFT `logN=14`, batch 16 | direct: 0.20 waves/SM, one 1024-thread CTA/SM | strong local cuFFTDx core | `Ub` grid underfill |
| FFT `logN=18`, batch 64 | online: 60.6% active warps | 1.60x cuFFT warp instructions, 9.50x bank conflicts | exchange/synchronization ceiling |
| FFT `logN=20`, batch 16 | equal aggregate waves to cuFFT | 2.00x warp instructions; first pass is 61.8% of time | prefix/core/layout imbalance |
| FWHT `logN=15`, batch 4/16 | warp-register: 0.05/0.20 waves/SM | at batch 16, 0.07x online warp instructions | `Ud` decomposition to `Td` residence crossover |

Thus `active_warps` is not an optimization objective by itself. Grid coverage,
residency, and useful work per active warp are independent selector features.
See [V100 Counter Attribution](v100_ncu_attribution.md) for the complete data.

## 10. Portable architecture-selection procedure

1. Enumerate ordered graph factorizations and per-dimension `(Us, Ud)` points;
   independently enumerate batch `(Ub,Tb)`. Derive `Ts`, `Td`, utilization,
   arithmetic cells, boundary bandwidth, inter-stage bandwidth, feedback
   bandwidth, and state capacity.
2. Select candidate residency levels and reject points whose state or feedback
   demand cannot be supported.
3. Record the hardware vector: SM count, warp size, scheduler/warp capacity,
   registers and shared memory per SM/CTA, L2 size, memory bandwidth, and
   coefficient-width integer throughput.
4. Calibrate small device microbenchmarks for 64/32-bit modular multiply,
   barriers, shared-memory access, coalesced bandwidth, and permutation stores.
5. Factor logical `Us`, `Ud`, and `Ub` over lane, thread, warp, CTA, cluster,
   grid, device, and node levels; map survivors to legal pipelines, tile sizes,
   block shapes, processing units, root forms, and layouts.
6. Apply register, shared-memory, working-set, minimum-wave, and correctness
   constraints.
7. Benchmark surviving candidates with identical modulus, length, batch,
   warmup, input/output residency, and output order.
8. Profile the Pareto candidates with NCU and classify each stage group using
   the signatures above.
9. Refine the search around the measured bottleneck and retain the complete
   configuration plus hardware vector as the result, not only the winning time.

The feasibility model is executable. For the current V100 mapping:

```bash
python3 scripts/analyze_hardware_mapping.py \
  --hardware configs/hardware/v100_sxm2_16gb.json \
  --mapping configs/mappings/compact_logN20.json \
  --batch 4 \
  --output results/v100_compact_mapping_model.csv
```

The JSON files deliberately separate hardware facts from schedule choices.
Register and shared-memory allocation granularities are architecture-specific
inputs and should be checked against the vendor occupancy model for each GPU.

The architecture-level demand model is separate:

```bash
python3 scripts/analyze_unfolding.py \
  --logN 16 --stage-space 8 --data-space 16 --word-bytes 8
```

This reproduces the APPT/Hermes U280 point: `Ts=2`, `Td=2048`, 128 spatial
butterfly cells, 32 coefficient words/cycle at the pipeline boundary, and an
upper bound of 128 twiddle words/cycle across the stage cells. GPU values for
`Ud_physical` must be calibrated rather than copied from thread counts.

Calibrate and rank a target GPU with:

```bash
./scripts/benchmark_hardware_capabilities.sh
python3 scripts/summarize_hardware_capabilities.py \
  results/hardware_capabilities_raw.csv \
  --output results/hardware_capabilities_v100.json
python3 scripts/rank_unfolding.py \
  --capabilities results/hardware_capabilities_v100.json \
  --logN 16 --spatial-budget 128 --word-bytes 8 \
  --stage-space 1 2 4 8 16 \
  --output results/unfolding_rank_v100_logN16_C128.csv
```

The default ranker uses a conservative full-CTA barrier for every inter-stage
handoff. `--barrier-model none` gives the communication/compute bound for an
idealized asynchronous or warp-local handoff. The gap between the two is the
value of a better synchronization mechanism and must be measured by the
explicit stage-pipeline prototype.

The selector should optimize a workload objective, for example latency at
batch 1, steady-state throughput at batch `B`, or a weighted length/precision
distribution. There is no hardware-independent single winner.

## 11. Paper structure and required evidence

The paper can be organized around three claims:

1. The `2 x 2` unfolding model separates the hardware demands of stage-space,
   stage-time, data-space, and data-time choices independently of butterfly
   implementation.
2. Residency and feedback are the coupling mechanisms that explain when a
   four-way hybrid point is beneficial on an FPGA or GPU.
3. A hardware-constrained mapping method selects different `(Us, Ud)` points
   and physical realizations across devices, and counters explain the changes.

Required experiments are: an explicit `Us=1` versus `Us>1` ablation, multiple
`Ud` points at equal `Us*Ud` where possible, at least two materially different
GPU generations, multiple lengths and coefficient widths, batch sensitivity,
native and natural output semantics, and processing-unit/root/layout ablations.
Report derived bandwidth/capacity demand, the selected mapping, and measured
hardware evidence. The existing compact-stage result alone is insufficient to
support the full architecture claim.
