# Physical-Chain Architecture Cost Model

The primary v0.8 model is a parameterized architecture equation. It does not
learn timing from candidate names or add batch-polynomial features. The older
ridge/histogram model is retained only as a historical search baseline. The
three-level measurements below identify one physical realization; they do not
fix the architecture to three segments.

## Runtime Butterfly Model

`src/butterfly_performance_model.cpp` is the first deployment of the same
architecture-first idea for the operator-independent butterfly API. For a
candidate `q`, the runtime score is decomposed rather than fit to an
implementation label:

```text
T(q) = T_startup(q) + T_compute(q) + T_memory(q) + T_occupancy(q)
```

`T_compute` scales with `B * 2^K * K/2`, an operator arithmetic factor, the
precision factor, and the selected radix service. `T_memory` accounts for the
two-buffer global traffic. Resident candidates add a shared-state ratio and
batch-amortization term; mature hierarchical candidates add the materialized
stage/startup term. Candidates are rejected before scoring when their shared
state exceeds the device envelope or when the physical point is not generated.

The selector currently compares two generated physical templates: a resident
hybrid-dataflow radix-4 point and the mature hierarchical radix-4 point. It
uses the active V100's SM count and 96 KiB opt-in shared-memory envelope, then
stores the predicted latency, resource terms, and rationale in `SelectionInfo`.
The equation's service coefficients are explicit (`compute_service_per_s`,
`memory_bandwidth_bytes_per_s`, startup, and occupancy terms) in
`ButterflyModelCoefficients`; the defaults are the V100 values and can be
replaced by a per-device calibration record.
The dispatch domain is intentionally limited to measured `logN=10/12/14`
cells. This is a model-guided shortlist, not a claim that unmeasured lengths
or arbitrary physical-core choices are already safe for automatic dispatch.

## Logical M, Physical G, And The Core Template

The model separates three choices that were previously conflated:

1. logical stage partition `L=(l0,...,lM-1)`, which describes the APPT graph;
2. boundary fusion vector `F`, which lowers adjacent logical groups into a
   physical partition `P=(p0,...,pG-1)`;
3. the physical worker template used by each resident execution group.

Thus `M` and `G` are independent parameters. For example, logical
`5+5+5+5` with fusion decisions `1,0,1` lowers to physical `10+10`. It has the
same physical work descriptor as direct logical `10+10`, while retaining its
different logical provenance. Only the `G-1` materialized boundaries incur
global handoff traffic.

The physical template is also explicit. The specialized measured three-level
kernel uses one worker containing 32 rows and padded row-tile shared memory.
The arbitrary-`G` kernel uses eight warp workers per 256-thread CTA and one
private `2^k` tile per warp. These implementations have different work and
shared-memory equations even when their stage partitions match, so services
from one realization are never silently applied to the other.

## What Is Parameterized

For a transform with `K=log2(N)`, physical partition `(k0,...,kG-1)`, batch
`B`, `R` rows per packet, `W` warp workers per CTA, packet grouping
`(D0,...,DG-1)`, and group weights
`(w0,...,wG-1)`, the model first replays the CUDA launcher's integer CTA
allocator. If `C` cooperative CTAs are resident, group `i` receives `Ci` CTAs.
The physical work is:

```text
packets_i      = B * ceil(2^(K-ki) / R)
packet_groups_i = ceil(packets_i / Di)
workers_i      = Ci * W
iterations_i   = ceil(packet_groups_i / workers_i)
elements_i     = iterations_i * Di * R * 2^ki
butterflies_i  = elements_i * ki / 2
radix4_items_i = Di * R * 2^ki / 4
lane_use_i     = min(1, radix4_items_i / 32)
odd_residue_i  = elements_i / 2, when ki is odd; otherwise 0
boundary_mul_i = elements_i, for non-final groups; 0 for final
```

`C` is not a constant. It is bounded by the target CTA count and the shared
memory footprint:

```text
row-tile:  shared_bytes = R * max(Di * 2^ki + padding) * word_bytes
warp-tile: shared_bytes = W * max(Di * 2^ki + padding) * word_bytes
resident_ctas_per_sm = min(target_ctas_per_sm,
                           floor(shared_bytes_per_sm / shared_bytes))
C = SM_count * resident_ctas_per_sm
```

`Di` is the physical realization of data-space unfolding for the generic warp
core. It groups independent packets inside one warp instruction stream; it
does not merge readiness tokens or change NTT dependencies. The generated
values are `D=1/2/4/8`. For radix-4, warp-fill selects `D=8` at `k=4`, `D=2`
at `k=6`, and `D=1` at `k>=8`.

The aggregate-calibration fallback role service and kernel time are:

```text
role_i = alpha * butterflies_i
       + beta  * odd_residue_i
       + gamma * boundary_mul_i

Tideal = launch + max(max_i(role_i), 2 * G * B * N * word_bytes / DRAM_BW)
```

`alpha`, `beta`, `gamma`, and `launch` are the only aggregate calibrated constants.
They respectively describe the selected radix-4 codelet, its odd-stage
residue, the non-final twiddle/modular-multiply boundary, and launch service.
All other quantities come from the graph, generated mapping, or GPU profile.

The descriptor also derives prefix/suffix topology, static global load/store
sectors, readiness polls/publications, barriers, and shared memory. Sector
counts are currently diagnostic rather than given a fitted coefficient: a separate stride
microbenchmark or matched NCU measurement must identify that service before it
is added to the time equation.

## Identified Physical Service Table

The aggregate equation is no longer the only service source. A separate
diagnostic kernel selected by `PlanConfig::profile_pipeline_roles` (CLI
`--profile-pipeline-roles`) accumulates, for each persistent role:

- readiness polling time;
- resident radix-4 codelet time;
- boundary input, twiddle/output, and publication time;
- completed dependency-closed tasks.

The normal and diagnostic kernels are separate template instances, so timing
instrumentation is absent from performance runs. The reducer divides measured
work by graph-derived butterflies and values. It indexes physical work service
by `(realization_id, word_bits, resident_ctas_per_sm, startup/steady,
stage_log, prefix_log, suffix_log, upstream_stage_log, role_class, role_ctas,
packet_group)`.
The upstream stage depth is required because it determines producer packet
granularity and therefore downstream dependency fan-in. Topology replaces
a fixed three-role identity, so an equivalent group can transfer across `M/G`
only when its physical context matches. `role_ctas` is required: increasing
one role's concurrent CTAs can reduce its nominal iterations while worsening
its strided boundary service.
Interpolation is permitted only between measured CTA counts with every other
dimension equal. Missing or out-of-range points remain uncovered.

Readiness is stored separately with the complete partition, weights, and role
allocation. It is an induced queueing observation, not a transferable codelet
constant. The physical role equation is:

```text
role_i = iterations_i * (
    wait_service[key_i]
  + butterflies_per_task_i * codelet_service[key_i]
  + values_per_task_i      * boundary_service[key_i])

Tdiagnostic = launch + max_i(role_i)
```

The generated V100 uint64 table is a `three-level-row-tile` service table:
`config/v100_three_level_physical_services_u64.json`. Reproduce it with:

```bash
./scripts/benchmark_three_level_service_terms.sh

OUTPUT_DIR="$PWD/results/three_level_service_terms_u64/allocation_probe_p668_b48_v2" \
  BATCHES=48 CASES="6,6,8:10,6,4 6,6,8:8,6,6" \
  ./scripts/benchmark_three_level_service_terms.sh

python3 scripts/analyze_three_level_service_terms.py \
  results/three_level_service_terms_u64/logs \
  results/three_level_service_terms_u64/allocation_probe_p668_b48_v2/logs \
  --output-dir results/three_level_service_terms_u64/combined \
  --service-table-output config/v100_three_level_physical_services_u64.json
```

The arbitrary-`G` generic-warp realization has its own table and collector:

```bash
WORD_BITS=64 ./scripts/profile_physical_chain_service_terms.sh
WORD_BITS=32 ./scripts/profile_physical_chain_service_terms.sh

python3 scripts/analyze_physical_chain_service_terms.py \
  results/physical_chain_service_terms_u64/logs \
  --output-dir results/physical_chain_service_terms_u64 \
  --service-table-output config/v100_ntt_generic_warp_services_u64.json
```

This collector profiles scalar and warp-filled `10+10`, `6+6+8`, `8+6+6`,
`6+8+6`, and `4+4+4+8` at batch 1/4/16/48 for both numeric widths.
Each width-specific table contains 38 work-service entries and 88 complete
readiness observations.
The timer is intrusive, so uninstrumented event timings remain the performance
authority. The matched uninstrumented batch-48 core comparison is:

| bits | G | partition | packet groups | scalar ms | packed ms | packed/scalar |
|---:|---:|:---:|:---:|---:|---:|---:|
| 32 | 3 | `6+6+8` | `2+2+1` | 48.9856 | 35.6399 | 1.374x |
| 32 | 4 | `4+4+4+8` | `8+8+8+1` | 50.5845 | 18.4835 | 2.737x |
| 64 | 3 | `6+6+8` | `2+2+1` | 59.4773 | 47.8010 | 1.244x |
| 64 | 4 | `4+4+4+8` | `8+8+8+1` | 61.9292 | 30.7539 | 2.014x |

The new core removes the first-order SIMD underfill: stage-6 moves from 50% to
100% lane use and stage-4 from 12.5% to 100%. The remaining cross-`G` gap is
not arithmetic underfill. At uint64 batch 48, packed G3/G4 still reach final
role wait fractions of 0.753/0.779 and materialize two/three boundaries, while
G2 waits 0.288 and materializes one. Thus packet packing is necessary but not
sufficient: readiness fan-in, role allocation, and extra boundary traffic are
now the dominant modeled dimensions.

## Minimal Calibration

The V100 uint64 profile uses a small factorial design, not a full search:

- batches 1 and 48 separate startup/tail behavior from steady service;
- `7+6+7 / 7:6:7` exposes the odd radix-2 residue and two-CTA residency;
- `8+6+6 / 10:5:5`, `8:6:6`, and `8:8:4` move pressure among producer,
  middle, and final roles.

That is four configurations at two batches, or eight aggregate timings for
four service constants. Generate and execute exactly those cases with:

```bash
python3 scripts/generate_three_level_architecture_calibration.py \
  --profile config/v100_three_level_architecture_model.json \
  --output results/three_level_architecture_model_u64/calibration_manifest.json \
  --word-bits 64 --logN 20

python3 scripts/run_three_level_model_followup.py \
  results/three_level_architecture_model_u64/calibration_manifest.json \
  --output results/three_level_architecture_model_u64/calibration_raw.csv \
  --trials 3 --warmup 10 --repeat 50 --resume
```

The existing dense scan contains these points, so the current profile is
reproduced without collecting new data:

```bash
python3 scripts/calibrate_three_level_architecture_model.py \
  results/three_level_model_u64/followup_raw.csv \
  --profile config/v100_three_level_architecture_model.json \
  --output-dir results/three_level_architecture_model_u64

python3 scripts/evaluate_three_level_architecture_partitions.py \
  results/three_level_partitions_u64/summary.csv \
  --profile results/three_level_architecture_model_u64/calibrated_profile.json \
  --output-dir results/three_level_architecture_model_u64
```

After calibration, rank the complete supported partition/weight space without
measuring every point:

```bash
python3 scripts/rank_three_level_architecture_candidates.py \
  --profile results/three_level_architecture_model_u64/calibrated_profile.json \
  --word-bits 64 --logN 20 --batch 16 \
  --weight-total 20 --weight-min 4 --top-k 20 \
  --output results/three_level_architecture_model_u64/ranked_b16.csv
```

To enumerate the architecture dimension itself rather than only weights within
the three-level calibration slice, run:

```bash
python3 scripts/rank_physical_chain_architecture_candidates.py \
  --profile config/v100_ntt_physical_chain_model.json \
  --physical-service-table config/v100_ntt_generic_warp_services_u64.json \
  --word-bits 64 --logN 20 --batch 48 \
  --physical-groups 2,3,4 --weight-policy stage-proportional \
  --packet-policy warp-fill \
  --output results/physical_chain_architecture_packed/u64_b48.csv \
  --group-summary-output \
    results/physical_chain_architecture_packed/u64_b48_groups.csv
```

The summary always retains a best candidate for every requested `G`, even when
the global Top-K is dominated by another group count. It prefers fully
identified candidates over lower but uncovered aggregate estimates. The
current packed table admits 1/3/1 measured candidates for G=2/3/4; all
other partitions remain `measurement_required=1` and form the next measurement
queue rather than an automatic runtime policy.

Weights are integer compositions, not a hard-coded candidate list. The output
includes the predicted critical role and all residency/iteration diagnostics.
Candidates in an unobserved readiness topology or stage order remain ranked
with a model-domain reason. A candidate becomes runtime-admissible only when
every physical role has matching active and readiness service; coverage is
never inferred from its G value.

Equivalent `partition` and `role` source labels are collapsed when they lower
to the same `(partition, weights, numeric contract)`. Treating those labels as
different candidates previously corrupted top-k evaluation.

## Current Validation

Only batches 1 and 48 participate in calibration. Across the other eleven
batches, the model obtains:

| aggregate calibration points | validation batches | top-1 | top-4 | geometric regret | p95 regret | worst regret |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 11 | 63.6% | 100% | 1.0047x | 1.0182x | 1.0218x |

These eleven batches validate CTA-weight selection inside the calibrated
`8+6+6` topology; they are not a claim of arbitrary-partition transfer. On the
separate six-partition, stage-proportional scan, the equation selects the
measured winner at three of four batches, with 1.0397x geometric and 1.1687x
worst regret. The batch-1 miss and the `6+8+6` residual show that local
stage-log/role service is still under-parameterized.

The model is therefore adequate to prioritize a reduced search, not to
dispatch top-1 at runtime. In particular, the ideal `max(role_i)` equation
underestimates the large-batch `7+6+7` point. Its two resident CTAs share an SM
while downstream roles poll readiness, so waiting CTAs interfere with producer
service. The model must abstain for this unmodeled regime.

An immediate unmeasured-weight spot check supports that boundary. At uint64
`logN=20,batch=16`, the model ranked `8+6+6 / 9:7:4` ahead of the measured
`8:8:4` incumbent. Three rotated-order runs measured 8.1564 ms versus 8.0646
ms, or 1.0114x regret. Their trial ranges overlap, so this is not a stable
crossover, but it directly rejects automatic top-1 admission while confirming
that the analytic shortlist reached the near-optimal region.

The role trace experiment now identifies stage-6/7/8 codelet and boundary
service directly. At uint64 batch 48, the one-CTA/SM `8+6+6 / 8:6:6` point is
predicted at 23.851 ms and measured by the normal kernel at 23.908 ms. The
model also ranks `6+6+8 / 8:6:6` first among the five confirmed allocation
controls: 22.083 ms predicted versus 22.926 ms measured.

The remaining mechanism is role-concurrency/readiness interaction. Changing
`6+6+8` from `8:6:6` to `10:6:4` raises producer CTAs from 31 to 39 and raises
producer boundary service from 11.97 to 17.71 globaltimer ticks/value. The
normal kernel slows from 22.93 to 29.22 ms even though consumer wait fractions
fall. Adding `role_ctas` to the work-service table corrects the ordering and
predicts 25.23 ms, but the remaining 13.6% underestimate is an explicit
readiness/queueing residual. It is retained as a measurement-required region;
no partition identity coefficient is introduced.

## Historical Regression Baseline

`scripts/fit_three_level_performance_model.py` and
`scripts/select_three_level_candidates.py` remain reproducibility artifacts.
Their dense nonlinear regression reached 1.018x interpolation-only geometric
regret and 100% top-5 recall, but it required measurements across thirteen
batches and did not reveal shared-memory residency or role-service causes.
It is therefore not the generation policy and must not be presented as the
research model.
