# Roadmap

This roadmap separates implementation goals from evidence required for the
research claim. Ordering may change after profiling or access to new hardware.

## Current Release Freeze: v0.8

The v0.8 nested-physical-space architecture is temporarily frozen as the
single-V100 research baseline. Its confirmed selection closure, public API,
processing-unit lowering contract, cross-operator manifests, and reproducible
reports are maintained as release evidence. New work should extend numeric or
batch coverage without silently changing the frozen comparison protocol. Any
new mapping must state whether it is a compatible v0.6 candidate, a native
resident realization, or an experimental processing-unit lowering.

## v0.8.0 Status: Nested Physical Space

Status: native implementation and comprehensive V100 envelope complete.
Logical four-axis mapping, execution grouping, CUDA ownership, and
physical core are separate descriptors. The v0.6 Hybrid2D radix-4 family is a
strict GridTiled compatibility subset and the generated cooperative 10+10 core
is available as a native transform-ready resident pipeline with independent
producer/consumer codelets. Plan/CSV wave diagnostics, generator v3, the
384-sample three-generation envelope, regression gate, NCU command, docs, and
tests are checked in. The subsequent five-trial matrix covers 60 semantic
points and 1,380 candidate executions. DD GridTiled wins every uint32
`logN=20` batch from 1 through 160; the v0.6-compatible GridTiled mapping wins
all other covered width/length regimes. This two-branch rule matches the
measured oracle at 60/60 points and is generated into the runtime selector.

Matched NCU attribution is complete: CooperativePhased is arithmetic- and
sector-equivalent to GridTiled, but exposes fewer active warps and lower IPC.
The dual 128-thread subgraph control validates inter-subgraph latency hiding
with a 1.466x gain at equal CTA residency, but its reused shared-memory codelet
still trails the full-width phased and Grid controls. The next gate is
held-out-batch/modulus validation of the deployed selector and a uint64 DD
codelet study. CUDA Graph replay remains a separate repeated-call optimization
and must not merge phase-specialized kernels. Generation of partitions whose
dependency-closed publication unit is smaller than a whole transform follows.
Cross-operator ownership lowering follows the NTT contract; cross-GPU
calibration remains deferred.

The explicit `producer_ahead` ResidentQueue startup gate is now available for
controlled experiments. Its V100 screen degrades as the gate grows, confirming
that delaying resident consumers is insufficient; the next implementation gate
is per-dependency-closure publication with online reorder and work-conserving
consumer scheduling.

An experimental bounded FFT adapter now permits `HybridDataflow` to own a
single block-resident FP32 `logN=3..10` cuFFTDx block tile (`stage_space` and
`flow_tile_log_n` equal to the local transform, one role, `data_time=1`, linear
layout, in-place state). This
proves the processing-unit/ownership boundary without claiming multi-role
overlap. The next implementation gate is embedding the cuFFTDx execute into
multiple packet roles so block-resident subgraphs can actually pipeline; see
[`Architecture Guardrails`](docs/architecture_guardrails.md).

The first role-service screen is reproducible through
`scripts/benchmark_three_level_role_weights.sh`. Its V100 data shows a
batch-dependent crossover (`6:6:8` for small batches, `4:4:12` for larger
batches). The matched uint64 confirmation selects `4:4:12` for all tested
batches, so the eventual policy must key on numeric width as well as batch;
repeated trials across nearby partitions and physical cores are still required
before feeding this policy into automatic selection.

The first matched uint64 partition screen now covers all six `6/7/8` stage
orders at batch 1/4/16/64. `8+6+6` wins every batch with stage-proportional
weights, while `6+6+8` is the nearest alternative. This establishes the
search order for the next gate: choose the stage partition first, then tune
role weights and physical-core parameters within that partition. The resident
launcher also opts into V100's larger dynamic shared-memory budget for valid
64-bit 8-stage points.

The first within-partition uint64 role screen for `8+6+6` is also complete:
`8:6:6` wins at batch 1/64 and `10:5:5` wins at batch 4/16. This confirms that
partition and role service must be calibrated jointly; neither stage order nor
role weights can be treated as a standalone global default.

The primary model is now a generic physical-chain architecture equation in
`scripts/three_level_architecture_model.py` (legacy filename retained). It
separates logical segment count `M`, fused physical group count `G`, worker
template, CTA allocation, and prefix/suffix topology. Existing three-level
data identifies one `three-level-row-tile` realization; it is not a fixed
global segmentation. `scripts/rank_physical_chain_architecture_candidates.py`
enumerates arbitrary `G` and emits a per-`G` summary so no group count vanishes
from a global Top-K. See [`docs/performance_model.md`](docs/performance_model.md).

The next model layer is also implemented for the V100 uint64 three-level core.
An isolated diagnostic template measures wait, codelet, and boundary service
per role. The generated service table is indexed by realization, residency,
stage depth, prefix/suffix topology, load regime, and role CTA concurrency; interpolation is bounded by actual
measurements. It restores the measured batch-48 ordering between `6+6+8` and
`8+6+6`. Readiness queueing outside those bounds remains an explicit
measurement-required item rather than a fitted partition feature.

The first generic-warp evidence gate across `G=2/3/4` is complete. A generated
warp-packed radix-4 core maps data-space unfolding to `D=1/2/4/8` independent
packets per worker and restores full lane use for stage-4/6 packets. Matched
batch-48 G3/G4 speedups are 1.244x/2.014x at uint64 and 1.374x/2.737x at
uint32. Width-specific service tables now cover batch 1/4/16/48, three G3
orders, scalar/packed controls, and admit 1/3/1 exact candidates for G=2/3/4.
The next gate is readiness/boundary co-design: packed large-batch G3/G4 remain
limited by downstream wait and their additional materialized boundaries.

The first model-guided follow-up is generated as a structured manifest. It
keeps the union of each measured batch's model top five and measured winner,
then evaluates six retained candidates at thirteen dense batch anchors. The
runner supports verification, repeated trials, and resumable CSV collection;
the result feeds directly back into the same grouped-holdout model.

The historical regression follow-up is complete: all 234 executions pass verification. Dense
batch anchors reduce nonlinear leave-one-batch-out geometric-mean regret from
1.155x to 1.032x and raise top-5 recall from 75.0% to 92.3%. On interpolation-
only holdouts, top-5 recall is 100% with 1.018x geometric-mean top-1 regret;
therefore model-guided top-5 measurement is admitted inside the calibrated
interval, while direct top-1 runtime dispatch and endpoint extrapolation still
abstain. The executable policy is `scripts/select_three_level_candidates.py`.

## v0.6.0 Status: General Shapes And Professional Plan Surface

Status: complete on the single-V100 release target. General-shape semantics,
the professional plan surface, boundary selection, tests, raw evidence, and
release documentation are checked in.
The new C plan API separates standard exact-length semantics from explicit
zero-extended embedding, lowers rank-two workloads to per-axis mappings, and
adds exact static-profile/cache selection with actionable miss logs. Exact FFT
selects between native cuFFT and Bluestein; modular Bluestein NTT is enabled
only when its root requirements hold and now reuses the selected radix-4
Hybrid2D core. Embedded and rank-two plans select and cache each physical axis.
The initial Bluestein-only FFT reached 0.105x-0.179x of cuFFT; the new selector
reaches parity by abstaining to direct cuFFT. Boundary fusion for embedded
operators now removes contiguous output scatter, and selected FP32
warp-register FWHT also fuses logical input and zero fill. The complete
logical `32767 -> 32768,batch=256` FWHT path reaches 0.11759 ms, within 1.01x
of the archived pre-padded Dao core. Structured fused input remains a generated
but unselected candidate because it regresses on V100. See
[v0.6 Implementation Status](docs/v0.6_implementation_status.md).

## v0.4.0 Status: Library And Resource-Model Integration

Status: complete on V100. This release adds the public device API and package,
generated Structured 2x2 register units, coefficient-policy specialization,
runtime selection, and portable resource-cliff features. The measured
`logN=13..15` Structured transition demonstrates why length alone is not an
adequate mapping key: shared-memory, register, occupancy, and batch-derived
grid-wave effects interact.

## v0.5.0 Status: Numeric-Regime Mapping

Status: complete for the fixed-V100 release scope. FP16/BF16 storage and
accumulation contracts, the 736-case quick screen, all 29 confirmed-event
full-protocol follow-ups, 12 paired NCU profiles, 99 adaptive coverage anchors,
and the library/base/search matrix are checked in. The abstaining piecewise
selector covers 27.74% of leave-one-batch-out shapes with 1.00013x geometric-
mean and 1.01112x worst regret. Global selection remains
`measurement-required`; that negative boundary is a result of this milestone,
not an unfinished promotion step.

This release studies the mapping method on fixed hardware before adding GPU
generation as another independent variable. Its primary question is:

> How do numeric representation, arithmetic semantics, workload shape, and
> processing-unit cost move resource cliffs and change the preferred
> space-time mapping?

The study covers precision/word and accumulator width, FFT/NTT/FWHT/Structured
arithmetic and coefficient policies, length, batch, direction, normalization,
stride, placement, output order, and the generated physical-core choices. It
uses orthogonal screening followed by focused scans around predicted and
observed launch, occupancy, register, shared-memory, and bandwidth boundaries.

The deliverable is a conditional regime model with explicit abstention and a
selector evaluated on held-out regimes, not a larger table of isolated winners.
See [Numeric-Regime Mapping Study](docs/next_phase_numeric_regimes.md) for the
hypotheses, protocol, evidence, and remaining generalization boundary.

## Completed: v0.3.0 Counter-Calibrated Selection

Status: the three V100 work packages, vectorized FFT physical unit, FFT
leave-one-batch-out selector evaluation, and matching-protocol long-FFT refresh
are complete. The post-vector fixed-mapping NCU capture attributes the gain to
19.6% fewer warp instructions with unchanged DRAM write volume. Cross-GPU
validation is deferred until the repository moves to a host with another GPU
generation.

This release turned the measured design space into a reproducible mapping
method. Its primary question was:

> Given an operator contract, transform length, batch, and GPU service profile,
> which mapping family and processing unit should be selected, and why?

The release is complete when all of the following are true:

1. NCU counters explain the FFT `logN=14` concurrency threshold, the
   `logN=18/20` throughput ceilings, and the FWHT `logN=15` mapping crossover.
2. A selector consumes workload and hardware descriptors, returns a ranked
   top-k candidate set, and reports the resource or service constraints behind
   the ranking.
3. Leave-one-shape-out validation records top-1/top-k accuracy and performance
   regret against the exhaustive V100 measurements.
4. Dao-AILab FHT and GPU-NTT are refreshed under the same correctness, clock,
   trial, and semantic protocol as the internal candidates.
5. One release command regenerates the scaling manifest, summaries, selector
   evaluation, and the paper-facing tables without modifying raw measurements.

See [Next Phase: v0.3.0](docs/next_phase_v0.3.md) for the execution order,
artifacts, and decision gates.

## Near Term Backlog

- Search boundary residence jointly with processing unit, numeric contract,
  logical padding ratio, batch alignment, and layout instead of promoting a
  fusion family from one successful operator.
- Add a padded-layout reuse contract for adjacent plan executions so an
  application can retain physical data without repeated embedding work.
- Extend the fixed-V100 regime model to stride, placement, direction,
  normalization, coefficient policy, and output-order neighborhoods using the
  same screen-follow-up-counter hierarchy.
- Hold out complete numeric and workload regimes and measure where the
  piecewise selector can expand coverage without crossing its regret gates.
- Reduce the remaining saturated FP32 `logN=20` exchange cost and the five
  FP64 `logN=14..16` crossover deficits, preserving fixed-mapping NCU evidence.
- Explain the Structured `logN=13..15` resource-cliff response across batch,
  coefficient policy, decomposition, and physical core.
- Reduce template warning volume and record compiled resource envelopes as
  generated metadata.
- Add focused sanitizer jobs on an available self-hosted GPU runner.

## Future: Cross-GPU Validation

This work starts after the numeric/workload-conditioned regime model is
established and the repository is available on a second GPU generation. It is
not the next release gate.

- Capture A100/H100/RTX 4090 hardware service profiles using the checked-in
  schema.
- Predict a reduced `(Us,Ud,Ts,Td,F,Q)` candidate set before exhaustive timing.
- Regenerate legal processing units for each architecture.
- Compare predicted and measured rankings for FFT, NTT, FWHT, and XOR-zeta.
- Identify which mapping thresholds transfer and which require recalibration.

This milestone is required before claiming that the selection methodology is
portable across GPU generations.

## Processing Units

- FFT: Stockham/split-radix and established register/shared long-FFT units;
  Tensor Core blocks only where precision and composition costs are explicit.
- NTT: compressed or recurrent root generation, shorter-lived modular
  reduction temporaries, and carefully bounded lazy/Montgomery candidates.
- FWHT: multi-CTA composition beyond the current register-resident length.
- Layered operators: add another non-transform butterfly workload with an
  independently validated reference and external baseline where available.

## Paper Artifact

- Preserve the V100 comprehensive suite as the single-GPU baseline and keep
  focused historical protocols separate from its cross-workload tables.
- Freeze a versioned numeric/workload regime matrix; later add a reduced
  multi-GPU transfer matrix.
- Publish scripts and container/toolchain metadata for every main table.
- Separate architecture ablations, core-only comparisons, resident transforms,
  and application end-to-end results.
- Add statistical confidence and clock/power controls.
- Archive a release with a persistent identifier when the paper is submitted.

## Non-Goals For The Current Release

- claiming universal superiority over specialized libraries;
- treating V100-selected tile sizes as architecture constants;
- equating a Tensor Core instruction count with end-to-end FFT acceleration;
- mixing native bit-reversed and required natural-order timings;
- presenting placeholder GPU rows as measured results.
