# Changelog

All notable repository and research-artifact changes are recorded here.

## 0.8.0 - 2026-09-06 Nested Physical Space Baseline

### Added

- Added the install-time hardware-profile workflow. The installation wrapper
  builds the capability microbenchmark, records a device-fingerprinted local
  profile, emits a parameterized unfolding model table, and provides a
  mismatch check before tuning data is reused on another GPU.

- V100 confirmed-search closure is now compiled into the runtime selector as
  an exact resident mapping overlay. Structured 2x2 and zeta cells that were
  confirmed with repeated current-code measurements report
  `selection_confidence=confirmed-median`; unmeasured cells retain model or
  calibrated fallback behavior.

- Native confirmed resident mappings now clear inherited FFT processing-unit
  state before lowering. This prevents a reused `ButterflyConfig` from sending
  a zeta/structured point through the cuFFTDx block-adapter validator.

- Broad cross-operator external-baseline coverage report. It includes the
  current-build cuFFT/Dao-AILab-FHT matches and exact-contract archived
  GPU-NTT comparisons, while explicitly leaving modulus/layout-mismatched
  rows unmatched. Regenerate it with
  `scripts/summarize_external_baseline_coverage.py`.

- Corrected comparison closure to admit multiple mature incumbents and select
  the fastest measured incumbent per semantic point. The FWHT manifest now
  includes the v0.6 warp-register/radix-2 point, preventing the hierarchical
  compatibility profile from being mistaken for the v0.6 performance floor.

- Documented workload-conditional configuration as a first-class library
  property: hardware, operator, precision, scale, direction, placement, and
  layout select the architecture mapping and processing unit independently.

- Fixed-protocol FFT dispatch validation launcher and JSON configuration for
  reproducible auto-select, linear-layout, and cuFFT comparisons.

- Search-provenance analysis for the expanded FFT matrix, distinguishing
  physically measured candidates from runtime model selections and v0.6
  fallbacks. The generated V100 FFT dispatch now preserves per-shape shared
  layouts, enabling the NCU-validated XOR-swizzle choice for logN18 high
  batch without applying it to unfavorable logN20 shapes.

- Public `query_butterfly_lowering()` contract that separates architecture
  mapping admission from processing-unit implementation. It reports role-local
  versus CTA-collective ownership and rejects unsupported multi-role/data-time
  lowerings before generation; the bounded cuFFTDx adapter remains explicitly
  single-role until a collective subgroup codelet is generated.

- Independent NTT work-distribution IR for GridTiled, TransformResident, and
  ResidentQueue ownership after logical stage and execution-group lowering.
- Strict v0.6 Hybrid2D radix-4 compatibility candidates and a generated
  physical-core capability registry.
- Stable mapping IDs plus CTA, resident-slot, grid-wave, and tail-fill
  diagnostics in the plan and benchmark CSV.
- v0.6/v0.7/v0.8 envelope benchmark with a 3% compatibility-regression gate
  and a focused NCU collection script.
- Native transform-ready `10+10` Hybrid2D radix-4 pipeline with resident
  producer/consumer CTA roles and fused physical-core coefficient layout.
- Independent producer/consumer physical-core selection inside the native
  resident kernel, including Hybrid2D/Dataflow radix-4 mixed compositions.
- Explicit `producer_ahead` startup-gate experiment for resident two-level and
  fine-wave three-level schedules
  mappings, with CLI/CSV identity and correctness/validation coverage.
- Native `cooperative-phased` control and a phase-specialized DD GridTiled
  control that isolate codelet cost from persistent CTA ownership.
- Matched work-distribution NCU collection and analysis for v0.6, DD
  GridTiled, DD ResidentQueue, and DD CooperativePhased at configurable batch.
- Native dual-tile CooperativePhased mapping with two independent 128-thread
  subgraphs per CTA, per-subgraph `data_space={1,2,4}`, SM70 named barriers,
  exact forward/inverse coverage, and a focused NCU attribution protocol.
- Five-trial comprehensive V100 protocol over two numeric widths, five
  lengths, and six batches, plus a generated NTT runtime selector table and
  machine-readable regime map.
- Three-trial three-level role-service screens parameterized by numeric width;
  the V100 uint32/uint64 evidence records the batch and precision-dependent
  `stage_partition` role policy without changing the default selector.
- Fixed cooperative three-level launches for 64-bit 8-stage partitions by
  opting into the required dynamic shared-memory limit before occupancy
  calculation.
- Added width-specific partition and within-partition role-weight evidence for
  the V100 resident mapping search, including the `8+6+6` uint64 candidate
  screen and its batch-dependent `8:6:6`/`10:5:5` policy split.
- Added a generic warp-packed radix-4 physical core. Per-role data-space
  grouping fills stage-4/6 SIMD lanes with independent packets while retaining
  packet-granular readiness and natural-order correctness.
- Expanded generic physical-chain service tables to uint32/uint64, batch
  1/4/16/48, G2/G3/G4, three G3 stage orders, and scalar/packed controls.
- Added a bounded `HybridDataflow` plus cuFFTDx block adapter for FP32
  `logN=3..10`. The adapter preserves the mixed-dataflow ownership contract,
  is excluded from automatic selection, and is measured by a four-way
  native/adapter/TemporalTile/cuFFT envelope script with explicit failure
  records.
- Recorded the cuFFTDx multi-role processing-unit boundary: a naive shared-tile
  composition is rejected after V100 illegal-access evidence, so no unverified
  core lowering is exposed; the generated workspace/role ownership contract
  remains the next implementation milestone.
- Moved HybridDataflow processing-unit lowering checks into plan validation:
  native radix-2 admits only generated role/data points, while resident radix-4
  admits its flow and CTA envelope; unsupported combinations now abstain before
  kernel launch.
- Added `ButterflyPlan::execution_realization()` and matching benchmark fields
  for the post-lowering physical contract: resident single-role versus native
  resident multi-role, data-time reuse, visible launches, whole-transform
  on-chip residency, and materialized global boundaries. Added a reproducible
  logN=8 architecture realization matrix separating native HybridDataflow,
  bounded cuFFTDx, standalone cuFFTDx, and opaque cuFFT.
- Added the public processing-unit capability registry, separating
  `ComputeUnit`/`FftCore` ownership and multi-role lowering flags from the
  backend mapping capability table.
- Made the butterfly performance model consult the processing-unit registry
  before scoring candidates, including the distinction between local-stage
  envelopes and full FFT codelet lengths.
- Added the first explicit subgroup role-local cuFFTDx codelet probe
  (`Size=8`, `ElementsPerThread=8`, `FFTsPerBlock=32`) and its benchmark. It
  validates one-warp packet ownership independently of the stage handoff
  scheduler; it is not yet promoted as a multi-role HybridDataflow lowering.
- Added an experimental complete two-role cuFFTDx resident adapter for
  64-point FFTs. Two warp roles use a shared double-buffered packet channel and
  per-slot ready flags in one launch, with a reproducible batch benchmark;
  arbitrary stage partitions remain planner work.
- Exposed the generated two-role adapter as the explicit
  `FftCore::CufftDxRolePipeline64` processing-unit choice. Plan validation now
  fills its exact N=64/FP32 mapping defaults, lowering queries report a real
  multi-role resident execution, and the plan owns the cross-twiddle table
  used by the packet consumer.
- Generalized the resident role pipeline around a compile-time `PacketSize`
  processing-unit point. The generated FP32 V100 family now instantiates
  packet widths 4, 8, and 16 (complete transforms `N=16,64,256`); packet count,
  shared-channel capacity, cross-twiddle table size, and normalization are
  derived from the template. `CufftDxRolePipeline64` remains the 8-point
  compatibility alias, while the generic point is selectable through
  `ButterflyConfig::processing_unit_points`.

### Current Evidence Boundary

- GridTiled and the native resident-ready path pass forward correctness on
  V100; the native path also has explicit inverse smoke coverage.
- The complete 1,380-execution matrix covers 60 width/length/batch points.
  Over 57 stable points, its candidate oracle is 1.138x the radix-2 base,
  1.017x v0.6, and 2.705x one fixed v0.7 resident mapping.
- DD GridTiled wins all six uint32 `logN=20` batches and the v0.6-compatible
  mapping wins every other point. The deployed two-branch policy agrees with
  the measured oracle at 60/60 points with 1.000000x stable regret.
- Resident-ready remains generated for 10+10 and is retained as an explicit
  research candidate. The complete matrix does not select it as a default;
  large-batch resident ownership and the uint64 DD core remain search targets.
- The phase-specialized DD GridTiled core takes 0.910/4.125/8.213 ms at
  batch 16/80/160, beating paired v0.6 controls by 1.27x/1.25x/1.26x. The
  persistent DD variants fall behind at large batch, so work distribution,
  phase specialization, and physical codelet are now separate search axes.
- The batch-80 dual-tile control improves one-CTA/SM CooperativePhased from
  9.525 to 6.499 ms, validating inter-subgraph latency hiding, but remains
  1.574x Grid time because the reused shared-memory codelet is inefficient at
  128-thread width. It is an attribution candidate, not a selected default.

## 0.7.0 - 2026-08-26 Hybrid Dataflow Research Baseline

### Added

- Variable logical stage partitions and resident execution-group lowering.
- Hierarchical readiness, homogeneous warp subgraphs, APPT online/static
  layouts, packet-shared radix-4, and matched NCU attribution experiments.
- Frozen research baseline tagged `v0.7.0`; v0.8 retains these points as
  candidates rather than rewriting their evidence.

## 0.6.0 - 2026-08-11 General Shapes And Plan API

### Added

- CUDA-library-style C handle/descriptor/plan API with streams, explicit
  workspace, status codes, selection queries, and structured logging.
- Exact arbitrary-length complex FP32/FP64 FFT through Bluestein and exact
  uint64 modular Bluestein NTT when the requested root domain exists.
- Explicit zero-extended power-of-two embedding for all public operators.
- Rank-two FFT, NTT, FWHT, zeta/Mobius, and Structured 2x2 composition through
  per-axis plans and transpose boundaries.
- Versioned V100 application profile table generated at build time, exact-key
  user cache, bounded runtime measurement policy, and profile-miss guidance.
- A `cubutterfly_plan_bench` CLI for logical/physical shape, batch, policy,
  cache, workspace, and kernel-time inspection.
- Per-axis static/cache/measurement selection for embedded and rank-two
  compositions, selectable Hybrid2D Bluestein NTT cores, and complete direct
  cuFFT versus Bluestein measurement for exact arbitrary FFT.
- Structured 2x2 matrices and explicit composite algorithm IDs in the plan
  benchmark surface.
- Direct physical-output lowering for contiguous embedded and rank-two
  compositions, eliminating the final full-buffer scatter while retaining the
  general strided-output path.
- Generated zero-extended input dispatch for warp-register FP32 FWHT and
  Structured 2x2 units, with the runtime selecting the one-kernel fused-input
  lowering for validated FWHT workloads and retaining pack fallback otherwise.

### Current Evidence Boundary

- All 49 CTest entries pass on V100/CUDA 11.8, including exact arbitrary
  length, embedding, rank-two, strided fallback, cache, and generated boundary
  candidate coverage.
- Direct packed arbitrary FFT selects cuFFT at timing parity; this is
  processing-unit abstention, not a claim of improving the vendor core.
- Fused-input FWHT reaches 0.11759 ms for `32767 -> 32768,batch=256`, within
  1.01x of the archived pre-padded Dao result, and uses zero composition
  workspace.
- Structured fused input is retained as a correct generated candidate but is
  not selected because the measured V100 path regresses.
- Cross-GPU selection transfer and universal specialist-library superiority
  remain outside the release claim.

## 0.5.0 - 2026-08-08 Numeric-Regime Mapping

### Added

- Public FP16 and BF16 real/complex storage contracts for FFT, FWHT, and
  Structured 2x2, each with explicit native-width or FP32 accumulation.
- V100 BF16-native emulation labeling so FP32-compute-and-narrow measurements
  cannot be mistaken for native SM70 BF16 arithmetic.
- A generated 185-cell numeric-regime space covering floating-point contracts,
  uint32 zeta, uint32/uint64 NTT, nine length anchors where legal, and matched
  batch neighborhoods around predicted grid-wave boundaries.
- Resource, grid-wave, working-set, winner-crossover, and numeric-pipeline cliff
  detection with repeated-trial confirmation rules.
- Complete-regime selector evaluation across numeric contract, arithmetic
  policy, batch region, and operator, with explicit promotion/regret gates.
- An abstaining piecewise batch selector that emits stable intervals, candidate
  sets at boundaries, and explicit unseen-contract/out-of-range policies.
- A matching-protocol library/base/search matrix and table generator covering
  representative FP32 FFT/FWHT and 60-bit NTT shapes, with separate search/base
  and search/external-library ratios.

### Current Evidence Boundary

- Correctness, focused smoke timing, and the 736-case/2,208-sample quick screen
  pass on V100. The screen identifies 29 confirmed and 67 ambiguous mapping
  crossovers. This screen is discovery evidence rather than a cross-library
  performance claim; the focused timing, counter attribution, and separate
  matching-protocol library comparison below provide the confirmation layers.
- Complete-regime top-3 recall is 1.0, but geometric-mean/worst regret are
  1.047/2.078. The new selector is intentionally not promoted to runtime use.
- Focused full-protocol timing repeats all 29 quick-confirmed neighborhoods:
  23 retain the same direction, 3 reverse, and 3 have overlapping trial ranges.
  All nine length-axis crossovers reproduce; instability is confined to batch
  boundaries.
- Twelve paired NCU profiles attribute the six unstable batch events to
  instruction/register tradeoffs, online-composition synchronization, and one
  actual FP64 resident-CTA capacity change. None contradicts the stable
  length-axis regime result.
- The conservative piecewise selector auto-selects 13.61% of leave-one-batch-
  out shapes with 1.0016x geometric-mean and 1.0567x worst regret. It is
  validated only for those calibrated intervals; global runtime status remains
  `measurement-required`.
- Adaptive full timing covers 99 weak non-boundary anchors: 66 become stable,
  33 remain near-ties, and none reverse direction. Incorporating them raises
  leave-one-batch-out coverage to 27.74%, with 98.84% top-1 agreement,
  1.00013x geometric-mean regret, and 1.01112x worst regret.
- In the representative ten-shape library/base/search matrix, searched
  configurations improve over fixed radix-2 base mappings by 2.349x geometric
  mean and over matched external-library rows by 1.109x. Operator-level
  search/library ratios are 1.015x FFT, 1.054x FWHT, and 1.313x NTT. The NTT
  external rows are archived same-machine measurements with the same protocol,
  not interleaved measurements from the current refresh.
- Cross-GPU transfer remains deferred until the fixed-hardware numeric and
  workload regime model is measured and validated.

## 0.4.0 - 2026-08-05

### Added

- Canonical uint32 subset-zeta/Mobius and superset-zeta/Mobius operators across
  temporal, hierarchical, online-reorder, warp-hybrid, and stage-pipeline
  mappings, including radix-2/4/8 and CPU references.
- A parameterized FP32/FP64 structured-2x2 operator with broadcast or per-stage
  matrices, validated local inversion, device-resident coefficients, radix
  fusion, CLI/CSV representation, and CPU/GPU regression targets.
- A 762-sample V100 structured-2x2 length/precision mapping scan, matched FWHT
  control, forward/inverse best-point verification, and focused NCU A/B script.
- Structured/FWHT counter attribution and an alternating-order `logN=20`
  confirmation that rejects a frequency-sensitive sequential-scan advantage.
- A generated FP32 Structured 2x2 warp-register core for `logN=3..15`, with
  broadcast/per-stage matrices, inverse and strided/in-place coverage, and a
  V100 register-versus-shared/FWHT performance study.
- Refreshed Structured/FWHT NCU attribution showing the generated `logN=12`
  matrix core cuts DRAM traffic by 3.02x and base-clock time by 2.64x; its
  remaining same-transport gap is dense arithmetic and register pressure.
- Separate generated Structured coefficient policies for one-matrix broadcast
  register reuse and arbitrary per-stage tables, with equivalent-semantics V100
  A/B timing and compiled register/spill evidence.
- NCU coefficient-policy attribution confirming broadcast reuse preserves the
  FP32 work while reducing registers, long-scoreboard stalls, and base-clock
  time relative to the per-stage table path.
- Portable per-candidate residency/cliff feature extraction, hardware/resource
  profiles, and resource-aware sweep annotations for register, shared, thread,
  warp, occupancy, grid-wave, and temporal-state search signals.
- An operator catalog defining transform semantics, the legacy `xor-zeta`
  compatibility behavior, mapping coverage, and the next operator candidates.
- Allocation-free asynchronous device-pointer execution for common butterfly
  and NTT plans, with caller-selected CUDA streams and queryable workspaces.
- Installable CMake package metadata exposing the
  `cuButterfly::cuButterfly` target and a standalone device API example.
- Device API lifecycle, memory, placement, type, and concurrency documentation.
- Product-oriented programming guide, public C++ API reference, error model,
  capability/compatibility matrix, and task-oriented example index.
- Build-checked basic FFT and asynchronous NTT examples alongside the common
  butterfly device-pointer example.
- A unified V100 runtime selector for FFT, NTT, FWHT, and XOR-zeta plans, with
  resolved configurations, prediction confidence, and auditable reasons.
- Build-time generation of runtime latency anchors from the archived scaling
  summary, plus selector generation and mapping-crossover regression tests.

### Changed

- Host-vector execution now submits through the same stream-aware kernel path
  as the application API and lazily creates its private staging buffers.
- Added a benchmark-only `--shared-bank-width default|32|64` probe. On the
  measured V100, an explicit 64-bit request is rejected after the runtime
  reports fixed 32-bit banks; the valid direct1024 logN14/batch64 point has a
  0.034652 ms median under the five-trial probe. Bank width is therefore not
  included as a V100 selector dimension.
- Added a current-build broad comparison screen covering six butterfly
  operators at logN10/12/14 and a separate logN20 NTT screen. Results and
  the publication-grade coverage boundary are recorded in
  `docs/comprehensive_butterfly_comparison.md`.

### Current Evidence Boundary

- The generated register core makes resident FP32 Structured 2x2 transforms
  1.208x-3.108x faster than the best measured shared-core controls at
  `logN=8/10/12/15`; the coefficient policy adds up to another 1.107x.
- On V100, the Structured register path enters a shared-memory resource cliff
  at `logN=13`, a register-limited region at `logN=14`, and the one-CTA/SM
  boundary at `logN=15`. These are measured single-GPU boundaries, not fixed
  architectural constants.
- The public device API, package target, examples, and V100 runtime selector
  are functional and regression tested. Unsupported contracts remain explicit.
- The next research phase conditions mapping preference and resource cliffs on
  numeric representation and workload shape before attempting cross-GPU
  transfer. Cross-GPU profiles remain placeholders only.

## 0.3.0 - 2026-08-02

### Added

- An orthogonal 184-case V100 length/batch suite and generated manifest for
  FFT, NTT, FWHT, and XOR-zeta.
- Scaling summaries that identify timing quality, throughput saturation, and
  mapping crossovers without using sub-0.020-ms rows as stable claims.
- Targeted NCU collection for the FFT saturation gaps and FWHT processing-unit
  crossover.
- A V100-calibrated mapping selector with held-out-shape evaluation and
  explicit separation of mapping family from processing unit.
- A matching-protocol 120-sample Dao FHT and GPU-NTT baseline refresh.
- Targeted NCU attribution over 22 FFT/FWHT implementation-shape pairs and 37
  kernels, plus one command that regenerates all derived V100 analyses.
- An FP64 prefix address-path experiment that separately reproduces scalar,
  table, recurrence, XOR-swizzled, and cuFFT timing under one protocol.
- FP64 online cuFFTDx segment specializations at logN 7/8/9, a 648-point
  neighboring-length mapping scan, and an interleaved 35-shape batch matrix.

### Changed

- Reworked the root, design, documentation-index, and experiment narratives
  around the two logical dimensions and their four spatial/temporal unfolding
  factors, with an explicit mapping from graph work to GPU services.
- Replaced the high-level concept figure with a hardware-mapping diagram that
  connects `Ud/Td/Us/Ts`, residence and transport, replaceable processing units,
  online reordering, and counter-calibrated selection.
- Updated paper-facing experiment summaries to include the expanded FP64
  length/batch robustness matrix and fixed-mapping NCU attribution.
- Regenerated checked-in V100 summaries with the current `shared_layout` and
  `fp64_thread_instructions` schemas; existing measurement values are unchanged.
- The comprehensive runner now supports exact case selection, validates batch
  consistency within comparison groups, and caches identical correctness
  preflights.
- The next release is scoped around counter-calibrated mapping selection and
  refreshed same-protocol external baselines.
- Cross-GPU validation is deferred until the repository moves to a host with a
  second GPU generation; it is not a `v0.3.0` release gate.
- FP64 online staging now hoists invariant global addresses and advances
  same-period shared permutations by compile-time strides.
- FP64 online composition now covers total `logN=14..18`; each length selects
  its split and two physical mappings independently.

### Current Evidence Boundary

- The new scaling evidence remains V100-only.
- FP32 FFT matches or exceeds cuFFT at selected saturated shapes, but cuFFT has
  a higher large-batch ceiling at `logN=18` and remains ahead at `logN=20`.
- The profiled FP64 `logN=16`, batch-64 mapping reaches cuFFT parity at 1.002x.
  Its updated counter capture attributes this to 26.8% fewer prefix warp
  instructions and 8.3% lower prefix replay than the prior XOR kernel.
- In the expanded FP64 matrix, 20/25 stable shapes have higher median
  throughput than cuFFT and all 13 stable `logN=17/18` shapes are faster.
- Cross-GPU portability and selector accuracy have not yet been measured.

## 0.2.0 - 2026-07-24

### Added

- Operator-independent butterfly and FFT architecture-space specifications,
  validation, exploration, and generated-code selection tools.
- Optional cuFFTDx, TurboFFT, and VkFFT integration boundaries.
- A controlled 37-case V100 comprehensive suite with correctness preflight,
  randomized trials, stability classification, raw records, and detailed
  length/batch tables.
- FFT architecture sweeps, processing-unit comparisons, and resident-kernel
  Nsight Compute evidence.

### Changed

- Extended FFT, FWHT, NTT, and XOR-zeta configuration and capability coverage.
- Updated the architecture, hardware-mapping, processing-unit, experiment, and
  reproducibility documentation to distinguish fresh comprehensive results
  from focused historical protocols.
- Defined the next phase as an orthogonal length/batch scan, refreshed external
  baselines, and targeted profiling of long FP32 and FP64 FFT gaps.

### Current Evidence Boundary

- V100 remains the only comprehensively measured GPU.
- Measured FP32 FFT reaches cuFFT parity at selected lengths, while FP32
  `logN=20` and FP64 `logN=16` remain open performance gaps.
- Dao-AILab FHT and GPU-NTT results were not rerun under the comprehensive
  protocol and remain contextual evidence.

## 0.1.0 - 2026-07-23

- Initial public cuButterfly research artifact with the common space-time
  mapping model, NTT/FFT/FWHT/XOR-zeta kernels, V100 measurements, and
  reproducibility documentation.
