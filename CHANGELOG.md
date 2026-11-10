# Changelog

All notable repository and research-artifact changes are recorded here.

## Unreleased — v0.9 source-release documentation

- Describe the common dependency/resource method, tile subgraph execution and
  replaceable cores through task-oriented public documentation.
- Separate stable installation/API guides, experimental results, platform
  references and development records; retain historical evidence paths.
- Document native/CMake 0.9.3 and Python frontend 0.9.3.11 as component identities
  within the v0.9 family, preserving build fingerprints and module ABI contracts.
- Check nested documentation links and support explicit source-contract scope
  validation without presenting missing result evidence as verified.


## Unreleased - Fixed-core FFT controls and full refresh preparation, 2026-10-25

- Correct the multi-slot factor prefetch wait distance and report effective zero
  prefetch for a single consumed tile. Preserve both synchronous and asynchronous
  legal organizations; deeper buffering is not assumed faster.
- Complete nine matched FP64 Graph measurements on A100 80GB. One-step prefetch
  improves two fixed cores, while the strongest synchronous controls reduce the
  gains to 1.034x/1.093x and complete cuFFT remains faster at this shape. Preserve
  all trials, numerical errors and mapping differences in the new paper evidence.
- Prepare all 66 frozen operator requests on CPU and add a guarded calibration
  and asynchronous measurement driver. Freeze the seven-track full refresh plan;
  pending GPU measurements and application bindings are not reported as complete.
- Fix current-comparison registry preparation so actual runner paths use the
  requested build. Update the manuscript and add a reproducible fixed-core figure.

## Unreleased - Independent RNS stage mappings, 2026-10-18

- Extend fused CT/GS local stages to logs 2..10 and row groups through 128;
  expose per-pass row/thread overrides with independent resource checks.
  Keep coupled search candidates and add equal tile capacities with independent
  threads. This bounded subset now represents previously excluded native shapes.
- Pass 148 exact RNS cases and 12 adapter cases, including measured-candidate
  coverage. A100 40GB and 80GB each complete 52 exact CKKS Graph pairs.
  Method-I/II geometric mean speedups are 0.969x/0.925x on 40GB and
  0.972x/0.924x on 80GB. Method-I N=65536 averages 1.050x/1.055x respectively,
  with 12 measured wins per card. Four fixed-core organization controls show no speedup.
- Reject occupied-GPU starts and sampled runtime contention in the CKKS runner;
  retain raw records while excluding known contaminated performance. The attempted
  contaminated 80GB attempt is excluded; subsequent monitored reruns pass. See the
  [stage-mapping report](results/e2e_rns_workload_20261018/stage_mapping_optimization.md).
- Rebuild C++ consumers after the RnsMapping layout change. Historical snapshots
  retain matching binaries and libraries; the shared conda install is unchanged.

## Unreleased - RNS register codelets and workload attribution, 2026-10-18

- Add register-fused radix4 to the existing fused-root CT/GS candidate family,
  retaining radix2 and explicit rejection of unavailable radix8 mappings.
  Dispatch, validation and automatic tuning consume the same compute-unit field.
- Pass 124 exact CPU-reference RNS cases and 12 adapter cases. Re-run all 52 CKKS
  Graph pairs with exact ciphertexts and unchanged kernel counts. Method-I/II
  geometric mean speedups are 0.961x/0.927x; method-I N=65536 averages 1.016x.
- Reconstruct actual request frequency from application traces, distinguishing
  ciphertext batch from each NTT request's 2–12 polynomials. Retain workload
  attribution, core controls and negative results in the
  [optimization report](results/e2e_rns_workload_20261018/optimization.md).

## Unreleased - RNS core and boundary candidates, 2026-10-18

- Add physical bit-reversed boundary traversal and resource-checked row groups
  through 32. Direct bit-reversed input/output were already supported; this
  changes lane-to-address mapping without adding a permutation kernel.
- Add a parameterized fused-root CT/GS candidate using mature negacyclic NTT
  strategy and existing Shoup arithmetic. Bind explicit roots, stage intervals,
  polynomial addresses and orders; specialize direction and power-of-two
  indexing. Select it alongside Hybrid2D rather than by application name.
- Pass 104 exact RNS cases, 12 native-library adapter cases and six legacy
  Hybrid2D checks. The new 52 complete CKKS Graph pairs remain slower than
  HEonGPU, with method-I/II geometric mean speedups of 0.919x/0.899x. Preserve
  the negative rows and counter evidence in [the report](results/e2e_rns_coalesced_20261018/README.md).

## Unreleased - Explicit-root RNS NTT, 2026-10-18

- Add a public C++ RNS plan with per-domain roots and per-polynomial addresses,
  cyclic/negacyclic rings, both directions and natural/bit-reversed boundaries.
  Reuse the existing Hybrid2D arithmetic; combine RNS domains per pass and fuse
  boundary transformations. Request-local Graph tuning records its bounded
  candidate family separately from installation cost-model calibration.
- Connect HEonGPU complete CKKS operations to this plan and retain the grouped
  adapter as an explicit control. All 104 current paired Graph rows agree
  exactly in integer ciphertexts. Native-RNS remains slower than GPU-NTT inside
  HEonGPU despite a substantial improvement over the old adapter; see
  [the current report](results/e2e_rns_20261018/ckks_comparison.md).
- Keep unchanged generated register kernels' timestamps stable by rendering
  intermediate templates in memory. This avoids unrelated CUDA recompilation
  when CMake reconfigures; generated output bytes remain identical.

## 0.9.3.11-search.1 - Candidate Family Coverage, 2026-10-17

- Distribute finite public search budgets across execution families before
  sampling each family; preserve the incumbent, candidate semantics and
  exhaustive budget-zero behavior. Add optional NTT candidate trace records.
- Add grouped external FFT baseline adapters for logical rank above three.
  Include packing, copies and normalization in Graph timing and identify
  these results separately from native vendor rank1--3 measurements.
- Complete the finite comparison's same-precision numerical review and
  trace historical NTT means to their distinct populations. See the
  [repair and measurement scope](docs/ntt_search_repair_20261017.md).

## 0.9.3.11-provider.3 - Mature Core Reuse, 2026-10-14

- Adopt into the main build and shared conda environment for the continuing
  development campaign. Add explicit build-prior search/stage continuation and
  driver adoption, retaining prior evidence while confirming current finalists.
- Prioritize imported mature cores over inventing arithmetic to fill absent
  provider parameter combinations; distinguish semantic requests from explicit
  physical mappings in the project-wide research principles.
- Enumerate the online register FFT's cuFFTDx suffix over provider-supported
  power-of-two EPT values, preserving historical preference and geometry checks.
  Include previously omitted EPT2 configurations without changing user semantics.
- Keep the newly added native suffix available for explicit experiments and
  replay, but remove it from automatic candidate enumeration. Preserve existing
  prefix and factor cores, and all recorded negative measurements.
- Initial validation ran in isolation; the user subsequently authorized adoption
  into the continuing experiment. See the [validation and scope](docs/provider_reuse_20261014.md).

## 0.9.3.11-provider.2 - Explicit Native Suffix, 2026-10-14

- Add a parameterized native suffix core to the existing register FFT template,
  including the missing FP64 local N1024/N4096 EPT32 geometries, without changing
  precision, workload or EPT and without modifying the vendor database.
- Thread `suffix_core` through public mapping/replay, candidates, JIT preparation,
  physical group records and calibration/service identities. Preserve the default
  cuFFTDx mapping and reject conflicting or irrelevant core requests.
- Account for full native shared storage independently of writeback chunking.
  Pass 209 CPU tests and 432 bounded GPU checks; report numerical errors and
  slower native timings without promoting a new default. See the
  [implementation and evidence](docs/native_suffix_core_20261014.md).

## 0.9.3.11-provider.1 - Explicit Core Capability, 2026-10-14

- Read installed cuFFTDx C2C implementation records through one parser shared
  by C++ candidate generation, explicit mapping validation and Python JIT.
- Distinguish missing provider combinations from invalid transform semantics;
  retain unknown capabilities for actual compilation and export a candidate audit.
- Include provider fragments and the parser in compilation/build identities,
  and install the parser with calibration and JIT tools.
- Bind the generated table to provider contents, including upgrades in place;
  return `NOT_SUPPORTED` rather than `INVALID_VALUE` for missing implementations.
- Preserve valid FP32/FP64 alternatives and native cores. Pass 90 affected CPU
  tests, compile affected C++/CUDA objects and both legal EPT16 counterparts of
  the rejected EPT32 mappings. This revision is isolated from the active campaign;
  see [contract and validation](docs/provider_capability_contract_20261014.md).

## 0.9.3.11-calibration.3 - Finite Curve Refinement, 2026-10-14

- Give each full-calibration curve its own finite refinement budget and report
  that actual cap. Preserve explicit caps, holdout criteria and measurement
  identities; deeper refinement no longer stops at the old one-refinement cap.
- Resume the frozen stage inventory before plan search, preserving 457 clean
  stage records, one excluded contaminated record and 322 candidate records.
- Distinguish unsupported provider configurations from unsupported operator
  requests. Two FP64 EPT32 candidates lack implementations in the installed
  cuFFTDx SM800 table; the same FP64 N65536 B64 request has correct measurements
  through other mappings in this build. Retain both rejections and document the
  missing common capability check; do not call this a mathematical limitation.
- Affected CPU checks passed: 61 stage/recovery tests and 41 calibration gate
  tests. GPU supplementation passed the finite stage gate with 562 clean records
  and every earlier payload preserved; plan search resumed. Final confirmation
  and the 66-request comparison remain pending and are tracked in the
  [recovery record](docs/calibration_repair_20261014.md).

## 0.9.3.11-calibration.2 - Recovery Source and Interruption, 2026-10-14

- Bind historical input explicitly with `--seed-registry` and validate the frozen
  seed snapshot before GPU work; preserve strict source-path and input-hash checks.
- Save interrupted stage checkpoints and exit 130 before search; record
  `interrupted-calibration` in the recovery driver.
- Preserve the failed v1 attempt and resume compatible 412 clean stage
  measurements and 321 candidate records with the same repaired probe and GPU
  library; retain one contaminated record for remeasurement.
- Record 116/117 integration tests passing, with new preflight/interruption
  regressions passing and the existing bounded stratification coverage failure
  reproduced from the unchanged pre-repair source. Recovery reached 439 correct
  stage records, then paused under the foreign-GPU guard; the saved checkpoint
  was reviewed for compatible resume. CPU comparison retains 83 rows and GPU
  comparison remains 0/66. See the
  [v1 history and v2 recovery status](docs/calibration_repair_20261014.md).

## 0.9.3.11-calibration.1 - Calibration Recovery, 2026-10-14

- Prepare fixed CPU predictor features and use exact native CPU event simulation;
  preserve candidate scores and stable ordering while reducing the measured
  19,539-candidate scoring pass from 167.65655 s to 2.20227 s. Preparation and
  projection costs are separate; this is not a GPU execution speedup.
- Correct FWHT FP16/BF16 stage-reference storage rounding and replace only the
  stage probe; preserve the GPU library and whole-plan benchmark binaries.
- Distinguish certified curves eligible for ranking from complete declared-scope
  calibration; require validated service coverage and `complete-observed`
  composition before publishing a registry.
- Resume in a separate recovery directory with compatible candidate/incumbent
  timings retained and stage measurements repeated for the new probe. Calibration
  and the 66-request GPU comparison remain pending; see
  [repair and recovery status](docs/calibration_repair_20261014.md).

## 0.9.3.11-common-organization-20261013 - Cross-Operator Organization

- Share capacity, local residence, layout, thread, batch-folding and fusion
  candidate generation across the six arithmetic operator families.
- Execute shared bulk batch widths 1..4 through IR, dynamic/JIT launches and
  calibration identities. Preserve the separate overlap chunk semantics.
- Lower NTT shared resident boundaries to physical groups while retaining the
  public logical mapping; align offline compilation and service projection.
- Extend xD resident organization to the common rank-eight layout limit,
  arbitrary axis permutations and device-valid warp-multiple launch widths.
  Keep arithmetic kernels and numeric contracts unchanged; version xD cache v22.
- Make serialized mappings authoritative for calibration and retain abstract
  status when enumerating all declared cores. Record bounded coverage and
  remaining core-specific constraints in `docs/common_organization_20261013.md`.

## 0.9.3.10-unified-template-parameter-20261013 - Unified Template Parameters

- Expose `register_threads` as a shared FWHT/structured template resource, with
  zero retaining the legacy shape-specific choice and nonzero values requiring
  a generated, shape-valid instance.
- Generate and launch real thread variants, preserve the resolved resource in
  mappings and calibration identities, and expose it through the bench frontend.
- Add the existing supported NTT homogeneous/static-I/O and two-level/three-level
  dataflow families plus the descriptor-complete base APPT register-tail family
  to the public candidate inventory through a bounded factory.
  Validate modulus, word-size, batch and root-of-unity support before emitting candidates.
  APPT warp/radix8/grouped/resident variants remain explicitly excluded until
  their separate role and liveness contracts are executable.
- Add CPU contract tests, GPU arithmetic checks for 20 register variants, and
  six-operator public mapping replay coverage. The NTT template inventory is
  replayed over 16 word64 and 26 word32 templates in both directions (84/84
  exact cases); platform CTA weights are resolved by the plan before launch.

## 0.9.3.9-core-template-contract-20261013 - Shared Core Parameters

- Move xD axis-core enums, resource fields, validation and compiled candidate
  enumeration into one CPU-only contract consumed by public parsing, lowering
  and per-axis refinement.
- Add explicit packed exchange-4 and length-64 local 32-line candidates while
  retaining all earlier candidates and recording shape/device omissions.
- Include output scale and final-storage rounding in platform semantic matching;
  add CPU regressions for exact and mismatched FWHT rules.
- Keep imported VkFFT as a composable provider. This revision does not claim
  that its internal implementation has been transplanted into every butterfly
  template; the audit records that boundary and the remaining NTT/FWHT gaps.
- Use v21 xD cache identities so mappings generated under the old parameter
  contract are not replayed.

## 0.9.3.8-e2e-external-axis-20261013 - Imported Axis Core Selection

- Add an optional VkFFT single-axis provider to the shared xD core inventory,
  uniform selection and per-axis refinement for complex and compact-real FFTs.
  Axis order, layout, real boundaries and scaling remain owned by cuButterfly.
- Support FP32/FP64, both directions and non-power-of-two suffixes with a
  power-of-two transformed axis. Record unavailable requests rather than
  changing their precision, shape or operation.
- Distinguish provider-owned scratch/LUTs from caller workspace and preserve
  the final scale pass when the imported core cannot fuse arbitrary scaling.
- Add optional CMake integration, disabled stub, v20 cache separation, exact
  mapping replay and focused provider/public/mixed-core Graph validation.
- Keep the prior grouped-axis binary immutable and compare the same complete
  application requests with an isolated build. Importing a faster core is
  recorded separately from fixed-core organization gains.

## 0.9.3.7-e2e-grouped-20261013 - Grouped Axes and Scale Placement

- Add explicit 2/4-line CTA candidates for 2048/4096-point xD cuFFTDx axes,
  including FP32/FP64 and both directions, with device resource checks.
- Add inverse compact-real scale placement at the last complex-axis writeback;
  retain the old final scale pass as a measured candidate and preserve old replay.
- Include these parameters in public selection, per-axis refinement and v19 cache
  identity. No application-name or GPU-model performance overrides are added.
- Keep the prior binary and produce an isolated incremental research build with
  source/object/library SHA receipts. The embedded build fingerprint is inherited;
  the receipt and library SHA identify this revision. Public ABI version is unchanged.
- Add finite correctness coverage and same-request complete CUDA Graph application
  comparisons. This revision does not declare the full research matrix complete.
- Correct the validation resource probe so unsupported runtime attribute queries
  on driver-loaded functions do not contaminate subsequent CUDA launch checks.
  Retain the full grouped inventory; no workload-size or direction exclusions
  are inferred from those diagnostic errors.

## 0.9.3.6-e2e-coverage-20261013 - Public Application Paths

- Extend xD cuFFTDx axes to 2048/4096 in FP32/FP64, with device resource
  checks, and add packed contiguous-core candidates to public xD selection.
- Add explicit full-FP32/final-FP16 FWHT rounding and output scaling through
  C/C++/Python descriptors, candidate selection, registry and cache replay.
  Keep intermediate-storage rounding distinct from the fused short provider.
- Add reusable changing-input inference graphs, fixed-filter scale caching,
  and explicit fast-padding application controls. Complete application timing
  includes required input/output copies and conversions.
- Use same-precision CPU numerical references in the new coverage drivers;
  keep GPU comparator errors and historical higher-precision diagnostics
  separate. Record the bounded xD tuning memory budget before cache checks.
- Build and validate separately from the active calibration runtime. This
  revision is a bounded integration milestone, not a full-matrix performance
  or universal speedup claim; see `docs/e2e_unified_coverage_20261013.md`.

## 0.9.3.5-library-rebound-20261011 - Explicit Search Budget

- Add opt-in `CUBUTTERFLY_SEARCH_BUDGET` to the public butterfly and NTT
  measurement search. Zero keeps exhaustive search; positive values measure a
  deterministic subset and retain full versus scheduled counts in the trace.
- Restart A100 80GB experiments under a distinct library/calibration identity.
  Bounded searches and finite calibration remain provisional evidence, not
  exhaustive design-space coverage or full migration qualification.
- Validate calibration, manifest provenance and compile transport before
  preparing a rebuilt-library campaign; preserve the source experiment files.

## 0.9.3.4-jit-runtime.1 - 2026-10-10 Shared JIT Runtime

- Link JIT specializations with a shared CUDA runtime and include linkage in
  module identities. This removes measured stream/event/launch overhead from
  thousands of private static runtimes without changing arithmetic or search.
- Add optional candidate-phase tracing and bind research calibration to the
  immutable broker's compiler, templates and runtime inputs. Async runner
  revision `0.9.3.4-async-experiments.4` rejects incompatible formal timing.
- Preserve static-runtime evidence and CPU references. Prepare fresh per-GPU
  calibration and a distinct shared-runtime comparison; old timing rows are
  not relabeled. The linked library and public ABI versions remain unchanged.

## 0.9.3.4-async-experiments.3 - 2026-10-10 Persistent Compile Queries

- Replace per-module Python startup with an optional native client and persistent
  CPU broker service, preserving runtime validation and complete-search failures.
- Group measurement workers by module family and verify each case before the
  next search, reducing repeated module loading and partial-search recycling.
- Audit a new handoff with frozen source snapshots and retained measurements;
  keep the CUDA library, calibration and full timing protocol unchanged.

## 0.9.3.4-cpu-preparation.1 - 2026-10-10 Compile Queue Scheduling

- Add an independent CPU producer that refills compilation slots directly,
  avoiding repeated full request/record scans on each completion and heartbeat.
- Prioritize known case dependency groups with background fairness; retain every
  candidate, existing failures, broker identity and exact compiler cache keys.
- Run 24 CPU slots and reuse complete input/reference preparation. Keep the
  active GPU library, calibration, worker sources and timing protocol unchanged;
  complete offline export of adaptive xD candidates remains a separate gap.

## 0.9.3.4-async-experiments.2 - 2026-10-10 Bounded Worker Recovery

- Audit interrupted indexes and artifacts, retain 29,170 processed requests and
  retry both unindexed resource-error cases among the remaining 9,246 requests.
- Bound GPU worker lifetime and record process resources; guard JIT queries
  inside selection and reject incomplete or unrecorded resource-failed searches.
  Preserve complete candidate selection and the existing timing protocol.
- Complete all 120 original-device scaled controls before resuming GPU1 timing;
  continue the independent GPU2 timing subset. No library or calibration change.

## 0.9.3.4-real-range-controls.2 - 2026-10-09 Numeric Experiment Controls

- Extend separate scaled-input controls to FP16 R2C/C2R, including xD packed
  Hermitian bounds and scaling before C2R spectrum quantization. Record input
  provenance and quantization loss without changing raw requests or kernels.
- Audit nonfinite inputs and references separately from finite-reference kernel
  failures. Preserve original verdicts and complete error artifacts.
- Prepare complete CPU inputs/oracles and queue original-device replay after
  active measurements. Distinguish the completed 24-case A100 40GB diagnostic
  from pending original-mapping verification and floating accuracy acceptance.

## 0.9.3.4-async-experiments.1 - 2026-10-09 Experiment Scheduling

- Prepare full inputs and exact references with a resumable CPU process pool.
  Compile templates in an independent queue; incomplete search attempts cannot
  silently exclude candidates and are retried only after preparation finishes.
- Move snapshot verification and error artifacts off the GPU submission path.
  Keep each backend's seven timing samples contiguous, preserving the original
  invocation counts and complete conversion costs.
- Retain checked historical results, per-GPU calibration and separate scaled
  controls. The CUDA library version and arithmetic kernels do not change.

## 0.9.3.4 - 2026-10-09 Range-Aware Inverse Normalization

- Refine the intermediate 0.9.3.3 low-precision inverse repair: use exact
  magnitude-bit bounds to normalize after a butterfly when it has sufficient
  range, and before otherwise. Keep the total 1/N factor and requested arithmetic
  precision while avoiding the observed minimum-subnormal constant regression.
- Retain both intermediate builds and their complete error evidence. Stop the
  intermediate calibration queues before rebuilding; restart final-build
  measurements with fresh identities and no inherited passes or timings.

## 0.9.3.3 - 2026-10-09 NTT Replay and Low-Precision Inverse Normalization

- Complete public algorithm identification for valid NTT backends through the
  existing serialized mapping interface. Preserve full physical parameters and
  descriptor semantics for CompactStage, StagePipeline and HybridDataflow replay.
- Distribute normalized low-precision FFT/FWHT inverse scaling over radix-two
  stages, before arithmetic can overflow. Retain FP16/BF16 storage, the requested
  computation precision, input values and total 1/N normalization.
- Preserve the stopped campaign and original failures. Prepare fresh affected
  and unfinished requests, and a complete new-version timing population with
  explicit disjoint GPU assignments and independent calibration per device.
- Add validated timing as a separate continuation mode. Every timed request
  undergoes fresh correctness and mapping-replay checks; historical passes
  cannot be imported into this mode. Floating accuracy acceptance remains pending.
- The public ABI remains 0.9.3; the library and runtime fingerprints distinguish
  this implementation. See the dated repair record for completed checks and
  ongoing comprehensive experiments.

## 0.9.3.2 - 2026-10-08 Composition Calibration Evidence

- Add bounded, design-selected bulk/overlap calibration pairs, using verified
  operator and independent-stage descriptors. Retain tail workloads and omitted
  families explicitly rather than treating serial winners as overlap coverage.
- Freeze stage-model predictions before whole-plan timing, verify every batch,
  and report paired speedups, ranking regret and prediction errors separately.
  Resume checks bind the library, source, device, protocol and saved evidence.
- Wire this check into staged hardware migration and fresh recalibration. A
  completed finite sample cannot satisfy the full inventory completion gate;
  unknown concurrent-resource parameters remain unknown.
- This revision changes calibration tooling, not the 0.9.3 public ABI or the
  current 0.9.3.1 GPU library. GPU combination results are pending.

## 0.9.3.1 - 2026-10-08 UINT64 Calibration and JIT Completion

- Execute and verify UINT64 zeta in both whole-plan and independent-stage
  calibration tools using full-width unsigned inputs and exact comparisons.
- Complete UINT64 shared-iteration JIT type and operator dispatch. Keep the
  0.9.3 public ABI; runtime fingerprints distinguish the generator revision.
- Restart the bounded campaign with fresh mapping and stage measurements;
  retain the initial failed JIT probe and previous runtime as separate evidence.

## 0.9.3 - 2026-10-08 Bounded Loads and Numeric Lowerings

- Add native UINT64 subset/superset zeta and inverse kernels, including xD
  residency, exact candidate checking and eight-byte resource accounting.
- Add explicit FP32-storage/FP64-compute adapters and same-precision
  FP16/BF16 full-spectrum R2C/C2R adapters. Their mappings report conversion
  kernels and workspace; complete execution includes every boundary operation.
- Extend compact FP32/FP64 real FFT composition to rank eight and arbitrary
  standard-length FFT composition beyond two axes. Bound default high-rank
  axis-order candidates without restricting legal explicit order replay.
- Replace the extreme size/batch Cartesian product with a declared bounded
  population and an exclusion audit. Functional gaps remain visible.
- Use an exact integer radix-two NTT reference for power-of-two axes, retaining
  a direct oracle for other lengths and all-element validation in both cases.
- Archive the previous library and stop the old matrix between shards. New
  kernels require fresh measurement identities and calibration binding; old
  results and unaccepted floating-point evidence are not relabelled.

## 0.9.2.2 - 2026-10-08 Experiment Continuation Repair

- Validate each closed shard once per controller process, check its evidence
  for changes between launches, and fully validate history on process restart.
  Remove repeated large-set construction and redundant parent-case copying.
- Support explicit controller-only source upgrades with archived state and
  migration hashes. Measurement sources, kernels, calibration and saved rows
  retain their original identities.
- Reject baseline execution/binding errors in every continuation branch, and
  require matching conformance/comparison execution configuration.
- Correct continuation heartbeat counts and historical-population reporting.
- Extend separately labelled input-range controls to standard FP16 FWHT with
  an input L1 bound, preserving the original pressure-test inputs and errors.

## 0.9.2.1 - 2026-10-08 Research Harness Numeric Reporting

- Reconstruct FP16/BF16 complex outputs by component assignment, preserving
  infinities, NaNs and signed zero without arithmetic-induced extra NaNs.
- Report reference storage-range violations and actual nonfinite components
  separately. These diagnostics do not relax correctness or floating acceptance.
- Continue unaffected semantic requests in separately fingerprinted shards,
  retaining failed rows and artifacts in an explicit quarantine. Runtime errors
  still stop execution. Historical measurements are not rewritten or relabelled.
- Add a CPU half-arithmetic stage diagnostic and continuation heartbeat reporting.
  The library remains 0.9.2 with the same kernels and calibration binding.
- Add user-approved, separately identified FP16 FFT input-range controls. Their
  power-of-two scaling comes from the complete input L1 bound, with input hashes
  and quantization losses recorded. Original stress tests remain unchanged.

## 0.9.2 - 2026-10-07 Unified Public Semantics

- Added CPU descriptor validation, canonical JSON roundtrip and capability
  reasons shared by the C, C++ and installable Python frontends.
- Preserved independent axis/batch strides and introduced per-axis logical
  order across operators, with a common boundary lowering and native NTT reuse.
- Bound public mapping replay and full-workload cache identity to the same
  semantic record; extended generic composite execution through rank eight.
- Cancelled the preceding comparison campaign while retaining its frozen
  inputs and historical results. No replacement performance plan is implied.

## 0.9 Research Snapshot - 2026-09-20

- Added public mapping and module interfaces, research-time specialization,
  staged service calibration, evolutionary search and cross-operator acceptance.
- Added portable SM70 build, CPU precompilation and target-local research entry
  points with GPU/build/protocol identity checks and resumable results.
- Froze 738 experiment contracts covering batch/precision, inverse transforms,
  cryptographic compositions and a bounded general-shape FFT extension.
- Added a V100 runbook and source packaging with explicit legacy build inputs.
  SM70 compilation and CPU checks passed; new V100 GPU measurements are pending.

## 0.9.0 - 2026-09-09 Layered Cross-Hardware Framework

- Added the v0.9 architecture manifest and configure-time layer validation.
- Separated reusable templates, platform profiles, parameter spaces, search and
  calibration, and runtime selection in the documented architecture contract.
- Kept `v100_*` inputs as explicitly named frozen compatibility evidence while
  target-local GPU measurements remain fingerprinted and non-portable.
- Promoted large-transform CTA/EPT mapping to install-time calibration data.

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
