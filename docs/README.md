# Documentation Index

The [QuaRot fused-path validation and reporting audit](quarot_fused_path_20261107.md)
records matched short-transform measurements and distinguishes supplied-output,
native-return, device-event and host-wall comparison contracts.

The [November 7 GitHub handoff](pro6000_handoff_20261107.md) identifies the target entry, frozen paper inputs and validation boundary for work on the PRO6000 server.

The [PRO6000 Blackwell Server96 runbook](hardware/pro6000_server96.md) covers separate CUDA/MathDx installation, CPU preparation, target-local calibration and quick12/full66 comparison through `scripts/research_platform.py`. Its [population contract](../config/paper_portability/README.md) keeps later paper tracks separate.

The [November 7 A100 status and FFT-L scope update](research_status_20261107.md)
is the current summary. The [November 4 research and paper status](research_status_20261104.md)
retains the earlier scope and evidence; older campaign notes below retain their
original scope.

The [October 25 full benchmark refresh](full_benchmark_refresh_20261025.md)
defines the frozen seven-track rerun, prepared operator inputs, guarded execution
and outstanding application bindings. Its pending state must not be confused
with completed historical measurements.

The [static-profile architecture decision](adr/0001-static-profile-driven-planning.md)
defines shared hardware inputs, default local/bundled reuse and explicit refresh.
The [constraint pipeline audit](constraint_pipeline_audit_20261024.md) separates
the implemented common checks from remaining descriptor and model coverage gaps.
The [common planner validation](common_planner_validation_20261024.md) records
the GPU matrix, controlled tile-prefetch measurements and matched cuFFT results.
The [A100 recalibration record](recalibration_20261024.md) records the completed
finite stage-service calibration, its exact B=12–16 scope, and the remaining
physical launch descriptor gap.
The [physical launch audit](physical_launch_descriptor_audit_20261024.md) identifies
which legacy NTT descriptors still need to distinguish logical roles from launches.

The canonical benchmark scope is declared in
[`benchmark/scopes/`](../benchmark/scopes/). Start with the active
[A100 80GB research scope](../benchmark/scopes/a100_80gb_research_v1.json) and
run `python scripts/validate_benchmark_scope.py benchmark/scopes/a100_80gb_research_v1.json`
before starting or summarizing a comparison.

The [project research principles](research_principles.md) define the shared
design goal and evidence contract for **all butterfly operators**: adaptive
mechanisms, replaceable cores, connected calibration/tuning/selection, two
distinct ablations and matched-batch application comparisons. These principles
are separate from the completion status of any particular experiment.

The [reporting and evidence contract](reporting_contract.md) is the canonical
format for new reports. It separates scope, protocol, results, attribution,
limitations and reproduction, and keeps measured, derived, external,
unsupported and placeholder records distinguishable. The [project and authors](about.md)
page is the source for software attribution and citation metadata.

The [persistent-query and grouped-worker update](async_fast_dispatch_20261010.md)
addresses Plan initialization after the known compilation queue drained. It
records transport parity, CPU overhead and the checked experiment transition.

The [bounded-worker recovery](async_recovery_20261010.md) records the October 10
resource failure, audited continuation, bounded process lifetimes and completed
120 original-device scaled controls. Its CPU preparation follow-up documents
the independent parallel compiler and the remaining adaptive-export boundary.
The full comparison remains in progress.

The [real FFT range-control repair](numeric_range_controls_20261009_v2.md)
audits 120 raw numeric failures, adds the 24 missing R2C/C2R controls and records
their completed A100 40GB diagnostic separately from pending 80GB replay.

The [asynchronous experiment queue](async_experiments_20261009.md) separates CPU
input/oracle preparation, compilation and verification from serialized GPU
measurement. It preserves v0.9.3.4 calibration and checked historical results.

The [0.9.3 bounded support milestone](bounded_support_20261008.md) records the
current workload limits, new arithmetic lowerings, targeted validation and
remaining full-campaign obligations. The earlier
[descriptor-based 0.9.2 retest](semantic_campaign_20261007.md) retains its
original matrix, measurements, calibration scope and comparison decisions.

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

cuButterfly has two contracts: a usable CUDA library and a research artifact
that explains how mappings were generated and measured. Choose a path below;
the long experiment files remain available as source evidence rather than
being repeated in the root README.

The [2026-09-12 all-application method and implementation audit](method_implementation_audit_20260912.md)
records the executable coverage and gaps observed at that audit. Later changes
and pending work are maintained in the current section of
[refactor progress](framework_refactor_progress.md).

The [unified planner guide](unified_planner.md) documents the implemented public
API, compile policy, incremental registry and installation search. The
[refactor progress](framework_refactor_progress.md) separates verified changes
from remaining performance acceptance work.

The [prepared RNS integration guide](rns_prepared_integration.md) explains how
to export actual CKKS NTT requests, prepare measured mappings offline, require
compatible cache replay, and replace a Python NTT callback while preserving the
native application baseline.

The [RNS exchange follow-up](../results/rns_exchange_20261021/README.md) records
the dependency-scoped synchronization option, exact and sanitizer checks,
controlled comparisons, and the remaining small-request/CKKS performance gap.

The [RNS single-warp and CKKS attribution report](../results/rns_single_warp_20261021/README.md)
records 32-thread coverage, allocation-failure recovery, retuned request replay,
complete-Graph profiling and explicit native-dispatch controls. It separates
historical NTT gains from the small/medium requests limiting the current application.

The [RNS stage-role report](../results/rns_stage_role_20261021/README.md)
records compile-time role specialization, retuned real requests, cache
sensitivity controls and complete-Graph attribution. Its six operator points
reach 1.131x geometric mean; complete CKKS bootstrap improves to about 1.033x,
with the remaining small/medium request losses retained.

The [runtime RNS tuning report](../results/rns_tile_io_20261021/README.md)
records the public tuning-budget control and an all-request cache-only replay.
Using 12 bounded tuning repetitions, complete CKKS Graph bootstrap reaches
1.034x at B=1 and 1.033x at B=8 with 118/118 runtime cache hits.

The [NTT mechanism audit](ntt_common_mechanisms_20261023.md) records which
organization mechanisms are genuinely shared with other butterfly operators and
which APPT role, readiness and coefficient-supply protocols remain NTT-specific.
The [NTT APPT variant audit](ntt_appt_variant_audit_20261023.md) maps every
APPT core to its launcher, validation, and portable-search status, with explicit
omission reasons for variants that lack an independent descriptor.

The [weak-region quick replay](weak_regions_quick_20261023.md) records the
post-audit A100 diagnostic matrix and separates automatic vendor whole-FFT
selection from forced local hybrid paths.

The [open-source template exploration](open_source_template_exploration_20261023.md)
records the matched A100 experiments for the shared axis-pair route, the
generated stage-local route, and an actual VkFFT provider plus mixed-provider
Graph. It separates reusable VkFFT/rocFFT/cuFFTDx mechanisms from
complete-library baseline claims.

The [radix-4 core and resident-subgraph experiment](fft_radix4_subgraph_20261024.md)
records corrected mixed-radix mapping, matched-core resident/split controls,
CPU FP32 error distributions, and the shared-memory alignment repair. Its
32×32 FP32 diagnostic results remain separate from public-selector performance.

The [public radix-4 integration report](public_radix4_integration_20261024.md)
records the public xD candidate, strict mapping replay, cache reuse, complete
cuFFT provider baseline, and bounded two-dimensional/three-dimensional results.
It keeps long 1D, FP64, and compact-real pair packing outside the verified range.

The [radix-4 completion report](radix4_completion_20261024.md) extends the public
path to FP64 and compact-real pair packing, and connects mixed-radix register
leaves to long-1D JIT and candidate enumeration. It reports automatic xD
selection separately from fixed-mapping long-1D core replacement, including
negative results and complete conversion costs.

The [FP64 factor-graph exploration](factor_graph_exploration_20261025.md) reports
the follow-up CTA tile-group scan on the existing cuFFTDx block core, including
forward/inverse, B=4/16/64, neighboring factor points, and the remaining gap to
cuFFT. It is an explicit mapping experiment and does not certify AUTO selection.

The [cuFFT Xt strategy audit](cufft_xt_strategy_audit.md) separates Xt's
observable plan, layout and workspace controls from its opaque provider core,
and records the BF16 adapter boundary for future measured experiments.

The [fixed-core organization controls](factor_core_organization_experiment_20261025.md)
separate native/cuFFTDx core identity from tile scheduling and prefetch depth.
They record the async wait correction, GPU correctness and race checks, and the
completed nine-case A100 80GB experiment, including its cuFFT gap. This is a matched
organization diagnostic, not a claim of complete open-source core transfer.

The [template and parameter audit](template_parameter_audit_20261013.md)
records which core and organization dimensions are actually connected across
FFT, NTT, FWHT, subset/superset and structured operators. It distinguishes an
imported provider from an absorbed, parameterized template strategy and lists
the remaining concrete lowering gaps.
The accompanying [core contract implementation record](core_template_contract_20261013.md)
contains the v21 parameter fixes, build identity, focused correctness coverage
and matched application regression results.

The [October 7 engineering review](engineering_review_20261007.md) records
search/statistics checks and the reviewed comparison entry. Preparation is
CPU-only; measurements require matching calibration and an available GPU.

The [public frontend dimension audit](public_frontend_dimension_audit_20261007.md)
checks which semantic dimensions reach public APIs and lowering, identifies
silent C++ field losses, and separates finite regression coverage from the
unfinished general-library contract.

The [0.9.2 unified semantics](unified_semantics.md) documents the repair,
installed Python frontend, common logical-order/layout boundary and explicit
lowering gaps. The preceding performance campaign is cancelled; correctness
regressions do not constitute a replacement performance matrix.

The [xD lowering guide](xd_lowering.md) covers rank-two/rank-three planning,
resident and per-axis candidates, compact real FFT, and the measured selection
and replay contracts introduced in 0.9.1.

## Reading by Task

The [integrated rc.2 repair acceptance plan](integrated_acceptance_rc2.md)
tracks the FFT pipeline/exchange, NTT boundary and FWHT retained-state fixes
under one future build. It distinguishes planned integration, prepared inputs
and measured results; older queues do not establish acceptance of these fixes.

Use [unified planner](unified_planner.md) for current API/lowering/search behavior,
[hardware calibration](hardware_profile_install.md) for migration and experiment
recovery, and [refactor progress](framework_refactor_progress.md) for the active
milestone. [AGENTS.md](../AGENTS.md) records the concise research working conventions.
Per-model and memory-capacity commands and evidence are in the
[hardware notes](hardware/README.md), with a template for adding another GPU.
The [research acceptance workflow](research_acceptance.md) runs searched
cross-operator implementations against matching baselines with resumable trials.
The [portable V100 campaign](hardware/v100.md) prepares a target-local build,
calibrates the target and runs the shared experiment matrix. The
[application coverage roadmap](application_coverage_roadmap.md) separates
existing semantics needing measurement from missing benchmark contracts and new lowerings.
The cuButterfly paper working draft (local artifact: `../paper/README.md`; not included in this source release) organizes the method,
theoretical conditions, frozen V100/A100 evidence and pending experiments in
the supplied ISCA2026 format. Its partial results do not qualify the full matrix.
Consult paper and architecture sections when the task changes method assumptions.

The following is an optional first-time orientation, not a prerequisite for edits:

1. [v0.9 Framework Release](release_v09.md) establishes the layered framework
   and hardware-profile boundary. [v0.8 Release Overview](release_v08.md)
   establishes the frozen V100 evidence scope,
   terminology, evidence labels, and V100 limitations.
2. [Getting Started](getting_started.md) and the [Programming Guide](programming_guide.md)
   explain the user-facing CUDA plan and workspace workflow.
3. [Design Overview](design_overview.md) explains the Roofline-to-hybrid-dataflow
   design argument, and [Hardware Mapping Methodology](hardware_mapping_methodology.md)
   formalizes the two-dimensional space/time unfolding.
4. [v0.8 Nested Physical Space](v0.8_nested_physical_space.md),
   [Architecture Guardrails](architecture_guardrails.md), and
   [Processing-Unit Design Space](processing_unit_design_space.md) define the
   architecture/core boundary and legal lowering contracts.
5. [Unified Planner](unified_planner.md) and [Hardware Profile Initialization](hardware_profile_install.md)
   explain how a target GPU chooses a point in that space.
6. [Reproducibility](reproducibility.md), [V100 Three-Way Comparison](v100_three_way_comparison.md),
   and [Comprehensive Butterfly Comparison](comprehensive_butterfly_comparison.md)
   provide the measurement protocol and performance evidence.

## Visual Overview

- [Architecture map](../figures/cubutterfly_concept.svg) shows the graph,
  unfolding factors, GPU realization, replaceable processing unit, and
  measurement feedback loop.
- [V100 external-library summary](../figures/v100_library_comparison.svg)
  shows the selected-matrix throughput ratios behind the headline comparison.
- The editable architecture source is
  [`../figures/cubutterfly_concept.dot`](../figures/cubutterfly_concept.dot).

## Use The Library

| Need | Document |
|:--|:--|
| Short overview and a verified workload | [Summary And Example](summary_example.md) |
| Build and first run | [Getting Started](getting_started.md) |
| C plan and device API | [C Plan API](c_api.md), [Device API](device_api.md) |
| C++ types and fields | [API Reference](api_reference.md) |
| Streams, workspaces, layouts, lifetime | [Programming Guide](programming_guide.md) |
| Errors and unsupported combinations | [Error Handling](error_handling.md) |
| Supported operators and numeric forms | [Capability Matrix](support_matrix.md), [Operator Catalog](operator_catalog.md) |
| Examples | [Examples](examples.md) |
| Current runtime selection | [Unified Planner](unified_planner.md) |
| Platform tuning priors and identity | [Platform Tuning Rules](platform_tuning.md) |
| Multidimensional transforms and compact real FFT | [xD Lowering](xd_lowering.md) |
| xD lowering prior art and adopted mechanisms | [xD Related Work](xd_lowering_related_work.md) |
| Unified external comparison targets | [Comparison Catalog](../config/external_comparison_catalog.json) |
| Historical generated selector | [Runtime Mapping Selector](runtime_selector.md) |
| Unified runtime migration and remaining gaps | [Mixed-Dataflow Runtime](mixed_dataflow_runtime.md) |
| Install-time GPU calibration | [Hardware Profile Initialization](hardware_profile_install.md) |

## Understand The Method

| Question | Document |
|:--|:--|
| What is the architecture claim? | [Design Overview](design_overview.md) |
| How do both dimensions unfold in space and time? | [Hardware Mapping Methodology](hardware_mapping_methodology.md) |
| How are graph, mapping, core, layout, and hardware separated? | [Complete Butterfly Design Space](butterfly_design_space.md) |
| How are logical segments lowered to physical groups? | [v0.8 Nested Physical Space](v0.8_nested_physical_space.md) |
| How are arbitrary physical group counts ranked? | [Performance Model](performance_model.md) |
| Which local arithmetic cores are available? | [Processing-Unit Design Space](processing_unit_design_space.md), [Candidate Implementations](implementation_candidates.md) |
| How are FFT stages generated and dispatched? | [FFT Design Space](fft_design_space.md), [FFT Pipeline Generator](fft_pipeline_generator.md) |
| What must an implementation preserve? | [Architecture Guardrails](architecture_guardrails.md) |
| How does APPT-style online layout fit the architecture? | [APPT Static Layout](appt_static_layout.md) |
| How does cuButterfly relate to prior work and libraries? | [Research Positioning](cubutterfly_positioning.md), [FFT Library Comparison](fft_library_comparison.md) |
| Where has the method been migrated to another platform? | [Cross-Platform Ports](platform_ports.md) |

## Reproduce The Evidence

| Evidence | Document or entry point |
|:--|:--|
| v0.8 frozen release summary | [v0.8 Release Overview](release_v08.md) |
| Current research status and recovery | [Refactor Progress](framework_refactor_progress.md) |
| Earlier research status and limitations | [Research Status](research_status.md) |
| All-operator protocol | [Comprehensive Butterfly Comparison](comprehensive_butterfly_comparison.md) |
| v0.6/v0.8 and library reconciliation | [Library Reconciliation](v08_v06_cufft_reconciliation.md) |
| V100 library/base/search matrix | [V100 Three-Way Comparison](v100_three_way_comparison.md) |
| V100 external baselines | [V100 External Baselines](v100_external_baselines.md) |
| Length and batch scaling | [V100 Scaling](v100_scaling_results.md) |
| NCU methodology | [NCU Profiling](ncu_profiling.md) |
| Reproduction commands and evidence labels | [Reproducibility](reproducibility.md) |
| Report structure and provenance | [Reporting and Evidence Contract](reporting_contract.md) |
| Project attribution and citation | [Project and Authors](about.md) |
| Hardware capability and model generation | [Hardware Profile Initialization](hardware_profile_install.md) |
| Raw and reduced artifacts | [`../results/README.md`](../results/README.md) |

Run the broad timing matrix with:

```bash
python3 scripts/run_cross_operator_comparison.py --help
```

Run the privileged counter collection only on a machine with Nsight Compute:

```bash
sudo -E ./scripts/profile_cross_operator_v08_ncu.sh
```

## Research Positioning

[Research Positioning](cubutterfly_positioning.md) explains the distinction
between this architecture mapping claim and operator-specific cores such as
cuFFTDx, TurboFFT, Dao FHT, and GPU-NTT. The project does not claim that the
arithmetic unit itself is novel in every case; its contribution is the
parameterized mapping, residence, dataflow, lowering, and measured selection
framework around regular layered butterfly graphs.

For direct library tables and protocol notes, continue to [FFT Library
Comparison](fft_library_comparison.md) and [V100 External Baselines](v100_external_baselines.md).

## Release And Maintenance

- [Changelog](../CHANGELOG.md) records versioned implementation and evidence changes.
- [Roadmap](../ROADMAP.md) separates frozen v0.8 work from future cross-GPU validation.
- [Contributing](../CONTRIBUTING.md) defines correctness, benchmark, and provenance requirements.
- [Citation](../CITATION.cff) and [License](../LICENSE) contain the redistribution metadata.

## Evidence Labels

| Label | Meaning |
|:--|:--|
| `measured` | Direct result from a checked-in command on the stated hardware. |
| `derived` | Computed from measured values with a documented reduction. |
| `external` | Third-party result under a matched semantic and timing protocol. |
| `placeholder` | Future hardware or schema entry without a performance claim. |

V100 remains the fully measured, reproducible GPU in the v0.8 release
boundary. Other GPUs can now generate target-local profiles during
installation; those local measurements do not change the frozen V100 baseline
or imply portable performance claims.
