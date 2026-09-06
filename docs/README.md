# Documentation Index

cuButterfly has two contracts: a usable CUDA library and a research artifact
that explains how mappings were generated and measured. Choose a path below;
the long experiment files remain available as source evidence rather than
being repeated in the root README.

## Recommended Reading Order

Read the repository in this order when encountering it for the first time:

1. [v0.8 Release Overview](release_v08.md) establishes the frozen scope,
   terminology, evidence labels, and V100 limitations.
2. [Getting Started](getting_started.md) and the [Programming Guide](programming_guide.md)
   explain the user-facing CUDA plan and workspace workflow.
3. [Design Overview](design_overview.md) and [Hardware Mapping Methodology](hardware_mapping_methodology.md)
   introduce the two-dimensional space/time unfolding.
4. [v0.8 Nested Physical Space](v0.8_nested_physical_space.md),
   [Architecture Guardrails](architecture_guardrails.md), and
   [Processing-Unit Design Space](processing_unit_design_space.md) define the
   architecture/core boundary and legal lowering contracts.
5. [Runtime Selector](runtime_selector.md) and [Hardware Profile Initialization](hardware_profile_install.md)
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
| Build and first run | [Getting Started](getting_started.md) |
| C plan and device API | [C Plan API](c_api.md), [Device API](device_api.md) |
| C++ types and fields | [API Reference](api_reference.md) |
| Streams, workspaces, layouts, lifetime | [Programming Guide](programming_guide.md) |
| Errors and unsupported combinations | [Error Handling](error_handling.md) |
| Supported operators and numeric forms | [Capability Matrix](support_matrix.md), [Operator Catalog](operator_catalog.md) |
| Examples | [Examples](examples.md) |
| Runtime selection | [Runtime Mapping Selector](runtime_selector.md) |
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

## Reproduce The Evidence

| Evidence | Document or entry point |
|:--|:--|
| v0.8 frozen release summary | [v0.8 Release Overview](release_v08.md) |
| Research status and limitations | [Research Status](research_status.md) |
| All-operator protocol | [Comprehensive Butterfly Comparison](comprehensive_butterfly_comparison.md) |
| v0.6/v0.8 and library reconciliation | [Library Reconciliation](v08_v06_cufft_reconciliation.md) |
| V100 library/base/search matrix | [V100 Three-Way Comparison](v100_three_way_comparison.md) |
| V100 external baselines | [V100 External Baselines](v100_external_baselines.md) |
| Length and batch scaling | [V100 Scaling](v100_scaling_results.md) |
| NCU methodology | [NCU Profiling](ncu_profiling.md) |
| Reproduction commands and evidence labels | [Reproducibility](reproducibility.md) |
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

V100 is the only fully measured GPU in the v0.8 release boundary.
