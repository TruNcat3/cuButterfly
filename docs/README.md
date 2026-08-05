# Documentation Index

The documentation has two layers. The library layer starts from installation,
plans, streams, workspace, errors, and API contracts. The research layer
separates the architecture claim from kernel implementation and measurement
evidence. Application users should begin with [Getting
Started](getting_started.md); architecture and paper readers should begin with
[Design Overview](design_overview.md).

## Library Documentation

| Document | Question answered |
|:--|:--|
| [Getting Started](getting_started.md) | How is the library configured, built, installed, tested, and called? |
| [Programming Guide](programming_guide.md) | What are the plan, device, stream, layout, workspace, concurrency, and lifetime contracts? |
| [C++ API Reference](api_reference.md) | What does each public type, configuration field, and plan member mean? |
| [Error Handling](error_handling.md) | Which exceptions are reported synchronously and where can asynchronous CUDA errors surface? |
| [Capability and Compatibility Matrix](support_matrix.md) | Which operators, types, toolchains, optional components, and runtime features are supported? |
| [Operator Catalog](operator_catalog.md) | What does each transform compute, which mappings does it reuse, and which operators come next? |
| [Examples](examples.md) | Which minimal, build-checked example matches each integration style? |
| [Device API](device_api.md) | What is the shortest device-pointer integration recipe? |
| [Runtime Mapping Selector](runtime_selector.md) | How does a supported workload resolve to a calibrated mapping? |

## Reading Paths

**Application integration.** Follow [Getting Started](getting_started.md), run
the [Examples](examples.md), then use the [Programming
Guide](programming_guide.md) and [C++ API Reference](api_reference.md) as the
contract. Check the [Capability and Compatibility Matrix](support_matrix.md)
before depending on an optional backend, GPU, or execution feature.

**Architecture and paper review.** Begin with the [Design
Overview](design_overview.md) for the `D x S` iteration domain and its four
space-time factors. Continue with [Hardware Mapping
Methodology](hardware_mapping_methodology.md) for resource demand and selection,
then [Complete Butterfly Design Space](butterfly_design_space.md) for the formal
object boundaries. [Research Positioning](cubutterfly_positioning.md) and
[V100 Research Status](research_status.md) delimit the contribution and claims.

**Kernel and generator development.** Read [FFT Design
Space](fft_design_space.md), [FFT Pipeline
Generator](fft_pipeline_generator.md), and [Processing-Unit Design
Space](processing_unit_design_space.md). Together they show how a logical stage
decomposition is lowered into physical execution groups, how online layout
boundaries are generated, and where operator-specific codelets plug in.

**Experimental reproduction.** Start with
[Reproducibility](reproducibility.md), regenerate the derived tables, and use
[Single-GPU Comprehensive Benchmark](comprehensive_benchmark.md) only when new
timings are needed. The focused NCU reports explain mechanisms; the
comprehensive and scaling reports define the paper-facing comparison protocol.

## Primary Documents

| Document | Question answered |
|:--|:--|
| [Design Overview](design_overview.md) | Why are data and stage work each unfolded in space and time, and how does that map to a GPU? |
| [Hardware Mapping Methodology](hardware_mapping_methodology.md) | How do the unfolding factors translate into GPU resource demand? |
| [Complete Butterfly Design Space](butterfly_design_space.md) | How are graph, architecture, processing unit, layout, realization, selection, and hardware separated? |
| [FFT Design Space](fft_design_space.md) | Which semantic, factorization, two-axis mapping, processing-unit, and hardware parameters are enumerable? |
| [FFT Pipeline Generator](fft_pipeline_generator.md) | How are stage partitions, local units, boundaries, measured search, and runtime dispatch separated? |
| [Candidate Implementations](implementation_candidates.md) | Which processing units and kernel forms are implemented or planned? |
| [Getting Started](getting_started.md) | How is the project built and used? |
| [Programming Guide](programming_guide.md) | How do plans, layouts, streams, workspaces, devices, concurrency, and determinism compose? |
| [C++ API Reference](api_reference.md) | What is the contract of each installed public API type and member? |
| [Error Handling](error_handling.md) | How are validation, backend, CUDA launch, and asynchronous failures reported? |
| [Capability and Compatibility Matrix](support_matrix.md) | Which public semantics, optional builds, and runtime integrations are supported? |
| [Operator Catalog](operator_catalog.md) | Which local pair updates are implemented and how do they share the architecture mapping? |
| [Examples](examples.md) | Which build-checked program demonstrates each primary API path? |
| [Device API](device_api.md) | How are plans, caller-owned CUDA memory, streams, and workspaces composed in an application? |
| [Runtime Mapping Selector](runtime_selector.md) | How does a semantic workload resolve to a measured V100 mapping and an auditable decision? |
| [Experimental Results](experiments.md) | Which experiment tests each architecture claim, what has been measured, and where are the gaps? |
| [Single-GPU Comprehensive Benchmark](comprehensive_benchmark.md) | How are lengths, precision, semantics, implementations, and external baselines compared together? |
| [V100 Comprehensive Results](comprehensive_v100_results.md) | What does the controlled full-suite comparison currently establish? |
| [V100 Length/Batch Scaling](v100_scaling_results.md) | How do saturation and the best mapping change when length and batch are swept independently? |
| [V100 Mapping Selector](v100_mapping_selector.md) | How accurately can the measured V100 mappings be selected with a held-out shape? |
| [V100 Counter Attribution](v100_ncu_attribution.md) | Which hardware services explain the observed saturation and mapping crossovers? |
| [Structured 2x2 V100 Results](structured_2x2_v100_results.md) | Does a parameterized dense pair unit preserve the mapping methodology across precision and length? |
| [V100 External Baselines](v100_external_baselines.md) | What do matching-protocol Dao FHT and GPU-NTT comparisons establish? |
| [V100 Research Status](research_status.md) | Which claims are closed, what are the current performance boundaries, and what remains? |
| [v0.3.0 Mapping-Selection Milestone](next_phase_v0.3.md) | Which falsifiable goal, execution order, and definition of done shaped this release? |
| [Numeric-Regime Mapping Study](next_phase_numeric_regimes.md) | How will precision, arithmetic, length, batch, layout, and core choice be related to resource cliffs and mapping preference? |
| [Reproducibility](reproducibility.md) | How are sweeps, external baselines, and profiler data reproduced? |
| [Research Positioning](cubutterfly_positioning.md) | What is the intended contribution relative to prior work? |

Repository maintenance and research priorities are documented in
[`CONTRIBUTING.md`](../CONTRIBUTING.md) and [`ROADMAP.md`](../ROADMAP.md).

## Architecture And Mapping

- [Hybrid2D GPU Mapping](hybrid2d_architecture.md): Cooley-Tukey/NTT
  factorization and current CUDA realization.
- [Processing-Unit Design Space](processing_unit_design_space.md): local
  arithmetic, coefficient, radix, and exchange choices.
- [Generated Processing Units](generated_design_points.md): build-time
  specialization for register DFT8, CTA DFT8, WMMA, register FWHT, and
  register-resident Structured 2x2.
- [Processing-Core Integration](core_integration_strategy.md): boundary between
  an imported codelet and a complete external library.
- [Feature Coverage](cubutterfly_feature_coverage.md): implemented semantic and
  backend matrix.

## Measurement Reports

- [V100 Comprehensive Results](comprehensive_v100_results.md)
- [V100 Length/Batch Scaling](v100_scaling_results.md)
- [V100 Mapping Selector](v100_mapping_selector.md)
- [V100 Counter Attribution](v100_ncu_attribution.md)
- [V100 Matching-Protocol External Baselines](v100_external_baselines.md)
- [V100 Initial NTT Results](v100_initial_results.md)
- [V100 NTT Parameter Matrix](v100_matrix_results.md)
- [GPU-NTT Gap Analysis](gpu_ntt_gap_analysis.md)
- [Cross-Operator Results](cubutterfly_cross_operator_results.md)
- [Structured 2x2 V100 Results](structured_2x2_v100_results.md)
- [Structured 2x2 Counter Attribution](../results/ncu_structured_2x2_attribution.md)
- [Large-Length Results](cubutterfly_large_results.md)
- [CTA DFT8 Space-Time Mapping](fft_cta_space_time_results.md)
- [FFT Library Comparison](fft_library_comparison.md)
- [Nsight Compute Profiling](ncu_profiling.md)
- [Cross-GPU Experiment Matrix](cross_gpu_experiment.md)

## Evidence Labels

Every performance statement should be read under one of these labels:

| Label | Meaning |
|:--|:--|
| measured | Produced on the stated machine using a checked-in command and raw record |
| derived | Computed from measured counters or timing using a stated equation |
| external | Same-machine result from a pinned third-party implementation |
| placeholder | Capture schema only; no performance value is claimed |

V100 is the only complete measured hardware profile in this revision.
