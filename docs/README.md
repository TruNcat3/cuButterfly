# Documentation Index

The documentation is organized by abstraction level. Start with the first row
for the research idea, then move toward implementation and evidence.

## Primary Documents

| Document | Question answered |
|:--|:--|
| [Design Overview](design_overview.md) | What is the reusable space-time parallel paradigm? |
| [Hardware Mapping Methodology](hardware_mapping_methodology.md) | How do the unfolding factors translate into GPU resource demand? |
| [Complete Butterfly Design Space](butterfly_design_space.md) | How are graph, architecture, processing unit, layout, realization, selection, and hardware separated? |
| [FFT Design Space](fft_design_space.md) | Which semantic, two-dimensional, processing-unit, and hardware axes are enumerable? |
| [Candidate Implementations](implementation_candidates.md) | Which processing units and kernel forms are implemented or planned? |
| [Getting Started](getting_started.md) | How is the project built and used? |
| [Experimental Results](experiments.md) | What has been measured, and where are the gaps? |
| [Single-GPU Comprehensive Benchmark](comprehensive_benchmark.md) | How are lengths, precision, semantics, implementations, and external baselines compared together? |
| [V100 Comprehensive Results](comprehensive_v100_results.md) | What does the controlled full-suite comparison currently establish? |
| [V100 Length/Batch Scaling](v100_scaling_results.md) | How do saturation and the best mapping change when length and batch are swept independently? |
| [V100 Mapping Selector](v100_mapping_selector.md) | How accurately can the measured V100 mappings be selected with a held-out shape? |
| [V100 Counter Attribution](v100_ncu_attribution.md) | Which hardware services explain the observed saturation and mapping crossovers? |
| [V100 External Baselines](v100_external_baselines.md) | What do matching-protocol Dao FHT and GPU-NTT comparisons establish? |
| [Next Phase: v0.3.0](next_phase_v0.3.md) | What is the next falsifiable goal, execution order, and definition of done? |
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
  specialization for register DFT8, CTA DFT8, WMMA, and register FWHT.
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
