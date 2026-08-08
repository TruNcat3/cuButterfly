# Roadmap

This roadmap separates implementation goals from evidence required for the
research claim. Ordering may change after profiling or access to new hardware.

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
