# Documentation

cuButterfly combines a common butterfly organization method with replaceable
arithmetic cores and hardware-aware plan generation. Choose a route by task.

## Use the library

[Getting started](getting_started.md) → [Public planner](unified_planner.md) →
[Support contract](support_matrix.md).

Use the [C plan API](c_api.md), [C++ reference](api_reference.md) and
[device API](device_api.md) for integration. The [operator catalog](operator_catalog.md),
[examples](examples.md) and [error handling](error_handling.md) describe requests
and execution. [xD lowering](xd_lowering.md) covers multidimensional transforms;
[prepared RNS integration](rns_prepared_integration.md) covers finite-field
application integration.

## Understand the method

[Design overview](design_overview.md) → [Research principles](research_principles.md) →
[Hardware mapping methodology](hardware_mapping_methodology.md) →
[Butterfly design space](butterfly_design_space.md).

The [performance model](performance_model.md) connects resource demand to
service estimates. [Processing-unit contracts](processing_unit_design_space.md)
and [core integration](core_integration_strategy.md) explain how local arithmetic
is replaced while preserving semantics. [Research positioning](cubutterfly_positioning.md)
and [platform ports](platform_ports.md) place the method in context.

## Configure hardware and selection

[Hardware profiles](hardware/README.md) → [Installation and calibration](hardware_profile_install.md) →
[Platform tuning rules](platform_tuning.md).

Hardware capacities, measured services and compiled resources have distinct
roles. The [static-profile policy](adr/0001-static-profile-driven-planning.md)
defines compatible reuse and explicit refresh; the [planner guide](unified_planner.md)
defines compilation, selection, tuning and replay.

## Evaluate and reproduce

[Benchmark guide](../benchmark/README.md) → [Measurement and reporting](reporting_contract.md) →
[Reproducibility](reproducibility.md).

Use configuration-driven workloads for operator comparisons, controlled
organization experiments and applications. [Profiling](ncu_profiling.md) explains
how counters and traces complement complete-call measurements.

<a id="read-the-a100-paper-evidence"></a>

## Read the paper and results

[Paper](../paper/main.pdf) → [Paper guide](../paper/README.md) →
[Experimental results](benchmarks/results.md).

Results identify the hardware, workload, numerical contract and timing protocol
for each comparison. The [evidence index](../paper/evidence/README.md) links frozen
inputs and derivations; [terminology](../paper/TERMINOLOGY.md) aligns the method
and its implementation.

## Extend and maintain

[Contributing](../CONTRIBUTING.md) → [Engineering map](development/repository-layout.md) →
[Documentation maintenance](development/documentation.md).

The engineering map locates semantics, organization constraints, cores, hardware
profiles, generation and evaluation. The [v0.9 overview](release_v09.md) describes
the release family. [Project and authors](about.md) maintains attribution and
citation; the [changelog](../CHANGELOG.md) and [archive](archive/README.md) retain
version history and development records.
