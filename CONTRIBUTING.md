# Contributing

Preserve the public semantic contract when adding operators, arithmetic cores or
hardware profiles. Keep semantics, hardware constraints, core capabilities and
selection policy separate. Do not silently change precision, shape, layout or
normalization to satisfy a missing implementation.

Run affected CPU correctness and configuration checks before GPU experiments.
Performance changes need matching complete-call measurements with compilation and
plan initialization excluded. Retain negative results and unavailable baselines.

See the [documentation](docs/README.md), [planner](docs/unified_planner.md) and
[benchmark contracts](benchmark/README.md) for extension and validation interfaces.
