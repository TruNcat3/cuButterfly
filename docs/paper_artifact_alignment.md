# Paper-to-artifact alignment

This page maps the four experiment families used by the current evaluation to
the anonymous artifact. It is an index of reproducibility entry points, not a
claim that every declared request has been measured. A declared matrix, a
frozen snapshot, and a new target measurement have different status and must
remain separate. The artifact intentionally publishes the experiment
contracts and compact evidence needed to reproduce or audit the claims; it
does not bundle the manuscript or private host-specific result directories.

| Evaluation family | Artifact entry points | What can be reproduced or inspected | Current boundary / gap |
| --- | --- | --- | --- |
| **Fixed-kernel organization** | [`config/benchmark_populations/mechanism_profile_controlled.json`](../config/benchmark_populations/mechanism_profile_controlled.json), [`scripts/prepare_current_mechanism_cohort.py`](../scripts/prepare_current_mechanism_cohort.py), [`scripts/run_controlled_experiment.py`](../scripts/run_controlled_experiment.py), [`scripts/profile_current_mechanism.py`](../scripts/profile_current_mechanism.py), and [`experiments/tools/prepare_scalar_organization.py`](../experiments/tools/prepare_scalar_organization.py) | The controlled population fixes the arithmetic realization while changing grouping. The journal and profile tools preserve mappings, timing boundaries, correctness, and resource records. | The public population is a bounded control set. The exact frozen records used for a particular paper panel remain evidence snapshots; a new target must regenerate them with its own build and profile. |
| **Complete FFT/NTT/FWHT comparison** | [`config/paper_portability/full.json`](../config/paper_portability/full.json), [`benchmark/README.md`](../benchmark/README.md), [`config/benchmark_full.json`](../config/benchmark_full.json), [`config/external_comparison_catalog.json`](../config/external_comparison_catalog.json), [`scripts/research_platform.py`](../scripts/research_platform.py), and [`benchmark/scopes/`](../benchmark/scopes/) | The portable main66 population, CPU-normalized denominators, external-provider contracts, and target-local calibration form the operator comparison route. `scripts/run_external_baseline_suite.py` supplies matching GPU/CPU comparator adapters when installed. | `full` is a request inventory, not proof of coverage. Missing providers, unsupported precision/layouts, and unavailable baselines remain explicit. The checked-in `figures/cpu_normalized_performance.*` is an A100 frozen orientation artifact; it is not a new measurement for another GPU. |
| **End-to-end applications** | [`config/benchmark_full.json`](../config/benchmark_full.json) (`e2e` cases), [`experiments/tools/prepare_end_to_end.py`](../experiments/tools/prepare_end_to_end.py), [`experiments/tools/run_application_batch_20261012.py`](../experiments/tools/run_application_batch_20261012.py), [`experiments/tools/export_application_closeout.py`](../experiments/tools/export_application_closeout.py), [`scripts/benchmark_quarot_contracts.py`](../scripts/benchmark_quarot_contracts.py), [`scripts/benchmark_xd.py`](../scripts/benchmark_xd.py), and [`experiments/snapshots/current_refresh/`](../experiments/snapshots/current_refresh/) | Preparation, complete-call timing contracts, output checks, and application closeout summaries are available. The snapshot retains CKKS, QuaRot, filter, bootstrap and Poisson/FNO evidence as separate cohorts. | Some application runners require pinned external dependencies or omitted historical raw inputs. A clean checkout can inspect the frozen compact evidence and prepare the contracts, but cannot silently recreate those missing dependencies. Application rows must not be pooled with operator cohorts. |
| **Resource / boundary analysis** | [`config/benchmark_populations/mechanism_profile_controlled.json`](../config/benchmark_populations/mechanism_profile_controlled.json), [`docs/hardware_mapping_methodology.md`](hardware_mapping_methodology.md), [`docs/performance_model.md`](performance_model.md), [`experiments/tools/run_boundary_io_control.py`](../experiments/tools/run_boundary_io_control.py), [`experiments/tools/run_boundary_bank_control.py`](../experiments/tools/run_boundary_bank_control.py), and [`experiments/tools/export_boundary_io_control.py`](../experiments/tools/export_boundary_io_control.py) | Dependency/resource reasoning, fixed-core profiles, boundary I/O controls and recorded resource decisions connect logical stage boundaries to measurable complete-call costs. | Analytical bounds and profiler records do not establish a universal hardware limit or cross-architecture transfer. Recalibration is required for a new GPU; compiled-resource and spill behavior remain target-specific. |

## Paper evidence map

The current manuscript uses the following evidence order. The names below are
stable artifact concepts rather than manuscript file paths.

| Paper evidence | Canonical artifact record | Status in a clean checkout |
| --- | --- | --- |
| Motivation and model | [`docs/design_overview.md`](design_overview.md), [`docs/butterfly_design_space.md`](butterfly_design_space.md), [`docs/mixed_dataflow_runtime.md`](mixed_dataflow_runtime.md), [`docs/xd_lowering.md`](xd_lowering.md) | Readable and inspectable; no GPU required |
| Fixed-core organization | `mechanism_profile_controlled.json` plus controlled-study journal/profile tools | Preparation and contract checks are runnable; target timing requires a GPU |
| Complete operator comparison | `paper_portability/full.json`, `benchmark_full.json`, CPU baseline guide and external comparator registry | Manifest expansion and CPU preparation are runnable; provider coverage is target- and dependency-dependent |
| End-to-end propagation | `benchmark_full.json` e2e cases plus `current_refresh/` snapshots and application tools | Generic contracts are runnable; historical CKKS/QuaRot/FNO records remain frozen evidence unless optional dependencies and raw inputs are supplied |
| Resource and boundary attribution | mechanism population, boundary control tools, and compact snapshots | Offline metadata checks are runnable; profiler collection requires the target GPU and profiling permissions |

## Reproduction order

1. Read the [benchmark contract](../benchmark/README.md) and choose a bounded
   manifest ([`config/benchmark_quick.json`](../config/benchmark_quick.json)) or
   the declared full inventory ([`config/benchmark_full.json`](../config/benchmark_full.json)).
2. Validate a frozen scope with
   `python scripts/validate_benchmark_scope.py <scope.json>` when referenced
   result files are available. Use `--source-only` only for source-level checks.
3. Prepare CPU inputs and application queues before GPU execution; follow the
   [experiment guide](../experiments/README.md) and target-local profile docs.
4. Report each result with semantic scope, hardware/build identity, timing
   boundary, and status (`measured`, `derived`, `external`, `unsupported`,
   `diagnostic`, or `placeholder`).

The artifact intentionally contains no manuscript source or author metadata.
Historical platform material is retained only for provenance and must not be
relabeled as evidence for the current target. Files whose names contain
`v100` are archived historical studies; they are not part of the current
A100/PRO 6000 comparison contract.
