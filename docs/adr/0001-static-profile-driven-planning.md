# ADR 0001: Static profiles supply platform facts

Date: 2026-10-24. Status: Accepted. Scope: all butterfly operators.

## Context

Joint choices of partition, tile, parallelism, buffering and core must be
constrained by the target hardware. Repeated live queries, per-operator
capacity rules and unconditional install-time profiling make this contract
hard to inspect and reuse. Existing calibration files also mix hardware
capacity, sampled rates and model readiness, although these have different
validity conditions.

The project owner requested reusable static files generated during installation
or manual profiling. Compatible repository profiles should be used by default;
updating an existing profile must be an explicit operation.

## Decision

Keep four distinct inputs. `hardware_model.json` holds exact device identity
and capacity constraints. Existing hardware/stage profiles and cost-model files
hold measured performance and their coverage. Core declarations specify legal
arithmetic and mapping parameters. Compiled module manifests supply actual
registers, static/shared scratch and supported specializations.

Use explicit, local, then bundled compatible artifacts. Match capacity files
by device model, compute capability and memory capacity; preserve UUIDs as
observation provenance. Performance and compiled artifacts additionally retain
their build, core, protocol and semantic identities. A matching hardware file
does not make an old cost model current or validated.

Installation reuses matching artifacts without launching calibration. Missing
targets may initialize the declared calibration scope. `--refresh-profile`
requests replacement; `--resume-search` requests continuation of measurement.
Ordinary installation must not overwrite local profiles with bundled seeds.
Legacy partial profiles remain visible and reusable as partial data.

The common candidate flow combines parameters, applies dependency/core rules,
then compares resource demand against the static capacities. An infeasible
combination is removed, not its entire buffering or tiling mechanism. Unknown
compiler resources require compilation and feedback before execution. These
universal rules remain code; hardware model names and measured thresholds do
not become scattered branches. Live identity and launch-safety checks remain
permitted and do not trigger performance profiling.

## Consequences and implementation status

Static capacity loading, strict identity validation, repository A100 80GB
capacity/rate seeds and install reuse are implemented. New hardware acquisition
exports a separate capacity file. CPU tests cover reuse, refresh, invalid and
partial artifacts, and resource rejection.

Rank-one production constructors now call the common validator, and the resource
estimator consumes each physical group's compiler feedback. Registry misses use
static cost ranking with bounded compilation and reranking. Missing model or
physical descriptor coverage remains explicit; it does not turn an executable
fallback into a calibrated selection. The
[constraint audit](../constraint_pipeline_audit_20261024.md) records implementation
coverage and verification separately. No performance claim follows from this
architecture decision alone.

Maintaining static artifacts adds an explicit compatibility responsibility.
Missing layers must be reported, not silently remeasured or labelled complete.
The [installation guide](../hardware_profile_install.md) records operational use.
