# Composition calibration repair, 0.9.3.2

The earlier calibration completed its selected independent-stage measurements.
It did not measure native cross-group overlap combinations. Those scopes must
not share a single "calibration complete" claim.

The public search already had overlap choices. The missing connection was the
calibration input: finalization replayed retained mappings, whose overlap flags
were disabled, without constructing paired composition requests. A stage service
curve alone cannot establish stream/event overhead or concurrent SM/HBM sharing.
This is a calibration coverage gap, not evidence of a theoretical failure or
proof that overlap must improve every workload.

## Implemented measurement contract

The reusable [inventory builder](../scripts/composition_inventory.py) consumes
verified operator and stage descriptors. It preserves physical partitions and
semantic fields, then constructs serial/overlap pairs for shared-iteration and
online-reorder execution. Legal factor-streamed requests additionally compare
serial slices against overlapped slices. Single physical groups and factor
configurations without meaningful overlap are explicitly omitted.

The [runner](../scripts/calibrate_composition.py) freezes independent-stage
predictions before timing, verifies every batch, and retains commands, raw
outputs, all trials and resume identities. Repeated bulk designs are measured
once and referenced by their pairs. Model error statistics count each distinct
execution once. Positive and negative speedups remain in the same report.

Staged installation and fresh recalibration invoke this runner. The migration
manifest distinguishes observed timing, prediction validation, inventory coverage
and unidentified concurrency parameters. A finite sample cannot certify a full
inventory. Factor-sliced services still lack an independent projection and are
reported as uncovered. No overlapping whole-plan measurement trains its own
holdout prediction, and no concurrency-calibrated flag is fabricated.

## Current A100 preparation

The [frozen plan](../results/composition_calibration_20261008_v0932/plan.json)
contains **168 requests in 84 pairs** derived from the existing verified
measurements. It covers online FFT using cuFFTDx-block and register-tile cores,
plus shared-iteration UINT64 subset and superset zeta. Original and additional
tail workloads are labelled separately. Each workload stays within the declared
`N * batch <= 2^20` limit; no original request is silently reduced.

The inventory still lacks feasible verified seeds for shared-iteration FFT,
FWHT, NTT, XOR-zeta and structured 2x2, and for FP32/FP64 factor streaming under
this bound. These are missing calibration samples, not claims that their public
operators are unsupported. Independent factor-slice service modelling is also
unfinished. Those gaps are retained in the machine-readable coverage report.

The target is A100 80GB, UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`.
The library remains the frozen 0.9.3.1 build, SHA-256
`eef2fe786c3b23df3c72a8b72761c9beef3f1b148f1d075ac6a176b6bb2a9eea`.
Revision 0.9.3.2 identifies the new calibration tooling, not a kernel change.

The [queued entry](../results/composition_calibration_20261008_v0932/run_repair.py)
waits for the active conformance child to finish. It holds only the owned pipeline
parent so the next GPU phase cannot race with calibration. On completion or
handled failure it releases the parent to continue scaled controls and comparison.
It does not wait for or interrupt foreign GPU workloads; an exclusivity failure
stops this diagnostic. The original campaign profile, registry, manifest and
measurements remain unchanged. New results stay in their separate output directory.

Read [queue status](../results/composition_calibration_20261008_v0932/queue_status.json)
for execution state. Preparation and CPU regression success are not GPU results.
The expected report is `measurements/composition_calibration.json`, with measured
overlap benefit, model ranking, relative error and explicit uncovered cases.
Floating diagnostic correctness remains separate from final accuracy acceptance.

## Verification

The affected calibration, inventory, entrypoint, scheduling and resume checks
pass **132 CPU tests** in `/home/wt/yes/envs/cubutterfly`. Checks include structured
matrix and NTT semantics, exact UINT64 inputs, nondivisible tiles, identical-mapping
cache reuse, changed-runtime rejection, negative overlap benefit and incomplete
coverage gates. GPU combination timing remains pending; no new speedup is claimed.
