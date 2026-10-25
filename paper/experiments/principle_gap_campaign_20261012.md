# Finite Principle Gap Campaign, 2026-10-12

This contract implements [the repository research principles](../../docs/research_principles.md).
The JSON alongside this file is the machine-readable task and budget inventory.
Preparation is CPU-only. Nothing in a prepared manifest claims runtime feasibility.

Execution update: the four local layout pairs now have three trials per variant.
The initial attempt exposed two pending compiled modules; v2 replayed only the
three incomplete pairs after CPU readiness checks. See
[the actual run audit](../evidence/PRINCIPLE_GAP_RUN_20261012.md) for the partial
v1 gate, clean v2 gate, negative results and remaining floating error-report gap.
The commands below document the original frozen preparation; do not relaunch
completed pairs to reproduce a planning step.

## Reuse First

Reuse the completed standalone radix16 FFT leaf-core factorial on A100 80GB and
40GB. Its two leaf identities are `NativeCodelet<4,false>` and
`DxCodelet<4,false>` in stagewise HBM and resident-prefix-2 organizations.
Keep this historical build and hardware cohort separate from current public
planner timing. N256/B1 has different block sizes between organizations; the
fixed contract is the arithmetic leaf, not identical occupancy or launch shape.

Reuse accepted FWHT FP32 and NTT word64 historical E3 rows from
`results/paper_completion_20260918/e3_cross_operator/summary.json`: N4096,
B1/B1024, fixed `[8,4]` physical groups, `leaf0` versus `leaf2` local partitions,
24 accepted units with complete-plan and independent-group correctness checks.
Keep this historical runtime and hardware identity separate from current timing.

## New Executable Local Mechanism

The initial queue contains exactly four cases: FWHT FP32 and cyclic NTT word64,
N4096, B1/B16, forward, out of place, no normalization, natural NTT order.
NTT uses modulus 576460756061519873 and Shoup multiplication with table coefficients.
All cases retain radix2, partition `[6,6]`, 32 threads, no overlap, one batch tile,
and the original one-stage shared primitive. No local register partition is added.

The baseline is linear shared placement. The treatment is writer-aligned shared
placement, spelled `hermes-xor` by the NTT API. Each 64-value tile has 32
butterflies and a fully active warp, with 64 independent tiles per transform.
This baseline has reasonable parallelism at both batches and is not the
deliberately underfilled `[1]*12` residency diagnostic. This is a matched
conventional parallel layout, not an external library's original organization.

This is mechanism ablation one. It measures a shared-address organization choice
with identical operator traits. It does not establish full mixed-template core
transfer, optimization of both organizations, or component-disabled re-search.
The operator benchmarks replay the public mapping schema. Their CUDA-event
launch-sequence latency is reported as kernel timing, not full C API/application
end-to-end latency or automatic public selection.

Budget: eight variants, eight full-batch preflights, three independent process
trials per variant, 100 warmups and 100 repeats. Largest batch is 65,536 elements.
No search candidates, extra sizes, inverse cases or profiler matrix are implicit.
Negative results complete this finite task; they do not trigger more tuning.

## Preparation and Guarded Launch

```bash
/home/wt/yes/envs/cubutterfly/bin/python paper/tools/prepare_principle_gap_campaign.py
```

This verifies the frozen library, benchmark binaries and GPU2 calibration hashes,
then writes `study.json`, `prepared.json`, `primitive_sources.json`, and `launch.json` in
`results/paper_targeted_20261012/principle_gap_attempt_safe`. `launch.json` contains exact argv
and shell commands with the existing compile broker and runtime environment.
Primitive source hashes refer to the frozen compile snapshot used by that runtime.
The earlier unexecuted `principle_gap/launch.json` is superseded; do not launch it.
Only the main agent starts the generated `launch_command`.

The command uses `scripts/run_isolated_gpu.py` and telemetry on
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6` (A100 80GB GPU2), with the finite
deadline 2026-10-12 16:30 Asia/Shanghai. It waits for the target and quarantines
conflicted or incomplete attempts. It never stops foreign jobs. GPU1 QuaRot is
protected and is not an allowed fallback. Do not use the older GPU1 E2 wrapper,
whose global foreign-GPU check is incompatible with independent GPU2 execution.

The immutable parent specification launches this script's `--guard-child` mode.
Every attempt uses the guard's `XD_REPORT_DIR` for `timing.json`, all telemetry,
preparation events and child ownership logs. The child verifies its guard parent
and session/group leadership. SIGTERM/SIGINT become `KeyboardInterrupt` inside
the existing telemetry call, invoking its owned-child session cleanup and final
telemetry save. No global process-spawn or signal-function patches are used.
A conflicted attempt cannot seed a retry. Only successful artifacts renamed by
the guard to `guard/release` are audited.

GPU acceptance requires an existing guard `release`, clean target telemetry,
successful preparation-event validation using
`semantic_gpu_pipeline.check_preparation_events`, and the following CPU audit:

```bash
/home/wt/yes/envs/cubutterfly/bin/python paper/tools/prepare_principle_gap_campaign.py \
  --audit-journal results/paper_targeted_20261012/principle_gap_attempt_safe/guard/release/timing.json
```

The audit checks the frozen semantic request, full-batch counts, all three trials,
every requested resolved control, and the actual mapping difference. It keeps
derived core/group identities in the comparison: only the declared layout axis
may differ. It rejects aliases and unexpected core/partition/parallel-shape changes.
Resolved mappings and actual axes remain in `mapping_audit.json` for review.
The existing FP32 FWHT benchmark uses an absolute max-error limit 1e-4 for
internal sanity. This is not an adopted user numerical-acceptance threshold.
NTT is exact elementwise. Failure does not authorize threshold relaxation.
The v2 audit explicitly leaves FWHT numerical acceptance unassessed and its
timings exploratory. Full error distributions remain a reporting-interface
dependency; profiler values remain separate from ordinary timing.

## Blocked Dependencies

Component-disabled re-search requires a public search constraint for component
exclusions and immutable core identity, complete candidate provenance, equal
budget and independent replay of selected mappings. Explicit `PlanOptions.mapping`
replay does not enforce these constraints on the adaptive candidate generator.
The JSON records an intended budget of eight candidates per space; no GPU work
for this task is queued until the interface exists.

Core transfer requires verifiable leaf/subgraph identity across conventional
parallel and mixed organizations, plus two compatible leaves per operator.
FWHT, both zeta transforms and structured butterflies have equivalent radix2/4/8
implementations elsewhere, but shared-iterative only accepts scalar radix2.
Their second portable leaf is missing. NTT has multiple generic subgraph cores,
but regrouping stages changes the physical subgraph and does not itself establish
the required fixed-leaf conventional-versus-mixed contract. FFT's standalone
factorial exists; the public factor-overlap study alone is only a local scheduler
control. Complete external library calls remain external baselines.

These tasks retain `blocked_implementation` status and explicit dependencies.
The launchable layout study and reused historical evidence do not close them.
