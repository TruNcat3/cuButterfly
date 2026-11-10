# Experiment status for cuButterfly 0.9.3

Historical snapshot. This campaign stopped at 04:50 CST on October 9 with
33,076 records; see the [0.9.3.3 repair and new queues](numeric_plan_repair_20261009_v0933.md)
for current execution. Its measurements retain their original build identity.

Snapshot: 2026-10-08, 23:01 CST, during bounded conformance.
The immediate objective is a complete bounded semantic campaign on the new
binary, followed by matched performance comparison. The new population is
prepared; full conformance and performance results are still pending.

The latest conformance checkpoint contains **15,148 / 38,416** requests:
2,688 exact integer passes and 12,460 floating requests awaiting accuracy review.
There are no recorded cuButterfly runtime errors or unavailable requests in this
visited portion. Baseline binding gaps remain explicit; this is not a complete
correctness or performance result.

Calibration-tool revision **0.9.3.2** adds a separate
[composition calibration diagnostic](composition_calibration_20261008.md).
Its prepared population has 168 requests in 84 serial/overlap pairs. It will run
between the current conformance stage and the remaining scaled-control/comparison
stages on the same reserved GPU. Existing library and campaign evidence stay frozen.
The previous calibration measured 75 whole-plan mappings and 56 independent
stage classes, but had no observed overlap pairs. It remains a finite stage
profile with composition gaps, not a fully calibrated concurrency model.

## Current evidence and population

The [support status](../results/semantic_support_20261008_v0923/status.json)
records 334 targeted checks (332 GPU cases and two CPU oracle checks), 194 CPU
checks with 28 subtests, and 13 unique requests through the actual runner.
Six runner requests pass exact integer checks; seven floating requests await
accuracy review. These checks cover the new lowerings but do not establish
complete matrix correctness or performance.

The [bounded population](../results/semantic_campaign_20261008_v093/population/manifest.json)
contains **38,416 valid requests**, including **232 rank-nine boundary negatives**.
Those negatives belong to support/rejection coverage and are excluded from the
performance population. Every request has at most `N_total * batch <= 2^20`
logical elements. The shape/batch policy retains 20 scaling combinations and
excludes 24 extreme combinations independently of measured performance.
See the [support record](bounded_support_20261008.md) for the declared limits.

## Immediate work on the new binary

New campaign directory: `results/semantic_campaign_20261008_v093/`.
Its [hardware probe manifest](../results/semantic_campaign_20261008_v093/hardware_probe/manifest.json)
records completed hardware initialization. This is not completion of operator,
stage, or whole-plan calibration. Revision **0.9.3.1** fixes native UINT64
execution in the two calibration tools and UINT64 type/trait dispatch in the
shared-iteration JIT generator. Public ABI version remains 0.9.3. The library
SHA-256 is `eef2fe786c3b23df3c72a8b72761c9beef3f1b148f1d075ac6a176b6bb2a9eea`.
The previous runtime and failed probe are retained under this campaign.
Sixteen exact whole-plan cases and twelve independent stage probes passed,
covering both zeta directions, high-bit inputs, strided/in-place execution and
split shared-iteration groups. These are new generator/tool checks; the earlier
334 checks keep their original build identity.

| Priority | Stage | Completion evidence still needed |
| --- | --- | --- |
| P0 | Fresh calibration and measurement binding | New binary, device and protocol identities; explicit stage/model coverage gaps |
| P0 | Bounded conformance | Every manifest request recorded, including expected boundary rejection, exact integer checks and floating error artifacts |
| P0 | Separate scaled-input controls | FFT/FWHT pressure failures retained; fixed power-of-two scaled controls checked with their own input identities |
| P0 | Matched comparison | Same descriptor, precision, direction, order/layout and batch; full conversion-inclusive call timing |
| P0 | Accuracy review | Full floating error distributions and reference-library errors, followed by explicit threshold selection |

The execution order is fresh calibration, conformance, the separate composition
diagnostic, scaled controls, then comparison. The background pipeline runs on A100 80GB
(`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`). Live progress comes from the
[pipeline state](../results/semantic_campaign_20261008_v093/pipeline_status.json)
and the supervision state referenced by the [active queue index](../results/active_experiment_queue.json).
A launch command alone is not completion
evidence. Old measured rankings are not automatically certified for new adapters.

Float thresholds remain undecided under the user's error-report-first policy.
`accuracy-review-required` records cannot count as accepted passes or validated
speedup winners. Exact integer comparison remains all-element. Native and
adapted implementations are reported separately, with actual conversion costs.

## Historical evidence to retain

Earlier experiments remain useful evidence under their recorded build and
protocol identities. They should not be relabelled as 0.9.3 results or all
repeated automatically.

| Experiment | Existing evidence | Appropriate follow-up |
| --- | --- | --- |
| FFT core/organization controls | [Factorial v2 summary](../results/paper_experiments_20260917/a100-80gb/fft_factorial_v2_summary.md): 6/6 workloads, 36/36 variants, 54/54 comparisons | Representative same-build confirmation where implementation changed |
| Inverse semantic smoke | [E8 GPU summary](../results/paper_experiments_20260917/semantic_adaptation/e8_gpu_summary.json) completed | Reuse intent in the new descriptor matrix |
| Model ranking and search | [Historical search analysis](../results/paper_experiments_20260917/analysis/search_protocol_summary.json); [covered holdout](../results/paper_experiments_20260917/protocols/a100-80gb-gpu1/covered-holdout/heldout_validation.json) has only two eligible workloads | Freeze training/holdout identities and assess supported new paths; report uncovered classes |
| Hardware mechanism diagnostics | [Control/profile analysis](../results/paper_experiments_20260917/analysis/control_profile_analysis.json) and later FFT diagnostic artifacts | Refresh a finite set of changed positive/negative cases after ordinary timing |
| FNO and QuaRot applications | [E13 queue](../experiments/workloads/end_to_end_queue.json) records FNO 64/128 v9 and fixed-random-weight QuaRot small/large runs | Confirm integration identity before new-version claims; finish outstanding application diagnostics |

The broader historical model protocol recorded eight service gaps, whereas the
covered holdout only evaluates two workloads. Neither establishes general
ranking coverage for the new precision adapters. Preparation/search cost and
finite mechanism studies remain separate from steady-state operator timing.
The [paper experiment plan](../experiments/workloads/plan.json) defines these claims
and their completion criteria.

## Remaining baseline, application and hardware work

External baseline bindings are narrower than the executable library. A CPU-only
call to the current `baseline_contract` rules gives the following counts; these
are contract matches, not measured or accuracy-qualified results.

| Operator | Manifest requests | At least one external contract match |
| --- | ---: | ---: |
| FFT | 14,640 | 2,464 |
| FWHT | 9,632 | 4,416 |
| NTT | 2,336 | 448 |
| Subset zeta | 2,336 | 0 |
| Superset zeta | 2,336 | 0 |
| Structured 2x2 | 7,136 | 0 |

The totals include boundary negatives. Excluding all 232 rank-nine negatives
leaves 38,184 requests, of which 7,304 currently match an external contract.
The zero-match zeta/structured rows have no applicable library in the current
four-library baseline set; they are distinct from missing adapters for otherwise
applicable vendor routines.

All **3,200 real-FFT requests** lack a unified-runner external binding, including
FP32/FP64. Each of FP16, BF16, FP32 and FP64 contributes 296 R2C and 504 C2R
requests. This is a harness gap, not a claim that cuFFT/VkFFT lack real FFT.
The current FFT binding admits only matching-precision FP32/FP64 C2C, standard
length, rank at most three. GPU-NTT binds rank one only; its 1,888 xD requests
lack a matching adapter. Dao supports FP32 arithmetic with FP16/BF16/FP32
storage, total flattened points at most 32,768, and no embedding.
Separate follow-up cohorts must fill these adapter gaps without mutating a
running campaign's frozen measurement sources.

Missing bindings
must remain explicit; another precision, operator or smaller batch cannot stand
in for the requested contract. Complete paired statistics require matched
correctness and timing evidence, not just runnable library kernels.

HEonGPU currently has a successful build and single-prime adapter probe.
Full RNS integration and complete CKKS application comparison remain pending.
FNO still needs application trace/peak-memory diagnostics according to its
recorded queue; QuaRot random-weight timings do not establish trained-model
quality. The older [E13 source manifest](../experiments/workloads/end_to_end.json)
contains preparation-era statuses; the queue and measured reports give the
later execution evidence.

A100 40GB requires a separately bound same-version comparison subset after the
80GB campaign is established. Historical 40GB records are retained, not copied
into the new cohort. V100 execution remains assigned to the user on the other
machine after GitHub synchronization; cross-architecture claims require its
same-version results. A100 40GB/80GB are distinct targets but both SM80.

The old [experiment plan index](../results/experiment_plan_index.json) still
points to the October 1 RC2 queue. It is historical and must not be used to
resume the cancelled pre-unified-semantics performance plan.
