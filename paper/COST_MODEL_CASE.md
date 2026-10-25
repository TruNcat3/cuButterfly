# Reproducible Cost-Model Case

This note records one reproducible composition case and the implementation
contract that should be used when describing the cost model in the paper.  It
is a modular performance-model artifact for candidate ranking, not a new GPU
experiment, a prerequisite for hybrid dataflow or subgraph execution, or
evidence of a globally optimal schedule.

The two examples and their resource demands are also frozen in
[`evidence/method_cases.json`](evidence/method_cases.json), including the source
SHA-256. To replay the numerical composition on the CPU from the repository
root (no GPU or new fitting):

```bash
python - <<'PY'
import json, math, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from stage_cost_model import compose_stages
cases = json.loads(Path('paper/evidence/method_cases.json').read_text())
for row in cases['rows']:
    resource = compose_stages(row['stages'], row['schedule_tiles'], overlap=True)
    prediction = max(resource, row['scheduling_floor_ms'])
    assert math.isclose(resource, row['resource_estimate_ms'], rel_tol=1e-12)
    assert math.isclose(prediction, row['predicted_ms'], rel_tol=1e-12)
    print(row['tile_batch'], prediction, row['measured_ms'])
PY
```

## Search implementation constants

The paper describes the search procedure symbolically.  The current
implementation uses these engineering caps: each parent contributes at most
64 neighbors, an exploitation step retains at most 32 family-diverse
proposals, and the projected top 4 are resolved against their actual lowering
before the final model score.  These caps bound proposal work; they are not
limits on the candidate family, the hybrid-dataflow method, or the performance
model's validity.  The frozen acceptance budget remains 16 exploration
attempts, up to 4 historical-seed attempts, and at most 3 confirmed finalists
per workload; seed attempts count in equal-budget comparisons.

## Scope and provenance

The selected artifact is the A100 80GB `a10080_composition_diagnostic` cohort.
It contains 27 complete plans: nine FFT, nine word64 NTT, and nine FP32 FWHT
plans.  All use `logN=12`, 128 threads, batches 16/31/64, and tile batches
2/8/16; all 81 public correctness trials passed under warmup/repeat/trials
`100/100/3` ([report.md:51](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:51)).
The hardware, runtime fingerprint, compile mode, and stage-probe hash are
recorded in [report.md:3](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:3)-[report.md:8](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:8).

The numbers below are CUDA-event plan medians from
`public_completed_services.json`; JSON stores milliseconds, while this note
prints microseconds.  `public_plan_ms` is the timed complete plan, not a
sum of host allocation or compilation work.

## What the implementation computes

### Batch/tail descriptor resolution

`resolve_batch_services` in
[`stage_composition.py:335`](/home/wt/git/cuButterfly/scripts/stage_composition.py:335)
is a descriptor resolver.  It does not launch a benchmark or invent resource
data.  For a batch-ring point it computes

\[
  T = \min(B,\;\texttt{batch\_tile\_count}),\qquad
  R = B\bmod T,\qquad
  Q = \left\lceil B/T\right\rceil .
\]

Here `B` is the logical batch, `T` is the full tile batch, `R` is the tail
batch, and `Q` is the number of tiles.  The implementation obtains a normal
resolved source descriptor when needed, then describes the full tile at
`batch=T` and, only when `R>0`, a second serial descriptor at `batch=R`
([`stage_composition.py:358`](/home/wt/git/cuButterfly/scripts/stage_composition.py:358)-[`stage_composition.py:418`](/home/wt/git/cuButterfly/scripts/stage_composition.py:418)).  Each serial descriptor has
`stage_overlap=false`, `factor_overlap=false`, and `batch_tile_count=1`; the
overlapped whole-batch projection is not copied as a service curve.

The resolver accepts SharedIterative points with at least two physical groups,
and a restricted two-group OnlineReorder FFT path.  Factor overlap and other
unsupported backends are rejected ([`stage_composition.py:272`](/home/wt/git/cuButterfly/scripts/stage_composition.py:272)-[`stage_composition.py:289`](/home/wt/git/cuButterfly/scripts/stage_composition.py:289)).  Before returning `resolved`, it requires the source and serial views to
have the same runtime fingerprint, mapping identity, group shape
(`first_stage`, `stage_count`, `core`, `threads`), and known single-kernel
compiler resources.  Resource mismatches or unavailable serial resources are
an explicit uncovered result, not a nearest-curve guess
([`stage_composition.py:291`](/home/wt/git/cuButterfly/scripts/stage_composition.py:291)-[`stage_composition.py:332`](/home/wt/git/cuButterfly/scripts/stage_composition.py:332)).

### Group service curves

`stage_service_model.py` measures one physical execution group at a time.  A
measured duration is a complete group-kernel duration, including the launch
that produced it; a second launch charge must not be blindly added
([`stage_service_model.py:1`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:1)-[`stage_service_model.py:10`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:10)).
The normalized load axis is
`K=work_blocks` (`grid_ctas` is a legacy alias).  Batch size and absolute
load are deliberately excluded from the curve identity.

The curve key contains operator semantics, address geometry, mechanism and
group shape, plus compiler/live-resource regime
([`stage_service_model.py:281`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:281)-[`stage_service_model.py:391`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:391)).  An absent compiler-known flag keeps a numeric register/local value
unknown rather than treating a reported zero as measured zero
([`stage_service_model.py:175`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:175)-[`stage_service_model.py:197`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:197)).
`resolve_resources` may attach resources only when all same-shape candidates
collapse to one known regime ([`stage_service_model.py:439`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:439)-[`stage_service_model.py:469`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:469)).

Profile construction keeps only `measured`, correct, independent groups;
groups are bucketed by curve key, train/validation role, and actual load
`(K, work_blocks, grid_ctas)`.  Duplicate aliases at the same actual load are
merged, with the median of positive trials as the endpoint; a stronger
quality alias wins when the same `K` has different absolute loads.  Validation
points never enter the training hull
([`stage_service_model.py:680`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:680)-[`stage_service_model.py:815`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:815)).  Prediction is exact at a training endpoint or piecewise linear between
training endpoints.  Outside the measured `K` hull, or for an unknown curve
or resource regime, it returns uncovered rather than extrapolating
([`stage_service_model.py:593`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:593)-[`stage_service_model.py:609`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:609), [`stage_service_model.py:835`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:835)-[`stage_service_model.py:876`](/home/wt/git/cuButterfly/scripts/stage_service_model.py:876)).

### Scheduling terms and the composition floor

The independent `pipeline_schedule_model.py` is intentionally a measured
minimum BatchPipeline event/submission floor, not an operator concurrency
model ([`pipeline_schedule_model.py:1`](/home/wt/git/cuButterfly/scripts/pipeline_schedule_model.py:1)-[`pipeline_schedule_model.py:5`](/home/wt/git/cuButterfly/scripts/pipeline_schedule_model.py:5)).  It fits a separate curve for group counts 2, 3, and 4.  Training tile counts
are 1/2/4/8/16/32; 3/6/12/24 are independent holdouts.  A query at a
training endpoint is measured; an in-range missing endpoint is linearly
interpolated; outside the training range or with an unsupported group count it
is uncovered ([`pipeline_schedule_model.py:92`](/home/wt/git/cuButterfly/scripts/pipeline_schedule_model.py:92)-[`pipeline_schedule_model.py:124`](/home/wt/git/cuButterfly/scripts/pipeline_schedule_model.py:124), [`pipeline_schedule_model.py:173`](/home/wt/git/cuButterfly/scripts/pipeline_schedule_model.py:173)-[`pipeline_schedule_model.py:271`](/home/wt/git/cuButterfly/scripts/pipeline_schedule_model.py:271), [`pipeline_schedule_model.py:353`](/home/wt/git/cuButterfly/scripts/pipeline_schedule_model.py:353)-[`pipeline_schedule_model.py:393`](/home/wt/git/cuButterfly/scripts/pipeline_schedule_model.py:393)).
The probe uses one correctness-checked CTA store per group/tile and
`BatchPipeline::enqueue` with ready/consumed two-slot edges
([pipeline_schedule_calibration.json:68](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/pipeline_schedule_calibration.json:68)-[pipeline_schedule_calibration.json:96](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/pipeline_schedule_calibration.json:96)).

`schedule_cost_model.py` is a separate direct-launch curve.  For a thread
shape, it takes the one-CTA median as startup and fits the nonnegative slope

\[
  \hat{d} = \max\left(0,
    \frac{\sum_i (G_i-1)(m_i-m_1)}{\sum_i (G_i-1)^2}\right),
  \qquad C_{\rm dispatch}(G)=(G-1)\hat{d}.
\]

The query chooses the nearest measured thread shape in log2 space; when a
slope is not identified it uses the explicit unmeasured prior and marks grid
extrapolation ([`schedule_cost_model.py:13`](/home/wt/git/cuButterfly/scripts/schedule_cost_model.py:13)-[`schedule_cost_model.py:57`](/home/wt/git/cuButterfly/scripts/schedule_cost_model.py:57)).  These are minimal-launch estimates, not butterfly throughput; use them only for a service that does not already include its producing launch.

The recorded complete-plan predictor exposes the following composition
contract: per-group service durations feed a dependency/shared-resource
simulation (`resource_estimate_ms`), an independent pipeline floor is queried,
and the reported prediction is their maximum.  The report states this max
operation explicitly ([report.md:20](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:20)-[report.md:26](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:26)).  The production node recurrence is in
`stage_cost_model.py`: `schedule_nodes`/`compose` construct the active-node
schedule ([stage_cost_model.py:105](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:105)-[stage_cost_model.py:166](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:166)), batch/tail nodes are composed at
([stage_cost_model.py:316](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:316)-[stage_cost_model.py:367](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:367)), and measured service demands are assigned at
([stage_cost_model.py:368](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:368)-[stage_cost_model.py:446](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:446)).  The following is production-model pseudocode:

```text
compose(point, stage_profile, pipeline_profile):
    B = point.batch
    T = min(B, point.batch_tile_count)
    R = B mod T
    Q = ceil(B / T)

    full = resolve_serial_view(point, batch=T)
    tail = resolve_serial_view(point, batch=R) if R != 0 else None
    require same mapping/runtime/group/resource identity for full and tail

    for view in [full] + ([tail] if tail else []):
        for physical group g:
            duration[view,g] = exact_or_piecewise_stage_service(K=g.work_blocks)

    resource_estimate = simulate_active_nodes(
        duration, dependency_edges, service_resource_demands)
    pipeline_floor = pipeline_costs(pipeline_profile,
                                    groups=count(physical groups), tiles=Q)
    return max(resource_estimate, pipeline_floor)
```

For a node `(g,q)`, the ring dependencies cover the previous tile in the
same group `(g,q-1)`, the same tile in the previous group `(g-1,q)`, and the
two-earlier tile in the next group `(g+1,q-2)`, with boundary conditions from
the source graph.  Serial-control mode adds the submission-order predecessor
edge.  The event recurrence is:

```text
simulate_active_nodes(nodes, dependencies, demand):
    clock = 0
    remaining[n] = duration[n] for each node n
    completed = {}
    active = {n: all dependencies[n] are completed}

    while active is not empty:
        for n newly entering active:
            start[n] = clock

        p = max(1,
                sum(n.sm_demand for n in active),
                sum(n.hbm_demand for n in active),
                sum(n.dispatch_demand for n in active))
        w = min(remaining[n] for n in active)
        clock += p * w
        for n in active:
            remaining[n] -= w

        done = {n for n in active if remaining[n] == 0}
        for n in done:
            finish[n] = clock
            completed.add(n)
            release successors whose dependencies are now complete
            remove n from active
        add newly released successors to active
```

The measured branch of `_predict_measured` derives demand from each stage
term, rather than using fixed resource constants.  Before the final scheduler
clamp, its per-node terms are

\[
  s_n^{\rm raw}=\max(\texttt{term.occupancy},0),\qquad
  h_n^{\rm raw}=\begin{cases}
    \min(1,\texttt{term.memory\_ms}/\texttt{body\_ms}) & \text{if body\_ms}>0,\\
    0 & \text{otherwise},
  \end{cases}
  \qquad d_n=\min(1,\texttt{overhead\_ms}/\texttt{duration\_ms}).
\]

Here `duration_ms=overhead_ms+body_ms*fraction` in `schedule_nodes`; measured
overhead is zero, so `d_n=0` for this branch.  `schedule_nodes` then clips
`s_n^{raw}` and `h_n^{raw}` to `[0,1]` before forming `p`
([stage_cost_model.py:127](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:127)-[stage_cost_model.py:132](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:132)).  The
default `stage_terms` applies resident correction
`term.occupancy=min(1, grid/(sm_count*resident))`
([stage_cost_model.py:95](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:95)-[stage_cost_model.py:101](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:101)); only the missing-rate
`_minimal_terms` fallback uses `term.occupancy=min(1, grid/sm_count)` and
`term.memory_ms=0`
([stage_cost_model.py:300](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:300)-[stage_cost_model.py:312](/home/wt/git/cuButterfly/scripts/stage_cost_model.py:312)).  Thus `p` is a uniform slowdown factor for all currently active nodes, not a
discrete capacity-packing decision.  A node's `finish` time is the clock at
which its remaining service reaches zero; it is not in general `start +
fixed_duration` when other nodes are active.  The artifact's
`resource_estimate_ms` is the persisted result of this recurrence, while
`concurrency_calibrated=false` records that real-kernel overlap remains
uncalibrated.

## One same-load ranking example

Candidates 26 and 27 have the same logical workload: FP32 FWHT, `N=4096`,
forward, out-of-place, `batch=64`, SharedIterative `[8,4]`, and 128 threads.
They differ only in tile batch (`T=8` versus `T=16`), so they test the ranking
decision at one workload rather than comparing different operators.

The source descriptors are visible at
[public_completed_services.json:18510](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:18510)-[public_completed_services.json:18618](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:18618)
for candidate 26 and
[public_completed_services.json:19085](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19085)-[public_completed_services.json:19193](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19193)
for candidate 27.  The group resources and measured/interpolated service
values are:

| Candidate | `T`, `Q`, `R` | Serial group 0 (`K`, service, registers/shared) | Serial group 1 (`K`, service, registers/shared) |
|:--|--:|:--|:--|
| 26 | 8, 8, 0 | 128, 4.464640 us, 27 regs/thread / 2048 B | 2048, 5.560320 us, 18 regs/thread / 128 B |
| 27 | 16, 4, 0 | 256, 4.957091 us, 27 regs/thread / 2048 B | 4096, 7.148451 us, 18 regs/thread / 128 B |

Candidate 26's serial view has `tile_batch=8`, `tile_count=8`, and no tail;
its per-group values are exact training-curve lookups
([public_completed_services.json:18919](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:18919)-[public_completed_services.json:19018](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19018), [public_completed_services.json:19021](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19021)-[public_completed_services.json:19053](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19053)).
Candidate 27 has `tile_batch=16`, `tile_count=4`, and no tail; its two service
values are piecewise-linear training-curve predictions
([public_completed_services.json:19494](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19494)-[public_completed_services.json:19592](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19592), [public_completed_services.json:19596](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19596)-[public_completed_services.json:19628](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19628)).

| Candidate | Resource/dependency estimate | Pipeline floor | Predicted complete plan | Public complete-plan median | Absolute relative error |
|:--|--:|--:|--:|--:|--:|
| 26 (`T=8`) | 51.262197 us | 56.442881 us | **56.442881 us** | 82.276 us | 31.398% |
| 27 (`T=16`) | **35.754048 us** | 28.692479 us | **35.754048 us** | 47.538 us | 24.788% |

The candidate-26 plan record reports the 82.276 us public median, 56.442881 us
prediction, resource estimate, and 2-group/8-tile pipeline floor at
[public_completed_services.json:18843](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:18843)-[public_completed_services.json:19080](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19080).
Candidate 27 reports the corresponding 47.538 us public median, 35.754048 us
prediction, and 2-group/4-tile floor at
[public_completed_services.json:19418](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19418)-[public_completed_services.json:19655](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19655).

Thus the model ranks `T=16` below `T=8` (35.754 < 56.443 us), and the public
measurements make the same ordering (47.538 < 82.276 us).  This supports the
ranking for this pair only; both absolute predictions are low, with the
candidate-26 error showing the known short-kernel composition problem.

## Uncertainty and uncalibrated concurrency

The independent pipeline floor is well covered as a *floor*: the calibration
has 30 shapes, 12 holdouts, and mean/maximum interpolation errors 0.193%/0.419%
([report.md:34](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:34)-[report.md:47](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:47)).  The 27-plan composition
diagnostic has median errors 4.998% (FFT), 4.109% (NTT), and 12.846% (FWHT),
with a 40.287% maximum FWHT error ([report.md:58](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:58)-[report.md:63](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:63)).

For candidate 26 specifically, the independent serial-service sum for eight
tiles is 80.200 us, close to the unprofiled 82.276 us public result, while the
simulator's resource estimate is 51.262 us and the max-with-floor prediction is
56.443 us.  A profiled trace showed zero concurrent kernel time, but profiling
raised the public result to 110.843 us; therefore its gaps cannot be reused as
an unprofiled coefficient or proof that every unprofiled schedule serializes
([report.md:92](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:92)-[report.md:105](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/report.md:105)).

The artifact consequently marks `concurrency_calibrated=false`, and its
complete-plan serial holdout population is zero
([public_completed_services.json:19062](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19062)-[public_completed_services.json:19080](/home/wt/git/cuButterfly/results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json:19080)).  The correct paper wording is therefore "measured group service curves plus a measured BatchPipeline scheduling floor and an unvalidated dependency/resource composition estimate," not "calibrated concurrent execution time" or "theoretical lower bound."
