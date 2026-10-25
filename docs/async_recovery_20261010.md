# Bounded GPU workers and checked experiment recovery

Experiment revision `0.9.3.4-async-experiments.2`, October 10, 2026.
Library v0.9.3.4 and both A100 80GB calibration identities are unchanged.

## Failure and evidence

The previous GPU2 and GPU1 queues stopped at 04:55 and 05:24 CST respectively.
Both emitted specialization compiler spawn failures, followed by Python
`MemoryError`, including while serializing a checkpoint. The last indexed
reports therefore still said `running`, although the processes had exited.

`src/jit_module.cpp` keeps every loaded specialization in its process-global
`modules` map. Candidate Plan destruction does not remove these `dlopen`
handles. Long MEASURE sweeps accumulate candidate modules across requests.
The host's `vm.max_map_count` is 65,530. This retention path and failure sequence
are consistent with mapping/resource exhaustion; the dead processes' peak VMA
counts are unavailable, so reaching that exact kernel limit is not a directly
observed historical fact. This is an experiment-process lifetime problem, not
evidence that a batch's transform data exceeded GPU memory.

The [checked handoff](../results/async_recovery_20261010_v0934_2/evidence/handoff.json)
verifies every indexed row and error artifact, descriptors, UUIDs, library,
calibration, protocol and original source identities. Snapshots retain the two
old worker sources that change in this repair. No successful case was committed
after the first compiler spawn failure in either old log. The two unindexed
MemoryError rows are retained as incident evidence and remain in the retry
population, rather than being adopted as completed measurements.

| Original performance population | GPU1 | GPU2 | Total |
|---|---:|---:|---:|
| Indexed requests retained | 14,585 | 14,585, including 35 earlier records | 29,170 |
| Explicit rank-nine missing lowerings within retained records | 116 | 116 | 232 |
| Remaining requests | 4,623 | 4,623 | 9,246 |

The completed 11,802-request conformance recovery remains separate evidence.
Its 120 raw numerical failures are not turned into passes by the controls below.

## Repair

[run_async_semantic_campaign.py](../scripts/run_async_semantic_campaign.py)
now records VMA, RSS, virtual memory, FD and module mapping counts. A worker
recycles at 128 completed cases, 64 plan attempts, 8 GiB RSS, or 45,000 VMAs,
with the latter also capped at three quarters of the host mapping limit.
These limits govern process lifetime, not the legal candidate population.

[semantic_resource_compile_proxy.py](../scripts/semantic_resource_compile_proxy.py)
also checks the GPU parent before each JIT broker query, protecting a long
individual search. A resource event invalidates the entire selection and
requests a fresh process; it cannot silently exclude a candidate. Genuine
compile errors retain investigation priority. Native compiler-start and module
load failures that occur before the broker can record an event are captured
from C++ stderr and block acceptance as well.

On recycling, incomplete plans close, only fully verified rows remain committed,
and all other requests remain pending. Exit code 75 distinguishes this expected
boundary from runtime failure. The controller checks every resulting row and
artifact before adopting it. Repeated no-progress recycling or a per-case
preparation budget violation stops for investigation rather than pruning search.

[run_bounded_semantic_queue.py](../scripts/run_bounded_semantic_queue.py)
runs at most 128 requests per child manifest, reuses the original prepared
inputs and CPU compile broker, and prefers requests whose known compilation
dependencies are ready. When all known dependencies are pending, it waits on
CPU with no GPU worker resident, within a bounded twelve-hour wait budget.
It never waits for or stops foreign GPU jobs. Existing valid measurements are
retained with their original identities, not relabelled as measurements of the
new experiment revision. Interrupted new runs require an audited handoff rather
than an unverified automatic retry.

The seven-sample timing group, full conversions, research candidate semantics,
arithmetic precision, error rules and calibration remain unchanged. Compilation,
planning and CPU checking remain outside kernel timing. Process teardown releases
the retained JIT modules; this repair does not add module eviction to the public
library or change the ownership ABI of existing Plan/module pointers.

## Completed checks and resumed work

CPU regressions cover snapshot accounting, limits, compile error priority,
native stderr capture, whole-search rejection, pending-work preservation,
evidence corruption, ready-case selection and controller exit classification.
The audit's six additional tests are in
`scripts/tests/test_interrupted_async_handoff.py`.

A real A100 40GB lifecycle diagnostic deliberately limits a worker to one
completed case. Two successive processes committed distinct cases and exited
75, preserving 23 then 22 remaining requests. Its
[report](../results/async_recovery_20261010_v0934_2/recycle_pilot/results.json)
remains a scheduling diagnostic, not a replacement performance sample.

A separate [in-search guard diagnostic](../results/async_recovery_20261010_v0934_2/proxy_guard_pilot/status.json)
sets an intentionally low 799-VMA threshold on A100 40GB. Starting from 775
VMAs, loading four specialization libraries created 24 module mappings and
triggered the guard at 805 VMAs. The complete selection returned WorkerRecycle
and was discarded without timing. This directly verifies the mid-search guard
and module-mapping growth; it does not reproduce the historical 65,530 limit.

The original A100 80GB GPU1 has now completed all **120 separate scaled controls**:
96 C2C/FWHT plus 24 R2C/C2R, retaining the original mapping in every case.
All outputs are finite, with complete errors and input/padding checks retained.
The largest relative L2 errors are approximately 0.1331% for the 96 controls
and 0.1180% for the real FFT controls. These summaries do not establish an
acceptance threshold; all 120 remain `accuracy-review-required`. Raw inputs,
their failures and baseline binding gaps remain visible.

Both GPUs then continue their disjoint 4,623-request performance subsets.
Adaptive candidate discovery can still request additional CPU compilation;
launching a queue does not establish a completed comparison. Current state is
in the [GPU1 queue](../results/async_recovery_20261010_v0934_2/gpu1/queue_status.json)
and [GPU2 queue](../results/async_recovery_20261010_v0934_2/gpu2/queue_status.json),
with child logs and complete measurements under each phase's shard directory.
The active index is `results/active_experiment_queue.json`. The old expired
real-FFT dependency waiter is superseded by these completed control phases.

## Reproducible entry

The CPU audit is `scripts/prepare_interrupted_async_handoff.py`. It emits
remaining manifests whose case objects exactly match the existing preparation
cache. The prepared plans are `gpu1_plan.json` and `gpu2_plan.json` under
`results/async_recovery_20261010_v0934_2/`. Each plan binds its library, GPU UUID,
calibration, CPU preparation and compilation directories and ordered phases.
Invoke the bounded queue with `--plan PLAN --output FRESH_DIRECTORY`, using
`/home/wt/yes/envs/cubutterfly/bin/python` and research compilation mode.
Do not relaunch an already active plan on the same GPU.

## CPU preparation follow-up

Preparation revision `0.9.3.4-cpu-preparation.1` changes only the independent
CPU producer. The GPU experiment revision remains `0.9.3.4-async-experiments.2`.
Input/oracle preparation already covers all 38,416 requests: 38,184 ready,
232 explicit rank-nine gaps, and no pending or failed preparation records.
The separate 120 scaled controls are also fully prepared and measured.

The previous compiler producer repeatedly scanned every request and record
when reporting its heartbeat and refilling a worker slot. With approximately
73,000 requests, this host-side scheduling work left most of its six worker
slots unused. The replacement
[semantic_compile_producer.py](../scripts/semantic_compile_producer.py)
discovers new request names periodically, retains state in memory and refills
slots directly after completion. It prioritizes the smallest known unresolved
case dependency groups, reserving every fourth dispatch for the background
queue. This changes compilation order only; it never removes a candidate or
accepts a partially searched plan.

The producer uses the existing broker identity, compiler, cache keys, request
locks and atomic ready records. Existing compile failures stay failed. SIGTERM
stops dispatch and lets active compilations finish. A producer lock rejects
duplicate instances of the new entry; the old broker producer must be stopped
before switching, because it does not acquire that producer lock.

On this 64-CPU host the new producer runs 24 compilation slots with
`CUDA_VISIBLE_DEVICES=''`. The original producer exited cleanly with 35,569
ready requests and no running or failed compilation. Its state and the new
command are retained in `evidence/cpu_compile_launch.json` under this campaign.
The producer heartbeat reports active slots, pending/ready/failed requests and
completion rate without rereading every terminal record. Counts refer to
compiled modules, not completed GPU comparisons.
An initial 90-second observation completed 564 requests (about 6.2/s) with
all 24 slots occupied and no failed compilation. This is a scheduling
observation on the current queue, not a controlled kernel speedup result.
The source hash, test outcome and a later producer snapshot are retained in
`evidence/cpu_compile_observation.json`.

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/semantic_compile_producer.py \
  --broker results/async_experiments_20261009_v0934_1/compile \
  --jobs 24 --watch \
  --priority-report 'results/async_recovery_20261010_v0934_2/gpu1/timing/shard-*/measurement/results.json' \
  --priority-report 'results/async_recovery_20261010_v0934_2/gpu2/timing/shard-*/measurement/results.json'
```

The quoted report globs include future recycled workers automatically. Without
`--watch`, the same entry drains available work and exits. Do not start a second
producer while the recorded one is alive. CPU-only regressions for scheduling
and the unchanged broker passed together (34 tests).

One additional cache scan uses the existing `seed_cache_since` entry to enqueue
exact mappings from matching-SM manifests newer than the frozen library. These
still pass through current compiler/cache-key validation. The scan does not
infer a new design space or trust old module paths; its result is recorded in
`evidence/cpu_cache_seed.json`.

Complete CPU-only candidate export is still a separate implementation gap.
The existing rank-one `precompile_research.py` export queries device properties,
and semantic xD/real refinement still discovers some candidates during Plan
preparation. The new producer can compile all known requests ahead of timing
and consume later discoveries, but cannot promise that every future adaptive
candidate was known before GPU preparation. Output verification and error
analysis also still depend on the corresponding GPU results. No frozen GPU
source, arithmetic kernel, calibration or measurement protocol changed here.

For large-batch runtime measurement, the research searcher exposes an explicit
upper bound without deleting candidates from the inventory. Set
`CUBUTTERFLY_SEARCH_BUDGET=N` before starting a new worker to measure a
deterministic, uniformly spaced subset of `N` candidates; `0` (the default)
retains exhaustive measurement. The search trace records both the complete
inventory and scheduled counts, and the worker emits progress every 64
candidates. A budget is a measurement policy for a new run and must be
recorded with its build and hardware identity; it must not be applied when
claiming exhaustive search coverage.
