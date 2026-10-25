# Search-budget validation

This is an isolated validation of the bounded runtime search introduced in
`0.9.3.5-library-rebound-20261011`. The isolated check below is separate
from the subsequent bounded campaign.

Command:

```bash
CUDA_VISIBLE_DEVICES=GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e \
CUBUTTERFLY_SEARCH_BUDGET=64 CUBUTTERFLY_COMPILE_MODE=research \
LD_LIBRARY_PATH=build-e13-cufftdx:$LD_LIBRARY_PATH \
build-e13-cufftdx/cubutterfly_plan_bench \
  --operator fft --shape 256 --batch 4096 --precision fp32 \
  --policy measure --verify --warmup 1 --repeat 2 --trials 1
```

The run scheduled 64 of the complete candidate inventory and emitted a
`64/64` search-progress record. All 4,096 batches passed verification. The
reported warm kernel measurements were:

| Implementation | Kernel time | Relative to cuFFT |
| --- | ---: | ---: |
| cuButterfly | 0.019968 ms | 1.103x |
| cuFFT | 0.022016 ms | 1.000x |

This result validates the bounded-search execution path and is an isolated
sanity check. It is not part of the frozen comprehensive cohort and does not
replace exhaustive search for final evidence.

## A100 80GB bounded continuation

The active queue is indexed by `results/active_experiment_queue.json`, under
`bounded_search_campaign`. GPU UUID is
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`. The new library SHA-256 is
`24486975585fa2ab6399a0cfeae4b9449c398490cb76c5ea5a73a3560abe8c66`.
The queue measures at most 512 candidates per public butterfly/NTT search.
Input size, batch, storage/compute precision, mathematical semantics, all-element
validation and full-call kernel timing are unchanged. The reduced candidate
budget can change the selected mapping and its performance.

The old GPU2 worker was stopped while inside a long Plan search. Its 14 indexed
floating rows are retained, but none are adopted into this new build's count.
The controller's identity mismatch is retained as an incident record. Early
calibration attempts in `recalibration_gpu2_search8_20261011` and
`recalibration_gpu2_bounded2_20261011` were interrupted; they are checkpoints,
not completed calibrations. The low-level four-workload calibration completed,
and the final confirmed calibration is
`results/plan_batch_trace_20261010/recalibration_gpu2_finalized_bound_20261011/`.
It binds the exact library and shared JIT broker, remeasures eight mappings,
and releases seven registry records after selector replay passes.

This is a deliberately **finite** calibration. The model reports
`calibrated-local-serial`, not complete concurrency calibration. Composition
measurements covered 36 requests, but predictions covered only 9 of 18 paired
comparisons. All nine covered pairs had the correct relative ordering; the
median latency error was about 12.7%. The other nine pairs remain uncovered,
and `inventory_complete` is false. One-trial confirmation is a pipeline check,
not a stability-qualified final performance result.

One matched BF16 R2C case with N=256 and B=4096 took 18,273.6 seconds to plan
in the historical exhaustive run and 189.1 seconds in the new 512-candidate
run. The corresponding warm kernel medians were 0.030669 and 0.031898 ms.
These are different-build, different-search observations, not a controlled
kernel-speedup experiment. They show the planning benefit and the possible
selection-quality cost of reducing the budget.

The worker captures C++ stderr until each Plan construction ends. A quiet
worker log therefore does not prove a stalled search. For diagnosis, the live
worker's stderr file can be inspected read-only via `/proc/PID/fd/2` after
verifying the PID. CPU reference generation and full-batch verification still
occur within candidate search; compiled modules being ready does not mean all
CPU planning work has been eliminated.

The first active phase contains 4,600 cases. The 14,608-case refresh and the
other device's phases remain part of the full goal; a completed first phase is
not full-matrix completion. Floating rows remain `accuracy-review-required`
until the previously agreed error-distribution review determines acceptance.

The refresh is prepared at
`results/search_budget_20261011/gpu2_refresh/gpu2_plan.json`, with a separate
copy of the same library in `results/search_budget_20261011/frozen_runtime/`.
`results/search_budget_20261011/run_gpu2_refresh_after_remaining.sh` is running
as a dependency watcher. It checks the predecessor's process identity and
requires all 4,600 case IDs to be indexed before starting refresh. An abnormal
stop blocks the successor instead of silently retrying. GPU1's later recovery
uses fresh calibration on its own UUID and retains its original population;
see the [parallel continuation record](parallel_resume_20261011.md) and the
active index for its current state.

The reusable preparation entrypoint is
`scripts/prepare_library_rebound_campaign.py`. Before writing a new campaign,
it checks source/config hashes, finalized calibration, GPU UUID, shared-JIT
identity, and Unix socket pathname length. It selects exactly the requested
phase, records all omitted phases, and requires explicit C API source
rebinding when the historical source differs. The 24 focused CPU tests passed.
These checks prevent startup identity failures; they do not certify model
coverage or floating accuracy. The first 16-case shard has completed and the
next shard is running; the first shard's rows all require floating-error review.

During recovery the hardware-rate probe reused the prior `gpu2_bound/hardware`
directory. Its former hashes must not be treated as immutable historical rate
evidence. Do not rerun probes into that directory while the current queue is
using it; later measurements need fresh hardware and calibration directories.
Startup failures due to unfinalized calibration, missing JIT identity and a
long Unix socket pathname were retained separately and produced no timing rows.
