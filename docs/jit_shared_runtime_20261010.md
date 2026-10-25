# JIT runtime scaling investigation

Experiment revision `0.9.3.4-jit-runtime.1`, October 10, 2026.
This is a distinct diagnostic build, not a replacement label for existing
`0.9.3.4-async-experiments.3` measurements.

## Cause and scope

Precompilation and persistent broker queries removed compiler waiting and
repeated interpreter startup. However, the public FFT MEASURE search still
took roughly 28–34 minutes for N=256, batch=16 in the running 80GB campaign.
Its N=256 FP32 inventory has 26,282 candidates, including 20,960 overlap
candidates. Every candidate retains three warmups, twenty executions and full
reference checking. Larger batches expose more legal overlap tile sizes.

The JIT compiler used NVCC `--shared` without `--cudart=shared`. Each resulting
module therefore carried a private static CUDA runtime. Loading thousands of
modules increased the cost of stream, event and launch calls. This particularly
penalized candidates that submit many groups across streams. It is an execution
infrastructure defect; it does not establish a limitation of hybrid dataflow,
nor does fixing it establish theoretical optimality.

An initial, explicitly interrupted A100 40GB trace completed 5,322 bulk and
6,525 overlap candidates. Overlap execution and submission took 222.55 seconds
out of 240.09 seconds in those candidate totals. Reference checking took only
2.15 seconds. Compilation was already complete. These are partial trace totals,
not a completed search or FFT throughput result.

## Controlled mechanism test

Fresh processes loaded distinct modules compiled from the same trivial shader,
with static or shared CUDA runtime linkage. All modules were initialized before
timing. At 2,048 modules, median host API times were:

| Operation | Static runtime, us | Shared runtime, us |
|---|---:|---:|
| Event record | 23.231 | 0.380 |
| Stream wait event | 22.320 | 0.330 |
| Module kernel launch | 31.691 | 3.860 |
| Stream create | 24.451 | 1.960 |
| Stream destroy | 25.711 | 1.970 |

The benchmark varies module count as well as linkage. One-module static launch
cost was 2.48 us, while 512-module static launch cost was 9.25 us. The same
shared-runtime measurements were 2.82 us at 512 modules and 3.86 us at 2,048.
These are host-call microbenchmarks, not cuFFT speedups.

Reproduction and raw results are under
`results/plan_batch_trace_20261010/module_stream_bench_2048/`, using
`module_stream_bench.py` and `module_stream_bench.cpp` in the parent directory.

## Change and verification

The main compiler and its isolated snapshot use `--cudart=shared` and record
`cuda_runtime=shared` in both its cache identity and manifest. Templates,
rendering, mappings, arithmetic, candidate count and timing protocol are
unchanged. Old modules are not overwritten. The new broker binds the snapshot
and a separate cache; requests are re-keyed from exact original mappings.

Real FFT FP32, FWHT FP32 and NTT word64 modules compiled successfully and depend
on `libcudart.so.12`. Eighteen GPU checks cover batch 1/16, forward, normalized
inverse and roundtrip. FFT's maximum relative L2 error was 1.48e-7. FWHT's
binary-fraction input was exact. All NTT outputs matched the integer reference
exactly. Full arrays and module/UUID identities are in
`shared_validation/gpu_smoke.json`. These checks do not set the campaign's
pending floating-point acceptance thresholds.

An optional `CUBUTTERFLY_PLAN_TRACE=/absolute/path.jsonl` records candidate
construction, input generation, execution, checking and destruction. It is off
by default. Trace failure emits a warning and disables diagnostics without
rejecting a candidate. The trace library is linked separately from unchanged
production GPU objects; its build identity is recorded in
`trace_final_build.json`. `summarize_search.py` checks exact ordered candidate
identities before computing a complete-search wall-time ratio.

Two cold public-Plan diagnostics on separate A100 80GB GPUs completed without
preparation events: N=256, batch=16 FP32 took 250.508 seconds and FP64 took
231.659 seconds. These use the unchanged production library with shared JIT
modules. The observed wall times and selected mapping summaries are recorded
in `public_shared_pilots.json`; they are not cuFFT throughput ratios.

The complete paired diagnostic on A100 40GB measured 2,394.044 seconds with
static linkage and 386.144 seconds with shared linkage, a 6.200x reduction in
search wall time. Both runs used the same instrumented library and completed
the same 26,282 candidates in identical order: 5,320 bulk and 20,960 overlap
accepted, with the same two illegal-thread-count rejections. No compilation
preparation events occurred. `paired_search.json` contains complete phase sums,
candidate-order hashes, outcomes and errors. This is one matched search pair,
not a repeated statistical sample or an operator-throughput comparison.

Overlap execution/submission fell from 1,978.315 to 63.952 seconds. Candidate
construction still occupies 235.449 seconds in the shared run; removing static
runtime scaling does not eliminate module loading and plan-construction costs.
Reference-checking totals remained approximately ten seconds across all
candidates, supporting the conclusion that CPU reference generation was not
the dominant cause of this search slowdown.

The original two A100 80GB experiment queues have checkpointed and stopped. The original CPU
producer and query services for those queues are retired. The 40GB baseline
used a frozen ready-module service without a running compiler.
Both 80GB GPUs completed the shared-runtime FP32/FP64 search diagnostics and
now perform fresh, independent runtime-bound calibration.
All 72,654 known requests have been re-keyed for CPU precompilation in the new
broker. Ready modules and completed GPU cases are counted separately.

## Promotion boundary

The calibration identity now optionally binds the exact compiler broker through
`CUBUTTERFLY_CALIBRATION_BROKER`. The async formal-timing entry rejects shared
runtime without a matching binding. Stage and composition checkpoints carry
the same identity. A real old calibration was rejected as expected, and the
affected compiler/trace/calibration/async CPU regression suites passed.
This prevents the observed linkage mismatch; the recorded runtime inputs are
not a complete ELF loader trace. The paired full-search comparison is complete;
fresh affected whole-plan/composition calibration and formal timing remain
separate prerequisites for campaign promotion.

Calibration preparation retained all 75 fixed mappings and the existing finite
stage/composition consumers. Their 757 consumer descriptions projected to 20
unique JIT requests with no projection or module gap; 314 consumers use linked
lowerings. All 20 exact broker IDs were ready before calibration began. Existing
stage-adapter coverage gaps remain recorded separately. The new async runner
revision is `0.9.3.4-async-experiments.4`; its source identity includes the JIT
calibration validator. Calibration readiness does not certify the full design
space or set floating acceptance thresholds.

Old results retain their original identities and remain historical evidence.
CPU inputs and exact references can be reused. A single shared-version full
performance matrix requires new timing for previously measured static-version
cases, not a relabeling of their records. The current diagnostic does not claim
that matrix is complete.

The stopped static-runtime campaign has an audited handoff in
`static_evidence/handoff.json`: 29,215 processed requests, including 232 explicit
missing-lowering records, and 9,201 remaining requests. The separate 120 scaled
controls are preserved with `accuracy-review-required`; they do not turn the
original stress failures into passes. Historical source archives match every
original recorded hash. The shared campaign starts its own completion counts
and schedules each GPU's unmeasured requests before refreshing its historical
timings. These two populations must not be added into one version's progress.

The prepared continuation is recorded in
`new_campaign_preparation/population_audit.json`. Each original 80GB GPU keeps
19,208 timing requests, with 116 explicit gaps per device; the 120 untimed
scaled controls retain their original GPU1 assignment. All 38,416 timing case
objects and 120 control preparations were checked on CPU. Independent launchers
wait for their own fresh calibration and the same-GPU paired diagnostic, then
start the existing bounded controller with one active GPU case and 128 cases
per worker. They verify source hashes and calibration identity before launch;
dependency failures, GPU conflicts and expired waits stop explicitly.
Status files are `gpu1_shared_launch.json` and `gpu2_shared_launch.json`.
