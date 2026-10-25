# NTT planning and low-precision inverse repair

Intermediate revision 0.9.3.3, October 9, 2026. The two calibration queues were
stopped after low-amplitude review found a subnormal regression; see the
[0.9.3.4 follow-up](numeric_plan_repair_20261009_v0934.md). This record and its
artifacts describe the intermediate implementation, not the final active build.
Public ABI remains 0.9.3. Intermediate library SHA-256:
`51a520197caca02e84917f41c6caadd6016f949e6862811c9bf17b1559348ee2`.

## Failures and their causes

The previous campaign stopped at 04:50 CST after recording 33,076 of 38,416
requests. Its 12 numerical failures, one runtime error, original inputs and
error arrays remain under `results/semantic_campaign_20261008_v093/`.
The previous library and tools are also retained in
`results/numeric_plan_repair_20261009_v0933/before/`.

The NTT runtime error affected the public request with 64-bit words,
`N=1048576`, `batch=1`, modulus `576460756061519873`, forward direction and
natural orders. Search selected a legal CompactStage configuration, but the
public algorithm-name function omitted that backend. StagePipeline had the
same omission. HybridDataflow emitted a parameterized name that the inverse
parser did not accept. The repair reuses the complete mapping-JSON replay
interface for these configurations. Workload semantics, including modulus,
word width, direction and orders, remain in the descriptor. This is an interface
coverage error; it does not require a different NTT arithmetic kernel.

Eight FP16 C2R cases at `N=65536`, batch 1 or 16, failed with Inf despite finite
reference outputs of magnitude about one. The low-precision complex child
previously formed its unnormalized inverse in FP16 and divided by N only at
finalization. Rounding near the last butterfly exceeded FP16's finite range
before normalization could reduce the result. Changing input/output order or
placement did not remove this arithmetic issue.

Normalized low-precision FFT and FWHT now distribute the same total 1/N scale
over their radix-two butterflies, multiplying each input by 1/2 before that
butterfly. Finalization does not divide again. The shared trait covers C2C,
full-spectrum real adapters, stage splitting and generated subgraphs. It keeps
the original input, precision and normalization semantics. Rounding locations
change, so the new build retains fresh errors and cannot inherit old acceptance.

The [CPU trace](../results/numeric_plan_repair_20261009_v0933/cpu_inverse_trace.json)
finds first overflow at stage 15 of the original 16-stage recurrence and no
overflow with distributed normalization. This trace is not bitwise identical
to the GPU lowering: it reports 964 nonfinite values versus the GPU's 928.
It explains the range mechanism; the GPU rerun below establishes the actual fix.

Four other original FFT/FWHT stress cases have reference outputs exceeding
FP16's range. Their failures remain valid pressure-test observations. Separate
input-scaled controls retain their existing policy and cannot replace them in
the original performance population.

## Completed checks

The [original-request rerun](../results/numeric_plan_repair_20261009_v0933/original_failures/results.json)
contains all eight C2R failures and the NTT runtime failure, using unchanged
descriptors, full inputs, references and exact mapping replay. NTT passes every
integer element. All eight C2R outputs are finite. Batch-one nonfinite counts
fall from 928 to zero; batch-sixteen counts fall from 14,583 to zero.
For the natural out-of-place batch-one case, relative L2 error is 0.07494%,
p99 absolute error is 0.001417 and maximum absolute error is 0.002626. Relative
errors near zero remain in the full artifact; no floating threshold is accepted.

GPU regression evidence comprises 64 low-precision normalization checks and
eight public NTT algorithm/replay/support checks. The normalization matrix covers
FP16/BF16, native/FP32 computation, N=4096/65536/262144, rank-two combinations,
batch 1/16, explicit stage splits, forward/unscaled controls and mapping replay.
Exact constant/impulse identities pass equality checks; other floating cases
retain errors and require finite output, without a new tolerance. Two initially
misconfigured CompactStage test variants requested unsupported sizes/direction;
their failure report is preserved, and the final NTT tests cover both the legal
log20 forward configuration and explicit rejection boundaries.

The affected CPU calibration, recovery, continuation and codec suite passes
178 tests plus two subtests. The new validated-timing mode rejects imported
prior results and requires calibration. Existing comparison mode keeps its
full-conformance prerequisite. Every new timed plan still undergoes full
reference checking, padding checks and mapping replay before timing.

The [same-core diagnostic](../results/numeric_plan_repair_20261009_v0933/same_core_latency_gpu40.json)
uses A100 40GB, native FP16 hierarchical/radix4 inverse FFT and FWHT, three
interleaved trials per version and eight N/batch combinations. Observed
before/after ratios are 0.934–1.086 for FFT and 0.998–1.079 for FWHT. The slowest
FFT point takes about 7% longer after repair. These finite diagnostic timings
do not establish a general speedup; comprehensive new-build timings follow.

## Continuing experiments

The [frozen execution plan](../results/semantic_recovery_20261009_v0933/execution_plan.json)
and [selection audit](../results/semantic_recovery_20261009_v0933/prepared/recovery_selection.json)
preserve every semantic ID in the original population. No previous pass or
timing becomes a new-version measurement.

- Recovery conformance contains 11,802 requests: the 5,340 previously unvisited
  requests, previous failures, and affected NTT and normalized low-precision
  inverse combinations, with overlap removed.
- The complete 38,416-request population is split into two disjoint 19,208-request
  timing shards by original case order. Rank-nine boundary negatives remain in
  the coverage population and are excluded only from performance statistics.
- Each A100 80GB independently measures hardware rates, scheduling, stage services
  and whole-plan composition. The earlier 0.9.3.2 overlap queue stays historical;
  new finalization regenerates its compatible composition requests.
- GPU1 runs recovery conformance, separate scaled-input controls, then its timing
  shard. GPU2 starts its independently calibrated timing shard directly; each
  request includes fresh full correctness and baseline verification.

Read [GPU1 status](../results/semantic_recovery_20261009_v0933/gpu1_retry1/pipeline_status.json)
and [GPU2 status](../results/semantic_recovery_20261009_v0933/gpu2/pipeline_status.json).
Their UUIDs are respectively `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02` and
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`. Each comparison uses its own GPU's
baseline timings and preserves the device identity. The active queue index points
to both pipelines and supervisor heartbeats. Unexpected execution errors stop
the corresponding pipeline; there is no silent retry or foreign-workload waiter.
The first GPU1 calibration stopped at 09:47 CST when an external tdflow process
appeared. Its [interrupted record](../results/semantic_recovery_20261009_v0933/gpu1/pipeline_status.json)
is preserved. After that GPU became free, one inspected attempt started at
09:53 CST in `gpu1_retry1`, with fresh hardware and calibration measurements.
It imports no interrupted timing. GPU2's original attempt continues independently.

These intermediate pipelines were stopped before completion. Model coverage gaps, missing external
bindings and floating accuracy review remain explicit. A100 40GB's diagnostic
above does not replace its comprehensive comparison, and V100 remains a separate
user-run target.
