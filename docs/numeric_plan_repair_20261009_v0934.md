# Range-aware inverse normalization and fresh experiment recovery

Revision 0.9.3.4, October 9, 2026. Public ABI remains 0.9.3. Final library SHA-256:
`90c37f196d945b3cd179612c0fc7a9763a0b06dc1e5357b324c9055c057f8901`.
This follow-up
supersedes the intermediate [0.9.3.3 repair](numeric_plan_repair_20261009_v0933.md)
and its incomplete calibration attempts. Old data and binaries are retained.

## Root causes and implementation

The original NTT runtime failure was a public algorithm-identification omission:
valid CompactStage and StagePipeline configurations had no identifier, while
HybridDataflow's parameterized name did not round-trip through the parser.
These paths now return their existing full serialized mapping. This correction
does not change modular arithmetic or the user's transform descriptor.

Eight original FP16 real inverse FFT failures instead overflowed an unnormalized
intermediate before final division by N, although the reference output was
finite. Moving 1/N into one factor of 1/2 per radix-two stage removed those Inf
values. Review then found a different boundary failure in that intermediate
implementation: halving the smallest nonzero operand before adding it loses
the value. The 32-case constant-input diagnostic was exact on the original
build; the intermediate build lost all 16 minimum-subnormal cases while its
16 minimum-normal cases remained exact. No tolerance was loosened to accept
this regression, and both running intermediate calibration queues were stopped.

The final implementation keeps one factor of 1/2 per stage and chooses its
placement using operand magnitude bits. When all butterfly input components
are at most max(storage)/4, computation precedes scaling. Otherwise scaling
precedes computation. FP16's bit bound is `0x73ff` (16376); BF16's is `0x7e7f`.
In the first branch each complex-product term is at most max/4, its two-term
sum at most max/2, and the final addition at most 3max/4. Unit twiddle components
have magnitude at most one. This leaves rounding headroom while preserving
the small constant identities lost by unconditional prescaling.

The two branches have the same mathematical factor; an axis contributes 1/N
and xD factors multiply across axes. Finalization does not divide again.
The policy remains in the shared low-precision traits and applies to FFT,
FWHT, real adapters, requested native/FP32 computation and physical stage
partitions. It does not promote arithmetic, scale the user's input or change
forward/unnormalized semantics. Its rounding order differs from both previous
builds, so general floating-point acceptance still requires fresh error review.
Outputs whose mathematical value exceeds the storage range remain unrepresentable.

## Evidence and experiment protocol

Original failures, the intermediate CPU trace, both prior binaries, all before/
after low-amplitude arrays and the intermediate performance diagnostic remain
under `results/numeric_plan_repair_20261009_v0933/`. In particular, the intermediate
low-amplitude XML records a completed error-report diagnostic, not acceptable
accuracy for the 16 zeroed subnormal outputs. Its first attempt failed before
execution because the new test incorrectly supplied a pointer for zero workspace;
the test was corrected and the first report retained.

Final-build verification and diagnostic timing are recorded separately under
`results/numeric_plan_repair_20261009_v0934/`. The exact constant identities are
regression assertions; other floating requests retain complete errors without
an adopted threshold. NTT uses full integer equality. No old-build pass or timing
can become a final-build measurement.

The [final GPU regression report](../results/numeric_plan_repair_20261009_v0934/gpu_regressions.xml)
contains 104 completed checks: 64 normalization cases, 32 low-amplitude exact
constant identities and eight NTT public identifier/replay/support cases.
The [original-request rerun](../results/numeric_plan_repair_20261009_v0934/original_failures/results.json)
has nine requests and no remaining cases: the NTT passes full integer equality,
and all eight original real inverse FFT failures have finite output with full
error arrays retained. The representative N=65536, batch-one natural case has
relative L2 error 0.07494%, p99 absolute error 0.001417 and maximum absolute error
0.002626. Its largest relative error near a reference zero is 18.34; the full
distribution is retained instead of treating aggregate L2 as an acceptance rule.
The two FP16 real-FFT baseline bindings remain missing in these eight cases;
this reproduction does not establish vendor speedups or matched FP16 accuracy.

The eight-point [same-core timing diagnostic](../results/numeric_plan_repair_20261009_v0934/same_core_latency_gpu40.json)
uses native FP16 normalized inverse FFT/FWHT on A100 40GB, three interleaved
trials per version and complete-batch checks. The old/new ratios range from
0.753 to 0.971 for FFT and 0.911 to 1.088 for FWHT. The largest FFT diagnostic
latency increase is 32.9% at N=65536, batch=16 (87.921 to 116.828 microseconds).
This range protection has a measured execution cost. It is a correctness repair,
not evidence of a performance improvement; the full comparison retains that cost.
Library hashes differ and each build's runtime fingerprint is stable across
trials and distinct from the other's. The previous CPU tooling suite remains
178 passed plus two subtests; no calibration/continuation source changed afterward.

The frozen population remains 38,416 semantic requests, including 232 explicit
rank-nine rejection boundaries. The recovery subset contains 11,802 affected,
previously failed or unvisited requests. Timing shards contain disjoint halves
of the unchanged full population. Each A100 80GB independently measures hardware
rates, scheduling, stage services and composition before its comparisons.
Every timed request freshly verifies the selected mapping, output and baseline.
Conversions are included; compilation, planning and CPU reference checks are
outside device timing. Separate scaled-input controls do not replace the four
original pressure requests whose reference exceeds FP16 range.

The versioned entry is
`results/semantic_recovery_20261009_v0934/run_campaign.py`. It checks manifest,
library, source, baseline and seed identities. An explicitly selected fresh
output suffix permits an inspected new attempt without overwriting previous
evidence. Failures stop their queue; there is no automatic foreign-GPU waiter.
Current phase and process liveness are indexed in
`results/active_experiment_queue.json`.

Both final-build supervisors started at 10:13 CST on October 9. Hardware probes
completed on both A100 80GB cards, and both entered mapping/stage/composition
calibration. The [GPU1 status](../results/semantic_recovery_20261009_v0934/gpu1/pipeline_status.json)
and [GPU2 status](../results/semantic_recovery_20261009_v0934/gpu2/pipeline_status.json)
are live phase records, not completion claims.

Calibration covers a finite population, not the entire legal design space.
Comprehensive performance and floating acceptance are pending until their
corresponding artifacts and reviews are complete. A100 40GB diagnostic timing
does not substitute for its comprehensive comparison; V100 remains user-run.
