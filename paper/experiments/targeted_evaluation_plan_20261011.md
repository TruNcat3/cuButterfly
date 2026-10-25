# Targeted evaluation plan for the cuButterfly paper

This plan defines the smallest new evidence set needed for the paper's three
claims: operator-level competitiveness, hardware-mechanism attribution, and
end-to-end application value. It is separate from the 38,416-case broad
acceptance queue. The broad queue remains useful for coverage and migration
auditing, but its completion is not a prerequisite for these three figures.

The [2026-10-12 audit and follow-up](targeted_evaluation_followup_20261012.md)
records the NTT denominator differences, QuaRot validity repair, deeper
mechanism controls, and CKKS integration gates. The original cohort below
remains frozen; the addendum does not imply those follow-up runs are complete.

## Identity and reporting rules

The first paper cohort uses the current `0.9.3.5-library-rebound-20261011`
build on an A100 80GB PCIe. Every row records the GPU UUID, library and source
hashes, calibration manifest, compile policy, semantic contract, and timing
protocol. Old results are used only as historical context or as a candidate
for reuse after an exact identity and protocol check. A different build or
timing contract creates a new row.

Kernel timing excludes compilation, calibration, search, plan construction,
first-use setup, host transfers, and correctness reference generation. These
costs are reported in a separate preparation table. Integer and NTT results
must match element by element. Floating-point rows retain the complete error
distribution and are not accepted by silently relaxing a failed threshold.

For every matched pair, report

* `speedup = baseline_latency / cuButterfly_latency`;
* median, geometric mean, quartiles, minimum and maximum over paired points;
* the number of wins, ties, losses, unavailable contracts, and correctness
  failures; and
* the paired point list and denominator for every aggregate.

No aggregate combines FFT, FWHT, NTT, and internal-only transforms. A missing
external implementation is reported as `no-matched-baseline`, not replaced by
an unrelated library or by a slower synthetic path.

The expected results are hypotheses, not pre-filled outcomes. The FFT target is
non-inferiority to cuFFT with a point estimate near parity and an opportunity
for a small positive margin. FWHT and NTT are expected to retain the positive
historical trend. Zeta and structured 2x2 are reported against matched in-tree
controls because no generally accepted external GPU baseline is currently
available. The final paper uses measured ratios even when they contradict these
hypotheses.

## E1: operator-level comparison

The main figure is one absolute-throughput boxplot with operator-separated
panels. Each panel contains cuButterfly and every compatible external baseline;
the paired-ratio plot and coverage matrix are the companion views.

| Family | A100 anchor cells | External comparison | Required semantic controls |
| --- | ---: | --- | --- |
| FFT C2C | 12 | cuFFT and VkFFT when the contract matches | FP32/FP64, forward/inverse, `logN` 12/18, plus FP32 large-N anchors |
| FWHT | 8 | Dao-AILab FHT where precision and direction match | FP32 external anchors and FP64/internal precision controls |
| NTT | 8 | GPU-NTT with identical modulus and output order | word32/word64, forward/inverse, `logN` 12/18 |
| subset/superset/xor zeta | 6 | matched in-tree scalar or stagewise control | uint32, forward/inverse or the operator's defined dual |
| structured 2x2 | 2 | matched in-tree baseline | FP32, two lengths, identical matrix and normalization |

The 36-cell target is a fixed anchor set, not a post-hoc selection of winning
points. Add a small batch/stride scale slice before freezing the manifest: one
small and one large batch for each external family, plus one non-contiguous FFT
case. If a baseline cannot represent a requested direction, layout, or modulus,
the row remains in the coverage table and is excluded from the speedup
aggregate.

The first current-build collection uses the existing public semantic runner:
10 warmup calls followed by seven CUDA-event samples, each containing 20
complete calls in a CUDA graph for out-of-place requests. In-place requests
use seven single calls with resets outside the events. Complete logical outputs
are checked before timing and after every sample. These seven samples are
within one process and are not seven independent process trials. Independent
process repeats are a subsequent stability check, reported separately.
The measured-search budget is 512 candidates, matching the current runtime and
GPU1 calibration binding; it is not an exhaustive-search claim. Compilation,
search and plan creation remain outside kernel timing. Freeze the manifest
before GPU execution and do not mix this protocol with historical 100/100
measurements.

The main acceptance statements are deliberately statistical. For FFT, report
the geometric mean and paired distribution relative to cuFFT and VkFFT and use
the predeclared non-inferiority margin of 5 percent as a diagnostic gate. For
FWHT and NTT, report the same statistics without hiding losing points. Internal
zeta and structured-2x2 results support coverage and mechanism claims, not an
external-library superiority claim.

## E2: matched NCU/NSYS mechanism evidence

Use the five cases already frozen in `paper/experiments/profile_rc2.json`:

1. FFT resident-prefix positive case;
2. FFT pipeline negative case;
3. FFT core substitution at `logN=18`;
4. FWHT register-resident case; and
5. word64 NTT large-N boundary case.

For each case, first record ordinary CUDA-event timing under the exact mapping,
then run NCU and NSYS on the same UUID, build, input, and correctness journal.
The profile cohort reports kernel count, launch interval, registers per thread,
shared memory, active warps, DRAM/L2 traffic, barrier stalls, and long-scoreboard
stalls. Profiler duration is diagnostic and never replaces the ordinary timing.

The figure has one panel per mechanism and shows the matched before/after pair,
the latency ratio, and the resource change that explains it. A positive case,
a near-tie, and a negative case are all retained. The result must explain why
the dataflow helps under one resource regime and fails under another; it must
not turn one profile into a universal claim.

## E3: end-to-end application value

The existing E13 contract supplies six payloads and should remain the single
application manifest:

| Application | Butterfly | Main payloads |
| --- | --- | --- |
| NeuralOperator FNO Darcy | real FFT | 64x64 and 128x128 |
| QuaRot Llama-2-7B | FWHT | prefill and decode |
| HEonGPU CKKS bootstrap | NTT | N=4096 smoke and N=16384 resource case |

The baseline and treatment use the same pinned source, model parameters or
deterministic random weights, input tensors, precision, stream configuration,
and output-quality checks. Only the butterfly backend changes. Report complete
GPU-resident application latency as the primary metric, with setup, compile,
first-use, adapter/conversion overhead, peak memory, and quality checks in
separate columns. Use three process trials, five correctness warmups, and ten
steady-state calls.

The real-FFT adapter and the HEonGPU RNS/root/order/stream contract are gates.
If either is not valid, the payload is explicitly `blocked` and does not receive
a substituted C2C or reduced-precision result. FNO and QuaRot can be reported
independently when their contracts pass. An operator speedup never substitutes
for an application measurement.

## A100 execution order

1. Finish CPU-only manifest freezing, source/build identity checks, and external
   baseline availability checks.
2. Reserve GPU1 (A100 80GB) for E1 and then E2, keeping its UUID and output
   directories independent from the broad acceptance queue. GPU2's broad queue
   remains separately managed; an abnormal stop must be audited before resume.
3. Run E3 after the selected backend adapters pass correctness. Run blocked
   payloads only after their stated gate is cleared.
4. Generate the operator boxplot, five-case profile figure, application paired
   latency figure, and one semantic coverage table. Do not merge historical
   rows into the v0.9.3.5 aggregates.
5. Only after the A100 cohort is internally consistent, repeat the same anchor
   manifest on H100 and Ascend with fresh calibration and platform-specific
   baselines. Keep each platform's ratios separate; a combined mean would hide
   migration failures.

## H100 and Ascend follow-up

H100 and Ascend are portability cohorts, not substitutes for missing A100
evidence. Reuse the 36 semantic anchor descriptions, regenerate executable
lowerings, recalibrate the hardware services, and revalidate every mapping.
For H100, compare only matched cuFFT/VkFFT/Dao/GPU-NTT contracts. For Ascend,
first inventory CANN and operator availability; unsupported external contracts
remain explicit gaps. Report calibration, compilation, search, and first-use
costs separately from steady-state performance.

## Completion checklist

The paper evidence set is complete only when every E1 row has a correctness
record and a paired timing or an explicit unavailable reason, all five E2 cases
have normal timing plus matching profiler identity, and every E3 payload has a
baseline/treatment pair or a documented blocker. The final text must state the
actual FFT margin, FWHT/NTT distributions, internal-only baselines, and blocked
applications rather than replacing them with the expected outcome.
