# Unified framework implementation

> Native FP16 local-core and factor integration, 2026-11-05

The [integration report](fft_l_local_core_20261105.md) now includes completed
local-core geometry, packed I/O and multi-stage factor studies. Native FP16
cuFFTDx half2 execution reuses the shared factor layout, tile loop, prefetch and
slice scheduler, with separate logical-transform and physical-thread counts.
Forward and unnormalized inverse are supported. Normalized inverse, native
BF16 and explicit FP32 accumulation remain unsupported by this factor provider.

The bounded large-N campaign completed 76 measurements and four NCU captures
on A100 80GB; all outputs were finite, with CPU errors reported without a new
threshold. Best explicit factor points range from 0.754x to 1.070x cuFFT and
1.740x to 3.732x the original scalar path. N262144/B32 reaches 145.459 us versus
155.597 us for cuFFT. Holding the local core and factor geometry fixed, the
three large-batch cases improve 1.325--1.523x over one-tile/no-prefetch execution.
These are bounded measured candidates, not a newly calibrated AUTO result.

Validation completed 56 FP16 factor cases, 288 old/new local-I/O checks at each
of two lengths, and 16 FP32/FP64 factor regressions. A separate six-task public
I/O confirmation and four NCU captures also completed. CPU compilation and
reference preparation preceded GPU measurement; the GPU phase used cache-only
module loading. The finite monitors have exited successfully. Evidence lives in
`results/a100_fp16_factor_20261105/`; its diagnostic library receipt is in
`build/receipt.json`. That diagnostic stage did not replace the shared conda installation.
The subsequent bounded stage calibration is complete in
`results/a100_fp16_acceptance_20261105/stage/checkpoint.json`: all 16 configurations
are described, with 334 records and 330 measured points. Coverage and validation
are complete, with no adapter gaps, omitted configurations or pending descriptions.
The acceptance library SHA256 is
`775a85c8f76e44bbfee886fd90489e7c309bd097b60e968183e446aa006177b2`,
with research runtime identity
`3f24ebf748992e13153782f818b5503a1438510555609a1b5291af913edb3de5:research`.
Earlier diagnostic timings retain their original build identities.

`stage/model-receipt.json` reports a usable `calibrated-local-serial` model for
the explicit current native FFT-L mappings and validated measured intervals.
Across 72 valid multi-candidate holdout workloads, top-1 accuracy is 94.444%,
whole-plan relative error is 0.712% median / 6.993% P90, and mean latency regret
is 1.00683. Independent stage-service holdout error is 0.938% median / 6.986% P90.
The finite acceptance completed on November 6: production seed-only confirmation
covered 12 workloads / 32 mappings and promoted 32 records. All nine public
DEFAULT, nine MEASURE and nine installed DEFAULT calls resolved identical mappings.
Four installed semantic controls and three legacy-installed FP64/FWHT continuity
cells also completed; the shared conda installation is `installed-validated`.
`status.json` is `complete` with no remaining tasks; `summary.json` retains
`complete-error-report-only` rather than introducing a floating threshold.

The nine DEFAULT/cuFFT ratios have GM 0.853583x, range 0.733471--1.066737x,
with 1/9 nominal wins. Separate MEASURE and installed DEFAULT GMs are
0.853388x and 0.855909x; no minimum is taken across paths. The
[public acceptance evidence](../experiments/reports/FP16_PUBLIC_ACCEPTANCE_20261105.md)
lists all nine latency pairs, resolved designs and source hashes.

Two repairs addressed false seed rejection of the provider's radix2-to-auto alias
and an eager source-frontend import during CPU error-metric loading. Frontend
access is now lazy; the CPU reference regression suite passed 12 tests.
BF16 has semantic smoke coverage here, while its full automatic external comparison
remains pending. These nine requests do not close the full A100 matrix, change the
original 66-request means or relabel historical diagnostics as this build.

> FFT-L opt-in capacity and calibration adapter completion, 2026-11-04

The current [FFT-L completion campaign](../results/a100_fft_l_completion_20261104/README.md)
uses frozen library `79d9b09abdf4ba24b4501af0a71941944732e7a6bb5861800e84560e716f7976`.
Scalar temporal R2/R4/R8 now opts into the hardware's dynamic shared-memory
capacity and compiles through log16. A100 native low FFT N16384/N32768 can use
64/128 KiB single-stage points; N65536 still exceeds that hardware capacity.
This supersedes the previous implementation's 48 KiB temporal launch limit,
without changing online-reorder's separate grouped-tile contract. Candidate
selection and explicit execution share the same capacity predicate.

The stage microbenchmark now accepts Complex16/ComplexBf16 FFT with native or
explicit FP32 arithmetic. Same-precision CPU reference errors are report-only;
its `correct` field certifies finite output and byte-identical whole-plan versus
independent-group composition, not an agreed floating error threshold. Twelve
CPU reference tests match the existing Python oracle at three sizes each.
Forty-eight affected CPU tests and 144 GPU impulse/guard cases pass, including
first-use capture for 64/128 KiB FFT/FWHT kernels across supported precisions.

Calibration has begun for 34 native FFT-L mapping configurations. Existing
hardware rates retain their provenance; physical group timings are reacquired.
Completed clean rows survive foreign GPU interruptions. Validation is not yet
complete, and no installation or usable-model promotion is claimed. The queued
order is stage acquisition, model evaluation, the 22-request scale matrix and
the original 32-request 1D/2D precision/direction cohort. Full raw outputs and
same-precision CPU references are retained; measured timings exclude planning
and compilation. See campaign-status.json in the completion directory for the
current queue phase. A negative validation result is reported, never rewritten.


> FFT-L single-stage endpoints and bounded scale check, 2026-11-04

The [scale and selection report](../results/a100_fft_l_scale_selection_20261104/REPORT.md)
separates fixed organization controls from public selection. Single-stage execution
and expansion factors of one remain legal mixed-dataflow endpoints. Public
MEASURE chooses a scalar single-stage radix-2 point for native FP16 N4096 B256;
a larger batch does not make the measured two-stage point universally better.
For N8192 B32, grouped two-stage radix-4 beats the same-core single-stage control,
while replaying the public hierarchical winner is slightly faster still.
The report records the complete bounded matrix and actual mappings as it fills.

Scalar temporal log13 was missing from compiled dispatch despite fitting 32 KiB
for native low-precision FFT. Dispatch and the common capacity check now include
it; 45 exact-impulse GPU cases cover relevant FFT/FWHT precisions, radices and
forward/inverse normalization. The unchanged non-opt-in 48 KiB limit remains a
lowering limit, not a theoretical restriction on larger single-stage transforms.

The shared campaign Graph timer now binds replay and events to the supplied
stream. Timed output must match the same plan's direct output bytes before
reference error reporting. Two temporary probes run outside the required caller
stream context had invalid timings and are explicitly excluded. Properly scoped
previous controls remain separate evidence. Thirty-six affected CPU tests pass.

Static ranking in this experiment uses hardware-rate priors: the selected A100
profile directory has no cost_model.json. Existing refit/tuning models cannot be
promoted for native FFT-L because of runtime identity, validation status and/or
precision coverage. Public MEASURE is independent measured selection. Neither
full native FFT-L calibration nor installation of this build is claimed complete.


> Scalar online boundary grouping, 2026-11-04

The [bounded implementation report](../results/a100_fft_l_boundary_20261104/REPORT.md)
adds independent prefix `data_tiles_per_cta` and suffix `reorder_columns` to the
shared scalar online lowering. Adjacent output lanes now own adjacent subgraphs;
an internal XOR shared-memory permutation supports that ownership exchange.
Power-of-two geometry uses shifts/masks instead of runtime integer division.
Arithmetic traits and per-primitive low-precision rounding are unchanged.

Execution, resource projection and public candidates use a common capacity
contract. The former 1024-value group cap is replaced by actual byte capacity
within the executor's 48 KiB non-opt-in limit, independently of compiled local
stage sizes. The contract accounts for normalized integer-zeta storage widths.
The tuner can grow or shrink either grouping axis; this is a generic scalar
mechanism, not an FFT/GPU-name rule. Imported FFT cores and the separate NTT
executor retain their existing paths.

On A100 80GB, all 24 new fixed controls match old-library output bitwise. At
N4096 B256, FP16 changes from 125.901 to 40.755 us and BF16 from 128.000 to
43.110 us when selecting the measured grouped configuration. These are controlled
two-stage results, not AUTO selections or improvements over the overall best
known implementation. Keeping G1/C1 at large batch regresses about 4--5%; small
batch does not benefit from grouping. A matched 40GB one-stage control remains faster
than its measured two-stage point.

A separate A100 40GB mechanism check uses the grouped implementation. FP16 N4096 B256 drops
from 126.259 to 43.162 us with the same scalar radix-4 arithmetic. Paired NCU
captures show both stages' store sectors/request fall from 32 to 4, while total
warp instructions fall from 34.316M to 11.477M; arithmetic instruction counts
remain equal. Shared-bank conflicts are not eliminated and cuFFT remains faster.
Do not combine 40GB timing with the 80GB comparison matrix. The final CTest passes
80 bitwise pairs plus four invalid-configuration checks; 23 affected Python tests
and the mixed-dataflow resource test also pass. Twenty-four affected non-online
controls pass after keeping their single-tile geometry compile-time constant.
A100 80GB resumed after a busy interval; all 18 larger-group controls and six
final-build confirmations are complete. Public retuning and a new comprehensive
matrix remain pending. Eighteen proposals across six requests, including
one-stage rivals, are CPU-prepared; no installed-cache calibration claim is made.

> FFT-L instruction and boundary attribution, 2026-11-04

The [focused trace report](../results/a100_fft_l_trace_20261103/README.md)
contains 14 guarded NCU captures and 30 ordinary Graph controls with the same
production build. The native low-precision scalar online path is not equivalent
to the high-precision register core: its prefix/suffix writebacks use about 32
sectors per warp request versus cuFFT's four, and its named writer-aligned option
only permutes the intermediate address bits. It does not remap final-store thread
ownership or remove the observed transaction fragmentation. Native FFT-L still
lacks the register-tile and imported block adapters used by FP32/FP64 paths.

At FP16 N4096 B256, the frozen two-stage point takes 124.877 us, versus 11.213 us
for cuFFT. This is a fixed-organization diagnostic, not re-searched large-batch
library performance. Keeping the scalar radix-4 core and T256 but selecting one
stage reduces it to 34.202 us; changing only the two-stage thread count to T64
reduces it to 78.438 us. At this batch, the original two stages have 85–86%
active-warp occupancy, so their slowdown cannot be explained by too little
independent work alone. Dynamic warp instructions remain about 29 times the
external implementation. Two-dimensional execution separately retains two
explicit transpose kernels. These results qualify the earlier broad statement
that only a core gap remained: lowering coverage, boundary layout, geometry and
selection all matter. No production code or selection cache was changed during
this diagnosis, and no claim of complete theoretical optimality is made.

> A100 weak-path closure and FFT-L cache confirmation, 2026-11-03

The final A100 80GB build now contains compile-time FFT-H codelet coefficients,
the generic rank-2 direct-input boundary, the shared scalar temporal-tile capacity
contract, and explicit round-to-nearest BF16 primitive traits. The attempted FWHT
writeback interleaving was correct but neutral in three-process timing and was
reverted. The affected controls pass on the final build: 15 temporal-domain
capacity/direction cases, six mixed input/output order rank-2 cases, 60 fixed
register-tile cases, and the BF16 primitive bit-pattern study.

The low-precision public selector initially exposed a real engineering issue:
the static rate prior ranked a newly legal one-stage point above the measured
two-stage point. Official `Plan(MEASURE)` then measured a bounded 32-candidate
cohort per FP16/BF16 request and wrote the shared cache. `Plan(DEFAULT)` replayed
the same mappings, and the installed conda library reproduced them in three
independent processes without changing the cache. The selected point is two
stages `[5,7]`, tiles 32/128, T256, scalar radix-4, direct-strided boundaries.
Installed medians are 9.677 us (FP16) and 9.728 us (BF16) for N=4096, B=4.
This is a measured-cache confirmation for two requests; it is not a full model
refit or a proof of global search optimality. The public MEASURE timer and final
Graph timer are recorded separately, and the existing low-precision internal
relative-error screening remains distinct from the report-only CPU error policy.

The remaining FFT-L gap is therefore localized to the scalar low-precision core,
stage synchronization, and two-dimensional transpose path. The current controls
show that removing one rank-2 input copy improves 2-D execution, while the
one-dimensional path remains slower than cuFFT. This is an engineering/core
coverage gap for the finite requests, not evidence against the hierarchical
hybrid dataflow method. Full A100 cross-operator acceptance and PRO 6000
migration remain separate milestones.

> Focused remaining-gap diagnosis, 2026-11-03 evening

The [current instruction-level trace](../results/a100_closure_20261103/final_acceptance/trace/README.md)
adds four guarded NCU captures without changing the accepted performance records
or production code. FFT's fixed local phases still execute FP64 polynomial
arithmetic; a CPU-only same-radix compiler probe reduces FFT16 static instructions
from 504 to 256 and registers from 96 to 76 with literal coefficients. This is
code-generation evidence, not a measured full-transform speedup. The prior
diagonal-table fix did not cover these codelet/lane sites. FWHT's PC samples
localize most LG-throttle symptoms to concentrated vector output stores; Dao
interleaves those stores with later butterfly arithmetic. Aligned I/O widths and
counts match, so scalarization does not explain that case. FFT-L's separate CPU
audit identifies small-tile geometry and axis-boundary work but has no dedicated
low-precision NCU attribution yet. No new migration or unbounded sweep started.

> A100 affected-request closure, 2026-11-03

The [acceptance report](../results/a100_closure_20261103/final_acceptance/README.md)
closes the new strategy calibration and public-call checks. The finite competing
cohort has 34 certified physical service keys and 68 independent group holdouts.
All 15 complete-plan holdouts are covered and configuration ranking matches the
measured winner in all three request cells. The refitted model is validated for
this serial/direct-launch scope; uncovered requests and other timing protocols
retain the existing compatibility guards. Public DEFAULT uses current confirmed
records, and public MEASURE accepts all 24 scheduled candidates. The shared conda
library, templates, profile and registry are updated; installed default execution
also completes full CPU-reference checks. The two external weak points remain
0.838x cuFFT and 0.939x DaoFHT under matched complete Graph timing.

FFT-L now has declared same-precision CPU references for all 32 saved requests and
48 output comparisons. The regular runner invokes this CPU sidecar finalization
and supports `--finalize-cpu` without CUDA. No float threshold or vendor arithmetic
contract is invented. Calibration checks contradictory known runtime identities
before further GPU acquisition; 49 targeted tests pass, alongside 30 FFT-L tests.
The frozen main/application evidence remains associated with its original builds.
PRO 6000 migration has not started. Remaining kernel optimizations are research
extensions, not silently added acceptance gates for this finite cohort.

> Shared per-group thread geometry, 2026-11-03 afternoon

The [stage-thread report](../results/a100_closure_20261103/stage_threads/README.md)
closes the shared per-group launch gap described below. Physical group mappings
now reach dynamic/JIT launches, probes, batch overlap, common candidates, GA and
service identities for Butterfly and NTT. Forty-four cross-operator GPU checks
completed, with exact integer validation and floating error reports. On A100
80GB, fixed-scalar FFT FP64 N=2^20 B=16 with `[8,8,4]` stages improves from
3.852206 ms at threads `[128,128,128]` to 2.834104 ms at `[128,128,64]`, a 1.359x
gain. This also improves the preceding uniform-64 result by 1.110x. NCU confirms
the first two stages retain their 128-thread performance while the tail improves.
These are controlled explicit mappings, not a new external-library comparison
or evidence that automatic selection already picks the winner. Internal multi-tile
packing, service calibration, prior refit and FWHT scheduling remain open.

> Local FFT coefficients and shared thread geometry, 2026-11-03

The [bounded optimization report](../results/a100_closure_20261103/optimization/README.md)
records a completed A100 80GB intervention after the negative-result trace.
Register FFT prefixes now expose independent runtime/table local diagonal-root
supply through mapping, compilation, inventory, GA and stage service identities.
The table keeps the requested FP32/FP64 arithmetic and is prepared before timing.
For FP64 N=2^20 B=16, three-process complete Graph medians improve 1.150628 to
1.077156 ms (1.0682x); cuFFT remains faster at 0.900680 ms. Twenty-four small
same-precision CPU error reports cover both directions and three prefix shapes.
Public inventory contains the table geometry; automatic selection is unconfirmed.

The independent fixed-scalar-core `[8,8,4]` selective-resident experiment improves
3.851684 to 3.146066 ms by changing all groups from 128 to 64 threads. NCU shows
the 16-point tail improves while the first two groups regress. Per-group shared
thread geometry and internal multi-tile packing remain lowering/interface gaps;
neither is implemented by this thread-count intervention. Shared v8 prior refit,
new prefix service coverage and FWHT scheduling remain open. Earlier three-process
80GB regression and all 24 cross-operator execution checks also completed; the
expired-wait notes below describe their earlier state, not the current status.

> Shared organization repair, 2026-11-02 night

The user renewed the wait for performance verification. A detached UUID guard
now waits until 2026-11-03 09:00 Asia/Shanghai, then expires if the target remains
busy. Its [live status](../results/a100_closure_20261102/framework_repair_20261102/validation_until_20261103_0900/status.json)
and eventual comparison summary belong to the fixed two-request regression below.

The [implementation and validation ledger](../results/a100_closure_20261102/framework_repair_20261102/README.md)
records the common partition priorities, per-group resident candidate basis and
GA toggles, corrected leaf geometry, and the serial service view for a one-chunk
shared pipeline. The library rebuild and affected CPU checks pass, including
isolated installation dependencies. The v8 hardware-rate prior remains a heuristic:
it identifies the excessive FWHT thread allocation but does not recover the
observed FFT plain-tail ordering. Refitting/validation and current-build A100 80GB
performance confirmation remain open; the finite wait expired with a foreign job
on the target GPU. A separate A100 40GB run completed 24 shared/resident execution
cases across all six operators, with exact integer checks and same-precision float
error reports. Public inventory export also contains the selective FFT resident
mapping and both FWHT thread configurations. Actual finite-budget selection and
new 80GB timings remain unconfirmed. Historical intervention timings remain
historical evidence.

> A100 negative-result diagnosis, 2026-11-02 evening

The [diagnostic report](../results/a100_closure_20261102/negative_trace_20261102/README.md)
records six NCU captures and three bounded, single-axis comparisons on the original
A100 80GB. Reducing FWHT threads from 1024 to 128 at the same 256-point tiles improves
108.022 us to 49.992 us. FFT resident leaves help its first two 256-point groups but
hurt the 16-point tail; retaining the plain tail improves 4.347832 ms to 3.852124 ms
with the same scalar core, partition, layout and threads. Execution supports this
per-group choice, but public enumeration and default GA resident neighbors omit it.
The finite random screen also heavily sampled leading one-stage groups, while the
FWHT overlap-labelled winner actually executes one batch chunk sequentially.
These are diagnosed coverage/mapping issues; production selector repairs remain
outstanding. Original negative measurements are retained. The corrected FFT
winner's separate full numerical replay also finished, without changing timings
or introducing a float threshold.

> A100 winner-statistics correction, 2026-11-02

The finite measurement queues finished, including all 32 FFT-L requests and
the repaired three-request fixed-core study. Review found that final float
winner selection compared CSV latency strings: 12.470016 sorted before
4.344893. Selection now parses finite positive numbers before both median and
minimum. Recorded timings and the numeric shortlist selection were unaffected.
The corrected FFT conventional winner's new full error report is complete; the
other three float reports are reusable and NTT retains full-batch exact checks.
See the [correction ledger](../results/a100_closure_20261102/winner_correction_20261102/REPORT.md).
The limited fixed-core conventional/hybrid ratios are 0.425384x FFT, 0.480422x
FWHT and 1.773924x NTT. These are internal organization comparisons, not external
library speedups. The controlled runner also adds clearly directed ratio fields
while preserving the historical aliases and numeric values.

> A100 finite completion and execution preflight, 2026-11-02

The [current completion ledger](../results/a100_closure_20261102/README.md)
separates finished measurements from the remaining application, mechanism and
FFT-L jobs. All 138 searched composition requests completed 552 exact checks.
Their 1.691x geometric mean compares searched cyclic mappings with the internal
default composition, not an external library or a fixed-core ablation. The
34 CPU external comparisons also finished, with two independent processes and
two additional instability checks. CPU/GPU and GPU/GPU ratios remain separate.

An A100 40GB execution smoke found a profile-script error that passed NTT-only
modulus parameters to floating transforms. The script now applies those fields
only to NTT; actual FFT/cuFFT, FWHT/Dao and NTT/GPU-NTT calls pass their numerical
checks. These small diagnostic requests do not enter the A100 80GB comparison.
FFT-L CPU preflight validates all 32 public descriptors and the 16 compatible
cuFFT storage contracts; GPU execution and the same-precision CPU oracle gap
remain distinct. The [application progress table](../results/a100_closure_20261102/applications/progress_summary.md)
records new measurements, including QuaRot losses, without attributing steady
Graph time to the excluded planning cost.

All nine application repetitions now completed. QuaRot B1/B4 remain slower
(0.968x/0.958x descriptive process means), while the new FP32 filtering pairs
reach 1.068x/1.048x at 1024²/2048², B16. The small bootstrap repeat reaches
1.033x/1.032x at B1/B8 with exact ciphertexts and the original `sec_level=none`
restriction. Its missing temporary NTL dependency is now recoverable through
[`prepare_heongpu_dependencies.py`](../scripts/prepare_heongpu_dependencies.py)
into persistent local storage; the restored ABI matches, but the historical
package identity is unproven. The remaining mechanism/profile and FFT-L jobs
continue under the finite GPU guard. A separately queued C0 recovery handles
equivalent empty default fields in public mapping exports, retaining original
candidates, selection budgets and conversion audits.

> Cyclic NTT completion and persistent baseline recovery, 2026-11-02

All 90 cyclic NTT cuButterfly requests completed exact verification. The
[baseline recovery report](../results/full_current_20261025/recovery_gpuntt_baseline_20261102/REPORT.md)
separates 18 existing GPU-NTT comparisons, 36 arithmetic-domain exclusions and
36 bindings lost after the reboot cleared temporary vendor files. GPU-NTT now
has a persistent, configurable build with binary-bound source evidence. A finite
baseline-only supplement completed all 36 missing bindings on the original A100
80GB, reusing the inputs, exact verification and Graph timing without repeating
cuButterfly search or modifying accepted rows. The combined 54 valid pairs have
1.308x geometric mean speedup and 51 measured wins; three 64-bit B=1 cases range
from 0.972x to 0.999x. The 36 true arithmetic exclusions remain unpaired. CPU
verification covers 40 bridge checks and 62 supplement/timing/monitor checks.
The idle monitor exited normally after completion. These results concern this
cyclic NTT population; other operator and application scopes remain separate.

> Follow-up repair and retest, 2026-10-28

The [repair report](../results/full_current_20261025/recovery_followups_20261028/REPORT.md)
records completed mechanism timing and profiler captures after repairing stale
preparation identities. The crypto stop was a pinned GPU-NTT 62-bit arithmetic
defect; cuButterfly was exact on the failed request. Adapter guards now check
both modulus range and vendor bit metadata. Four supported forward/inverse GPU
regressions pass, and the two unsupported boundary requests are rejected.
The idle queue retains all 66 main comparison rows, 46 public composition cells
and two cyclic rows while continuing the unfinished follow-ups. Their full
performance conclusions remain pending.

> First-run event-log recovery, 2026-10-28 morning

The [recovery record](../results/full_current_20261025/recovery_event_log_20261028/REPORT.md)
documents a regression in the preceding proxy change: a fresh comparison could
start ranking before its event-log directory existed. The proxy now opens that
log before spawning and reaps its child on receipt-write failure. All 97 affected
CPU tests pass, including the first-run condition omitted by the earlier fixture.
The failed attempt produced zero usable comparisons and remains archived. CPU
preparation reached 5,823 ready modules with no pending or failed requests. The
original finite comparison resumes with completed calibration preserved; module
counts remain separate from measured rows in the live campaign ledger.

> Ranking signal receipts and independent CPU preparation, 2026-10-27 evening

The [follow-up recovery](../results/full_current_20261025/recovery_signal_20261027/REPORT.md)
reproduces the worker/proxy/helper cancellation chain and records exact per-attempt
signal evidence. Only verified cancellation recovers the generic ranking error;
ordinary runtime, compiler and numerical errors remain investigation-required.
The full refresh can attach to an independent CPU producer, so GPU occupation no
longer terminates that producer's compilers. Completed comparison phases do not
require a live producer on resume. The affected 171 CPU tests pass, including real
process-group cancellation and ownership/liveness tests.

All 24 reviewed compiler failures contained `nvcc: Terminated`; their logs and
requests were archived before requeueing. The same A100 manifest and follow-up
queue are restored under the idle monitor, preserving completed calibration and
all numerical/timing policies. Live measurement progress remains separate from
CPU module readiness and is recorded in the campaign ledger.

> A100 comparison recovery and research scope, 2026-10-27

The [recovery record](../results/full_current_20261025/recovery_interruption_20261027/REPORT.md)
documents the foreign-GPU interruption fix and guarded restart of the existing
66-request comparison. Both runner files and their 87 CPU tests distinguish
retryable GPU occupation from numerical/runtime failures. Completed calibration,
finalization and xD preparation remain intact; the original zero-row interrupted
attempt and diagnostics are archived. Only the reviewed runner identities changed,
with no kernel, semantic, budget or timing-policy change. The idle monitor resumed
the comparison on the same A100 80GB UUID; live progress is in the campaign ledger.

The [research principles](research_principles.md) and paper now emphasize resource
characterization under hardware constraints. Search is engineering, and ranking
competitions are no longer mandatory research evidence. Finish the defined A100
experiment list before the planned PRO6000 migration. Internal mechanism losses
and external library comparisons retain separate scopes.

> Radix-4 precision, compact-real and long-1D completion, 2026-10-24

The [completion report](radix4_completion_20261024.md) records 24 public xD
requests and 16 fixed-mapping long-1D requests on an A100 80GB. The pair core
now supports native FP64, compact-real plans can pack two complex axes, and
mixed-radix register leaves reach JIT, candidate enumeration, calibration seed
import and post-install tuning. Public automatic 32×32 C2C selection reaches
1.588x FP32 and 1.415x FP64 geometric mean against complete cuFFT; FP64 chooses
existing cores rather than the new pair. Compact-real means are 1.136x/1.102x.
The separate fixed-mapping long-1D study reaches about 1.064x in each precision,
with 4/8 FP32 and 5/8 FP64 wins. All 40 requests completed numerical error
reporting and exact replay checks. Targeted GPU regressions and memory/sync
checks passed. Broader pair sizes, long-1D automatic tuning, full weak-region
acceptance and general installation-model calibration remain open.

> Public xD radix-4 pair integration, 2026-10-24

The [public integration report](public_radix4_integration_20261024.md) connects
the validated register radix-4 pair core to the common xD lowering, candidate
search, cache replay, and Python API. On an A100 80GB, the public selector chose
the new pair candidate for all six `32×32` FP32 cases and measured `1.252–2.191x`
against an explicit whole-transform cuFFT provider. A `[8,32,32]` sample chose
the existing mixed-axis plan instead, preserving context-sensitive selection.
The report retains a direct cuFFT baseline for rank two, PyTorch for rank three,
full error distributions, and explicit rejections for FP64, unsupported sizes,
and unknown pair cores. Long 1D, compact-real pair packing, and broader shape
coverage remain separate implementation work.

> Runtime RNS tuning budget and cache replay, 2026-10-21

The [runtime tuning report](../results/rns_tile_io_20261021/README.md) adds the
public `CUBUTTERFLY_RNS_TUNE_REPETITIONS` budget control while retaining the
lightweight default of five. A fresh 12-repetition cache prepared the actual
runtime semantic requests, which differ from the older exported manifest in
placement and domain descriptors. A subsequent `require` run replayed all 118
runtime plans without tuning. Complete CKKS Graph bootstrap measured 1.034x at
B=1 and 1.033x at B=8 with exact ciphertext agreement. This closes the cache
integration gap for the measured application, but it does not establish global
optimality or finish the general RNS cost-model integration.

> RNS stage specialization and CKKS improvement, 2026-10-21

The [stage-role report](../results/rns_stage_role_20261021/README.md) records
compile-time first/last and prefix/suffix roles in the existing fused lowering.
All 118 actual requests were retuned and replayed from required cache entries.
The six selected N4096 requests reach 1.131x geometric mean against contiguous
GPU-NTT, with five wins and the small forward loss retained. Complete CKKS
Graph bootstrap reaches 1.033x at B1 and 1.032x at B8, with exact ciphertext
agreement. Trace-weighted NTT improves to 1.073x; non-NTT names, order and
launch geometry match the native path. Small requests and medium forward
requests remain weak. Artificial cache-access controls are reported separately.
This measures lowering plus retuning, not a fixed-mapping mechanism ablation.
General RNS model integration and broader application acceptance remain open.

> RNS single-warp coverage and complete CKKS attribution, 2026-10-21

The [new report](../results/rns_single_warp_20261021/README.md) adds 32-thread
fused candidates and transactional mapping changes, with 241 exact cases and
focused sanitizer checks. All 118 actual requests were retuned; none selected
32 threads, so this closes a coverage gap without establishing a speedup.
Complete Graph bootstrap remains about 1.01x. A full trace attributes most of
the limitation to small/medium NTT requests: 434 of 516 calls have at most 64
rows and lose in aggregate, while larger requests win. Non-NTT kernel names,
geometry and order match; no gather/scatter kernels appear in this path.
An explicit contiguous GPU-NTT control also exposes the difference between
standalone ModulusOrdered measurements and the application's mixed dispatches.
Actual-workload core efficiency, context-sensitive selection and general RNS
model integration remain open; historical headline speedups retain their scope.

> RNS dependency-scoped exchange, 2026-10-21

The [follow-up report](../results/rns_exchange_20261021/README.md) adds an optional,
geometry-derived warp/CTA barrier policy to RNS selection and persistent replay.
All 217 exact RNS cases pass, with focused racecheck/synccheck acceptance.
Offline preparation covers 118 actual requests; both full-bootstrap processes
replay all 118 without tuning. Six operator points range from 0.901x to 1.262x
against native GPU-NTT. Full B1/B8 Graph bootstrap is about 1.011x, a small gain.
Controlled exchange changes alone produce only small effects; NCU still shows
high CTA barrier waiting in the small forward case. Register/warp core ownership,
32-thread candidate coverage, general RNS model integration and exception-safe
mapping transitions remain open; this is not a full-library completion claim.

> Prepared RNS replay and actual CKKS requests, 2026-10-21

The [implementation and measurement report](../results/rns_prepared_20261021/README.md)
records persistent request-specific mappings, the additive Python/C ABI, and
shared scalar/stage preparation. All 118 actual bootstrap requests replay in
a new process without tuning. Full Graph bootstrap at B=1/8 is essentially tied
with native HEonGPU; six real-request NTT diagnostics retain both the 0.876x
small forward case and gains up to 1.259x. NCU points to excess CTA barrier
waiting in the small case. General descriptor/model integration, dependency-aware
warp exchange candidates and complete weighted attribution remain open.
See the [integration guide](rns_prepared_integration.md) for reproducible use.

> Explicit CKKS stage batching and matched controls, 2026-10-19

The [integration report](../results/ckks_stage_batch_20261019/README.md) connects
bound RNS batches to complete independent-ciphertext DAGs. Each scalar capture
owns its scratch; explicit NTT markers become private gather/NTT/scatter stage
graphs while all other nodes and dependencies are preserved. Native GPU-NTT
uses the same batching control. All 36 configurations and four execution paths
pass exact ciphertext checks. The 24 B=8/32 configurations beat both native
loop and native batched controls; forced B=1 batching loses, and two large-N
rotation cases prefer cuButterfly's own loop. The report retains thermal
diagnostics and uses completed thermally guarded repeats for final statistics.
Application schedule selection, global RNS descriptor/model integration and
full CKKS/TFHE bootstrap remain separate work.

> Finite comparison and NTT search repair, 0.9.3.11-search.1, 2026-10-17

The [66-request comparison](install_comparison_results_20261017.md) has completed
measurement and same-precision review. The [follow-up](ntt_search_repair_20261017.md)
identifies an NTT family omitted by global budget subsampling, replaces that
sampler with family quotas, and adds exact grouped external FFT adapters.
Historical headline populations remain separate. Public finalist confirmation
and remaining GPU baseline gaps are tracked in the follow-up report.

> JIT runtime scaling diagnosis, 0.9.3.4-jit-runtime.1, 2026-10-10

The [runtime investigation](jit_shared_runtime_20261010.md) isolates another
batch-dependent search bottleneck: thousands of JIT modules each link a private
static CUDA runtime, inflating stream/event/launch costs. Shared-runtime modules
pass real FFT/FWHT/NTT GPU checks. The complete same-GPU N256/B16 FP32 search
fell from 2,394.044 to 386.144 seconds (6.200x), with identical candidate order
and verification outcomes. Both 80GB GPUs are recalibrating under the new
runtime; prepared full-matrix launchers follow their completion automatically.
Original results keep their identities. Search wall-time improvement does not
establish operator-throughput improvement or close full-matrix goals.

> Persistent queries and module reuse, 0.9.3.4-async-experiments.3, 2026-10-10

The [experiment update](async_fast_dispatch_20261010.md) removes repeated Python
startup from ready-module queries and groups complete test cases by module
family. The independent compiler finished all 72,654 known requests; remaining
host overhead came from query dispatch, loading and partial-search recycling.
The optional service preserves broker validation and resource guards. The
handoff retains completed evidence with original identities; full comparison
and floating accuracy acceptance remain separate obligations.

> CPU preparation, 0.9.3.4-cpu-preparation.1, 2026-10-10

The [CPU preparation follow-up](async_recovery_20261010.md#cpu-preparation-follow-up)
replaces repeated whole-queue scans with an independent incremental compiler
producer and 24 CPU slots. It prioritizes completing known case dependencies
while retaining background fairness, exact cache validation and every candidate.
All input/oracle preparation is already complete, with 232 explicit gaps.
Full CPU-only export of future adaptive xD candidates remains unfinished;
compilation readiness is reported separately from GPU measurement completion.

> Bounded experiment workers, 0.9.3.4-async-experiments.2, 2026-10-10

The [resource recovery](async_recovery_20261010.md) preserves 29,170 checked
performance records and schedules the remaining 9,246. Process-global JIT
modules accumulate across searches, so workers now recycle with recorded
resource limits; resource failures invalidate full searches instead of silently
dropping candidates. Both original-device control populations (96 FFT/FWHT plus
24 real FFT) are complete with finite outputs and pending accuracy review.
Two A100 80GB queues continue the remaining timing population, using the same
v0.9.3.4 library and per-device calibration.

> Real FFT range controls, 0.9.3.4-real-range-controls.2, 2026-10-09

The [numeric evidence audit and controls](numeric_range_controls_20261009_v2.md)
identify 108 raw output-range failures and 12 C2R inputs that overflow during
FP16 spectrum quantization. Separate R2C/C2R controls close the 24-request gap
left by the existing 96 C2C/FWHT controls. All 24 new controls completed on
A100 40GB with finite outputs and complete errors retained; floating acceptance
remains pending. Original 80GB mapping replay is CPU-prepared and queued after
the owned GPU1 experiments. Frozen library, calibration and campaign identities
are unchanged, and raw failures are retained rather than reclassified as passes.

> Asynchronous experiments, 0.9.3.4-async-experiments.1, 2026-10-09

The [queue record](async_experiments_20261009.md) moves deterministic input and
oracle preparation, compilation, error analysis and evidence processing into
independent CPU producers. GPU consumers retain complete research selection,
reject incomplete compile attempts, and verify every output before accepting
measurements. Seven timing samples remain a contiguous GPU group. The v0.9.3.4
library and each GPU's calibration are unchanged. The checked handoff retains
6,532 completed recovery cases, 19 isolated correctness failures and 35 timing
cases; remaining recovery is 5,251 cases, followed by the original timing
shards and separate scaled controls. The full comparison and floating accuracy
acceptance remain unfinished.

> Numeric and public-plan repair, 0.9.3.4, 2026-10-09

The [repair record](numeric_plan_repair_20261009_v0934.md) supersedes the active
October 8 queue. Its stopped 33,076 records remain historical evidence. Public
NTT algorithm serialization now covers valid selected backends, and normalized
FP16/BF16 FFT/FWHT distribute scaling with explicit range bounds. An intermediate
always-pre-scale implementation fixed the original Inf failures but lost minimum
subnormal constants; its queues were stopped and its evidence preserved. The
final range-aware implementation is verified on both failure classes before
restarting two independent A100 80GB queues for affected/unfinished conformance
and disjoint halves of the original full performance population.
Complete results and floating accuracy acceptance remain pending.

> Bounded support milestone, 0.9.3, 2026-10-08

The [bounded support record](bounded_support_20261008.md) supersedes the old
extreme size/batch population. The old queue stopped between shards with
35,343 records preserved. Five functional gap classes now have executable
lowerings, including explicit same-precision real and mixed-compute adapters.
The prepared population contains 38,416 unique valid requests, at most 2^20
elements each; its 232 static gaps are rank-nine boundary negatives. New-path
GPU checks and actual-runner integration completed; full bounded-matrix
conformance, fresh calibration binding and comprehensive comparison remain
separate unfinished stages. Floating accuracy acceptance is still pending.

> Continuation controller repair, 0.9.2.2, 2026-10-08 evening

The [continuation repair](continuation_repair_20261008.md) removes repeated
large-set construction and historical-shard revalidation within a running
process. Restart still validates every saved artifact. Controller-only upgrades
preserve old state/source evidence and cannot alter measurement identities.
Baseline execution errors now block every continuation path, and comparison
requires the same execution configuration as conformance. The queue adds
separate FFT/FWHT scaled-input controls before timing; historical controls are
reused with their original provenance. The new FWHT range control has no
nonfinite outputs, with full errors still awaiting review. Full large-request
reference coverage and missing lowerings remain unresolved.

> Numeric reporting repair and isolated continuation, 2026-10-08 afternoon

The first large FP16 FFT failure has two distinct causes. Complex output decoding
introduced extra component NaNs; harness revision `0.9.2.1-numeric-reporting`
fixes that defect. An unchanged-input GPU reproduction still has 44 nonfinite
complex outputs. The reference itself has 21 values beyond FP16's finite range,
and a CPU reproduction of the implemented half arithmetic matches all 44
nonfinite positions, first overflowing at the eighteenth butterfly level.
This evidence explains the recorded overflow without establishing a general
kernel correctness or accuracy verdict. No arithmetic kernel has changed.

The user authorized continuing other experiments while investigating failures.
The new queue is `results/semantic_campaign_20261008_v0921/run_unaffected.sh`.
It validates the 34,961 historical records, preserves their identities and
failure artifacts, then executes the remaining 13,823 conformance requests in
fresh shards. Further correctness failures are retained and isolated; other
execution failures stop. A separate measured comparison follows conformance,
with quarantined requests explicitly excluded, not counted as successes.
The active queue index and supervisor heartbeat establish current progress.
The user approved separately labelled scaled-input controls; no scaled input
replaces an original pressure-test case. Preparation and results are separate
from the unscaled continuation. The first unchanged-mapping GPU control used
an input-derived factor of 1/128 and produced zero nonfinite outputs. Relative
L2 error is 0.133%, while pointwise median/p99 relative errors are 29.4%/381.6%;
full error review and FP16 external bindings remain outstanding. This is range
diagnosis evidence, not an accepted accuracy or performance result.
See the [repair record](numeric_reporting_repair_20261008.md).

> Semantic campaign resumed, 2026-10-08 morning

The queue resumed from 14,971 verified saved records, including 13,899 semantic
crossing requests. All referenced error artifacts and frozen identities match;
new checkpoints confirm forward progress. The previous process disappeared
after 00:51 without an exit reason. An independent outer supervisor now records
heartbeat, process identity and exit status without changing the runner or
measurement protocol. The active queue index points to its current status file.
Full conformance and searched performance comparison remain in progress/pending;
floating accuracy is still awaiting threshold selection.

> Descriptor-based retest started, 2026-10-07 evening

The user authorized fresh calibration and comparisons and explicitly selected
complete-call comparison including layout/order conversions, with native and
adapted paths reported separately. The [new campaign](semantic_campaign_20261007.md)
uses public canonical descriptors rather than the cancelled benchmark schemas.
Fresh hardware probes, 59 mapping reconfirmations and the 40-class stage
calibration completed on A100 80GB UUID ending fff02. Selector replay passed.
The model retains a local warning: four serial records lack stage-curve coverage,
the resident FFT independent-stage adapter is missing, and concurrent composition
is unvalidated. This is not complete calibration of unseen public descriptors.
The finite manifest has 48,784 unique semantic requests, including 7,680 static
lowering gaps. These are planned requests, not execution passes. The user chose
to inspect floating error distributions and baseline errors before selecting
thresholds; diagnostic floating rows cannot count as passes.
The previous reviewed campaign remains cancelled.

> Unified public semantics, 0.9.2, 2026-10-07

The previous reviewed comparison is cancelled, with a persistent cancellation
record and execution/resume guards; its manifest and historical data are
preserved. The [semantic repair](unified_semantics.md) adds one CPU normalizer
and JSON contract for C/C++/Python, independent strides and per-axis logical
orders, capability reasons, canonical workload/replay identity and a common
boundary lowering. Generic composite execution now admits ranks through eight.
Runtime correctness checks are recorded under `results/semantic_repair_20261007`.
The final build passes 177 public GPU regressions, 33 actual-library CPU API
tests, two CPU CTest targets and the full legacy C API regression. The unchanged
ordered-boundary kernels also pass 256 raw cases. These results establish the
tested semantic paths, not exhaustive coverage or high-performance acceptance.
This repair does not start calibration or define a replacement performance plan.

> Public frontend and dimension audit, 2026-10-07 (pre-repair record)

The [dimension audit](public_frontend_dimension_audit_20261007.md) follows the
user's requirement to organize coverage by semantic dimensions. At that point,
C/C++ had a shared public plan, but order was NTT-specific, public rank was
limited to three, and research Python entry points reused test bindings. Batch
strides alone and non-NTT non-default order fields were silently ignored.
The audit itself changed documentation only. The 0.9.2 repair above addresses
these findings; the old reviewed campaign is cancelled, not pending execution.

> Engineering review before the next campaign, 2026-10-07

The current entry is the [engineering review](engineering_review_20261007.md).
The previous comparison waiter and its partial calibration are paused; no new
GPU comparison was started. Confirmed mappings and stage checkpoints remain
available. The reviewed `0.9.1-review.1` campaign uses strict calibration/build
binding, stops on failed correctness/profile gates, and quarantines partial
outputs before retry. CPU preparation is exposed through
`scripts/reviewed_comparison.py`; GPU validation and comparison remain pending.
The reviewed build, 299 CPU tests, 22 subtests and the C++ platform matching
test passed. The frozen campaign is
`results/reviewed_campaign_20261007_r1/manifest.json`; all 49 commands passed
CPU preparation checks. No GPU worker or automatic waiter was started.

> Platform-boundary and retest repair, 2026-10-07

The [repair record](platform_boundary_repair_20261007.md) describes the preceding
repair. Native bit-reversed NTT is now reachable through
the common inventory; platform preferences are isolated behind a shared rule
interface. Focused CTest passed 21/21, and three weak NTT diagnostics improved
by roughly 50–149x over the defective first-feasible fallback. These are
before/after diagnostics, not a completed external performance population.
The oversized FFT checker now verifies every output with bounded temporaries.
The xD NTT harness failure was reproduced as a forced-preload loading issue.

The preceding, now-paused confirmation is under
`results/local_hardware_profile_a100_gpu1_repaired_20261007/`; its dependent
full comparison is under `results/recalibrated_campaign_20261007_repair/`.
Its partial files remain available; the reviewed build requires fresh confirmation.
The 20261006 campaign is historical evidence and retains its recorded failures.

> Length-64 local-core follow-up, 2026-10-06

The residual 64^3/B4 FP32 case was traced to an executable-space gap: the
selector had no ThreadShared implementation for 64-point axes and therefore
could only compare Block cores on the two strided passes. A bounded N64
8x8 local factorization was added with constant-root and recurrence-root
variants. The recurrence variant uses the same SoA XOR shared handoff as the
existing local cores and keeps the natural layout across each axis.

The public selector now admits N64 alongside N128/N256 and can combine local
strided axes with a Block contiguous axis and batch tiling. On A100 80GB,
64^3 selected plans improved over cuFFT at B=1/4/16 in FP32 and at B=4/16
in FP64. The B4 FP32 plan is 22.886 microseconds versus cuFFT 25.958 and
VkFFT 20.787; B16 selects an 8-sample batch tile and reaches 97.894
microseconds versus cuFFT 109.517. Correctness, mapping, and resource
checks pass, with no register spills in the N64 kernels. This is a bounded
coverage repair and does not claim global optimality.

The post-change 64-case regression is complete. All 192 selected/cuFFT/VkFFT
rows pass. Across the population, cuButterfly reaches 1.2714x geometric
speedup over cuFFT and 1.0112x over VkFFT, with 58/64 points at least 3%
ahead of cuFFT. Forward and normalized-inverse geometric speedups are
1.1285x/1.0033x and 1.4819x/1.0213x, respectively. The focused weak-case
profile attributes the remaining tails to register pressure in paired and
cooperative FP64 cores and launch underfill in a small 2-D FP32 case. This
narrows the next engineering targets without changing the method boundary.

The follow-up lowered the configurable promotion margin from the historical
3% to the measured 1% installation default. A second complete 64-case run
passed all 192 rows and improved the geometric population result to 1.2850x
cuFFT and 1.0222x VkFFT, with 33/64 cases fastest overall. The gain comes
from promoting measured mixed-axis candidates that were previously just below
the guard, including 128^3/B4 FP64. The conservative 3% policy remains
available through `CUBUTTERFLY_XD_PROMOTION_MARGIN=0.03` for noisy shared
GPUs.

The next bounded core repair extends the cooperative pair inventory to 32-point
axes. Four lanes use local Thread FFT8 codelets, and the FP64 32x32
specialization uses 128 threads. The selected 32^3/B16 FP64 plan reaches
26.52 microseconds versus 27.65 microseconds for cuFFT; the forced cooperative
pair is 26.88 microseconds versus 28.88 microseconds for the original pair.
The pair resource record falls from 148 to 50 registers per thread with no
local-memory traffic. A guarded 26-case cooperative regression and a narrow
B=1/4/16 FP32/FP64 matrix pass. The selector retains axis or original-pair
families when the new core does not win, so this is a targeted executable-space
repair rather than a blanket preference.

The subsequent 64-case rerun remains complete with 192 passing rows. Its
measured population speedups are 1.2839x over cuFFT and 1.0142x over VkFFT;
the variation from the previous 1% run is concentrated in sub-10-microsecond
rows, while the 32^3 FP64 B16 rows improve through the new core.

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

## xD FP64 local-core probe, v17, 2026-10-05

The bounded seven-variant probe completes with 420 axis checks and 132
full-plan timing rows passing. At 256²/B16 FP64, eight-line recurrence reduces
complete-plan latency from a fixed Block control's 42.291 μs to 35.584 μs;
matched cuFFT is 35.712 μs. The fixed control is not the best v16 plan at
every adjacent batch. NCU isolates a large reduction in phase-table loads
and short-scoreboard waiting, with no new HBM boundary or local spills.
See the [v17 protocol and profile](../results/paper_profile_rc2/end_to_end/xd_local_v17/README.md).

Production v17 adds `local_twiddle_policy=1` as an optional per-axis recurrence
choice for the existing 128/256-point inventory in both precisions and
directions. The original constant-root choice and line-cap behavior remain
available. Mapping replay, compact-real complex axes, batch tails and the
bounded measured search carry the policy explicitly. Fourteen GPU CTests
and 204 API cases pass; memcheck/racecheck report zero errors/hazards. All
48 matched weak rows pass. Public selection improves 256²/B16 FP64 from
42.035 to 35.174 μs, reaching approximate cuFFT parity, and improves FP32
at that workload from 19.507 to 16.589 μs.

The unchanged full64 matrix completed at 20:44:06 +08 with all 192 rows
passing. Geometric speedup is 1.2089x cuFFT and 0.9653x VkFFT, versus the
preceding same-population v16 run's 1.1833x/0.9424x. Forward is
1.0789x/0.9602x; normalized inverse is 1.3994x/0.9718x and includes fused
normalization relative to cuFFT's separate scale kernel. FP64 reaches
1.2379x cuFFT and 1.0020x VkFFT. Fifty-four of 64 points beat cuFFT; 22 are
strictly faster than both baselines. All winners are freshly measured and
14 use recurrence. The residual tail is now 64³/B4 FP32, about 0.881x
cuFFT and 0.709x VkFFT forward. Universal superiority remains open.

One small 8³/B16 FP32 historical timing outlier has the same physical map
before and after this change. Three alternating-binary recheck rounds complete
with all twelve rows passing and paired differences within ±0.7%; the apparent
regression does not reproduce. This follow-up is kept separate from the
immutable full64 population. Implementation, affected verification, controlled
profiling and the full comparison are complete for this v17 milestone.

## xD local FFT realization, v16, 2026-10-05

The 128/256-point `ThreadShared` core composes Thread FFT8/16 codelets with
SoA XOR shared exchange. It adds no HBM boundary between its local factors.
The existing independent-axis search admits it alongside Block/register
cores, including compact-real complex axes, exact replay and batch tails.
Actual per-axis metadata reports the chosen realization; unsupported axes
retain existing fallbacks. This remains a finite executable inventory and
bounded measured search, not a global-optimality claim or new calibration.

The frozen v16 library (`f9fd176a...264517cb`) passes fourteen affected CTests,
191 distinct public API cases, memcheck and racecheck. The test record explains
two repaired test-oracle/mapping issues; the library did not change during
validation. The unchanged weak population has 56 passing rows. Mixed local
and Block axes improve matched old-map throughput by 23.63% at 128³/B1 FP32,
7.97% at 128³/B1 FP64, and 9.39% at 256²/B64 FP32. 256²/B16 FP64 remains
weak at about 0.837x cuFFT.

The full fixed64 population completed uninterrupted on A100 80GB PCIe at
19:24:36 +08, with all 192 rows passing. Geometric speedup is 1.1833x versus
cuFFT and 0.9424x versus VkFFT; the corresponding forward values are
1.0577x/0.9425x and normalized-inverse values 1.3670x/0.9421x. Inverse results
include fused normalization versus the cuFFT baseline's separate scale kernel.
cuButterfly is faster than cuFFT on 52/64 points and fastest of all three on
16/64. The latest full result therefore improves over the previous v13.1
population, while universal parity and the remaining VkFFT/FP64 tails are
still open. See the [v16 evidence and reproduction scripts](../results/paper_profile_rc2/end_to_end/xd_local_v16/README.md).

Exact-map NCU follow-up completes with seven reports and 28 profiled kernels,
all with zero local-memory sectors. On the representative 128³ FP32 strided
axis, registers fall 64→40, executed instructions fall 22.7%, and shared store
conflicts fall 98.4%. For 256²/B16 FP64, eliminating almost all shared store
conflicts does not improve the local axis: register use remains 80, active
warps fall and short-scoreboard waits rise. The measured sweep includes that
candidate and correctly retains the old plan. See the
[profile analysis](../results/paper_profile_rc2/end_to_end/xd_local_v16/PROFILE.md)
for counters, replay limitations and the bounded remaining core target.

## xD independent axis resources, v15, 2026-10-05

The implementation separates core, EPT, Block cap, lane count and exchange
policy by logical complex axis. It reuses existing kernels, propagates resources
to compact-real and batch-tile children, and preserves empty-vector legacy
mapping replay. A single coordinate sweep refines the fastest measured unpaired
axis plan, using complete-call timing and the existing promotion margin. This
removes a plan-level resource coupling; it does not add a new factorization or
claim exhaustive optimization. All 13 affected CTests, 177 API checks and both
sanitizers pass. Two further compact-real mixed API cases also pass, with exact
mapping replay and input preservation in both precisions and directions.
Forty rows in the unchanged ten-point weak probe pass, but no
mixed candidate wins promotion. Selected throughput stays within -0.78% to
+1.44% of revalidated old physical controls. The full64 promotion condition is
not met; v13.1 remains the latest complete population result. Implementation,
affected validation and this bounded comparison are complete.
See the [v15 protocol and measurements](../results/paper_profile_rc2/end_to_end/xd_axes_v15/README.md).

## xD cooperative register axis at length 256, v14, 2026-10-05

The initial v14 extension and validation are complete. It covers
256-point non-contiguous axes using eight or sixteen cooperative lanes and
the existing Thread FFT32/16 codelets. The v13.1 control remains frozen. This
addresses a local-core coverage gap exposed by the 256²/FP64 profile; it does
not change butterfly semantics or claim that register-only execution always
wins. Six GPU CTests, 172 API checks and both sanitizers pass. All 80 rows of
the ten predeclared probes pass, but no new register candidate wins selection.
NCU shows that shared conflicts disappear while dynamic constant loads and
short-scoreboard waits increase. The completed v14.1 follow-up replaces trivial
small-radix phase lookups with equivalent analytic rotations. Six CTests, 172
API checks, all 80 probe rows and both sanitizers pass again. At 256²/B16 the
forced register plans improve by 8.2–26.5%; NCU shows 67.4–76.2% fewer constant
loads, but FP64 instructions increase and active warps remain low. No new core
wins automatic selection, so this is not an overall speedup. The conditional
fixed64 rerun is not triggered; v13.1 remains the latest full comparison.
See the [v14/v14.1 measurements and controls](../results/paper_profile_rc2/end_to_end/xd_register256_v14/README.md).

## xD local resource candidates, v13, 2026-10-05

Implementation, affected validation, fixed64 comparison and NCU are complete
for this bounded milestone. Following v12's matched NCU, the inventory adds
EPT4 and a range-checked 32-bit element-index
version of unrolled shared exchange. Existing EPT8/16 and all old exchange
paths remain available. The changes propagate through the xD selector,
compact-real complex axes, batch-tile children and mapping replay. This is
an executable-lowering extension, not a change to butterfly mathematics.
See the [v13.1 results and protocol](../results/paper_profile_rc2/end_to_end/xd_resources_v13/README.md).
Five affected GPU CTests, all 159 public API checks and 180 exchange cases pass;
memcheck/racecheck report no errors or hazards. The initial v13 preflight
caught two diagnostic-format incompatibilities before any performance run.
They are fixed in the separately identified final v13.1 binary.

All 192 rows of the unchanged 64-case population are correct. Geometric
throughput reaches 1.1616× cuFFT and 0.9255× VkFFT, versus v12's
1.1417×/0.9097×. Forward and normalized-inverse results versus cuFFT are
1.0382× and 1.3420×; normalization fusion contributes to the inverse advantage.
Twenty-three measured winners use EPT4 or narrow indices. Seventeen points
improve by more than 3% over v12 and none regress by more than 3%, in this
single campaign. The six diagnostic controls retain negative candidate data:
EPT4 helps 64²/FP64 but can hurt 256²/FP64 and 128³/FP32.

Twenty frozen-mapping NCU replays explain the tradeoff. Narrow indices reduce
address instructions but preserve most shared conflicts and global sectors.
EPT4 lowers registers while increasing CTAs and, on large cases, global-load
sectors and shared conflicts. The profiled kernels show zero local-memory
sectors. Higher occupancy alone therefore does not resolve the remaining
core/exchange inefficiency. Independent per-axis resources and more efficient
local-core exchange remain next targets; this milestone does not establish
theoretical optimality or complete other hardware/paper work.

## xD strided-axis exchange, v12, 2026-10-05

Implementation, affected validation, fixed comparison and NCU are complete for
this bounded milestone. Two new optional exchange policies
reuse the existing cuFFTDx arithmetic core: direct global/register transfers and
compile-time-unrolled shared staging. Existing staged paths remain candidates.
The policies share the length/precision/direction/EPT/cap dispatch and propagate
through C2C, compact-real complex axes, batch tiles, metadata and exact replay.
See the [bounded v12 protocol](../results/paper_profile_rc2/end_to_end/xd_exchange_v12/README.md).
Five affected CTests, 151 public API/replay checks and both sanitizers pass.
The new regression covers 72 cases. All 102 weak-region rows pass; 64²/B16
FP32 falls from v11's 8.806 us to 7.066 us, versus same-run cuFFT 7.424 us.
At 128³/FP32, matched EPT8/cap0/order controls isolate an exchange improvement
from 65.075 to 60.058 us, while direct exchange takes 105.728 us. The sign
changes for other EPTs, so all three policies remain measured alternatives.
The unchanged 64-case comparison has 192 correct rows. Geometric throughput
versus cuFFT increases from v11's 1.0546× to 1.1417×, including 1.0267× on the
36 forward cases and 1.3087× on the 28 normalized inverse cases. Against VkFFT
it reaches 0.9097×, so independent-library gaps remain. The original 16 forward
controls reach 1.0560× cuFFT / 0.9474× VkFFT. Thirty-nine selected plans use new
exchange candidates; all 64 plans use measured selection. Normalization fusion
contributes to the inverse comparison; a single campaign is not significance.

The first run was interrupted near its end by foreign CUDA tasks. An explicit
pre-conflict cutoff retains 61 verified cases; the final three are remeasured
under the same guard. The new benchmark resume options validate binary, GPU,
population and timing identities and record provenance. CPU tests also cover
UUID spelling normalization across CUDA and nvidia-smi. No conflicting trial
enters the final release.

Matched 128³/EPT8 NCU isolates a 29.625% instruction reduction from unrolling,
but registers increase 46→64 and active warps fall. Direct exchange nearly
doubles global sectors there. Shared-store conflicts persist after removing
external staging, localizing a substantial remaining cost inside the imported
core. Next targets are register live ranges, coalesced transfer/core layouts
and independent per-axis resources. This does not establish a theoretical
optimum or finish other hardware/paper milestones.

## xD independent block granularity and FFT baselines, v11, 2026-10-04

Implementation and matched measurement are complete for this bounded milestone.
The axis Block processing unit now
supports FFTs-per-CTA caps 8/16 independently of EPT; the default policy and old
mapping replay remain intact. C2C and compact-real complex axes share the parameter.
The new cache namespaces are `xd-v11-axis-block-caps` and
`xd-real-v11-axis-block-caps`. This addresses a finite executable-inventory gap,
not a change to butterfly semantics or the hybrid-dataflow method.

The pre-change NCU control confirms that 64²/B16 launches only 32 CTAs per axis;
cuFFT's strided axis instead launches 128 CTAs. 128³/FP32 also has higher
instruction/shared-exchange cost, so increasing parallelism alone need not win.
The [v11 protocol](../results/paper_profile_rc2/end_to_end/xd_block_v11/README.md)
fixes the original 16-point control and a 64-point union population before new
measurement. Precision, direction, rank and per-library results remain separate.

An optional `vkfft_xd_bridge` target and `scripts/vkfft_xd.py` add an independent
rank-1/2/3 C2C VkFFT baseline using caller-owned buffers and streams. Both floating
precisions and normalized inverse are supported. `diagnose_xd_fft.py` accepts
`--vkfft-library`, checks input preservation and actual captured-graph output,
then uses the same warmed 20-call/7-replay protocol for each library. JIT and plan
construction remain outside timing.

Four affected CTests pass, including 96 new executions; 16 C-ABI forced/replay
checks and both memcheck/racecheck pass. Twelve independent-baseline smoke cases
and all 192 rows of the 64-case comparison are correct. All selected plans were
empirically timed; 13 chose a cap variant. 64²/B16 FP32/FP64 improves to
8.806/10.291 µs in the weak-region probe, versus v10.1's 10.240/12.442 µs.
The larger 256² and 128³/FP32 gaps remain.

Paired geometric means versus cuFFT are 1.0073× on the original 16 forward cases
and 1.0546× on all 64 cases; the extended forward/inverse strata are
0.9599×/1.1901×. The same populations versus VkFFT are 0.9054×/0.8396×.
cuButterfly is fastest of three in 6/64 cases. Normalization fusion contributes
to the inverse advantage over native cuFFT's separate scaling launch. These
numbers support a limited gain against cuFFT, not broad superiority over FFT
libraries. Do not pool different baselines, old 1D results, or only winning cases
into one claimed gain. Full coverage, identities and results are retained in the
linked v11 report; significance and cross-hardware acceptance remain separate.

The final three-library NCU traces replay the matrix's exact winning mappings.
At 64² the new cap raises the grid to 64 CTAs, matching VkFFT's block/grid shape,
but the strided axis still executes substantially more instructions. At 128³,
each strided stage executes 6.554M warp instructions versus VkFFT's 3.277M, with
about 527K versus 135K shared-store bank conflicts. All libraries use three axis
launches there. The next bounded target is a more efficient strided processing
unit and per-axis resource choices. The initial NCU-attached search selected
different families and is excluded from matched mechanism claims.

## xD scheduling and twiddle lowering, v10.1, 2026-10-04

Measurement note: the four large-cube forward cases and all correctness checks
completed before an external job started on the target GPU at 21:45:58. The
remaining cases were rerun under a one-second CUDA-process guard, completing
at 22:07:32 without a foreign process observed. The
[guard status](../results/paper_profile_rc2/end_to_end/xd_schedule_v10/recheck_status.json)
records completion. The uncertain trials are archived separately and excluded.

Packed C2C now supports complete-transform batch tiles and exact tail replay.
The bounded rank-three selector proposes tiles from the executing GPU's L2
capacity, retaining whole-batch plans. Non-contiguous 64/128-point cores admit
4/8/16 cooperative lanes; pairs gain an optional XOR shared-row layout.
All cooperative cores share a 128-root constant table. Cache namespaces are
`xd-v10.1-schedule-cores` / `xd-real-v10.1-schedule-cores`.

The constant-table change repairs a measured lowering cost: the first wider
lane implementation lowered `sincospi` into large inlined polynomials. Lower
register use improved occupancy but increased total issued instructions.
Table evaluation removes that redundant work; lane width and layout still
require complete-plan measurement. Independent fixed-v9-core controls also
show about 18% lower time with batch tiling, accompanied by roughly half the
DRAM traffic. These are separate effects, not a generic stage-overlap claim.

On A100 80GB UUID ending `e49309e6`, final 64³/B16 forward takes
112.538 / 220.006 us in FP32/FP64, versus cuFFT 109.926 / 222.925 us.
128³/B1 FP64 improves from v9's 148.275 to 123.648 us, versus cuFFT
125.440 us. FP32 128³ remains at 0.807x cuFFT. The sixteen matched forward
cases reach 0.988x cuFFT geometrically (v9: 0.951x) and 1.039x over v9
selected times. Near-one ratios are not statistical-significance claims.
Normalized inverse 64³/B16 reaches 1.360x / 1.464x, including both sides'
normalization. Compact-real transforms and 64²/256² remain below native.
The eight 3D cases average 1.000x geometrically. Additional guarded 64³/B32
probes reach 1.034x FP32 and 1.099x FP64, superseding the larger uncertain
estimates. These extra batches are not pooled into the fixed sixteen-case mean.

Eleven CTests cover 504 FFT cases plus rank-three permutation, and all 127
public C-ABI checks pass. Six sanitizer modes are clean, covering all three
lane widths, both shared layouts, and batch tiles. Input preservation,
normalization, workspace reuse and B=5 replay as 2+2+1 are checked.
The [v10.1 report](../results/paper_profile_rc2/end_to_end/xd_schedule_v10/README.md)
records versioned controls, the final binary, all matched timings, negative
trials, profiles and reproduction scripts. Cross-hardware acceptance and a
general within-transform cross-axis subblock pipeline remain separate work.

## xD cooperative FFT cores, 2026-10-04

The v9 inventory adds four-lane register FFTs for non-contiguous length-64/128
axes and cooperative two-axis planes. The three new families compete with
prior candidates using full-plan graph timing. FP64 pairs use 256 threads
and fold independent lines into two rounds where needed; availability uses
the device's opt-in shared budget and compiled occupancy. FP64 128² remains
outside this whole-plane lowering. Cache namespaces are
`xd-v9-cooperative-cores` / `xd-real-v9-cooperative-cores`.

On A100 80GB UUID ending `e49309e6`, 64³/B16 forward improves from 161.126 to
131.379 us in FP32 (1.226x) and from 299.827 to 288.717 us in FP64 (1.038x).
The sixteen matched forward cases reach 0.951x native cuFFT geometrically,
up from 0.935x; 128³ retains the prior core family. Normalized inverse 64³/B16
reaches 1.187x/1.136x native in FP32/FP64, including both sides' normalization.
This does not establish forward or compact-real parity.

NCU verifies one-third fewer global store sectors for the new 64³ FP32 path.
Its remaining register axis has no shared exchange or CTA barrier. A negative
128³ FP64 candidate uses 206 registers/thread and only about 11% active warps,
explaining why removing shared exchange is insufficient. This is a measured
lowering limitation, not a proof against the method or completion of general
cross-axis subblock streaming.

Seven CTests cover 430 FFT cases plus rank-three permutation. New register and
pair memcheck/racecheck are clean after adding explicit warp memory ordering.
All 108 final public checks pass, including truthful real-axis candidate
metadata for `(64,1)`. The [v9 report](../results/paper_profile_rc2/end_to_end/xd_register_v9/README.md)
records versioned binaries, the complete matched matrix, profiles, fixes,
negative trials and reproduction scripts. Broader acceptance queues were not
silently rerun or marked complete.

## xD paired-axis FFT fusion, 2026-10-04

`pair-tiled-cufftdx` now keeps two short, physically trailing FFT axes in one
CTA, using static Thread codelets and padded shared-memory exchange. It
removes an intermediate global read/write and one launch while preserving
natural layout and input ownership. FP32 pair axes cover 8–64 and FP64 8–32.
EPT16 axis variants compete with the prior EPT8 core; pair EPT16 affects the
remaining unpaired axis. The xD cache namespaces advance to
`xd-v8-core-fusion` / `xd-real-v8-core-fusion`, and raw normalization flags
now match the strict replay contract in the cache key.

On A100 80GB UUID ending `e49309e6`, the sixteen matched forward xD FFT cases
improve 1.202x geometrically over v7 and reach 0.935x native cuFFT (previously
0.775x). 16²/B16 reaches 1.475x/1.478x in FP32/FP64. FP64 normalized inverse
16×32×32/B16 reaches 1.395x. The final NCU replay verifies two kernels instead
of three and one-third fewer global store sectors for the fused 16×32×32 case.
Larger sizes and compact real FFT remain below cuFFT. Wider EPT and fewer
pair threads were not universally beneficial; the negative trials are saved.

Five GPU CTests cover 388 FFT layout/core cases plus rank-three permutation;
91 public API checks pass. Pair memcheck/racecheck and wide-axis racecheck
report zero errors/hazards. The FP32 cancellation-sensitive pair tests now
use an FP64 reference with observed maximum relative L2 error 1.74e-7.
The [scoped report](../results/paper_profile_rc2/end_to_end/xd_fusion_v8/README.md)
records the final binary, complete matrix, inverse/real checks, NCU evidence,
and rejected alternatives. This does not complete the broader acceptance
matrix, generic layout fusion, or the complete xD search space.

## xD axis-aware FFT cores and graph selection, 2026-10-04

The optional `axis-tiled-cufftdx` processing unit now performs axis access,
shared exchange and FFT in one kernel for FP32/FP64 power-of-two axes 8–1024.
Unsupported axes retain the packed-axis fallback. Complex and compact-real
selectors use a uniform complete-plan graph protocol, including callers on
the default stream; cache keys are `xd-v7-graph-steady` and
`xd-real-v7-graph-steady`. This is a finite executable inventory, not exhaustive
core/tile search or cross-axis overlap.

On A100 80GB UUID ending `e49309e6`, sixteen matched forward cases improve
1.231x geometrically over v5 and reach 0.775x native-cuFFT performance.
128³/B1 reaches 0.798x/0.850x in FP32/FP64; 16×32×32/B16 FP64 normalized
inverse reaches 0.984x. The three compact-real samples remain below the public
cuFFT-backed reference. Three CTests (including 220 FFT layout cases), 75
public checks, memcheck and racecheck pass. NCU confirms three axis kernels
without standalone permutations or spills, while shared exchange/core mapping
remains a measurable gap. See the [full scoped report](../results/paper_profile_rc2/end_to_end/xd_axis_v6/README.md)
for the final build, matrix, profiles and open work. Earlier queued experiments
were not silently rerun or reclassified by this milestone.

## Cryptographic application contracts, 2026-09-20

The application extension (local artifact: `../results/crypto_application_20260920/README.md`; not included in this source release)
adds 208 cyclic and 138 composed contracts per A100 target. The generator
distinguishes modulus bits, machine-word bits, batch and RNS channels; uses
named Dilithium/BabyBear/KoalaBear primes and controlled 30/40/50/60/62-bit
primes; and includes batch tails, inverse transforms and legal root domains.
Twelve overlapping single-modulus/RNS-L=1 contracts are deduplicated, retaining
both suite memberships. Together with the existing 358 contracts, the finite
combined population is 704 per target, not 704 completed measurements.

An isolated consumer benchmark adds GPU twists around public NTT plans for
negacyclic and coset semantics. It prepares all RNS channel plans/buffers
before timing and measures the complete channel sequence inside each timed
iteration. It records per-channel mappings, root alignment, summed workspace,
working-set allocation and exact checking of all batches/channels. Modular
boundary/random patterns are separate. Exact-contract cyclic winners can be
replayed against the public shared proposal, but this is not joint RNS/twist
schedule optimization or an external-library comparison.

The 50 preparation contracts deduplicate to eight SM80 modules, all compiled.
The isolated benchmark builds against the unchanged public archive. Thirty-five
CPU tests pass, including thirteen host-only arithmetic/contract tests; these
are not GPU execution evidence. Each target's queue requires a 24-case GPU
correctness gate before new full search and composition timing. The queues
wait behind both earlier studies and their locks, use their own fixed 48-hour
window, and do not reset earlier deadlines. GPU measurements are still pending.

The [coverage ledger](crypto_application_coverage.md) explicitly retains
Goldilocks, multi-limb scalar fields, extension fields and Kyber incomplete NTT
as implementation gaps. New samples or calibration cannot substitute for the
missing arithmetic/transform lowerings. The installed planner ABI and existing
running experiment inputs remain unchanged.

## Batch and precision expansion, 2026-09-20

The independent extension (local artifact: `../results/batch_precision_20260920/README.md`; not included in this source release)
freezes 284 additional semantic cells per A100 target, retaining the original
74-cell comparison untouched. Its combined report has 358 cells; 322 unique
cells serve the new suites, including 38 exact matches in the original matrix.
The original-only cells remain separately identifiable. Batch and byte caps
are fixed before measurement rather than chosen from favorable timings.

The experiment adds 91 FP32 FFT/word64 NTT scaling points, byte-limited FP64
FFT/word32 NTT sweeps, six floating precision/accumulation contracts for FFT,
FWHT and structured 2x2, FFT/FWHT and NTT inverse anchors, and an NTT
same-modulus width control. It records unavailable external baseline contracts
and keeps exact modular correctness separate from floating error metrics.
The 54 finite preparation contracts project to 40 standalone modules, all
successfully compiled before GPU execution; linked low-storage paths are
identified separately. Kernels, installation and the active original launchers
are unchanged.

The extension reuses the tested continuation controller behind an explicit
parent-completion/lock dependency. An original runner doing CPU scoring still
owns its GPU even when utilization is zero. The new queue has its own fixed
48-hour wait window and never renews the parent's window. Same-build/UUID
calibration records are snapshotted when the extension starts. The validated
28-configuration basis and later exact service samples have distinct coverage
claims: a one-point-only new curve is retained as such, not declared fully
validated. Missing finalist services are handled by the existing acceptance
workflow. The new performance population remains pending until measured.
Both continuation queues started at 15:11 CST on September 20, with their own
fixed September 22 15:11 CST deadlines. The matrix generator's seven tests
and the queue/import path's seventeen tests pass; six offline-report tests also
pass, for 30 focused CPU checks. The import checks include
parent-journal binding and content-addressed checkpoint sidecars; a lock
handoff race is retryable without treating numerical failures as occupancy.
The report joins only complete exact-semantic cells with three correct paired
trials, binds even a single available journal to the frozen target UUID, and
checks frozen input hashes. Its initial figures are explicitly partial and
contain only original-cohort measurements; extension cells remain unmeasured.

## Current-build supplement and automatic continuation, 2026-09-20

The 40GB supplement (local artifact: `../results/comprehensive_20260919/supplement_completion_20260920.json`; not included in this source release)
completes the four remaining uint32 Zeta configuration validations. All 28
declared stage configurations now validate, with 1,555 accepted records and
zero missing intervals or adapter gaps. It adds 225 measurements and preserves
every prior raw record; two historical contaminated observations remain
excluded. The 1,024-point ceiling matches the declared batch domain. Adaptive
validation closed the four configurations at 268/183/130/156 points; no
validation threshold, kernel, timing protocol or comparison population changed.

40GB acceptance is running. The 80GB continuation preserves its six completed
cells and waits for its original GPU UUID. A target-specific, explicitly
started controller observes two idle polls and resumes only explicit GPU
exclusivity interruptions within one fixed waiting window. Other failures and
expired windows remain terminal; a resolved matrix is still required for
promotion. Nine CPU controller tests and the existing parallel-score
equivalence check pass. Eight CPU scoring workers preserve proposal order and
kernel timing. Controller states and deadlines appear in the current report.

The supplement record also retains a launcher-only error after stage completion:
editing the active Bash file's worker override shifted the old process's read
position. The completed checkpoint and raw-record preservation were checked
before a fresh process entered acceptance. Active launcher files are now left
unchanged. This finite supplement completes its declared stage domain; the
two comprehensive matrices remain unfinished.

## Public-strategy installation and new comprehensive run, 2026-09-19

The current-build refresh (local artifact: `../results/comprehensive_20260919/README.md`; not included in this source release)
installs the verified public whole/chunked FFT portfolio into the shared conda
environment. The library and installed templates have matching recorded hashes;
the preceding installed core assets are backed up. Runtime fingerprint is
`d13333ae10871fec9de9637a87e8e945263f8e101cc5a4ae2ef67047766f011f`.
The slower private resident/radix-8 suffix candidates are not promoted.

The existing precompiler now accepts `--candidate-source seeds` for a finite
mapping portfolio, with explicit incomplete coverage for unmatched cells.
Default runtime export is preserved; neither mode claims theoretical-space
exhaustion. All 50 SM80 modules for the 79 search seeds and 32-point stage basis
compile before GPU measurements. The original 72 semantic cells plus two
separately reported N23 additions are frozen for each A100 40GB/80GB target.

Both target-specific current-build stage runs have started, followed by the
existing evolutionary search, exact public-selector replay and matching
baseline comparisons. Stage measurements and performance timings are new;
only same-UUID hardware capabilities and mapping proposals are reused.
No new comprehensive speedup is claimed while these journals are incomplete.

The promotion audit also fixes imported online FFT mapping seeds' legacy
`reorder_columns` alias: it is normalized to one exactly as the public runtime
does, while actual columns continue to derive from threads and EPT. Real
strategy-axis mismatches remain errors. All 83 focused CPU checks pass, and
eight installed N22/N23 whole/chunk8 forward/normalized-inverse replays pass
correctness and requested mapping identity on the separately identified 80GB
validation card. These correctness timings do not enter either performance cohort.

The initial search exposed a command-construction mismatch: CPU scoring's
top-level copies of mapping fields were also emitted as legacy CLI options.
The public mapping JSON now owns those implementation axes; independent
workload semantics still use their documented flags. The rejected attempt is
retained separately with no completed comparison cells. The stage probe and
GPU kernels are unchanged, so compatible current-build stage records are
retained. Finalization checks the resolved matrix and all paired selector
trials before merging fresh winners into the normal local registry.

## Radix-8 lowering and research convergence, 2026-09-19

Two bounded follow-ups complete the preceding resident-core experiment:
dedicated radix-8 lowering (local artifact: `../results/paper_completion_20260918/fft_radix8_suffix_20260919/README.md`; not included in this source release)
and compile-time lane phases (local artifact: `../results/paper_completion_20260918/fft_radix8_constants_20260919/README.md`; not included in this source release).
The first replaces the generic eight-lane computation with a specialized
radix-8 FFT and one selected phase multiply. The second forces template-known
phase coefficients to be evaluated at compile time. Runtime-dependent
inter-factor twiddles, the prefix and the global stage boundary are unchanged.
These are private core/output-handoff candidates, not public mapping options.

Across both A100 40GB/80GB cohorts, 312 accepted units include 72 passing GPU
numerical checks. The first cohort retains one rejected 40GB clock-window
attempt; the second has none. The 36-case independent CPU radix-8 oracle,
host coefficient test and six matched 40GB NCU captures also pass their
respective validation. Compilation precedes GPU use and is excluded from
ordinary timings. All comparisons use their own frozen cohort.

Dedicated radix-8 improves end-to-end throughput 1.23--1.28x over the generic
private K16 core. Compile-time phases improve another 1.19--1.22x over that
dedicated implementation. Matched N22/B1 suffix instruction counts fall from
53,575,680 to 29,032,448 and then 19,136,512. Nevertheless, the final private
candidate still takes 1.078--1.160x the current block/chunk8 control's time
across the eight card/workload combinations. Neither candidate is promoted;
production source, installed packages, selectors and calibration are unchanged.

The latest control's finite four-case cuFFT-relative throughput geomeans are
1.0018x / 1.0238x on 40GB / 80GB: N23 is ahead, while N22 remains behind.
These are local FP32 results, not comprehensive acceptance. Source-derived
shared excess is zero for the private candidates, but raw bank-conflict
counters remain nonzero. Radix-8 source-PC counts omit 491,520 instructions
relative to the aggregate counter; this residual remains unattributed. The
legacy parser's `other_wrapper` label also includes inlined lane computation
and must not be interpreted as address-only overhead.

The [research convergence note](research_convergence_20260919.md) separates
these local results from the latest comprehensive cohort: 71 of 72 cells
complete, one unavailable, with three externally paired inverse cells.
The next priority is a current-build, matched-library refresh and consolidated
paper evidence, retaining the faster existing core. This finite negative
experiment closes the private branch without claiming that the broader
research objective or a global optimum has been established.

## Resident FFT suffix core experiment, 2026-09-19

The local-core study (local artifact: `../results/paper_completion_20260918/fft_resident_suffix_core_20260919/README.md`; not included in this source release)
implements two private replacements for the 2048-point Block FFT suffix:
32x64 with two cooperating lanes, and 16x128 with eight lanes. Both reuse
thread FFT units and use scalar SoA shared storage with writer offsets for
the intermediate and final ownership handoffs. The prefix, global two-stage
boundary and operator semantics remain unchanged. This substitutes the core
and its output handoff together, not arithmetic alone. Frozen templates record
the actual realization; the private variants are not public mapping options.

Both A100 cohorts complete 78 accepted units (18 correctness, 12 stage
diagnostics, 48 three-trial timing observations); 80GB retains two clock-window
rejections. The 80GB target is UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`,
different from the preceding study because the original card was busy. No
historical timings are pooled. Compilation precedes GPU use; timings exclude
setup and checking. All 36 GPU numerical cases, 22 independent CPU oracle
cases and three matched N22/B1/40GB NCU captures complete successfully.

Neither private core beats the current block/chunk8 control. K32 is 2.0--9.6%
slower end to end; K16 is 59.5--75.2% slower. The control's finite four-case
cuFFT-relative throughput geomeans are 0.9995x / 1.0256x on 40GB / 80GB.
N23 remains ahead and N22 remains behind. Production source, installed
packages, calibration and selectors are unchanged by this experiment.

Source-correlated shared wavefronts fall from 2,621,440 to 1,048,576, with
excess falling from 1,048,576 to zero. Raw shared-load bank-conflict counters
remain nonzero and are not interchangeable with source-derived excess.
K32 increases instructions 51.4%, uses 80 rather than 64 registers, doubles
shared allocation to 64 KiB and reduces active warps from 45.51% to 23.44%.
K16 retains similar active warps but issues 6.1x as many instructions, mainly
in the generic eight-lane codelet. No observed local-memory traffic explains
either regression. See the counter analysis (local artifact: `../results/paper_completion_20260918/fft_resident_suffix_core_20260919/ANALYSIS.md`; not included in this source release).

The write-offset resident mapping is correct and reduces the targeted shared
cost; the complete lowering still needs efficient register/warp computation.
Next, test one selected twiddle multiply plus a specialized radix-8 lane FFT
instead of repeated predicated operations in the generic eight-lane path.
Retain the fast Block FFT as a control. These results neither prove a global
optimum nor establish a theoretical obstruction to hierarchical hybrid dataflow.

## FFT suffix address lowering, 2026-09-19

The address-lowering study (local artifact: `../results/paper_completion_20260918/fft_suffix_addressing_20260919/README.md`; not included in this source release)
completes 100 accepted units per A100 40GB/80GB (one retained clock-window
rejection on 40GB), four matched source-correlated NCU captures, and 15 public
mapping/module checks. Compilation precedes GPU work; setup and verification
are excluded from ordinary GPU-event timings. Three alternating trials use
>=500 ms warmup and 100 repeats. The matrix remains FP32 forward contiguous
out-of-place N22/N23, batch 1/4; correctness additionally covers normalized
strided/in-place inverse and smaller public FP64 configurations.

Public JIT chunked suffix lowering now specializes its known local extent
and hoists invariant 64-bit output base/stride arithmetic. This is a generic
lowering change, with no new hardware-specific search axis. The original
whole-tile path is retained: a separate full-unroll candidate reduces
instructions but regresses several cases and is not promoted. Public forward
and inverse suffix SASS matches the measured chunk candidate and original
whole control at both large sizes. The 23 focused compiler/search tests and
two independent CPU address-oracle tests pass; all 15 public GPU checks pass
with no compilation during that verification phase.

Chunk-only before/after medians improve 0.2--0.9% across eight card/workload
combinations, with overlapping ranges in several cases. The finite pool of
original whole plus promoted chunk reaches 1.0035x / 1.0272x cuFFT throughput
geomeans on 40GB / 80GB. N22 remains behind cuFFT and N23 remains ahead.
These are same-cohort cuFFT 11.2 / CUDA 12.4 results, not a comprehensive or
automatic-selector acceptance. Current source/JIT lowering is updated; the
installed conda package, hardware calibration and selector are not replaced.

On N22/B1/40GB, chunk wrapper instructions fall 34.25%, and total suffix
instructions fall 8.52%; core and explicit-exchange instruction counts are
unchanged. All source-correlated excessive shared wavefronts remain at the
imported core/call-site lines; explicit write-offset exchange has zero excess.
There is no observed local-memory traffic. Whole unrolling raises long-scoreboard
stalls in the matched capture despite reducing instructions. These counters
support investigating local-core exchange and scheduling next, but are not
wall-time fractions or evidence of a theoretical limit of hybrid dataflow.

## Public FFT suffix follow-up, 2026-09-18

The public follow-up (local artifact: `../results/paper_completion_20260918/fft_public_suffix_followup/README.md`; not included in this source release)
completes 122 accepted units on each A100 40GB/80GB using current public JIT
templates, CUDA 12.4 and same-cohort cuFFT 11.2. It measures N22/N23, batch 1/4,
FP32 forward contiguous out-of-place C2C, with 60 full-batch forward and
normalized strided/in-place inverse checks across the cards. Compilation and
setup are outside GPU-event timing. Three alternating trials use >=500 ms GPU
warmup and 100 repeats. Two frequency-window rejections on 40GB are preserved;
80GB has none. The initial host-harness pilot incorrectly initialized NVML after
warmup; it is archived separately, and the corrected matrix is rerun in full.

The EPT16/C4 controls remain best against new public EPT8/C4 and EPT16/C8
suffix candidates. At N23 the best controls exceed cuFFT by 2.25--3.08% on
40GB and 6.42--7.77% on 80GB, with nonoverlapping three-trial timing ranges.
N22 still reaches only 96.92--97.62% of cuFFT throughput. The four-workload
geometric means are 99.93% / 102.24%; these are finite-pool results, not a
comprehensive or automatic-selector acceptance result. No selector is promoted.

Two matched N22/B1 NCU captures on 40GB explain why lower register allocation
does not suffice: EPT8 halves suffix registers (64 -> 32) and increases active
warps (46.01% -> 90.84%), but executes 13.0% more warp instructions, with higher
barrier and MIO-throttle metrics. Neither has observed local-memory traffic.
The counters cover both the imported unit and wrapper; source attribution is
still required to split their costs. Next work should reduce codelet/address/
exchange instruction work, retaining EPT16 as the control. This negative EPT8
result is not evidence of a theoretical obstruction.

## Public resident promotion, 2026-09-18

The local register-subgraph and chunked suffix axes are now public on
`ButterflyConfig` and `PlanConfig` as `local_stage_partitions` and
`exchange_chunks`, indexed by physical execution group. Mapping replay, JIT
cache identity, candidate projection, evolutionary neighbors, stage-service
keys, and cost-model terms carry these fields. Empty axes preserve the old
lowering; unsupported backends reject them explicitly. Shared resident units
are available to all six butterfly operators and use the same operator traits,
inverse handling, normalization, and output ordering as the original kernel.

The public promotion cohort has 26 candidates and three alternating trials per
candidate on each pinned A100 40GB/80GB. Both journals contain 78/78 accepted
units, strict identity/exclusivity/thermal audit passes, and at least 500 ms of
GPU-event warmup per measured region. The six-operator N=4096, batch=128
comparison shows the best local-unit variant improving over the old shared
lowering by 15.0--47.9% across these cards; leaf width 4 is slower for FFT and
NTT, so the search must retain both widths. The focused FFT N=2^22/2^23
batch=1 pool improves the old square EPT32 control by 7.0--15.7% depending on
card and size; chunking and rectangular factors are separate measured axes.
These are same-dataflow controls, not fresh cuFFT ratios or a claim of global
optimality. Full raw journals and the analyzer are in
`results/paper_completion_20260918/public_resident_promotion/`.

The production build passes the affected C++ mapping/planner/operator/API tests,
59 focused compiler/mapping tests plus 64 search/cost projection tests, and the
installed package consumer. The conda environment
`/home/wt/yes/envs/cubutterfly` now contains the public headers, templates and
resident mapping helper. A later hardware migration should measure these new
service identities before promoting selector entries.

## FFT suffix core and exchange study, 2026-09-18 (preceding private study)

The suffix study (local artifact: `../results/paper_completion_20260918/fft_suffix_exchange/README.md`; not included in this source release)
completes 204 accepted units per A100 40GB/80GB: 60 correctness checks,
24 single-stage observations and 120 whole-plan timing trials per card.
Twelve CPU oracle tests and all 120 GPU correctness configurations pass.
Three frequency-window rejections remain recorded; only missing units were
repeated. Five matched 40GB NCU captures complete with source-line information.
Compilation, initialization and verification remain outside ordinary timing.

The experiment combines the private rectangular/square prefix pool with the
existing suffix EPT16/EPT32 axis and a new private chunked output exchange.
EPT16 was already expressible in public mapping/JIT/search; prior controlled
leaf/prefix studies held it at 32. This is not a newly repaired EPT option.
The installed selector, production templates and calibration are unchanged.

For FP32 forward contiguous out-of-place N=2^22/2^23 and batch=1/4, selecting
the best measured private candidate per workload reaches 100.77% / 102.82%
geometric-mean cuFFT-relative throughput on 40GB / 80GB. The same-build
base32 square/rectangular pool reaches 96.25% / 97.20%. N=2^22 still trails
cuFFT by 1.3--3.5%; N=2^23 exceeds it by 4.1--7.9%. System cuFFT 11.2 and
the CUDA 12.8 build are unchanged. This finite pool is not a comprehensive
performance result or an installed-selector acceptance claim.

For the 2048-point suffix, EPT16 reduces registers from 98 to 64 per thread
and doubles threads/CTA from 256 to 512 while retaining two CTAs/SM. Matched
NCU active warps increase from 21.55% to 46.21%, without local-memory traffic.
Shrinking only the EPT32 output tile cannot shrink its shared allocation:
the core itself needs the whole tile. EPT16 plus chunking can halve the
allocation but does not further increase CTA residency; whole-tile EPT16 has
slightly lower medians in the two batch-4 cases on 40GB, with an overlapping
range at N22. Direct global scatter loses badly.

Instruction attribution resolves the preceding suffix bank-counter ambiguity.
Inlined core operations are mostly attributed to the wrapper's FFT execute
call line, so file-name-only attribution is incorrect. Across five batch-1
captures, all source-correlated excessive shared wavefronts map to core
source/call lines; explicit write-offset output exchange source lines have
zero excess. Compiler line attribution is not exact instruction ownership,
and these excess wavefronts are not raw hardware bank-conflict counts. For N22
the core excess decreases from 1,835,008 to 1,048,576 with EPT16. Raw reports and the
superseded file-only analysis are retained; reparsing requires no GPU run.
These counters are not time attribution or a universal zero-conflict proof.

Next production integration should jointly expose independent local factors
and output-exchange granularity, reuse the existing EPT axis, retain the square
fallback and sample the compatible lowerings. The remaining observed gaps
concern core/lowering choices; no theoretical obstruction was found, and no
global-optimality claim follows from this experiment.

## Rectangular FFT resident prefix, 2026-09-18 (preceding snapshot)

The rectangular-prefix study (local artifact: `../results/paper_completion_20260918/fft_rectangular_prefix/README.md`; not included in this source release)
removes the square-local-factor restriction in a private research template.
Its two local FFTs may have different sizes and cooperative widths while
sharing a CTA and using the existing write-offset exchange principle. This
extends an executable lowering; the theoretical dataflow space never required
equal local factors. Public mapping/JIT/search promotion remains outstanding.

Both A100 40GB and 80GB complete 220 accepted units with no rejected trials:
48 full-batch correctness checks, 64 single-stage diagnostic observations and
108 whole-plan timings per card. The independent CPU oracle passes 349 checks.
The GPU correctness population includes normalized, strided, in-place inverse
FFT at N=2^20, batch 3. Performance covers FP32 forward contiguous out-of-place
N=2^22/2^23, batch 1/4, three trials, >=500 ms GPU-event warmup and system
cuFFT 11.2. All custom kernels and the original control share one CUDA 12.8
SM80 build; compilation and initialization are outside timing.

At N=2^22, 64x32/C4 improves whole-plan throughput over the old 64x64/C4
control by 5.36--6.35% on 40GB and 5.07--8.59% on 80GB. It reduces prefix
registers from 178 to 128 and shared memory from 128 to 64 KiB, allowing two
CTAs per SM instead of one. At N=2^23 the larger suffix cancels this gain;
the square variant remains best. Selecting the best measured candidate per
workload gives 96.57% / 97.45% geometric-mean cuFFT-relative throughput on
40GB / 80GB, versus 93.84% / 94.09% for the original control. This is an
exploratory finite-pool comparison, not a public-selector or full-matrix result.

Five matched 40GB NCU captures show greater prefix warp availability and zero
local-memory sectors. Reducing columns to two cuts useful bytes per global
sector from 100% to 50%, despite increased occupancy. CPU address tests do not
establish zero hardware bank counters: the rectangular prefix still records
small shared-store counters, and the suffix aggregate combines the vendor
Block FFT with our final ownership exchange. Instruction-level attribution
remains open. Replay timing is excluded from all ordinary speed comparisons.

The next production integration must expose the independent local factors,
derive both cooperative widths and resource limits, preserve the square
fallback, and include these axes in JIT/search/model sampling. The current
installed selector and calibration were deliberately not changed by this
private study. The results identify a useful missing mapping and a measured
stage tradeoff; they prove neither a theoretical limitation nor global optimality.

## cuFFT device-core comparison, 2026-09-18 (preceding snapshot)

The same-dataflow PTX/LTO comparison (local artifact: `../results/paper_completion_20260918/fft_leaf_lowering/LTO_RESULTS.md`; not included in this source release)
is complete on A100 40GB and 80GB: 113 accepted ordinary-study units per card,
including 60 whole-plan timings, 12 small forward/inverse device-core checks,
and 16 unique custom full-batch correctness configurations. System cuFFT 11.2
is the host reference; cuFFT 11.5 EA supplies device functions. CUDA 12.8
offline SM80 cubins run correctly on this host's 550.127.05 driver.

For N=2^22/2^23 and batch=1/4, the PTX control reaches 93.49% / 94.05% of
cuFFT throughput geometrically on 40GB / 80GB. Thread-LTO reaches
93.60% / 94.16%; this small change does not resolve the N=2^22 gap.
Matched 40GB NCU captures show exactly equal prefix executed instruction and
FP32 instruction counts. The N=2^23 LTO suffix uses 122 rather than 98
registers/thread with essentially unchanged work. No default core or production
selector has been changed based on these results.

The original short-stage samples exposed startup DVFS and are exploratory,
not ranking evidence. A separate probe accumulates at least 500 ms of GPU
warmup per stage and completes 56 units per card (eight correctness checks,
48 timing samples). All six GPU kernels have identical disassembled instruction
text in the original and supplementary binaries. The resulting stage data
does not constitute a new full migration calibration or installed model.
Next investigations should distinguish resource occupancy and local subgraph
decomposition from core substitution; this experiment proves neither global
optimality nor a limitation of hierarchical hybrid dataflow.

## FFT leaf lowering and device-core preparation, 2026-09-18 (preceding snapshot)

The matched DIF lowering study (local artifact: `../results/paper_completion_20260918/fft_leaf_lowering/REPORT.md`; not included in this source release)
is complete: 24 cells / 72 ordinary trials, independent stage samples, and
matched NCU diagnosis on A100 40GB/80GB. Carrying the radix-4 lane frequency
into the existing write offsets reduces cooperative G=4 whole-plan latency by
12.45–13.52%; G=2 improves by 1.23–1.65%. G=1 remains fastest. The four
workloads reach 84.91–101.43% of cuFFT throughput with G=1; this is a focused
large-1D FP32 experiment, not a renewed comprehensive comparison.

The optimized header is now in the production tree, matches the measured
private template byte-for-byte, and passes 20 affected public API checks
(16 GPU correctness cases and four rejection checks, including FP64 G=4
inverse). Ordinary timing and NCU completed before the
main build changed; keep their original identities. The independent stage
samples are not an installed full hardware profile. Source/compiler changes
require compatible samples before selector promotion.

An isolated cuFFT Device API EA experiment (local artifact: `../results/paper_completion_20260918/fft_leaf_lowering/LTO.md`; not included in this source release)
was compiled for SM80 using CUDA 12.8. It checks explicit LTO selection and
holds the cuButterfly dataflow fixed while substituting the thread core, block
core, or both. GPU validation was initially paused by external work; the
completed results are recorded in the newer section above. The LTO path
remains separate from production backend selection.

## Cooperative FFT prefix, 2026-09-18 (preceding snapshot)

The cooperative-prefix implementation and finite study (local artifact: `../results/paper_completion_20260918/fft_cooperative_prefix/REPORT.md`; not included in this source release)
are complete. `prefix_codelet_lanes` separates cooperative leaf width from
contiguous I/O columns across public replay, JIT, resource reporting,
portable candidates, search mutation and stage calibration. Static prefix
columns now use `threads*EPT/local_size`, with captured-kernel parity tests.
The corrected runtime cohort passes its 24-cell / 72-trial evidence audit,
18 covered stage predictions and six public selector replays on A100
40GB/80GB. Initial-runtime NCU captures remain separate diagnostics of
unchanged GPU modules.

G=2/4 reduce registers and dependency stalls but do not beat G=1 in these
six workloads: executed prefix warp instructions grow by 1.68x/3.44x.
G=1 reaches 84.90–101.82% of cuFFT throughput (six-case geometric mean
94.31%). These are large-1D FP32 results, not a full application matrix.
Independent stage ordering is 17/18 pairs and 5/6 best candidates; the
N22/batch1 miss includes unstable independent prefix timing and a
stage-sum/whole-plan discrepancy. Complete sampling must not be described
as perfect ranking. Reducing cooperative leaf instructions and isolating
this stage-context effect remain open; the overall FFT optimization goal
is not marked complete.

## Paper completion cohort, 2026-09-17 (historical snapshot)

The completion cohort (local artifact: `../results/paper_completion_20260917/README.md`; not included in this source release) now
finishes the focused NTT output-order integration, covered E6/E9 experiments,
and the strict same-leaf core/dataflow ablation. E6 pairwise ordering is 11/12;
E9 reaches its finite pool optimum at budget four but ties logical-traffic and
local-search baselines. Changing organization with the same leaf core improves
the six-case geometric mean by 1.217x (native) and 1.221x (cuFFTDx Thread).
These controlled gains are separate from production-library comparisons.

The finite rank2 supplement is complete on A100 80GB: all eight configurations
are correct, but the best throughput ratios against cuFFT are 0.203 for
2048x2048 and 0.247 for 4096x4096. The 12-cell large-1D extension is running
on A100 40GB. Its initial 28 stage records are correct and cover the requested
loads for the six source mappings; the original FP64 N=2^24 batch16 training
request remains explicitly incomplete because the probe memory bound is 15.
The actual matrix requests batch4 at that size. This is neither full design-space
calibration nor a completed 1D comparison. Earlier snapshots below retain their
original build and measurement scope.

The extension exposed a factor-streamed probe connection omission: the module
exports its group entry, but the constructor did not bind it. The one-line
repair (local artifact: `../results/paper_completion_20260917/factor_probe_repair/REPORT.md`; not included in this source release)
passes eight isolated forward/inverse, fp32/fp64, two/three-group correctness
checks and a large-N descriptor check. Its private binary is separate from
the ongoing frozen comparison; this fixes adapter access, not a measured
performance gap or full migration qualification.

## Paper experiment preparation, 2026-09-17

The paper experiment directory (local artifact: `../paper/experiments/README.md`; not included in this source release) now maps E1–E10
to questions, inputs, reusable scripts and acceptance criteria. CPU preparation
produced 18 fixed-core controls and nine distinct module requests; all nine
are compiled. GPU correctness and paired timing for these controls remain
pending. Strict core/organization factorial bindings, whole-workload holdout,
an independent equal-budget search pool and the code-reuse audit retain explicit
missing inputs. See the preparation record (local artifact: `../results/paper_preparation_20260917/a100-80gb-gpu1/report.md`; not included in this source release).

The GPU 1 acceptance checkpoint is now interrupted by contention at 5/72
completed cells, with the sixth search preserved. A CPU evidence snapshot keeps
confirmed-population ranking before/after service extension separate: top-1
1/5 and 5/5 respectively. The four completed FFT cells reach geometric-mean
throughput ratios of 0.983424 against cuFFT and 0.905910 against VkFFT. These
figures do not establish a full-matrix result or cold-start generalization.

## Independent A100 80GB continuation, 2026-09-17 (initial snapshot)

GPU 1 (`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`) is available while the
original GPU 2 remains occupied. A new target-local calibration (local artifact: `../results/staged_migration_20260917/a100-80gb-gpu1/report.md`; not included in this source release)
has completed capability probing and the existing cross-operator stage basis:
547 correct records, 52 physical curves, 259/259 covered stage holdouts
(median/P90 error 1.373%/5.268%). Target-local scheduling probes also completed;
the pipeline floor has 12/12 covered holdouts and 0.635% maximum error.
Its 72-cell acceptance (local artifact: `../results/comprehensive_20260917/a100-80gb-gpu1/report.md`; not included in this source release)
is running in a separate directory with fresh timings; the original five
completed cells remain intact. The existing SM80 build and module cache are reused.
Its first FFT N=4096/batch=32 cell passes search, exact selector replay and
paired comparison: cuButterfly/cuFFT/VkFFT medians are
6.543/6.758/6.4512 us. All three confirmed candidates are correctly ranked
(3/3 pairs, 1.0x regret). This is 1/72 completed cells at the recorded snapshot,
not a full-matrix result.

The offline compiler queue has finished, with 33,572 successful modules and
118 cuFFTDx-unsupported FP64 register-tile requests (`suffix_ept=32`). There is
no active compiler queue. The compilation report (local artifact: `../results/comprehensive_20260916/a100-80gb-precompile/report.md`; not included in this source release)
records these failures; the full exported inventory is not declared compiled.

## Cross-operator acceptance entry, 2026-09-16

The [research acceptance workflow](research_acceptance.md) now connects existing
search, finalist component-service extension, isolated registry promotion,
automatic-selector replay and paired baselines. Its 72-cell semantic matrix
contains no fixed searched mappings. Baseline adapters reject mismatched
contracts and distinguish external libraries from the in-tree radix2 ablation;
ranking loss and library speedups are reported separately.

The A100 80GB run (local artifact: `../results/comprehensive_20260916/a100-80gb/report.md`; not included in this source release) imports
the existing 695 records / 78 curves. Its journal snapshot has 5/72 complete
cells and is interrupted by another process on the target GPU. The sixth cell
retains 8/19 screening attempts (seven correct, one unsupported configuration).
Four FP32 FFT cells have geometric-mean throughput
ratios of 0.97355 against cuFFT and 0.88684 against VkFFT; N=262144/batch=16
is faster by 1.05971x/1.07603x, while the small-batch N=256 cases are slower.
FWHT N=4096/batch=32 measures 5.263 us versus the existing Dao interface's
19.425 us. All five populations have service coverage; top-1 is 3/5 with worst
selection regret 1.0024913x. These results do not qualify the full matrix or
full migration; see the report for per-case and baseline-contract details.

The probe-point type-boundary repair issued one missing-service request and
extended the cumulative curve inventory from 78 to 79 with `service_gaps=[]`.
The confirmed-population ranking changed from top-1 false, pairwise `1/3` and
regret `1.2115735x` to top-1 true, pairwise `3/3` and regret `1.0x`; the journal
retains the pre-repair ranking and gaps in its `*_before_probe_type_fix`
fields, while all search and paired timing rows remain intact.

The shared environment passes all 9 `tests/test_research_acceptance.py` tests
and 21 stage-service tests. `resource_projection_parity.json` (local artifact: `../results/comprehensive_20260916/a100-80gb/resource_projection_parity.json`; not included in this source release)
shows complete field-for-field agreement for 101 real candidate predictions
while reducing the CPU diagnostic from `0.1814671 s` to `0.1077384 s`
(`1.6843x`). The change projects resource keys directly from physical groups;
it changes no GPU kernel or model equation.

Large-batch schedule simulation now uses a bounded 512-result cache keyed by
all numerical service, resource and dependency inputs. The original simulation
and model version remain unchanged. All 42 affected cost/service/search tests
pass; a 12-point repeated prediction diagnostic preserves complete results and
reduces CPU elapsed time from 0.44494 s to 0.22912 s. This is a bounded CPU
diagnostic, not an end-to-end search or GPU speedup; see the linked run report.

Partial-search continuation restores completed selections without repeating
their evolutionary rankings, while preserving the stratified frontier, budget
and historical prediction evidence. The exact continuation command is unchanged;
the current run retains all five completed cells and its partial sixth-cell
prefix. Runtime binaries and kernel timing protocol remain unchanged.

The [research precompile entry](research_acceptance.md#prepare-research-compilation-before-the-gpu-window)
exports resource-dependent candidates once, then compiles with GPUs hidden.
All 72 workloads exported successfully: 475,203 candidate points deduplicate
to 33,690 requests. At the 2026-09-16 snapshot an eight-worker CPU queue was
running; its final outcomes are recorded above. Eleven CPU checks and one actual prepared
module's GPU cache-reuse/correctness check pass. A further CPU audit checks
32 prepared module manifests/hashes and 101 screening request projections with
no mismatches. No additional GPU timing was accepted during the busy window. See the
compilation report (local artifact: `../results/comprehensive_20260916/a100-80gb-precompile/report.md`; not included in this source release)
for live artifacts and remaining coverage boundaries.

## Batch-ring composition and OnlineReorder reuse, 2026-09-16

The composition report (local artifact: `../results/staged_migration_20260916/a100-80gb-composition/report.md`; not included in this source release)
records actual full-tile/remainder service resolution, an independent
`BatchPipeline` scheduling-floor probe wired into installation, and verified
two-group OnlineReorder stage reuse for block/register FFT cores.

All 27 FFT/NTT/FWHT pipeline plans pass three public correctness trials and
now have measured component services. Prediction median errors are
4.998%/4.109%/12.846%; the largest remaining FWHT error is 40.287%. The model
therefore retains its composition accuracy warning and unvalidated-concurrency
status. A minimal 3-group/32-tile pipeline alone costs 343.562 us, close to
the approximately 353-us FFT/NTT cases: runtime submission/event cost matters.
One residual FWHT trace shows no simultaneous kernels under profiling; its
unprofiled 82.276-us latency is close to the independent serial-service sum
of 80.200 us. The model's lower resource estimate assumes overlap that the
trace does not demonstrate. Profiler overhead is kept separate from calibration.

Four new OnlineReorder combinations reuse existing curves with no additional
stage timing rows. Six complete FFT plans pass with median/max prediction
error 0.196%/0.948%. The cumulative online-reuse checkpoint preserves the
657 prior records and adds 23 block-core plus 15 FWHT records: 695 records,
78 curves, 342/342 covered holdouts, interpolation median/P90 1.341%/5.418%.

Affected verification is 145 CPU tests passed, four optional projection checks
skipped, plus the new GPU scheduling probe and public-plan trials above. The
shared conda installation is updated and checked against source/build bytes.
Full inventory qualification, service-dependent concurrent handoff costs,
remaining adapter gaps, selector qualification and external baselines are still
unfinished. Component-service coverage is not complete migration qualification.

## Interleaved calibration and physical-stage reuse, 2026-09-16

The A100 80GB reuse report (local artifact: `../results/staged_migration_20260916/a100-80gb-stage-reuse/report.md`; not included in this source release)
records interleaved description/sampling, full pending-inventory persistence,
strict checkpoint imports through installation, and resource-verified reuse
of shared physical stages across different whole-plan partitions.

All 585 prior raw stage records remain. Four FFT/NTT source plans added 72
correct records; two new four-stage combinations reused curves without stage
resampling. Six public plans passed three trials each: median/P90 prediction
error 1.234%/1.900%, and 2/2 within-cell winner matches. The accumulated 657
records give 72 curves and 321/321 covered stage holdouts.

Affected regression: 128 CPU tests passed, four optional projection checks
skipped. The shared conda package is updated; installed helpers and the
unchanged stage-probe binary match source/build bytes.

This is a serial calibration/reuse milestone. The 17,444 static plans in the
three-cell inventory have not been fully qualified. Other backend reuse,
overlap calibration, selector qualification and paired external baselines
remain unfinished; the composed model retains its concurrency warning.

## Independent operator-stage service calibration, 2026-09-15

The current implementation adds `cubutterfly_stage_microbench` and an internal
`StageProbeAdapter`. The probe invokes the physical lowering, checks full-plan
and independent-group recomposition against the CPU reference, and measures
individual groups, adjacent serial pairs and the original plan. Compilation,
resource inspection, reference checks and input resets stay outside event
timing. CUDA graphs inspect launches; they are not the measured execution path.

`stage_service_calibration.py` owns sampling/checkpoints and
`stage_service_model.py` owns mechanism/resource-specific service curves.
The load coordinate is physical CTA work (`K = work_blocks`, with `grid_ctas`
as an alias). Piecewise linear interpolation stays inside measured load bounds;
unknown mechanisms/resources and out-of-range loads remain explicit gaps.
The search consumes actual stage descriptors and preserves compatible timing
evidence when only its descriptor needs upgrading.

Fresh installation defaults to `--stage-calibration full`, with no implicit
search or compilation time limit. `bounded` and `skip` are explicit policies.
Stage-service validation and whole-plan/composition validation have separate
gates; successful service interpolation alone cannot complete migration.
The hardware-specific execution and recovery commands are in the
[A100 80GB](hardware/a100-pcie-80gb.md) and
[A100 40GB](hardware/a100-pcie-40gb.md) runbooks.

The GPU stage report (local artifact: `../results/staged_migration_20260915/a100-80gb-v3-stage/report.md`; not included in this source release)
now records passing NTT lowering checks and 20 native stage recomposition cases.
The six-design FFT/FWHT diagnostic produced 36 correct records and 8 service
curves: all 17 independent holdouts are covered, with median/P90 interpolation
error 1.710%/8.723%. Matching public benchmark trials give median/P90 complete-
plan prediction error 0.166%/0.358% and correct selection in all three cells.
This qualifies the diagnostic's serial model, not the full hardware migration.

GPU testing exposed and fixed JIT resource-query failures, a load-dependent
`batch_time` curve key, and a per-launch synchronization protocol that inflated
small-stage timing. The probe now uses driver resource queries and the public
benchmark's batched direct-launch event protocol. Batch extensions refresh
descriptors; model revisions rebuild derived state without repeating compatible
trials. Composition audits use immutable sidecars: the observed per-save
checkpoint shrank from 212 MiB to 236 KiB without dropping requests.

The default 14 semantic cells also passed the explicit 128/256-thread shared-
stage basis: 549 correct records, 52 curves and 268/268 covered independent
holdouts (median/P90 error 1.419%/5.545%). All 28 original requests, including
batch loads merged during calibration, are covered in public whole-plan checks:
median/P90 error 1.098%/4.508%, 13/14 winner matches, mean regret 1.0000875x.
Full mode now reserves independent validation/refinement slots after boundary
sampling; explicit caps remain binding. NTT CSV mapping aliases and exact
uint64 modulus identities are fixed without repeating compatible GPU trials.

CPU acceptance is now 109 passed, with four optional projection checks skipped.
The shared conda package has been updated. Full inventory traversal, overlap
calibration, selector qualification and paired library baselines remain
unfinished. These explicit bases do not establish optimum kernels or complete
hardware migration. Low-precision references and some composite/overlap probe
paths remain explicit coverage gaps. The following v3 section describes the
preceding build and cannot qualify the current service model.

## Physical-stage inputs and scheduling calibration, 2026-09-15

The A100 80GB v3 report (local artifact: `../results/staged_migration_20260915/a100-80gb-v3/report.md`; not included in this source release)
records the preceding rebuilt and installed shared conda package. Generic
hierarchical tail metadata now counts 256-thread CTAs correctly; generic
online/hierarchical thread, shared-storage and data-space/time projections
agree between Python and actual constructed plans. An independent scheduling
probe is wired into staged installation and cached by target/probe/protocol.
On this card, minimal-kernel CTA dispatch slopes are 7–14x the former prior,
depending on thread shape; operator-specific service remains separate.

Three FFT/FWHT cells completed mapping/search/selector gates. A bounded tuner
round expanded budget from six to eight candidates per cell and preserved all
six initial confirmed timing arrays. Finalist confirmation now deduplicates
actual resolved designs, and known runtime rejections cannot retain an
optimistically low ranking. The report contains the final replay evidence and
the historical initial/tuner snapshots.

The final traversal returned 29 correct candidates out of 30 selections, with
two distinct confirmed designs per cell and all three mapping/search/replay
gates passing. Held-out top-1 is 1/3, mean regret 1.04850x and median relative
latency error 8.462%; every held-out split still lacks a primitive class.
The selected designs match the remeasured incumbents, so this is workflow and
calibration-coverage progress, not an established operator speedup.

The source and installed binaries/helpers match. Verification includes 102
affected CPU Python tests, four actual GPU plan/projection comparisons and two
C++ projection/mapping tests. Migration remains model-warning: independent
operator-stage service calibration, concurrent service measurements and broad
held-out coverage are unfinished. This is not a new comprehensive comparison
against cuFFT or proof of global optimality. A100 40GB has separate fresh-build
instructions in its [runbook](hardware/a100-pcie-40gb.md); its old timings cannot
resume against this rebuilt library.

## Compositional cost model and resumable tuning, 2026-09-15

The implementation and experiment report (local artifact: `../results/staged_migration_20260915/report.md`; not included in this source release)
records stage costs, actual-plan probing, bounded evolutionary proposals and the
installed-workflow tuner. Fresh migrations use `--cost-model staged`; old
manifests retain their historical model unless explicitly upgraded. Stage
functions share physical work/resource inputs across launch configurations.
Serial observations fit positive overhead/service coefficients. Batch-ring and
factor fan-in schedules are composed with shared resource capacity, while
concurrency calibration remains explicitly unmeasured.

A100 40GB completed three capability trials and saved ten search attempts: two
confirmed, six correct screens and two runtime rejections. Three proposals came
from outside the exported inventory; one passed correctness. A projection
failure exposed by an infeasible register neighbor was fixed. The retry then
stopped because another user's GPU work arrived. The final patched search,
FWHT cell and selector replay are still pending; the tuner saved a resumable
busy-GPU state. No new mapping was promoted by this incomplete invocation.

On the unchanged 80GB 70-row data, stage-model v2 holdout top-1 is 20% and mean
regret 1.55835x, worse than the historical legacy model's 53.33% and 1.26417x.
The model remains warning: the new decomposition is implemented, but sufficient
primitive calibration and reliable unseen-configuration ranking are unfinished.
This is not a new comprehensive performance result or proof of theoretical
optimality. The unchanged library and existing benchmark hashes are in the report.

Hardware-specific commands and evidence now live in [per-target notes](hardware/README.md),
with separate A100 40GB/80GB pages and a template for another model and capacity.
The shared conda environment contains the new helpers and plan probe. Affected
verification covers 117 distinct Python tests and actual plan construction for
all seven operator families; no full performance acceptance claim is made.

## Immediate search feedback verified, 2026-09-15

The installed follow-up (local artifact: `../results/unified_migration_20260914/seed_recovery/model_feedback_recovery/report.md`; not included in this source release)
completed the same three A100 80GB cells under the original `auto` policy and
unchanged binary. It traversed 44 attempts: 38 correct candidate records, six
runtime rejections, and six newly selected configurations; compatible timings
were reused. Mapping/search coverage and selector replay all pass 3/3, with
the same full selected mappings as the prior paired cuFFT experiment.

The model now incorporates each new correct screen before its next ranking,
after the eight-row bootstrap. At logN18/batch16, the first 28 ms shared point
enters the next update; the two previously repeated slow points are no longer
selected. Deterministic FFT projection and validation identity accounting are
also corrected. Installed helpers and 79 affected CPU tests pass, including
legacy CSV configurations that share a display name but have different launch axes.

This does not complete model calibration. The first unseen shared point and
some factor points remain severely underpriced, and runtime-incompatible scalar
online proposals still consume budget. The nine-row model has only two eligible
distinct multi-design workloads, top-1 accuracy 0.5 and mean regret 1.241687x;
it stays warning. A separate identity-corrected refit of the unchanged 70-row
data set reports 0.5333 top-1 accuracy and 1.264173x regret, also warning. These
accounting changes do not alter model coefficients or establish a performance
improvement. The full acceptance matrix and broad model goal remain unfinished.

## Automatic FFT recovery and cost-model input audit, 2026-09-15

The generic historical-mapping revalidation path now runs through the unified
calibration entry and installer. It imports mapping parameters from historical
registry winners or explicit seed files, revalidates them under target workload
semantics, reserves a separate seed budget, and freezes provenance across resume.
The shared conda installation contains the updated Python helpers; the CUDA
library and benchmark hashes are unchanged.

On A100 80GB, all eight compatible seed proposals for the three controlled FFT
regression cells were checked, alongside two new exploration candidates per
cell. Automatic selector replay passes 3/3. The paired automatic-selection
ratios are 1.0622x, 1.0011x and 0.9860x cuFFT for logN18/batch16 and
logN20/batches4/16; their geometric mean is 1.0159x. Six full-batch correctness
checks and 18 paired timing rows pass, with maximum timing range 0.267% of the
median. See the seed recovery report (local artifact: `../results/unified_migration_20260914/seed_recovery/report.md`; not included in this source release).
This completes automatic recovery for these cells, not the full application
matrix or the broader model-calibration objective.

The cost-model audit found that legacy zero group counts were interpreted as
one despite physical plans describing 18/20 groups. The extractor now uses the
physical plan and records feature version `physical-execution-groups-v2`.
However, refitting the original 70 rows (local artifact: `../results/unified_migration_20260914/seed_recovery/group_count_ablation/report.md`; not included in this source release)
still gives 46.7% held-out top-1 accuracy, with mean regret worsening from
1.195x to 1.264x. The model remains warning. The successful seed-recovery
experiment's separate nine-row, three-cell model does not override that result.
The extension to 12 exploration candidates per cell (local artifact: `../results/unified_migration_20260914/seed_recovery/model_group_recovery/report.md`; not included in this source release)
has now completed all three cells: 44 attempts, 37 correct candidate records,
seven runtime rejections, and selector replay 3/3. The selected full mappings
are unchanged from the paired recovery experiment. The migration remains
incomplete only because the model gate is warning; it is no longer waiting
for a GPU window.

The completed trace exposes stale search feedback: after a shared-iterative
candidate measured 28.03 ms, two more selections still used the old model's
0.075 ms prediction. Reconstructing a model with that observation predicts
28.19 ms for the next slow point. A separate parity audit finds deterministic
projection mismatches in 24/37 correct records; resolving the factor and
register core axes reduces this to nine, with legacy hierarchical/online
metadata and unavailable compiler resources still requiring attention.
The final model's name-based validation also confuses mapping aliases with
distinct designs. These findings concern model/search engineering; neither
the projection correction nor successful measured mappings certify unseen
workloads or the full application matrix.

## Controlled historical FFT mapping replay, 2026-09-15

The latest controlled replay confirms mapping-selection misses for FP32
logN18/batch16 and logN20/batches4/16. Under the current A100 80GB binary and
the migration's explicit `auto` compile policy, the historical mappings achieve
1.062x, 1.000x and 0.987x cuFFT throughput in-place. The latest selected
mappings achieve 0.753x, 0.700x and 0.665x in the same run. Out-of-place replay
gives the same conclusion; historical-mapping placement differences are at
most 1.35%. All 18 full-batch correctness preflights and 54 timing rows pass,
with the largest timing range below 0.49% of the median.

See the controlled replay report (local artifact: `../results/unified_migration_20260914/historical_mapping_replay/report.md`; not included in this source release)
for mappings, complete timings and provenance. Existing fast mappings remain
effective; their core shapes were enumerated but omitted by the latest
12-candidate search budget. Global-scratch boundaries occur in both fast and
slow mappings and do not establish a theoretical or lowering failure.

This earlier replay was diagnostic completion: it used explicit mappings and
changed no registry or installed code. The subsequent automatic seed-recovery
experiment above implements and verifies the workflow step it motivated.
The broad model warning, application matrix and two-device research goals
remain unfinished.

## Migration workflow recovery, 2026-09-15

The A100 80GB installation-calibration run resumed with its original `auto`
compile policy and unchanged binaries. Its 14 configured workloads have
confirmed mappings; 52 internal records were promoted, and all 19 distinct
selected semantic cells (including legacy smoke cells) passed automatic
selector replay with `calibrated-registry` confidence. This is the `:default`
runtime fingerprint, separate from earlier `:research` measurements.

The initial 64-candidate/300-second screening protocol completed its traversal,
but attempted only 35 candidates. Structured-2x2 and both NTT workload searches
received zero candidates after the total budget expired. Their incumbent
mappings were measured; their new candidate searches were not complete. The
preserved initial run (local artifact: `../results/unified_migration_20260914/initial_bounded_run_20260915/`; not included in this source release)
contains the calibration, model and replay evidence. The fitted model has 52
training rows, workload-holdout top-1 accuracy 0.4167 and mean latency regret
1.2042x; it remains `calibrated-local-warning`. The archived migration manifest
predates the stricter completion gate: its success status records that the
command finished, not that the requested search budget or model gate passed.

The first extension attempt reused those measurements and was interrupted while
confirming FP64 logN16/batch64. Its checkpoint was then resumed in a verified
exclusive A100 80GB window; the final result is recorded below.

The unified entry now audits individual workload coverage, search-protocol
completion, model validation and selector replay separately, and supports
`--resume-from` for protocol recovery. Installation uses that entry for replay.
FFT acceptance can read the exact calibration workload file via `--workloads`
instead of expanding silently to other batch/placement cells. See the
[migration guide](hardware_profile_install.md). The larger two-device research
goals remain unfinished.

The shared `/home/wt/yes/envs/cubutterfly` installation now contains the unified
entry and exact-workload FFT acceptance helper. Standalone installed-path
dry-runs resolve the bundled 14-cell configuration, preserve the real resume
manifest, and restore this checkpoint's `auto` policy, 12-candidate budget and
registry path. The library and both benchmark hashes are unchanged. The
affected calibration, installer, registry, model and FFT acceptance CPU suites
pass 75 tests. These checks validate orchestration and artifact contracts; the
paired FFT matrix (local artifact: `../results/unified_migration_20260914/paired_fft_comparison/prepared_matrix.json`; not included in this source release)
contains the six exact FFT calibration cells and their paired timings.

When the recorded A100 80GB UUID is exclusive, resume from the repository root:

```bash
CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
  /home/wt/yes/envs/cubutterfly/bin/python \
  /home/wt/yes/envs/cubutterfly/bin/calibrate_hardware.py \
  --resume-from results/unified_migration_20260914/NVIDIA-A100-80GB-PCIe-sm80-84974239744B/migration_manifest.json
```

After confirming the manifest's mapping and search coverage, run the paired
comparison using the same explicit policy and workload file:

```bash
CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
CUBUTTERFLY_COMPILE_MODE=auto \
  /home/wt/yes/envs/cubutterfly/bin/python \
  /home/wt/yes/envs/cubutterfly/bin/run_fft_acceptance.py \
  --build-dir build-a100-cufftdx \
  --output-dir results/unified_migration_20260914/paired_fft_comparison \
  --workloads config/install_search_workloads.json \
  --trials 3 --warmup 50 --repeat 50 --verify-batches 0 --resume
```

This paired result will assess the measured selector cells. A warning model
still requires separate diagnosis and cannot certify unseen workloads or the
remaining hierarchy-aware model work.

## Completed A100 80GB extension and paired FFT result, 2026-09-15

The resumed extension completed all 14 configured cells with 12 candidate
screenings and three finalist confirmations per cell. Mapping coverage and
search protocol coverage are both 14/14; selector replay passes 19 semantic
mappings. The updated model has 70 internal rows, held-out top-1 accuracy 46.7%
and mean latency regret 1.195x, so the migration manifest correctly remains
`incomplete` because the model gate is warning.

The six-cell paired FFT comparison completed on the same A100 80GB. FP32
geometric mean is 0.785x cuFFT (logN16/batch64 1.026x, logN18/batches16/64
0.753x/0.827x, logN20/batches4/16 0.701x/0.666x). The FP64 logN16/batch64
cell is 0.829x. All six use calibrated-registry mappings with no fallback.
The detailed gap analysis (local artifact: `../results/unified_migration_20260914/paired_fft_gap_analysis.md`; not included in this source release)
records the selected routes and measured overlap alternatives.

The original attribution of this gap to an unrealized cross-CTA tile-resident
schedule was too strong. Global-scratch boundaries and launch/event costs
describe an implementation, but do not prove that mixed dataflow is absent.
The later controlled replay above recovers near-cuFFT or better performance
for three of these cells using historical mappings on this same binary.
Their fast core shapes were budget-omitted by the bounded search. Search
coverage and selection are the demonstrated issues for those cells; other
performance gaps and the model warning still need their own evidence.

## Current State — Factor Cores and Partial Factor Scheduling, 2026-09-14

The current build and installed library have SHA256
`00908a56615ec151d50a8cbc471d32ec9a213dacaded76f6dae4488dc16c1e49`; the
benchmark executable is `9b0fc930a849b5e82e04b3e85ce5ee7578db18c20273c15e3b17969cbd99b8e9`.
The runtime fingerprint is
`3bc71ebee6de40d95cc15ac0f9fec44aa4d90bf8455f24d5349400e23d225f97:research`.
The earlier `4f6ae944a29a54591dfee428f2a5423c78147c502ef58d044d1661e3953b09a8`
build and its performance records remain historical evidence, not current
automatic-selector calibration.

The public FFT `factor-streamed` lowering separates logical macro stages,
physical FFT factors, data tiles per CTA and asynchronous input prefetch depth.
It includes imported and native cores, JIT compilation, mapping replay,
physical resource/boundary reporting and dependency-closed partial-factor
scheduling. See [the mapping contract](unified_planner.md#fft-factors-and-data-prefetch)
and the partial-factor readiness report (local artifact: `../results/framework_refactor_20260912/factor_partial_readiness_20260914.md`; not included in this source release).

The partial-factor public async test covers four scheduler-sensitive shapes,
both serial and overlapping modes, and three A/B/A caller-stream submissions
per mode; the CTest run passed in 24.85 seconds. Native and imported partial
correctness each passed 36 cases in
`factor_partial_{native,imported}_correctness.json`. The four-case memcheck
run reports zero errors. Focused racecheck and synccheck runs for the native
four-factor and imported three-factor cases report zero hazards/errors.

The native FP64 N=2^24, batch1, slices4 diagnostic trace in
`factor_partial_n24_s4_trace.json` reports 125.089 us factor-0/factor-1
overlap, three early consumer launches and zero dependency violations. The
final full-fan-in join has no overlap, as required by the schedule. These are
diagnostic trace measurements and scheduler evidence, not accepted benchmark
performance.

CPU-side validation has 45 passes across the dependency-oracle, trace,
compiler, install-search and FFT-acceptance checks; the model suite has a
further 14 passes. The eight-cell native-core schedule ablation completed on
A100 80GB: whole execution wins all eight cells; FP64 logN24/batch1 reaches
96.6% of cuFFT throughput. These fixed-core ablations are separate from automatic
selection. See the controlled results and build provenance (local artifact: `../results/framework_refactor_20260912/factor_partial_schedule_summary.md`; not included in this source release)
and NCU attribution (local artifact: `../results/framework_refactor_20260912/ncu_factor_partial_analysis.md`; not included in this source release).
The 56-point model remains `calibrated-local-warning`: complete-workload holdout
top-1 accuracy is 75%, with mean latency regret 1.0253x.

Cross-operator revalidation found an NTT object compiled across a shared-header
edit, with an obsolete 168-byte `ExecutionGroup` layout instead of 176 bytes.
Recompiling that translation unit fixes the crashing NTT mapping. Seven public
checks, including NTT, butterfly, C API and factor async execution, now pass on
the A100 40GB; the installed-package consumer also passes. The shared conda
installation has been refreshed; the installation checkpoint (local artifact: `../results/framework_refactor_20260912/factor_partial_install_state.json`; not included in this source release)
records the verified hashes and remaining work. Earlier FFT ablation records retain their
original `80e3...` benchmark identity; FFT source and modules did not change in
the NTT rebuild.

Both target cards became occupied before current-build calibration could finish.
Twenty-five mapping-only seeds are prepared for fresh A100 40GB measurement.
The resume entry (local artifact: `../results/framework_refactor_20260912/resume_factor_partial_a10040.sh`; not included in this source release)
revalidates them, runs bounded calibration and the eight-cell automatic paired
comparison, then prepares the full matrix. It uses a private capacity-specific
registry and never imports 80GB timings as 40GB measurements.
Run it with `bash results/framework_refactor_20260912/resume_factor_partial_a10040.sh`
when the named GPU is exclusive. Shell syntax and all 75 seed commands passed
offline checks; the full resume workflow still needs GPU execution.
A100 40GB current-build calibration and paired acceptance remain pending. Old-build
`4f6ae944a29a54591dfee428f2a5423c78147c502ef58d044d1661e3953b09a8`
performance history must not be promoted to current automatic calibrated
performance. General CTA warp-role lowering and the full two-device acceptance
matrix remain pending.

On 2026-09-14 the free A100 80GB device admitted a resumed calibration search.
Eight incumbent candidates and two FP32 workload searches completed before an
exclusive-GPU check detected a new external process during the FP32 logN=24,
batch=1 workload. The partial search checkpoint is retained in
`factor_partial_a10080_calibration/`; its status and exact resume protocol are
in factor_partial_a10080_resume_status.json (local artifact: `../results/framework_refactor_20260912/factor_partial_a10080_resume_status.json`; not included in this source release).
No contended measurements were retained. This interruption is historical;
the later completed run is recorded below.

The next exclusive A100 80GB window completed the resumed search and the selected
8-cell paired acceptance. FP32 geometric-mean throughput is 0.8083x cuFFT and
FP64 is 0.7954x; all eight cells used `calibrated-registry` mappings. The full
capacity-clamped matrix was prepared with 176 cells and has now completed. Its
geometric mean is 0.7044x cuFFT for FP32 and 0.6346x for FP64; 80/88 FP32 cells
and 84/88 FP64 cells used the `unmeasured-feasible` fallback because only eight
workloads were calibrated. Details and per-cell medians are in the A100 80GB
full-matrix report (local artifact: `../results/framework_refactor_20260912/factor_partial_a10080_full_acceptance_report.md`; not included in this source release).
These results expose incomplete calibration coverage and changes in selected
execution routes. They do not establish a dispatch bug or a causal kernel
bottleneck. Fallback cells are not promoted as measured optima.

The theory versus engineering audit (local artifact: `../results/framework_refactor_20260912/theory_vs_engineering_20260914.md`; not included in this source release)
records the current boundary: the hybrid-dataflow premise is not contradicted,
but cross-factor on-chip residency, CUDA port/II proof and a hierarchy-aware
cost model remain incomplete. The earlier shorthand that called the whole
cost-model/calibration path “完成” referred only to the bounded A100 80GB
8-workload scope and should not be read as full-matrix calibration.

A bounded two-route NCU diagnostic (local artifact: `../results/framework_refactor_20260912/factor_partial_a10080_full_acceptance/ncu_lowvalley_analysis.md`; not included in this source release)
compares FP32 logN20/21, batch16, using the full matrix's saved mappings. The
N21 scalar route executes more instructions and reads more DRAM per input point;
its suffix uses only 25% of global load/store sectors. Both routes report zero
local load/store sectors. These observations identify investigation targets,
but size, partition and core differ, so the comparison is not a causal ablation.
FP64 logN14/batch1's 4.35x ratio is a throughput win; earlier text calling it a
valley inverted the ratio and has been corrected.

## Historical — Whole-Factor Follow-up Before Partial Scheduling — 2026-09-14

This subsection records the whole-factor follow-up that preceded the partial
factor scheduler. Its evidence belongs to build
`4f6ae944a29a54591dfee428f2a5423c78147c502ef58d044d1661e3953b09a8`, not to
the current build. The implementation and experiment details are retained in
the factor-core follow-up report (local artifact: `../results/framework_refactor_20260912/factor_core_followup_20260914.md`; not included in this source release);
the installed state is recorded in
factor_final_install_state.json (local artifact: `../results/framework_refactor_20260912/factor_final_install_state.json`; not included in this source release).

The whole-factor public numerical checks passed 96 cases across native and
imported cores, FP32/64, directions, strides and placements. The final
registry retained 29 confirmed mappings, and 12 automatic-selector replays
passed after installation. The installed FP64 batch1 comparisons reached
96.7% of cuFFT at N=2^24 and 90.1% at N=2^18; their two-cell geometric mean
was 93.3%. The model remained `calibrated-local-warning`, and the full
performance matrix was not met. These numbers are historical follow-up
evidence, not current calibrated acceptance.

## Historical — Previous Verified Installed Build — 2026-09-13

This section records the previous installed build. The milestone sections below are historical
evidence for their named artifacts, not instructions to repeat completed work.
The latest experiment record is
2026-09-13 NCU and lowering experiments (local artifact: `../results/framework_refactor_20260912/ncu_followup_20260913.md`; not included in this source release),
following the calibration and paired comparison (local artifact: `../results/framework_refactor_20260912/research_followup_20260913.md`; not included in this source release).
The library build is unchanged; calibration, focused comparisons, NCU attribution,
public API checks and regression anchors now have GPU evidence.

The register FFT adapter now supports bulk and batch-pipeline
execution for FP32/64, including per-group V1 module entry points. The compiled
inventory also exposes its existing logN=22 points. Register group grid, shared
capacity and data-time account for the columns processed by each CTA, with JIT
resource reports attached after compilation. Single-column suffixes write from
registers without a redundant full-output shared tile; their actual core shared
requirement is validated after compilation. This admits the FP64 10+14 mapping
on A100 without raising the device capacity limit.

The latest verified installed library and build have SHA256
`3eba31ad4325be54ae38b874a1092e554210c4ad12d75ffd45302a0aee6f4669`.
The final register/async/mapping/IR GPU checks pass, as do FP64 logN=24 forward
streaming with a partial tile and inverse in-place strided verification.
The installed consumer now passes for this build, and 11 automatic-selector
replays verify the newly calibrated mappings. The final public/C API GPU checks
also pass after the suffix-capacity change.
Earlier milestone performance numbers in this document belong to their named
artifacts, not to an unmeasured newer build.

Calibration journals individual confirmation trials, preserves later cells
across repeated interruptions, and skips completed cells on an identical-protocol
resume. Calibration and paired tests completed in the available GPU window.
NCU of all three lagging FFT cells and cuFFT completed, as did a suffix-store
ablation. Another job arrived before the additional experimental-kernel traces
and three-factor GPU validation; recheck availability before resuming those.
Contended timings are not promoted. General CTA warp-role
lowering and full FP32/FP64 acceptance on both A100 capacities remain unfinished.

The approved scope covers the six mathematical operators, existing numeric
contracts, arbitrary-length and multidimensional composition, and migration
between GPU models. Distributed execution of one transform is outside this release.

### Completed Follow-up and Pending Experiments

The completed calibration directory is
`results/framework_refactor_20260912/register_pipeline_capacity_calibration/`.
All 11 workload cells finished their recorded search budgets: 186 candidates
screened, 33 records confirmed with three trials, and 31 deduplicated mappings.
The local model's leave-one-out median relative error is 17.34%. After 11
successful automatic-selector replays, the 31 mappings were merged into the
public registry while retaining earlier fingerprints. The latest host-only
verification remains 33 passes and 4 GPU-dependent skips.

The following is the command used to resume and complete calibration. It is
retained for provenance, not a pending step. The capability-profile directory
name is historical; the calibrator validates model, SM and memory capacity.

```bash
CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
CUBUTTERFLY_COMPILE_MODE=research \
CUBUTTERFLY_REGISTRY="$PWD/results/framework_refactor_20260912/register_pipeline_capacity_registry.json" \
/home/wt/yes/envs/cubutterfly/bin/python scripts/calibrate_local_hardware.py \
  --build-dir build-a100-cufftdx \
  --profile-dir results/local_hardware_profile_a100_cufftdx_gpu1 \
  --output-dir results/framework_refactor_20260912/register_pipeline_capacity_calibration \
  --search-only --search-workloads config/pipeline_followup_workloads.json \
  --search-budget 20 --search-finalists 2 --search-seconds 600 \
  --compile-seconds 1200 --operator-trials 3 \
  --operator-warmup 20 --operator-repeat 50 --verify-batches 2 --resume-search
```

The formal-build paired cuFFT comparison completed for six calibrated FFT cells.
The four FP32 cells have 90.59% geometric-mean throughput relative to cuFFT; the
two FP64 cells have 52.88%. These precisions have different workload subsets.
All three same-mapping bulk/pipeline ablations favored bulk execution. Exact
cells, timings and raw evidence are linked in the latest experiment report.

NCU identifies FP64 logN=24 suffix spills and inefficient output stores; its
255-register prefix has no measured local-memory traffic. cuFFT uses three
256-factor kernels for that cell, with no measured local traffic. An isolated,
fully verified output-layout ablation improves FP64 logN=24/batch1 from 2.446
to 1.659 ms (66.3% of cuFFT), including the extra transpose. It has not been
promoted to the installed library. The final public/C API checks pass, and the
six FP32 regression anchors reach 98.953% geometric-mean cuFFT throughput with
all per-cell changes versus the previous build below 0.1%.

Pending focused GPU work is validation/timing of `fft_factor_ablation.cu` and
additional NCU of the output and factor experiments. The three-factor probe
compiles without spills, and its digit-rotation formula passes a CPU oracle;
CUDA correctness and performance are not yet established. Full FP32/FP64 acceptance on both A100 capacities remains
a separate authorized milestone. Neither focused comparison satisfies it.
The target remains at least 80% of cuFFT geometric-mean throughput for each
precision and hardware target on the full matrix, with the existing regression
anchors checked separately against the 5% regression bound.
General CTA warp-role lowering and further compute-core optimization remain open.

Use the [shared GPU recovery procedure](hardware_profile_install.md#shared-gpu-waiting-and-recovery)
when the device is occupied. Do not resume the older `register_pipeline_calibration/`
against this build or repeat the completed calibration solely to continue profiling.

## Historical Milestone — Initial Unified Framework, 2026-09-12

These checks preceded the register-pipeline and suffix-capacity changes above.
Their performance numbers and test counts apply to the cited artifacts.

Implementation milestones at that point:

- M0: baseline snapshot saved in `results/framework_refactor_20260912/baseline`.
  Source archive SHA256: `29ba53d789d150e16f0852b83563e68b7cd27945a9095ac5596d3d18f3bf3749`.
  Installed/static library SHA256: `92bf141e34a651e12feb24404d8e739965c882d2d23e02fb6ef9df03ac65b0fa`.
  Existing source changes were included; no user changes were discarded.
- M1: implemented and validated — common C/C++ selection,
  full mapping JSON and replay, cumulative hardware registry, installed tool
  dependencies, and the new `cubutterfly::Transform/Context/Plan` interface.
  Package-consumer, C API, rank-2/Bluestein replay, host mapping roundtrip and
  registry tests pass. Fourteen GPU registry fixtures verify all operators,
  both directions, complete mapping replay and obsolete-code rejection.
  Effective inverse normalization is canonicalized for structured/zeta operators.
- M2: common executable projection implemented — per-group stage intervals, launch/resource/layout records,
  separate compilation/correctness/measurement states, and generic shared
  execution with arbitrary positive partitions. The common shared template now
  includes NTT. Its 42-case matrix passes for all operators, FP32/64 and low
  precision, NTT word32/64, strided/in-place butterfly execution, and 1/3/12
  NTT groups. Partition enumeration cycles over all feasible group counts;
  budget zero enumerates all compositions. NTT logical radix and duplicate
  shared-footprint multiplication were fixed. Abstract stage-role schedules
  without an implementation still require lowering; this is not full coverage
  of every theoretical combination of roles, pipelines and processing cores.
- M3: implemented — a standalone NVCC module compiler and versioned C ABI now
  exist, with template/compiler/SM cache keys and compiler resource queries,
  for register-prefix/grouped-suffix FP32/64 FFT (V1) and all six shared
  operator traits (V2). The latter specializes stage indices and writer layouts.
  Research execution uses modules without relinking the library. The library
  exports search candidates; installation fits the local model after valid
  bootstrap measurements and interleaves ranking with exploration. Separate
  600-second additional compilation and 300-second search budgets are configurable.
  Per-group memory, data reuse and grid features distinguish unequal partitions.
- M4: engineering implemented, performance optimization remains open — arithmetic
  traits are reusable across precompiled and specialized templates. FP64 uses
  the register FFT template; FP32/64 logN=24 correctness passes. This establishes
  legal high-throughput candidates, not the expanded performance target.
- M5: implemented for the existing composition contracts — Bluestein FFT defaults to internal forward/inverse
  power-of-two plans; external cuFFT remains an explicit option. Numerical
  tests pass. NTT propagates complete mappings into convolution cores. Rank-2,
  embedding and arbitrary-length plans serialize each axis, including distinct
  forward/inverse convolution mappings. Tests replay these through the public API.
- M6: partially validated — that milestone's paired FP32 six-cell regression on A100 80GB
  passes: geometric-mean throughput 98.880% of cuFFT, each current/previous ratio
  within 0.18%. Expanded precision/size/batch acceptance and A100 40GB comparison
  remain outstanding. Other jobs continuously occupy the 40GB card and
  intermittently occupy the available 80GB card; contended runs are not promoted.

The frozen six-cell FP32 FFT comparison and historical per-operator comparisons
remain the regression anchors. No new performance claim follows from a host
build or an inventory check. The 98.880% figure covers logN 18/20 and batches
1/4/16 only; it must not be extrapolated to logN 3..24 or to FP64.

The milestone's validation artifacts are under `results/framework_refactor_20260912`.
`common_traits_correctness.json` records 42 passing GPU cases.
`registry_replay_contract.json` records 14 passing correctness fixtures.
`paired_anchors_final.json` and its log contain that milestone's exclusive paired timing runs.
`fp32_log24_correctness.csv` and `fp64_log24_correctness.csv` are correctness-only
checks and contain no accepted performance comparison. The installed-package
check includes importing each calibration tool from the installed prefix.

The live installation search integration completed on the exclusive A100 80GB:
`install_contract/` records six small operator cells, model-guided search,
17 verified mapping records, and six successful automatic-selector replays.
The short budget omitted candidates and left the last two cells with too few
samples for a per-cell fitted model. These records validate the installation
workflow; they do not replace the full performance acceptance matrix.

The verified package is installed in `/home/wt/yes/envs/cubutterfly`, including
benchmarks, calibration tools, template sources and enabled MathDx headers.
`installed_jit_correctness.csv` validates FP64 specialization using only the
installed template and MathDx paths. Activation defaults to `research` while
preserving an explicitly configured compile policy; deactivation restores it.
The 17 confirmed small-cell records were merged into the user's cumulative
registry. Their `:default` fingerprint applies to precompiled/auto shared
execution; research specializations deliberately require their own measurements.

Validation at that milestone: 66 Python tests (81 parameterized subtests), 37 host CTest
checks, public/C API/package-consumer checks, 42 generic-module GPU cases,
14 registry replay fixtures, and FP32/64 logN=24 numerical checks pass. Two
historical v0.8 test files absent from the checkout are registered only when
present; no missing test is reported as a pass.
This document deliberately does not mark the full optimization target achieved:
the expanded performance acceptance matrix and additional abstract stage-role
lowerings remain necessary. The implementation guide is `unified_planner.md`.

## Historical Milestone — Multi-group Scheduling, 2026-09-12

This section records build `be139c478e3b7e6b3fafc3188cc9c46fa54da766168ac4c7da4ed484a8cc64b0`,
before the register-pipeline and suffix-capacity changes. Its implementation
status and check counts are superseded by the current handoff above.

The shared lowering now supports batch streaming through any positive number
of physical groups greater than one, across all six operators and both NTT
word sizes. The same two-slot-per-edge scheduler serves precompiled kernels,
standalone research modules, and the existing two-group imported FFT adapter.
Cross-stream calls fence plan-owned storage; an edge slot is reused only after
its immediate consumer finishes. Global ring storage is not described as CTA
shared residency. At this milestone, register-prefix FFT used its bulk adapter;
the current implementation also supports its batch pipeline. General warp-role
schedules inside a CTA remain a separate unfinished lowering task.

Both precompiled and research 42-case numerical matrices pass (84 runs total),
including three-group floating/low-precision/strided/in-place execution and
two/three/twelve-group NTT. Async regression additionally covers three/four/
twelve-group FFT, partial tiles, workspace canaries and caller-stream changes.
Public C/C++ mapping replay preserves streaming and per-edge workspace for all
six operators. See `pipeline_all_ops_{precompiled,research}.json` and
`multigroup_pipeline_{ctest,research_ctest}.log` in the current results directory.

The cost model now recognizes NTT's physical-group CSV fields. It distinguishes
the full-batch communication volume, tile launch grid, ring capacity, kernel
launch count and event handoffs. This repairs feature semantics; it is not a
claim of a validated new prediction error or optimal overlap policy.

`run_fft_acceptance.py` exports the memory-clamped calibration matrix for exact
CUDA device identity and runs resumable, alternating cuButterfly/cuFFT trials.
Defaults cover 176 cells on the A100 80GB. Clamped duplicates count once. The
full acceptance gate cannot pass for a selected subset or uncalibrated fallback
cells. Bounded CPU checking records how many transforms were checked within
the actual full GPU batch. The cuFFT reference's inherited logN=20 cap was
removed so the large-size comparison reaches the library's own validation.

A fresh 11-workload search was attempted with three confirmation trials and
the research policy. Other jobs occupied GPU 2 during screening, so exclusivity
checks stopped it; no new performance records were promoted. A retry was also
blocked by another active job. The first interruption exposed a checkpoint gap:
incumbent records and current-cell screens were held until the entire search
finished. Checkpoints now persist each completed candidate atomically, validate
both binaries/compile policy/verification coverage on resume, and retain
unconfirmed screens without promoting them. A contention-and-resume test passes.

Full calibrated performance results on either capacity remain outstanding.
The earlier 98.880% six-cell result belongs to the previous paired measurement;
it is not a performance claim for this follow-up build.

Checks for this historical build: 70 Python tests and 81 parameterized subtests pass;
the final mapping/IR/public API/async tests and C API/installed consumer pass.
cuFFT FP32/64 logN=24, batch=1 numerical checks pass; their CSV timings are not
accepted performance evidence because other GPU jobs were present. The package
and new acceptance runner are installed in the shared `cubutterfly` environment.
Build and installed `libcuntt.a` SHA256 match:
`be139c478e3b7e6b3fafc3188cc9c46fa54da766168ac4c7da4ed484a8cc64b0`.
Prepared matrices live in `fft_acceptance_a100_40gb/` and
`fft_acceptance_a100_80gb/`, including requested/effective batch and exact
CUDA device identity. Neither directory contains a completed performance gate.
