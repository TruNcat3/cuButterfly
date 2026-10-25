# Descriptor-based retest, 0.9.2

The user authorized a new run on 2026-10-07 after the unified semantics repair.
The old reviewed campaign remains cancelled. The new entry consumes public
Descriptor JSON directly, preserving every semantic dimension.

## Explicit decisions and current stage

The user selected comparison of both native and adapted execution. Actual
ordering/layout conversion kernels enter complete-call timing and are listed
in each result. Neither side may promote precision, shrink a batch, substitute
an operator, or use historical timings for a missing row. Compilation, planning,
allocation, input restoration and reference checks stay outside kernel timing.

Launch target: NVIDIA A100 80GB PCIe, 84974239744 bytes, UUID
GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02. Other cards had foreign processes.
Fresh hardware-rate probes completed with five trials. Primitive calibration
remeasured 59 confirmed mapping descriptions with three trials, 100 warmup and
100 timed calls, then completed 673 measured points for 40 stage configurations
at 22:53 +08. Selector replay passed and 50 records were promoted. This does
not calibrate every public descriptor or every theoretical design point.

The cost model reports `calibrated-local-warning`, not a globally qualified
model. Stage holdout median/p90 errors are 1.56%/7.64%; the serial whole-plan
subset covers 48 of 52 records with median/p90 errors of 1.20%/17.70%.
The resident FFT log12 independent-stage adapter is unavailable, four serial
records lack service-curve coverage, and concurrent composition has no validated
samples. `usable_for_unmeasured_ranking` remains false. Measured comparison
rows must not be inferred from these model estimates.

The source has 73 records. Its 13 screening-only records and one unavailable
record are not confirmed seeds. Only the remaining 59 configuration descriptions
are reused; old timings, stage services and hardware rates are not imported.
The performance model still predicts between measured service points; those
predictions never become observed benchmark results.

The user selected complete floating error reporting, including reference-library
errors, before threshold selection. Diagnostic floating rows cannot claim an
accuracy pass before this review. Integer and finite-field outputs are
checked exactly, including UINT64 inputs above FP64's exact-integer range.
There is no post-failure tolerance relaxation.

## Finite declared population

The executable definition is scripts/prepare_semantic_retest.py. Configuration
and immutable manifest are in results/semantic_campaign_20261007_v092/population/.
Each population crosses all its explicitly listed values. It does not cover all
possible sizes, primes, matrices or per-axis permutations.

| Population | Shapes and batches | Other dimensions | Unique requests / static gaps |
| --- | --- | --- | --- |
| Semantic crossing | 8; 4×8; 2×4×8; 2×2×2×4. B=1,3 | Both directions, normalization choices and placements; packed/independent padded layouts; four independent input/output order patterns | 28,704 / 4,544 |
| Rank boundary | Eight and nine axes of length 2. B=1 | Both directions; natural/opposite-alternating input/output; packed; out-of-place | 464 / 288 |
| Size/batch scaling | 1D N=256,4096,65536,1048576,16777216; 2D 64²,256²,1024²; 3D 16³,64³,128³. B=1,16,256,4096 | Both directions and placements; natural/reversed input/output; packed | 19,008 / 2,816 |
| Arbitrary/embedding | 7; 3×5; 3×5×7. B=1,3 | FFT FP32/FP64, FWHT/structured FP32; exact-length versus explicitly requested embedding; both directions/placements/layouts | 608 / 32 |

Numeric regimes include FP16/BF16 with native or FP32 accumulation, FP32/FP64,
word32/word64 NTT, UINT32/UINT64 zeta probes, real FFT in four storage precisions,
and a FP32-storage/FP64-compute FFT capability probe. NTT primes are 998244353
and 576460756061519873. Structured matrices are [[1,0.25],[-0.5,1]], repeated
per axis and saved in provenance. These are finite domain examples.

Order patterns are natural, every-axis reversed, alternating starting natural,
and alternating starting reversed. They cover all vectors at rank one/two;
higher ranks have additional vectors outside this declared set. Canonicalization
retains every original request. Of 94,800 variants, 19,360 are invalid contracts,
such as inverse R2C or reversing a non-power-of-two half-spectrum axis. Valid
variants resolve to 48,784 distinct semantics, including 7,680 static
missing-lowering requests. Neither invalid nor gap rows count as executed.

## Gates and result meaning

The runner verifies library/config/source identities, semantic hash, actual plan
descriptor and exact mapping replay. Every output element and batch, padding,
and out-of-place input preservation are checked. Oracle work limits are explicit
protocol fields; exceeding one means validation-unavailable, never sampled
validation or a timing-only pass. Large-request validation and resource
feasibility remain obligations even when a request appears in the manifest.

Out-of-place timing uses 20 complete calls per CUDA graph and seven replays.
In-place timing uses seven individual event measurements with input restored
before the start event. Implementations at one semantic point use the same
protocol. Public MEASURE may tune a child without its fixed conversion boundary;
reported timings include that boundary. This does not prove exhaustive search
or optimal fusion.

cuFFT and VkFFT C2C adapters bind only matching supported contracts. Dao-FHT
binds standard FWHT with FP32 computation and FP16/BF16/FP32 storage, after
checking the installed implementation's source contract. Other
baseline adapter gaps remain explicit, including real/low-precision cases the
bridge cannot express. Missing rows do not enter speedup denominators. Dao-FHT
integration includes its mandatory output copy and any tiny-length padding
inside timed calls. Historical GPU-NTT CLI timings do not participate because
they used a different timing and conversion contract.
The new prepared GPU-NTT bridge now binds rank-one standard cyclic UINT32/UINT64
contracts using the same primitive root, with forward/inverse order conversion
inside the measured call and setup outside it. CPU checks and the initial GPU
pilot's 18 all-element integer comparisons passed. xD NTT is not flattened into rank one.
Libraries outside an operator's domain are `not-applicable`, not missing rows.

Floating results preserve actual/reference arrays and all absolute/relative
errors in lossless NPZ files. JSON records exact nearest-rank quantiles, fixed
log-bin counts, zero-reference errors and L2-relative errors. Checkpoints and
resume validate their SHA256. CPU floating reference arithmetic is FP64 or
complex128, including forward embedding. Thus FP64 comparisons are independent
same-precision implementation checks, not errors against arbitrary-precision
truth. Integer NTT arithmetic uses exact Python integers. Reference arrays keep
their dtypes in the artifacts, and no kernel precision is changed.

Each row has an atomic checkpoint and hash. Correctness or runner failures stop
the tranche. Foreign GPU work prevents accepting the current row without killing
foreign processes. Resume requires matching manifest, library, environment,
protocol, sources and calibration.

## Initial execution and continuation

The first search/timing pilot exposed a harness error when binding workspace to
a zero-workspace plan; the failed row remains in `pilot_1`. The runner now binds
workspace only when requested. No library or arithmetic kernel changed.

The subsequent DEFAULT-policy conformance pilot completed 256 descriptors:
36 exact integer passes, 183 floating error reports awaiting acceptance, and
37 explicit missing lowerings. Applicable baseline rows contain 18 exact passes,
70 floating error reports and 129 missing bindings, with no execution errors.
The initial subset is small rank-one data, not the full performance population.
All 464 rank-boundary requests have also been visited; their gaps and pending
floating acceptance remain distinct from execution passes.

`results/semantic_campaign_20261007_v092/run_remaining.sh` resumes the frozen
conformance populations first, then runs public MEASURE comparisons over the
same manifest. It stops on correctness/runner failures or foreign GPU work,
preserves checkpointed cases and never renews an expired monitoring budget.
The explicit full-reference bounds remain one million elements and two million
quadratic-NTT operations; larger cases are validation gaps, not reduced workloads.
Floating comparison timings remain diagnostic pending threshold selection.

## Recovery on October 8

At 09:19 +08 the queue resumed after its previous process disappeared. Its last
saved row was at 00:51:07; no recorded exit reason establishes why it stopped.
Before restart, all 14,971 saved records, referenced error artifacts, semantic
IDs and build/runtime/calibration/baseline identities were verified. Rank boundary
and arbitrary/embedding populations were already visited. Semantic crossing
resumed from 13,899 of 28,704 requests and produced new checkpoints without
repeating completed cases. The runner, manifests and numerical/timing protocols
were kept frozen.

The outer `scripts/supervise_semantic_queue.py` process runs in an independent
session and holds a per-campaign lock. Its status includes a 15-second heartbeat,
queue PID, progress snapshots and exit code/signal. It does not retry failures
or wait for a busy GPU. Current supervision records are under
`results/semantic_campaign_20261007_v092/recovery_20261008_0920/`.
Use that heartbeat and live PIDs to establish liveness: the frozen runner can
retain its previous `tranche-complete` text while advancing a resumed tranche.
Three CPU process tests cover nonzero exits, signal termination and duplicate
supervisor rejection. This recovery does not qualify floating accuracy or fill
the documented calibration and large-reference gaps.

## Reporting repair and failure isolation on October 8

The morning queue stopped at 13:58 +08 on an unnormalized FP16 FFT with
N=1,048,576 and B=1. At that point 34,961 of 48,784 requests had records:
3,712 exact integer passes, 24,288 floating reports pending acceptance,
5,632 missing lowerings, 1,328 validation-unavailable rows and one correctness
failure. Semantic crossing, rank boundary and arbitrary/embedding were visited;
size/batch scaling had visited 5,185 of 19,008. These are coverage counts,
not an aggregate performance or accuracy result.

The [numeric repair](numeric_reporting_repair_20261008.md) fixes complex decoding
in the harness and adds component-wise range diagnostics. The original failure
is retained, as is its unchanged-input GPU reproduction with the fixed decoder.
The library, arithmetic, data generator and acceptance policy remain unchanged.

The user authorized isolated continuation of other requests. The new entry is
`results/semantic_campaign_20261008_v0921/run_unaffected.sh`, supervised under
`results/semantic_campaign_20261008_v0921/supervision_20261008/`. It first checks
all historical row envelopes and artifacts, then selects only unvisited
requests. Each fresh shard freezes the new runner identity and cites its parent
manifest and excluded failure evidence. A correctness failure ends its shard;
the controller retains that exact failed case and starts the remaining cases
in another shard. Runtime errors, foreign GPU work and unknown exits still stop.

Historical results keep their original runner identity. Completed conformance
cases are not rerun by this continuation. The later timing phase measures
nonquarantined parent cases under the same declared complete-call protocol;
it is a separate experiment, not reuse of old timings. Pending floating
acceptance, missing lowerings, large-reference limits and calibration gaps
remain visible. A completed controller can therefore say only
`complete-with-unresolved-evidence`, never that the full population passed.

The October 8 evening [controller repair](continuation_repair_20261008.md)
replaces the slow repeated-history loop while keeping measurement sources and
historical artifacts frozen. The active queue now uses
`results/semantic_campaign_20261008_v0921/run_repaired.sh`: original-pressure
conformance, separate FFT/FWHT scaled controls, then comparison of requests
outside the original quarantine. The failed-pressure rows remain excluded even
if a different input profile yields finite output. `results/active_experiment_queue.json`
points to the current supervisor and phase artifacts.
