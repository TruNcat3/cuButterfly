# Independent installation comparison, 2026-10-13

Current result, October 17: the repaired campaign completed all 66 requests.
See the [numerical and coverage report](install_comparison_results_20261017.md)
and [NTT search repair and FFT baseline supplement](ntt_search_repair_20261017.md).
The execution narrative below retains the earlier campaign history.

The conda activation hooks select the installed shared library and preserve an
explicit user override. Once `finalized/registry.json` is published, subsequent
activation also selects that verified registry, preserving an explicit registry
override. Before publication the hook does not point callers at partial records.

This first finite comparison targets cuButterfly 0.9.3.11 and the
`build-contract-20261013` build. Installation and CPU preparation are complete;
the repaired initial calibration is running under revision
`0.9.3.11-calibration.3` in a separate recovery directory. See the
[repair and recovery record](calibration_repair_20261014.md). This document records the requested
population, not completed GPU comparison results or full operator/design-space coverage.

## Frozen population

The [campaign configuration](../results/install_compare_20261013/campaign_config.json)
contains 66 requests. Its first 36 populations equal the earlier
[fixed E1 configuration](../results/paper_targeted_20261011/operator_prepare/e1_fixed_36_v4/config.json)
without changing their names, semantics, precision, shape, batch or direction.
Only that configuration is reused; historical manifests, compiled choices,
calibration identities and timing results are not adopted.

| Operator | Existing anchors | Added requests | Total |
| --- | ---: | ---: | ---: |
| FFT C2C | 12 | 5 | 17 |
| FWHT | 8 | 5 | 13 |
| NTT | 8 | 5 | 13 |
| Subset zeta | 3 | 5 | 8 |
| Superset zeta | 3 | 5 | 8 |
| Structured 2x2 | 2 | 5 | 7 |
| Total | 36 | 30 | 66 |

Every operator receives the following five additions, with no Cartesian
expansion. All additions use natural input/output order, packed layout and
out-of-place execution. Inverse requests use inverse normalization. These
choices do not remove any existing anchor's semantic controls.

| Rank | Shape | Batch | Direction |
| ---: | --- | ---: | --- |
| 1 | 65536 | 4 | Forward |
| 2 | 128 x 128 | 4 | Forward |
| 3 | 32 x 32 x 32 | 4 | Inverse |
| 4 | 8 x 8 x 8 x 8 | 4 | Forward |
| 8 | 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 | 16 | Inverse |

Added FFT and FWHT requests use FP32. Added NTT requests use word64 and prime
`576460756061519873`. Added zeta requests use uint64. Added structured requests
use FP64 and the existing per-axis `repeat-default` matrix
`[[1,0.25],[-0.5,1]]`, unchanged across stages. Arithmetic choices are explicit
in the configuration; no precision substitutions are allowed.

The combined population has 42 rank1 cases and six each at ranks 2, 3, 4 and 8;
40 forward and 26 inverse requests. Storage counts are complex FP32 11,
complex FP64 6, real FP32 13, real FP64 7, uint32 10 and uint64 19. NTT accounts
for four word32 and nine word64 requests within the integer storage counts.

## Oracle and measurement contract

The default all-element CPU oracle limit is `1048576`; the largest requested
case equals this bound. The exact NTT budget is `16777216` radix2 butterflies;
the largest requested NTT case uses `2097152`. All shapes are powers of two.
No requested batch, shape or precision is reduced to fit an oracle or baseline.
The CPU oracle is a correctness reference, not a timed external CPU baseline.

Use the existing [unified semantic runner](../scripts/run_semantic_campaign.py)
with `--float-verdict error-report`. Floating outputs must be finite; complete
absolute/relative error statistics and lossless first-validation artifacts are
retained. Floating rows remain `accuracy-review-required` until an explicit
accuracy assessment; historical fixed tolerances do not automatically establish
a new acceptance verdict. Integer outputs require exact full-output equality.
Input preservation and output padding checks remain enabled.

Timed cases use public MEASURE selection. Compilation, initialization, plan
construction, allocation, reference computation and warmup are excluded from
steady-state timing. Required normalization, permutations, gather/scatter and
other semantic adapter operations are included in complete execution. Public
mapping, compile policy, build/library hashes and GPU UUID identify each row.
The existing protocol uses ten warmup calls and seven samples of twenty graph
calls for out-of-place requests; these samples are not independent process trials.

## Baseline coverage

| Baseline | Applicable requested domain | Gaps retained |
| --- | --- | --- |
| Native cuFFT | C2C FP32/FP64, rank1-3 | Rank4/8 adapter absent |
| VkFFT bridge | C2C FP32/FP64, rank1-3 | Rank4/8 adapter absent |
| GPU-NTT bridge | Supported rank1 word32/64 prime/root contracts | Multidimensional bridge absent |
| Dao FWHT | FP32 arithmetic, flattened points <=32768 | FP64 and N65536 absent |
| External GPU zeta/structured | No matching GPU adapter | GPU baseline gap remains; external CPU rows are now available |

The [CPU comparison supplement](cpu_external_baselines.md) is complete for all
66 requests: 83 timing records from FFTW (one/eight threads), FXT, Intel HEXL
and the HazyResearch butterfly author CPU implementation. Integer rows are
exact; floating rows retain error distributions without an acceptance threshold.
These shared-host CPU timings are provisional and occupy separate columns in
the unified table. They do not close missing **GPU** baseline coverage or enter
GPU/GPU speedup aggregates. Required adapters and their costs are explicit,
including HEXL's exact word32-to-word64 storage adaptation.

Use a baseline only when its exact requested semantic and runtime contracts
pass. The existing runner binds native cuFFT directly; a CuPy cuFFT measurement
is a separate frontend comparison and must identify its allocation/dispatch
contract rather than silently replace native cuFFT rows. Baseline availability
must be rechecked against the installed artifacts. No in-tree control or CPU
oracle is labeled an external vendor baseline.

## Preparation and execution binding

CPU schema validation and request counting passed: 66 distinct requests, the
36 original populations preserved, and both oracle bounds satisfied. The fresh
manifest and CPU preparation queue are complete: 66/66 ready, zero known gaps,
failures or pending requests. This is CPU preparation coverage, not GPU
correctness or timing coverage.

The shared C++ library, tools and Python package `0.9.3.11` are installed into
`/home/wt/yes/envs/cubutterfly`. The package consumer configure/build/run passed;
installed calibration tools imported in isolation, and all six shared resident
templates rendered with their installed header dependencies. The installed
benchmark starts without `LD_LIBRARY_PATH`. The conda activation hook selects
the installed library for ordinary Python calls and was tested. The previous
static `libcuntt.a` is preserved at
`results/install_compare_20261013/previous_libcuntt.a`; the other prior installed
artifacts are backed up in `previous_conda_install.tar.gz` in that directory.

The installed GPU smoke returned `correct=1` with explicit
`shared-iterative` and stage partition `6,6`. The first attempt omitted the
backend and reached the benchmark's default `warp-hybrid`, which rejects
N=4096; that was a benchmark configuration error, not a selector failure.
This smoke checks execution only; its latency is not performance evidence.

Installation rewrites `INSTALL_RPATH`, so the build and installed library files
have different SHA256 values. They are recorded separately:

| Artifact | SHA256 |
| --- | --- |
| `build-contract-20261013/libcuntt.so` | `6f590d064e9167f7ff9285ca4b91a9d0788dd73a6d018b7e0fd0948346d7e2d1` |
| `/home/wt/yes/envs/cubutterfly/lib/libcuntt.so` | `a461c7d1bfa3e282b346b99cb990d842eb27dff91b0b470107ddc513f5c05a51` |

The formal calibration and comparison explicitly use the **build library** and
its SHA above; the installed library is not claimed to be byte-identical or
silently substituted. A later comparison of the installed bytes would require
its own compatible calibration binding. Installation smoke checks and the
current finite performance campaign are separate evidence.

The [recovery driver](../results/install_compare_repair_20261014/run.py), v2 PID
`1690709`, stopped at 2026-10-14 16:51 CST after foreign GPU activity on A100 80GB
PCIe UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`. Its checkpoint has been
reviewed: 439 correct exclusive stage records, 22/24 templates described, and the
earlier contaminated load successfully remeasured. After checking the target was
exclusive and all frozen inputs still matched, driver PID `1749222` resumed with
`--resume`, accompanied by join follower PID `1749223`. State is now `running`,
phase `calibrate`; final confirmation and the 66-request comparison remain pending.
The [original campaign](../results/install_compare_20261013) is preserved;
compatible whole-plan timings are reused. The v1 stage-probe replacement required
fresh stage measurements; v2 reuses 412 compatible clean measurements and
retains one contaminated record for remeasurement. CPU comparison retains 83/83 records; completed GPU
comparison requests remain 0/66. The [repair record](calibration_repair_20261014.md)
describes the CPU scoring improvement, validation and strict scope gates.
The v1 driver PID `1566575` ran from 14:17 to 14:41 CST, described 21/24
templates, and stopped after NCU PID `1588829` triggered post-exclusivity and
the subsequent search rejected a changed seed-source path. Its status, logs,
stage checkpoints and tools remain in
[`attempts/calibration1_failed`](../results/install_compare_repair_20261014/attempts/calibration1_failed).
Revision `0.9.3.11-calibration.2` fixes the source with `--seed-registry
results/install_compare_20261013/initial/registry.json` and checks the frozen
seed snapshot before GPU work. Interrupted stage calibration now saves its
checkpoint and exits 130 before search, with driver status
`interrupted-calibration`. Input and source checks remain strict. The repaired
probe and GPU library are unchanged; 321 candidate records passed compatibility
checks. The new CPU join follower is PID `1691634`.

V2 integration passed 116/117 tests, including the new preflight and interruption
regressions. The one existing bounded 24-candidate stratification coverage
assertion also fails with the identical pre-repair source; it is retained as
an independent issue. Current `model` search scores the complete remaining
frontier and uses stratified ordering for periodic exploration, rather than
truncating the pool to that test's 24 points. No full-axis coverage is inferred.

Its finite calibration workload contains 23 cells, with stage-point budget 24,
calibration search budget 32 and comparison public MEASURE budget 512. Fresh
hardware-rate measurements are present. The initial installation probe
deliberately skipped operator calibration, so its migration manifest reports
incomplete migration; it establishes the basic rates only. Full finite operator
calibration, measured mappings and selector replay are still pending. No
completed comparison rows are claimed here.

The V3 recovery initially retained 457 clean stage records and 322 candidate records. Two
FP16 service curves exhausted the old refinement cap, so the per-curve default
budget is repaired and the frozen stage scope must pass before search resumes.
The supplement subsequently passed the finite stage gate with 562 clean records,
and plan search has resumed. Two FP64 candidates remain recorded as cuFFTDx
provider-parameter rejections. The same FP64 N/B request has correct alternatives
in this build, so these rejections do not establish a mathematical or
user-semantic limitation. See the
[support classification and recovery](calibration_repair_20261014.md).

The calibration gate requires the same library SHA, GPU model/memory/SM,
`research` compile mode, full-batch verification (`verify_batches=0`), fresh
hardware rates and `ready-with-measured-mappings`. The asynchronous comparison
also validates the broker's source/compiler identity and JIT CUDA-runtime
binding before formal timing. The old E1 manifest is not resumed because it
binds a different source library.

The actual launch command was:

```bash
/home/wt/yes/envs/cubutterfly/bin/python -u \
  results/install_compare_repair_20261014/run.py
```

The driver records phase commands and immutable input hashes in
[recovery status.json](../results/install_compare_repair_20261014/status.json). It finalizes the
registry before launching `run_async_semantic_campaign.py`, consumes the fresh
`prepared` spool, and uses this campaign's `compile` broker with an eight-worker
CPU producer. The comparison schedules CUDA work serially with four CPU workers
and four requests in flight. Floating results use `error-report`; all seven
timing snapshots are verified before a row is accepted. A worker recycle resumes
completed rows; detected foreign GPU activity quarantines the phase for review.
Use `run.py --resume` only after an interruption and compatible identity checks;
do not launch a second driver while this one runs.

Rank4/8 cases exercise the common xD path and retain missing external comparisons.
This population does not qualify arbitrary lengths, all strides/orders,
FP16/BF16, all ranks or exhaustive organization choices.
