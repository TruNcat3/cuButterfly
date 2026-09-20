# Cross-operator research acceptance

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

For a separate machine, use the [portable V100 workflow](hardware/v100.md).
`config/research_campaign` freezes the existing 704 contracts plus 18 operator
and 16 general-shape FFT additions independently of this host's result paths.
`scripts/run_research_target.py` prepares finite CPU specializations, calibrates
the target before invoking this acceptance engine, and preserves explicit
memory deferrals and unavailable baselines. It does not modify the frozen A100 queues.

The cryptographic application extension (local artifact: `../results/crypto_application_20260920/README.md`; not included in this source release)
adds 208 new cyclic and 138 composed contracts per target. It distinguishes
modulus bit length from storage width and RNS channel count from batch, with
named 23/31-bit fields, controlled 30/40/50/60/62-bit primes, non-power-of-two
batches, negacyclic/coset semantics and normalized inverses. The isolated
composition entry includes GPU twists and the complete RNS invocation in
timing. Its external comparisons are unavailable; replaying searched cyclic
cores is an internal application-composition experiment. See the
[support ledger](crypto_application_coverage.md) for implementation gaps that
cannot be filled by calibration or more sampling.

The September 20 batch/precision extension (local artifact: `../results/batch_precision_20260920/README.md`; not included in this source release)
adds a separate, frozen population to the original comparison: 284 new cells
per target, with exact-semantic references to the original 74. It includes
doubling batch sweeps through a 2 GiB input cap, matched FP16/BF16 native-rounding
and FP32-accumulation anchors, FP32/FP64, NTT word32/word64 and same-modulus
controls, and inverse anchors. Unsupported baseline contracts remain explicit.
The CPU preparation is complete for its 40-module seed portfolio. Its queue
waits for the original target's controller to release the GPU, preserving the
original matrix and wait window. Compilation is not GPU correctness evidence;
both the original comparisons and the extension must finish before a complete
scaling or precision claim is made.

`scripts/run_research_acceptance.py` runs a finite workload matrix through the
existing search, physical-stage calibration, registry and benchmark interfaces.
It accepts GPU-specific calibration files as inputs; neither GPU model nor a
winning kernel is hardcoded in the workload matrix. Run it from the source
checkout using the shared Python environment. This entry does not replace
`calibrate_hardware.py` or certify full hardware migration.

The default research matrix is
[`config/research_comprehensive_workloads.json`](../config/research_comprehensive_workloads.json):
72 semantic cells spanning FFT (29), NTT (14), FWHT (12), subset/superset zeta
(5 each), structured 2x2 (6), and the xor-zeta interface (1). FFT reaches
`N=2^22`; the largest invocation contains `2^24` logical elements. Small and
large batches, fp32/fp64, 32/64-bit NTT, inverse transforms, output orders,
placement and one strided FFT are represented. Low precision, arbitrary lengths
and multidimensional transforms remain outside this finite acceptance matrix.

## Workflow and evidence

`--continue-unavailable` is an opt-in for completing a finite matrix after
explicit capability rejections. It records a cell as `unavailable` only when
all attempted records have no samples and report recognized capability limits;
runtime failures and incorrect samples still stop the run. Unavailable cells
keep their original errors and do not count as measured. A visited matrix with
such cells has status `complete-with-unavailable-cells`, and its summary sets
`all_workloads_measured=false`. This option does not add a missing lowering or
establish theoretical infeasibility. Without it, the existing fail-fast
behavior is retained.

1. Verify target model, memory, UUID, binary identity, compile policy and
   timing protocol. Import compatible component curves and scheduling probes.
2. For each cell, remeasure its current public incumbent and use the existing
   evolutionary search. The default budget is 16 screened candidates plus up
   to 4 historical mapping seeds; historical timing is not imported. Confirm
   3 finalists with three trials of 100 warmups and 100 timed repetitions.
3. Resolve the confirmed plans' actual physical groups and full/remainder
   tiles. Request missing independent services from the cumulative stage
   calibrator. Unsupported adapters remain explicit gaps. Report model ranking
   before and after this extension without fitting complete-plan timings.
4. Promote the measured winner into an isolated run-local registry and replay
   it through `--auto-select`, checking correctness and exact mapping identity.
5. Interleave public-selector and baseline trials, reversing order every other
   trial. Record unsupported baseline contracts. Compilation, initialization
   and checking are outside kernel timing; GPU exclusivity is checked before
   and after each accepted measurement.

`acceptance.json` is the resumable journal; `summary.json` contains per-cell
latencies, baseline speedups and operator/precision summaries. A speedup is
`baseline_ms / cuButterfly_ms`, so values above 1 favor cuButterfly. Ranking
reports top-1, top-2, pairwise agreement and selected-latency regret over the
confirmed population, together with component coverage. It does not establish
the best point among candidates omitted by the search budget.

The journal preserves completed trials. Resume rejects changed target/build/
protocol or changed baseline binaries, dependencies and Dao package identity.
The proposal profile is frozen within a cell; newly sampled services affect
its finalist audit and subsequent cells. The ordinary user registry is not
overwritten by this experiment. An occupied GPU interrupts and checkpoints the
run; there is no implicit background wait or automatic retry loop.

For a compatible partial search, the saved selection prefix is restored
directly. The next candidate retains the original stratified exploration order
and overall budget; earlier evolutionary rankings and accepted measurements
are not repeated. A protocol change or incomplete prefix evidence falls back
to the existing traversal while retaining compatible raw timing rows.

Resume invocations also record Python source revisions. These are analysis
provenance, separate from the raw measurement compatibility rules: a proven
equivalent CPU lookup optimization does not invalidate kernel timings. A
KeyboardInterrupt explicitly checkpoints the run as interrupted.

On Linux, `CUBUTTERFLY_SEARCH_SCORE_WORKERS=8` optionally parallelizes the
CPU scoring of a frozen candidate inventory. The default is one worker. The
parallel path requires a pure CPU scorer, a single active parent thread, and
no parent CUDA context; it falls back to serial for inventories below 128
points or when `fork` is unavailable. Each round gets a fresh process pool,
so model updates cannot leave stale worker state. Ordered scoring and stable
tie breaking preserve candidate selection and budgets. Compilation and CPU
scoring remain outside kernel time. This helps long scoring phases; small
inventories can cost more to parallelize. Independent GPUs can run separate
acceptance processes, each pinned to its UUID and its own calibrated profile
and output directory.

## Baseline contracts

| Baseline | Current adapter scope | Correctness evidence |
| --- | --- | --- |
| cuFFT | fp32/fp64 FFT with matching direction, normalization, placement and strides | Public benchmark CPU reference; all batches by default |
| VkFFT | fp32 forward contiguous FFT | cuFFT reference, all batches |
| Dao FWHT | Native fp16/bf16/fp32, logN 3–15, contiguous out-of-place | Existing adapter checks a normalized round trip for up to two batches |
| GPU-NTT | Forward cyclic word64, fixed prime `576460756061519873`, natural input | Existing harness CPU reference for one batch; exact modulus and output-order marker checked |
| In-tree radix2 | Non-NTT operator ablation; fixed shared scalar radix2 realization | Public benchmark CPU reference; all batches by default |

GPU-NTT natural-order timing includes its output permutation. Dao inverse FWHT
uses the same Hadamard primitive with matching scaling; its original CSV is
retained. Dao timing surrounds PyTorch calls, including output allocation.
These verification and API differences are recorded rather than presented as
identical harnesses. The in-tree comparison is an ablation, not an external
library result. The present matrix has 29 cuFFT, 18 VkFFT, 8 Dao and 7 GPU-NTT
matching contracts when all listed dependencies are available; these are
planned comparisons, not completed measurements.

## Prepare research compilation before the GPU window

`scripts/precompile_research.py` separates candidate export from module
compilation. Export uses the linked library's finite candidate inventory and
optional historical mapping seeds. It requires a brief query of the target GPU
to obtain the same resource-dependent candidates as runtime search. Compilation
then runs entirely on the CPU with CUDA devices hidden, using the existing
`compile_module.py` cache and lock protocol; it does not rebuild the main library.

```bash
CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
CUBUTTERFLY_COMPILE_MODE=research \
/home/wt/yes/envs/cubutterfly/bin/python scripts/precompile_research.py \
  --mode export --build-dir build-a100-cufftdx \
  --workloads config/research_comprehensive_workloads.json \
  --profile results/staged_migration_20260915/a100-80gb-v3-stage/calibration_device.json \
  --mapping-seeds /home/wt/.cache/cubutterfly/registry.json \
  --output-dir results/comprehensive_20260916/a100-80gb-precompile

/home/wt/yes/envs/cubutterfly/bin/python scripts/precompile_research.py \
  --mode compile --jobs 2 \
  --output-dir results/comprehensive_20260916/a100-80gb-precompile
```

The second command needs no available GPU, benchmark binary or live hardware
query. Run it while the GPU is occupied, and repeat it to resume after an
interruption: the compiler revalidates its normal source/compiler/SM cache
identity and reuses completed modules. `modules.json` records the exported
population; `compilation.json` records each completed or failed module. Failed
or unprojectable entries remain explicit and prevent a complete status.
`--jobs` bounds compiler concurrency. `--mode all` performs both phases in order.

Requests deduplicate batch sizes, directions, strides and other runtime-only
parameters. They retain precision, physical stage groups, processing-core
configuration and compile-time layout choices. This covers shared templates
across operators, factor-streamed FFT and register-prefix FFT. Other linked
cores require no standalone module. Research search still uses
`CUBUTTERFLY_COMPILE_MODE=research` to load these cached specializations; newly
generated evolutionary neighbors may require additional compilation. Thus a
completed finite manifest prepares its exported population, rather than every
possible future search point. Resource legality and correctness are checked
when the prepared module is used on the target GPU.

For a current-build refresh with a known mapping portfolio, add
`--candidate-source seeds` to export. This prepares only the frozen mapping
seed snapshot and avoids enumerating the full runtime inventory. At least one
explicit `--mapping-seeds` file is required; a workload with no matching seed
is recorded as incomplete. The default `--candidate-source runtime` preserves
the original runtime-inventory-plus-seeds export. Both modes report their
coverage explicitly and neither certifies the theoretical design space.
Module preparation imports mappings, never historical performance timings.
Evolutionary neighbors outside this finite portfolio may still compile when
searched; their compilation remains outside kernel timing.

The September 19 refresh (local artifact: `../results/comprehensive_20260919/README.md`; not included in this source release) uses
this bounded preparation for the public whole/chunked FFT strategies and
previously verified cross-operator mappings. It reports the original 72-cell
matrix separately from two N23 FFT additions, with independent 40GB and 80GB
GPU cohorts and current-build stage measurements.

Its September 20 supplement adds an explicit, optional continuation controller
around the existing entry point. It observes the same GPU UUID, uses a fixed
waiting deadline, and retries only recorded GPU-exclusivity interruptions.
Validation failures remain terminal. See the refresh README for its state,
logs and launch command; the base acceptance entry's fail-fast behavior is
unchanged.

## A100 80GB continuation

On 2026-09-17 a separate GPU 1 calibration (local artifact: `../results/staged_migration_20260917/a100-80gb-gpu1/report.md`; not included in this source release)
completed its requested stage basis and scheduling probes; its
72-cell acceptance (local artifact: `../results/comprehensive_20260917/a100-80gb-gpu1/report.md`; not included in this source release)
is running while the original GPU 2 remains occupied. The two physical cards'
raw timing journals are kept separate; compatible SM80 modules can be shared.

The original GPU 2 experiment is interrupted by GPU occupancy; the `acceptance.json`/`summary.json`
snapshot records **5/72 complete cells** (four FFT and one FWHT). The sixth,
FFT N=256/batch=16384, retains 8/19 screening attempts. Resume that journal's
GPU measurements when its target is exclusive.
The full matrix and full migration
are not complete; see the run report (local artifact: `../results/comprehensive_20260916/a100-80gb/report.md`; not included in this source release)
for per-case times and confirmed-candidate ranking. All 72 workloads also have
an exported module preparation manifest (local artifact: `../results/comprehensive_20260916/a100-80gb-precompile/report.md`; not included in this source release);
the CPU queue finished with 33,572 successful modules and 118 unsupported
FP64 register-tile requests. No compilation queue is currently running.

The probe-point type-boundary repair converted CSV integer fields to exact JSON
numbers without routing the word64 modulus through floating point. Reanalysis
added one service request, extended the cumulative curve inventory from 78 to
79, and left `service_gaps=[]`; ranking moved from top-1 false, pairwise `1/3`
and regret `1.2115735x` to top-1 true, pairwise `3/3` and regret `1.0x`. The
pre-repair ranking and gap lists remain in the journal's
`*_before_probe_type_fix` fields, and all prior timing rows are retained.

CPU verification in the shared environment passes all 9
`tests/test_research_acceptance.py` tests and 21 stage-service tests.
`resource_projection_parity.json` (local artifact: `../results/comprehensive_20260916/a100-80gb/resource_projection_parity.json`; not included in this source release)
records field-for-field parity for 101 real candidate predictions while
reducing the CPU diagnostic from `0.1814671 s` to `0.1077384 s` (`1.6843x`);
this is CPU model-evaluation evidence, not a GPU kernel speedup. The process
continues with the second cell under the command below; the full matrix and
full migration remain unfinished.
From the repository root, use the following exact continuation command. For
another GPU, change the target, calibration inputs and output directory together.

```bash
CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
CUBUTTERFLY_COMPILE_MODE=research \
/home/wt/yes/envs/cubutterfly/bin/python scripts/run_research_acceptance.py \
  --build-dir build-a100-cufftdx \
  --output-dir results/comprehensive_20260916/a100-80gb \
  --workloads config/research_comprehensive_workloads.json \
  --profile results/staged_migration_20260915/a100-80gb-v3-stage/calibration_device.json \
  --stage-checkpoint results/staged_migration_20260916/a100-80gb-online-reuse/stage_calibration.json \
  --pipeline-calibration results/staged_migration_20260916/a100-80gb-composition/pipeline_schedule_calibration.json \
  --schedule-calibration results/staged_migration_20260915/a100-80gb-v3-stage/schedule_calibration.json \
  --mapping-seeds /home/wt/.cache/cubutterfly/registry.json \
  --search-budget 16 --finalists 3 \
  --vkfft-binary build-a100-vkfft/vkfft_bench \
  --fht-python /home/wt/yes/envs/cubutterfly/bin/python \
  --gpuntt-binary /tmp/gpuntt_merge_gap_bench_a100 \
  --mode all --resume
```

For a fresh output directory omit `--resume`. `--mode prepare` imports compatible
calibration and creates the journal; `--mode search` stops after search,
component extension and selector replay for every cell. `--mode all --resume`
continues into baselines. Keep external paths stable once comparison identity
has been recorded, and retain the stage checkpoint's companion files.
