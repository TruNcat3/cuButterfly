# Hardware Profile Initialization

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

cuButterfly does not treat a mapping measured on one GPU as a portable
constant. Installation can therefore initialize a local hardware profile and
build a first model table from the current device before an application begins
using tuning data.

The [unified planner guide](unified_planner.md) defines the current mapping,
compile-policy and registry contracts. This page covers their operational use.

## Recommended Installation

The installation wrapper can configure a new build from the first visible GPU
automatically. For a preconfigured tree, pass the existing build directory as
before; for a new tree, the wrapper detects `sm_XX` from `nvidia-smi`:

```bash
./scripts/install_with_hardware_profile.sh \
  --build-dir build-local \
  --prefix "$HOME/.local" \
  --search-seconds 0
```

To include the optional cuFFTDx processing unit in a new or existing build:

```bash
./scripts/install_with_hardware_profile.sh \
  --build-dir build-local \
  --prefix "$HOME/.local" \
  --enable-cufftdx --search-seconds 0
```

Select the intended GPU through `CUDA_VISIBLE_DEVICES` after checking its UUID,
model and capacity. Without `--profile-dir`, the wrapper stores profiles under
`PREFIX/share/cuButterfly/hardware/<model>-sm<SM>-<memory-bytes>B`.
An explicit profile directory must belong to that same target.

On the existing research host, reuse `/home/wt/yes/envs/cubutterfly` and
`build-a100-cufftdx`. Set `CUBUTTERFLY_COMPILE_MODE=research` for research runs
unless another policy is requested. Continuing a paused experiment uses the
calibrator's checkpoint workflow below; it does not require running the installer.

After target calibration, use the [cross-operator research acceptance entry](research_acceptance.md)
for a finite matrix of searched implementations, automatic-selector replay and
matching external-library comparisons. It extends missing finalist services
before comparison and reports coverage gaps separately from performance.

Keep source files and shared headers stable while their build is running. A long
CUDA compilation can read a header before an edit and finish afterward, leaving
an object newer than the header but compiled against its old layout. If this
happens, explicitly rebuild the affected translation units before installation;
an ordinary incremental build can miss the inconsistency. Preserve prior
benchmark identities and confirm the affected public interfaces after rebuilding.

The optional cuFFT Device API LTO study (local artifact: `../results/paper_completion_20260918/fft_leaf_lowering/LTO.md`; not included in this source release)
uses a separate compiler, CUDA runtime and frozen template root. Its isolated
binaries and finite samples do not qualify a full hardware profile or change
installation defaults. The completed study also exposes startup DVFS in short
stage probes: a fixed iteration count alone need not settle the GPU clock.
Its supplement accumulates at least 500 ms of GPU warmup and retains clock
observations; old short-stage samples remain exploratory. See the
measurement report (local artifact: `../results/paper_completion_20260918/fft_leaf_lowering/LTO_RESULTS.md`; not included in this source release)
before using those samples for ordering or fitting.
Treat a core/compiler/lowering change as a new measurement identity: validate
correctness, sample the affected stage services, and then compare whole plans
before promoting selector entries. A finite research sample is not a claim
that all configurations have been recalibrated.

The wrapper runs the following install-time loop. The examples above disable
the total screening cutoff so each configured workload receives its candidate
budget; they can take longer than a bounded installation. Fresh installs use
independent stage calibration by default (`--stage-calibration full`) and set
the full-mode search-time budget to `0` (unrestricted). The `bounded` mode uses
the default 300-second budget, while `skip` records that stage calibration was
not requested.

1. build the configured tree;
2. probe the device and run the correctness-checked operator calibration plus the selected independent stage policy;
3. fit the local cost model, update the cumulative hardware registry and verify selector replay;
4. run architecture-appropriate tests and install the library and profile helpers.

The same migration loop is available after installation through the unified
`calibrate_hardware.py` entry point. It detects the visible GPU by model,
compute capability and memory capacity, resolves the default cross-operator
workload file, and writes a single `migration_manifest.json` describing the
plan and outcome. Use dry-run to inspect the exact workload and search protocol
without launching calibration:

```bash
python3 scripts/calibrate_hardware.py \
  --build-dir build-local \
  --profile-root "$HOME/.local/share/cuButterfly/hardware" \
  --mode dry-run
```

Execute that same plan with `--mode execute` (the default). Pass
`--search-workloads FILE` or its `--workloads FILE` alias to make the semantic
matrix explicit; the selected file is recorded in the manifest. The command
delegates candidate enumeration, correctness checks, repeated timing, registry
promotion and cost-model fitting to `calibrate_local_hardware.py`, so a
partial or interrupted run retains its existing checkpoints:

```bash
python3 scripts/calibrate_hardware.py \
  --build-dir build-local \
  --profile-root "$HOME/.local/share/cuButterfly/hardware" \
  --search-workloads config/install_search_workloads.json \
  --mode execute --resume-search
```

For a migration that must visit every configured workload, use a per-workload
candidate count without the total screening time cutoff. This is still a bounded
search of library-exported candidates; it is not an exhaustive theoretical-space
search. For example:

```bash
python3 scripts/calibrate_hardware.py \
  --build-dir build-local \
  --profile-root "$HOME/.local/share/cuButterfly/hardware" \
  --search-budget 64 --search-finalists 3 \
  --search-seconds 0 --compile-seconds 600 \
  --mode execute
```

`--stage-calibration bounded` retains a 300-second stage/search budget.
Enumeration, verification and finalist confirmation take additional time; a
bounded run can leave later workloads with no newly searched candidate. The
manifest reports those omissions. Set the workload file to the actual
deployment or acceptance matrix before calibrating; the 14-cell default is not
all sizes and batches.

Resume using the recorded paths and measurement protocol, without copying a
long command or reprobeing the hardware:

```bash
python3 scripts/calibrate_hardware.py --resume-from /path/to/migration_manifest.json
```

Independent-stage checkpoints keep large composition audits in an immutable
`*.composition-<hash>.json` sidecar. Keep the sidecar beside the checkpoint when
moving a run; recovery validates its content hash and request count. Older
embedded-audit checkpoints remain readable. Increasing the requested batch
refreshes the full physical descriptor while retaining compatible timing rows.
A changed curve-model revision rebuilds keys and holdout decisions from raw
trials; it does not silently reuse an older validation pass.

Acquisition now interleaves description and sampling: it finishes one physical
configuration before compiling the next plan. Shared-stage template novelty
only controls visitation order; it never removes candidates or establishes
coverage. The full requested static-key inventory is retained in an immutable
`*.inventory-<hash>.json` sidecar. Unvisited requests remain explicit in
`pending_description_count`, including after a resume with a smaller input
list. Keep both kinds of sidecar when moving a checkpoint.

After actual resource inspection, one-transform-per-CTA shared stages can reuse
an identical, independently validated physical curve from another partition.
Reuse requires the same strict semantic/address/resource key and a validated
load interval covering the new request. `service_reuse` cites source candidates;
it does not manufacture raw timings or certify new whole-plan composition.
Bulk, two-group OnlineReorder FFT also supports physical-stage reuse for the
`cufftdx-block` and `register-tile` cores. Its contract checks the captured
single-kernel grid against the lowering's exact batch-to-CTA formula, together
with known compiler register/local/shared resources. Plan-level core identity
is retained: the two suffix implementations are distinct even when both group
descriptors say `cufftdx-block`. Overlap and composite groups do not reuse a
bulk curve as a whole-plan timing.

FFT strategy choices also participate in physical service identity. Changing
`prefix_codelet`, `prefix_shared_layout`, or `prefix_codelet_lanes` changes the prefix service key;
changing `factor_io_policies[i]` changes factor `i`'s key. An unchanged stage
can still reuse a compatible service when the other stages change. New
strategies remain uncovered until matching measurements are available. The
service-model revision rebuilds keys from compatible checkpoint records; it
does not turn an older lowering's timing into evidence for a new strategy.
An explicit sampling interval or finite strategy inventory qualifies only
that declared domain, not every possible FFT configuration.

Cooperative prefix calibration uses the actual physical shape:
`threads=R*G*C`, `EPT=R/G`, and unchanged `R*R*C` shared values for fixed
I/O columns `C`. The suffix service is unaffected by `G`. Omitted and
explicit `G=1` have the same service identity; no `G=1` timing is treated
as a calibrated `G>1` prefix. Sample these variants with the normal staged
calibration entry before ranking them in a new hardware profile.

To reuse measurements from compatible earlier diagnostics, repeat
`--stage-import-checkpoint /path/to/stage_calibration.json` on the installer or
`calibrate_hardware.py`. The standalone stage driver calls this option
`--import-checkpoint`. Binary hash, hardware, compile policy and the exact
warmup/repeat/trial protocol must agree; mismatches fail before changing the
target checkpoint. Identical records are deduplicated, distinct timing trials
are retained, and imported subsets cannot certify the target inventory.

Full mode also reserves sampling slots for boundary seeds, each nontrivial
interval's independent midpoint, and one level of midpoint refinement (at
least 33 slots). Explicit `stage_service_calibration.py --max-points N` caps
remain binding; bounded mode retains the 33-slot default. Further nonlinear
refinement can still produce an explicit incomplete status. The count limits
distinct loads for one probe configuration, whose physical groups are timed
together; it is not a claim of minimal sampling across all plans.

Select the same GPU UUID via `CUDA_VISIBLE_DEVICES`. The entry restores the
compile policy, checks the workload-file hash when recorded, and delegates
binary/device compatibility checks to the existing checkpoint engine. A changed
candidate budget can reuse compatible trials with `--resume-search`; it starts
a new search traversal rather than treating the old search as complete.
Adding `--dry-run` writes `migration_plan.json` when a migration manifest already
exists, preserving the execution checkpoint. Installed commands are available
under `PREFIX/bin/`; they can use `--build-dir PREFIX/bin` for a new calibration
with the installed benchmark executables.

`migration_manifest.json` distinguishes `planned`, `running`, `complete`,
`incomplete` and `failed`. It reports per-workload confirmed candidates,
search-budget completion, cost-model validation and automatic-selector replay
as separate checks. `complete` requires all these checks, including the model
holdout gate. A warning model does not invalidate individually verified measured
mappings, but cannot certify performance on an unmeasured workload. A failed or
interrupted extension can leave a prior model and replay artifact on disk; use
the current migration status and protocol, not file existence, to assess it.
Missing required artifacts or failed replay return a nonzero exit status. A
`full` stage run whose required stage profile is incomplete or unvalidated
returns nonzero. A `bounded` run may return zero while recording
`incomplete`/`incomplete-*`, so automation must check that manifest before
claiming complete migration. Previously confirmed incumbent mappings count
toward mapping coverage, but do not replace the new search's finalists.

The stage probe and public benchmark use the same queued direct-launch timing:
one event pair brackets the repeated launches, and input reset occurs between
trials, outside the interval. Resource capture, compilation and correctness
checks are also outside timing. Do not merge historical per-launch-synchronized
stage timings with this protocol; the probe binary identity distinguishes them.

For an explicit stage checkpoint, `validate_stage_service_profile.py` remeasures
its requested-load mappings through the public benchmark and reports stage sums,
probe whole-plan timing and public whole-plan timing separately:

```bash
python3 scripts/validate_stage_service_profile.py \
  --build-dir build-local --profile /path/to/calibration_device.json \
  --checkpoint /path/to/stage_calibration.json \
  --output /path/to/stage_plan_validation.json
```

Pass `--points /path/to/original_points.json` to validate every original request
when calibration merged several batch sizes into one curve. For an unmeasured
interior load the helper resolves the actual descriptor and checks the public
plan against interpolation, without adding a stage training observation.
When only the model changes, `--reuse-public-trials /path/to/prior_validation.json`
reuses complete public trials after checking target, binaries, commands,
correctness and timing protocol. It preserves the original measurements and
writes a new diagnostic report.

Batch-ring validation resolves two actual serial service descriptors: the full
tile and, if present, its remainder. Both retain the same semantic, address,
mapping and compiler-resource identity. The model queries each load separately;
it never scales a full tile's startup cost by the tail fraction or treats the
overlapped whole-batch descriptor as an independent stage. Supported projections
are SharedIterative and two-group OnlineReorder FFT with the block/register
cores; unsupported factor/composite projections remain explicit gaps.

Staged installation with `--stage-calibration full` or `bounded` additionally
runs `cubutterfly_pipeline_microbench` through the real `BatchPipeline` runtime.
`pipeline_schedule_calibration.json` records an independent minimum submission
and event-scheduling cost for 2/3/4 groups and 1–32 tiles. Training endpoints
are 1/2/4/8/16/32 tiles; 3/6/12/24 are independent interpolation checks. The
model uses this result as a lower bound on its resource/dependency estimate,
not as another per-kernel launch charge. It does not establish calibrated
operator concurrency; unknown group counts and out-of-range loads remain
uncovered. Compatible probe/hardware/protocol results are cached, with derived
summaries rebuilt from raw measurements when needed.

For a standalone diagnostic, add
`--pipeline-calibration /path/to/pipeline_schedule_calibration.json` to the
validation command above. This performs the same checked calibration/cache
lookup. `--resume` resumes completed compatible public rows in `--output`,
including a report interrupted between requests. Neither option promotes
diagnostic measurements to selector qualification.

The installed helper is under `PREFIX/bin`. This is a diagnostic for the
checkpoint's candidates; it does not promote mappings or replace the migration
manifest gates. The initial A100 80GB serial evidence and remaining full
inventory/overlap work are described in the
[hardware runbook](hardware/a100-pcie-80gb.md).

`calibrate_local_hardware.py` and `initialize_hardware_profile.py` remain
lower-level tools. Installation and standalone migration use the unified entry,
which now also runs selector replay. The installer does not replay it twice.

For the subsequent FFT comparison, pass the **same workload file**:

```bash
python3 scripts/run_fft_acceptance.py \
  --build-dir build-local --output-dir results/migration_fft_comparison \
  --workloads config/install_search_workloads.json
```

This selects the FFT entries and preserves their batch, placement, direction and
strides. It does not expand to uncalibrated cells or label a subset as the full
FFT matrix. Without `--workloads`, the original full-matrix preparation and
comparison mode remains available.

The default operator calibration combines legacy smoke points, the selector's
current choices for the requested workloads, and a bounded search of mappings
exported by the library. The common search entry supports the registered operator families;
the workload file selects the semantic cells to explore. External baselines
are kept separate from internal model training and registry promotion.
Calibration does not require a second library build. The standalone calibrator's
`--embed-legacy-selector` option is an explicit compatibility path.

### Historical mapping revalidation

The unified entry and installer reserve `--seed-budget 8` historical mapping
checks per workload, in addition to `--search-budget` exploration. Seeds come
from the cumulative registry selected by `CUBUTTERFLY_REGISTRY` (or its normal
default path). The calibrator takes winners within each original
hardware/build/semantic context and proposes their mapping parameters for the
same operator, precision and transform length on the target. Batch, placement,
direction and other target semantics remain the requested workload's inputs.
Every seed must resolve on the target and pass target-side correctness checks;
only finalists completing repeated target timing are eligible for promotion.
Historical latency never becomes a target measurement or target training label.

Additional portable proposals or another registry can be passed explicitly:

```bash
python3 scripts/calibrate_hardware.py \
  --build-dir build-local --profile-root "$HOME/.local/share/cuButterfly/hardware" \
  --mapping-seeds /path/to/mapping_seeds.json --seed-budget 8 \
  --search-budget 64 --search-seconds 0
```

`--mapping-seeds` is repeatable and is also accepted by the installer. A seed
file has schema `cubutterfly-mapping-seeds-v1` and a `seeds` array; each entry
contains `semantics`, a public version-1 `mapping`, and optional `provenance`.
Registry files with schema `cubutterfly-registry-v1` are also accepted. Explicit
seed files precede automatically discovered registry winners. Full mappings
are deduplicated for each target workload. There are no GPU-specific winners
or fixed FFT partitions in the calibration script. Transfer currently requires
the same transform length; it does not synthesize a new-size mapping from an
old partition.

`--seed-budget 0` disables historical revalidation. A positive seed budget is
a cap, so it does not promise to test every historical proposal. Coverage
records include available, requested, attempted, confirmed and budget-omitted
seed counts, source provenance, and each selection's reason. The original
exploration allowance applies after reserved seeds; a total search-time cutoff
can still interrupt either phase and is reported as incomplete coverage.

`mapping_seed_snapshot.json` freezes the input mappings for an execution.
Resume uses that snapshot even if subsequent registry promotion changed the
live registry. Explicit seed-file paths and hashes are checked; use a new
output directory when changing seed sources. Resuming a historical manifest
without a seed policy preserves its original disabled-seed protocol.

This provides a reproducible way to retain and revalidate known candidates;
it does not establish that the cost model ranks unseen mappings correctly.
The final fitted model and the smaller temporary model used during a workload
search are distinct. Search records identify temporary-model updates and the
candidate IDs that supplied their target-local training data.
Each new update also records its coefficients, feature normalization and
feature-definition version, so later finalist confirmation cannot erase the
model that actually chose a candidate. After eight correct screens, each new
correct measurement is incorporated before the next ranking; a rejected or
incorrect candidate does not trigger an identical refit. Every fourth ordinary
selection still explores a stratum. Search protocols record both
`cost_projection_version` and `model_update_policy_version`: changes retraverse
the requested search while preserving compatible measured-trial reuse.

Model validation identifies a design by its complete exported mapping,
canonical workload semantics, runtime fingerprint and recorded hardware
identity. Different provenance names for one design do not create independent
ranking candidates. A workload with only one distinct design is excluded from
multi-candidate holdout admission, while the original timing rows remain in the
fit. Validation artifacts record `validation_identity_version`; the older
name-based validation results remain historical evidence and can be refitted
on their existing measurements without GPU timing. This corrects accounting,
not the model's predictions or admission thresholds.

When there is no historical registry
or explicit seed file, installation continues with ordinary bounded search;
the seed mechanism alone does not establish cold-start ranking quality.

Disable operator calibration when only the capability profile is needed:

```bash
./scripts/install_with_hardware_profile.sh \
  --build-dir build-local \
  --prefix "$HOME/.local" \
  --skip-operator-calibration
```

The default probe uses five trials. It measures global feedback, modular
butterfly service, shared-memory exchange, CTA barrier service, and `Us=1/2/4/8`
stage-pipeline rates. The result directory contains:

```text
hardware_capabilities_raw.csv   immutable trial records
hardware_profile.json           median capability profile
unfolding_model.csv             parameterized Us/Ud/Ts/Td model table
manifest.json                   device fingerprint and provenance
unfolding_rank.csv               local workload-facing unfolding ranking
operator_calibration.json        incumbent, screened and confirmed candidate records
operator_calibration.csv         compact candidate summary after calibration completes
operator_recommendations.json    fastest correctness-checked point per workload
cost_model.json                  target-local candidate model and holdout checks
stage_calibration.json           stage-probe checkpoint, including omissions
stage_service_profile.json       compact stage curves and validation status
pipeline_schedule_calibration.json measured BatchPipeline scheduling floor and holdouts
search_coverage.json             enumerated mappings, omissions and completion state
search_measurements.json         search trial checkpoints, including partial confirmation
calibration_manifest.json        one entry point for all local artifacts
```

The initialization step does not run Nsight Compute and does not require
administrator privileges. NCU collection remains a separate diagnostic step
because it changes clocks/replay behavior and requires profiler permissions.

## Capability Probing Only

Probe hardware capabilities without installing or searching operator mappings:

```bash
python3 scripts/initialize_hardware_profile.py \
  --microbench build-local/cuntt_hardware_microbench \
  --output-dir results/local_hardware_profile \
  --trials 5 \
  --logNs 8 10 12 14 16 18 20 \
  --spatial-budgets 32 64 128 256 \
  --word-bytes 4 8 \
  --force
```

The generated `hardware_profile.json` uses the same capability fields consumed
by `scripts/rank_unfolding.py`:

```bash
python3 scripts/rank_unfolding.py \
  --capabilities results/local_hardware_profile/hardware_profile.json \
  --logN 16 --spatial-budget 128 --word-bytes 8 \
  --stage-space 1 2 4 8 16 \
  --output results/local_hardware_profile/rank_logN16.csv
```

The initialization-generated `unfolding_model.csv` is a broad model table;
the ranker can produce a narrower workload-specific table without rerunning the
probe.

## Preventing Parameter Drift

Before reusing a local model, validate the GPU fingerprint:

```bash
~/.local/bin/check_hardware_profile.py \
  ~/.local/share/cuButterfly/hardware/local/hardware_profile.json
```

The command compares the recorded device name with `nvidia-smi`. A mismatch is
an error, not a warning. Use `--skip-device-check` only to inspect a profile on
a machine without CUDA. A profile is also tagged `calibrated-local` and carries
the CUDA-visible-device setting, host, probe executable, trial count, and
conditions in its provenance fields.

The profile is an input to model ranking, not an automatic claim that every
mapping has been measured. The install workflow now also writes
`cost_model.json`: its fit uses correctness-checked cuButterfly candidates,
keeps complete external libraries such as cuFFT out of the training set, and
records leave-one-out error plus workload-winner checks. A model with weak
holdout results is marked `calibrated-local-warning`, remains a diagnostic
artifact, and does not promote an unmeasured mapping. Runtime promotion still
requires a correctness check and repeated timing confirmation for the selected
semantic cells.

The calibration manifest reports `ready-with-measured-mappings` when records
are promoted, otherwise `ready-with-unmeasured-fallback`. C/C++ plans read the
cumulative registry directly and match the hardware, complete operator semantics
and runtime fingerprint. A registry miss uses a feasible portable candidate
labelled `unmeasured-feasible`; it is not evidence of calibrated performance.
Unsupported semantics still fail explicitly. Archived V100 mappings remain
historical target-specific evidence and are not migration defaults.

The installed `calibrate_local_hardware.py` command runs the same workflow
without reinstalling the library. It detects whether the build contains the
optional cuFFTDx processing unit and records unavailable candidates instead of
turning an optional dependency into an installation failure:

```bash
~/.local/bin/calibrate_local_hardware.py \
  --build-dir build-local \
  --profile-dir "$HOME/.local/share/cuButterfly/hardware/local"
```

Use the actual profile directory selected at installation. Without `--force`,
the calibrator reuses a matching existing capability profile. Add
`--resume-search` with the original workload, output directory and measurement
protocol to reuse compatible operator/search checkpoints. The installer requests
fresh capability probes for a new calibration; with `--resume-search` it preserves
the existing capability profile.

The model can also be regenerated directly after editing or extending the
candidate matrix:

```bash
~/.local/bin/fit_local_cost_model.py \
  --profile "$HOME/.local/share/cuButterfly/hardware/local/hardware_profile.json" \
  --operator-calibration "$HOME/.local/share/cuButterfly/hardware/local/operator_calibration.json" \
  --output "$HOME/.local/share/cuButterfly/hardware/local/cost_model.json"
```

For a wider post-install search, require correctness on every candidate and
feed the resulting CSV into the same model. This is intentionally opt-in
because it is more expensive than the bounded install smoke set:

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/sweep_cubutterfly_designs.py \
  --binary build-local/cubutterfly_bench \
  --operators fft --logNs 8 12 16 \
  --target-points 4194304 --verify --require-exclusive-gpu \
  --output results/local_verified_sweep.csv

~/.local/bin/fit_local_cost_model.py \
  --profile "$HOME/.local/share/cuButterfly/hardware/local/hardware_profile.json" \
  --operator-calibration "$HOME/.local/share/cuButterfly/hardware/local/operator_calibration.json" \
  --candidate-csv results/local_verified_sweep.csv \
  --output "$HOME/.local/share/cuButterfly/hardware/local/cost_model.json"
```

The persisted model reports holdout quality. During installation search, an
in-memory model ranks candidates after valid bootstrap samples, interleaved with
exploration; this does not certify prediction accuracy or promote an unmeasured
winner. Verified, repeatedly confirmed mappings determine registry selection.
Verified CSV inputs must also carry the
exclusive-GPU marker emitted by `--require-exclusive-gpu`; the fitter rejects
unmarked files unless `--allow-nonexclusive-csv` is explicitly requested for
exploratory analysis.

### Shared GPU Waiting and Recovery

If the user requests waiting, `sweep_cubutterfly_designs.py` supports
`--wait-for-exclusive-gpu --exclusive-timeout 1800`. These are sweep options,
not options of `calibrate_local_hardware.py`; 1800 seconds is an example timeout,
not a requirement for every task. A waiting budget is not silently renewed.
For sustained monitoring, use an available resumable task or bounded waits and
report meaningful state changes rather than repeatedly requesting a GPU window.

When the GPU is occupied, preserve progress and continue available CPU work.
Absent a continuing-monitoring request, report the checkpoint and remaining GPU
work. Do not stop other users' jobs or accept contended timings. The existing
checks can detect contention but do not reserve the device against arriving jobs.

Sweeps checkpoint completed candidates to `OUTPUT.partial` and use `--resume`.
Calibration uses `--resume-search`, preserving individual confirmation trials
and skipping completed cells under an identical search protocol. FFT acceptance
uses `run_fft_acceptance.py --resume`. Each workflow checks its recorded identity;
do not combine evidence from incompatible builds, hardware or timing protocols.

## Optional Complete FFT Baselines

VkFFT is an external complete FFT library, not a cuButterfly processing unit.
It is therefore built and benchmarked separately from local selector
generation:

```bash
./scripts/install_vkfft.sh
cmake -S . -B build-vkfft \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUBUTTERFLY_ENABLE_VKFFT=ON
cmake --build build-vkfft --target vkfft_bench -j2

CUDA_VISIBLE_DEVICES=1 ./build-vkfft/vkfft_bench \
  --library vkfft --logN 12 --total-points 4194304 \
  --warmup 20 --repeat 100 --verify --csv
```

Compare it with the same command using `--library cufft`. Run this only when
the selected GPU is not concurrently occupied; these rows are external
baseline evidence and are never used to train the cuButterfly local cost
model.

For the repeated complete-library protocol, match the calibrated cuButterfly
placement and require exclusive access throughout the run:

```bash
CUDA_VISIBLE_DEVICES=1 \
BIN=build-vkfft/vkfft_bench \
PLACEMENT=out-of-place \
REQUIRE_EXCLUSIVE_GPU=1 \
OUTPUT=results/fft_libraries_local_raw.csv \
./scripts/benchmark_fft_libraries.sh
```

For the migrated A100 profile, the checked-in comprehensive manifest extends
the same-point suite through `logN=18` and `logN=20`, and includes FP64,
inverse, strided, FWHT, zeta, and NTT groups:

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_comprehensive_suite.py \
  --manifest config/a100_comprehensive_suite.json \
  --mode full --require-exclusive-gpu \
  --output results/a100_comprehensive_full_raw.csv

python3 scripts/summarize_comprehensive_suite.py \
  results/a100_comprehensive_full_raw.csv \
  --manifest config/a100_comprehensive_suite.json \
  --output results/a100_comprehensive_full_summary.csv \
  --markdown results/a100_comprehensive_full_report.md \
  --require-stable
```

That manifest is a selected historical workload set. The expanded FP32/FP64
FFT acceptance matrix and its calibration-first procedure are documented in
the [unified planner guide](unified_planner.md#method-alignment-and-acceptance).
Full acceptance is a separate milestone from focused regression tests.

## Updating A Profile

Reassess capability and operator measurements after a driver/toolkit update,
GPU replacement, clock-policy change, or major processing-unit change. To
refresh the profile explicitly without reinstalling:

```bash
python3 scripts/calibrate_local_hardware.py \
  --build-dir build-local \
  --profile-dir results/new-hardware-profile \
  --trials 7 --force
```

Keep old profiles in versioned directories when comparing model drift. Do not
overwrite a checked-in V100 profile with data from another device; local
calibration belongs in a user-specific profile directory or a separately named
results directory.

## Stage costs and continued tuning

Hardware-specific commands, evidence and open coverage are indexed in
[hardware notes](hardware/README.md). Use a separate note and output directory
for each GPU model and memory capacity; copy the [template](hardware/TEMPLATE.md)
for another target. GPU indices are local selectors, not hardware identifiers.

Fresh installations default to `--cost-model staged`. This consumes the measured
hardware bandwidth/compute/synchronization profile in candidate ranking, models
each physical group, and composes the current runtime's serial, batch-ring or
factor-slice dependencies. `cubutterfly_plan_probe` resolves shortlisted mappings
through the real plan constructors before measurement. It does not launch the
operator; plan construction and any JIT compilation remain outside kernel timing.
The descriptor cache records the probe, benchmark, compile policy and profile
identity. An initial inventory projection remains an estimate, explicitly marked
`projected`; the selected point has a resolved plan or a recorded failure.

Staged calibration also runs `cubutterfly_schedule_microbench` once per compatible
target/probe/protocol identity. It records three trials across available thread
shapes and GPU-scaled CTA grids in `schedule_calibration.json`. The model consumes
the measured minimal-kernel launch latency and fitted incremental CTA dispatch
cost. Unidentifiable dispatch slopes and grids outside the measured range remain
explicit. These measurements characterize scheduling; they do not substitute for
operator-specific core throughput or separately measured operator stage latency.
The original bandwidth/compute hardware profile is reused when compatible.

Independent stage calibration is controlled by `--stage-calibration full`,
`bounded`, or `skip`. `full` has no implicit wall-time limit and is the fresh
installation default; `bounded` records a resumable incomplete status when its
budget is exhausted; `skip` preserves the legacy whole-kernel workflow. The
unified entry forwards this policy to the lower-level calibrator and includes it
in the migration manifest. A required stage profile passes the migration gate
only when `stage_service_profile.json` has both `validated: true` and
`calibration_status: "complete"`. A `complete-bounded` stage result is useful
checkpoint evidence but is not the full completion gate.

The stage probe checks target-GPU exclusivity before and after each invocation;
it refuses to launch when another compute process is active and does not wait or
mix timing rows. This is a reservation check rather than a race-free system
lock, so the target must remain reserved for the entire calibration.

The current `StageProbeAdapter` exposes the original launches for the supported
NTT `baseline`, `tile256`, `hybrid2d`, `stage-pipeline`, and
`shared-iterative` paths where a physical group split is available. It does not
substitute an alternate kernel. Low-precision paths and selected multi-subgraph
composite/overlap lowerings remain unsupported until their real group contracts
are exposed and validated.

This update adds the mode/manifest plumbing and CPU-only contract tests. The
current A100 target is occupied, so no GPU stage calibration or performance
acceptance was run for this documentation update. Do not call a target
calibrated or performance-passed until an exclusive target run produces a
validated complete stage profile and the migration manifest reports `complete`.

Per-stage startup plus tail and service coefficients are positive fits to serial
whole-plan observations, shared across workload sizes. The artifact records
unseen primitives, extrapolated ranges and whether overhead/service can be
separated by the observations. It does **not** claim separately measured stage
latencies. Concurrent SM/HBM sharing is still an approximation, so this initial
model remains `calibrated-local-warning` pending independent calibration and
holdout validation. Whole-workload holdout ranking is reported separately.

`--search-strategy evolutionary` adds bounded split/merge/boundary-move and
implementation-parameter neighbors around multiple measured parents and model
seeds. New points may lie outside the exported inventory. Runtime feasibility,
correctness and repeated timing still gate promotion. It imposes no fixed number
of stages and does not certify a global optimum.

A cached runtime rejection overrides symbolic ranking estimates. Confirmation
slots count distinct complete resolved designs, including workload and hardware
identity, so candidate/seed aliases do not consume separate finalist slots.
These policies are versioned in `search_coverage.json`; changing them retraverses
search while retaining compatible timing samples. Rejected proposals and budget
omissions remain visible in the audit.

For an already installed target, continue its exact checkpoint:

```bash
python scripts/tune_hardware.py \
  --resume-from results/MY_TARGET/migration_manifest.json \
  --gpu GPU-YOUR-UUID --time-budget 1800 --round-budget 16 --max-rounds 4
```

The driver increases the exploration budget, uses the staged model and
evolutionary strategy, and retains compatible timing trials. It stores
`tuner_state.json` beside the manifest. Without `--watch`, a busy GPU pauses the
driver; `--watch --poll-seconds 15` waits within the same total budget. Time spent
across resumptions counts against that budget. No background service or scheduled
job is installed. Only the driver's own process group is interrupted at expiry.
`--dry-run` records the next command without executing measurements.

Old manifests resume their recorded whole-kernel model unless explicitly changed
with `--cost-model staged`; the tuner makes that change explicitly. Changing
ranking/search strategy retraverses the candidate queue while compatible kernel
measurements remain reusable. `--cost-model legacy` is available for controlled
comparisons with the previous regression.

## Candidate Search and Coverage

Installation queries `--list-design-points` from `cubutterfly_bench` or
`cuntt_bench` for each workload in `config/install_search_workloads.json`.
Mappings include the operator's available shared-group and specialized-core
lowerings, including bulk and supported batch-pipeline schedules. The hardware
registry restores complete measured mappings through the public planner.

`--search-budget N` limits screening per workload; zero traverses the exported
inventory's count (also the finite exploration budget in evolutionary mode).
It does not remove time or partition-projection limits. `--search-finalists N`
controls repeated confirmation and `--search-workloads FILE` selects semantic
cells. Search time and additional compilation time have separate budgets.
Use each command's `--help` for current defaults; at this update the installer
and calibrator default to 64 screened mappings and 3 finalists. Fresh `full`
stage calibration uses `0` search seconds and `0` additional compilation
seconds (both unrestricted); `bounded` and `skip` use 300 search seconds and
600 compilation seconds unless explicit budgets are supplied. These are not
limits on total installation time or an already-started confirmation.
`search_coverage.json` records budget omissions and unsuccessful points.
`search_measurements.json` preserves candidate trials, including incomplete
confirmation; `search_coverage.json` marks completed cells. Only confirmed
correct internal mappings are promoted; cuFFT remains a reference.

This does not establish full coverage of the theoretical mixed-dataflow space.
See the current [refactor progress](framework_refactor_progress.md) for remaining
work. The [theory alignment audit](theory_alignment_audit.md) records earlier gaps
and should be interpreted with its historical scope.
