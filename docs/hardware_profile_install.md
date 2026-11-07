# Hardware Profile Initialization

Platform-specific candidate preferences use the
[platform tuning interface](platform_tuning.md). They are separate from
measured profiles and executable capabilities. The
[October 7 repair record](platform_boundary_repair_20261007.md) documents
the corrected candidate coverage, runtime loading and fresh retest identities.

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

cuButterfly does not treat a mapping measured on one GPU as a portable
constant. Installation first reuses compatible local or repository profiles.
A missing target can initialize a local profile; refreshing an existing profile
requires an explicit request. Reuse does not certify missing performance layers.

The [unified planner guide](unified_planner.md) defines the current mapping,
compile-policy and registry contracts. This page covers their operational use.

## Static profile policy

The [architecture decision](adr/0001-static-profile-driven-planning.md) applies to
all butterfly operators. Hardware constraints and performance observations are
separate files, not one undifferentiated calibration flag.

| Input | Meaning | Reuse condition |
| --- | --- | --- |
| `hardware_model.json` | Complete capacity limits, including per-CTA and per-SM resources | Exact model, compute capability and memory capacity |
| `hardware_profile.json` | Sampled hardware rates and sampling conditions | Preserve platform and observation provenance; not a complete cost model |
| `stage_service_profile.json`, `cost_model.json` | Service samples, coverage, model version and validation | Matching semantics, core/build and measurement protocol |
| Compiled module manifests | Actual registers, shared scratch and specialization support | Matching compiler, templates and target |

The installer selects an explicit `CUBUTTERFLY_HARDWARE_MODEL`, then complete
local artifacts, then compatible `config/hardware/<tag>` seeds. A compatible
legacy `calibration_device.json` remains reusable as partial data when no complete
capacity artifact exists. An invalid existing capacity file fails explicitly;
it is not silently replaced with another GPU's limits. A missing performance
layer or a warning model is reported and does not automatically start new tuning.

The repository's A100 80GB PCIe seed currently contains capacity facts and
historical hardware rates. It does not claim current whole-plan model validation.
Existing local performance files are preserved when missing capacity files are
supplied from that seed. Reinstalling also preserves refreshed local files.

New rate profiles include exact CUDA memory capacity from
`cubutterfly_bench --device-identity`. Explicit calibration also completes this
field in older profiles after checking the device and existing capacity model.
The verified profile is persisted beside `hardware_model.json`, even when stage
and search results use a separate output directory, so runtime ranking consumes
the same identity. Conflicting identities are rejected; capacity is not inferred
from a GPU name or rounded `nvidia-smi` output. This metadata repair preserves
existing rate observations and compatible stage/search checkpoints.

Default reuse invokes no calibration or search. Build and requested correctness
tests remain separate installation steps. To explicitly update a profile, add
`--refresh-profile` to the install command. To continue an interrupted search,
use `--resume-search`; these options are mutually exclusive. The manual
`calibrate_hardware.py --mode execute --force ...` entry remains available for
explicit acquisition. `--stage-calibration bounded --search-seconds 300` requests
a bounded scope; it must not be described as exhaustive calibration.

Capacity acquisition uses `cubutterfly_plan_probe --hardware-live` and exports
`hardware_model.json` even when operator calibration is skipped. Ordinary
`--hardware` reports the capacities actually consumed by the common resource
query and a `resource_source`. Missing static capacity data currently permits a
labelled live-properties fallback; this is not calibrated performance. The
CPU-only inspector needs no GPU:

```bash
python scripts/static_hardware_profile.py inspect \
  config/hardware/NVIDIA-A100-80GB-PCIe-sm80-84974239744B/hardware_model.json
```

Runtime lookup uses the explicit file, then `CUBUTTERFLY_PROFILE_ROOT/<tag>`, or
the configured installation data directory when no root is set, then the bundled
seed. For a custom profile directory, source its generated `profile_environment.sh`.
The installer binds the CMake install prefix; moving a separately configured
installation with `cmake --install --prefix` requires setting the profile root.
No application call automatically reruns a performance benchmark.

The rank-one constructors now apply the common legality gate before execution,
including after JIT resource feedback. The selector first looks for a compatible
measured registry entry. On a miss it ranks the projected inventory from static
performance artifacts, constructs a bounded set of finalists and reranks their
resolved resources. It does not benchmark during this fallback. The default
finalist budget is four; `CUBUTTERFLY_MODEL_FINALISTS` sets a positive alternative.
This budget is an exploration limit, not a claim of an exhaustive optimum.

Selection reports distinguish `validated-static-model`, `hardware-rate-prior`
and `unmeasured-feasible`. Only a compatible, validated model with covered
primitives supplies the first category. Hardware rates alone supply a prior;
missing performance data or an incomplete physical launch descriptor stays
unranked. A feasible fallback is reported by the public C API as
`CUBUTTERFLY_SELECTION_UNMEASURED_FALLBACK`, not as a static model prediction.
The [audit](constraint_pipeline_audit_20261024.md) records remaining coverage.

For model reuse, set `CUBUTTERFLY_MODEL_TIMING_PROTOCOL` to the JSON string
`"direct-launch"` only when the caller actually uses that timing protocol.
An unspecified or incompatible protocol does not admit a fitted model.
`CUBUTTERFLY_PLAN_TRACE=/path/to/trace.jsonl` records projection rejections,
ranking partitions, finalist construction and the final mapping. Ranking uses
the installed `rank_static_candidates.py` CPU helper; `CUBUTTERFLY_STATIC_RANKER`
can explicitly select its path. No GPU measurement or compiler is launched by
that helper. It reuses an existing compatible native CPU schedule library in
load-only mode, or evaluates the same schedule in Python. Explicit
`CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE=1` remains authoritative. CPU preparation
can populate the existing cache before GPU work with
`PYTHONPATH=scripts python -c 'import stage_schedule_native as n; print(n.backend())'`.
The ranking trace reports `schedule_backend`. Finalist construction may compile
GPU modules under the caller's compile policy, outside execution timing.
Resource-rejected finalist batches are replaced from the remaining ordered
inventory; they are never selected merely because the initial budget was reached.

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

JIT modules use a shared CUDA runtime. This avoids a separate static runtime in
every specialization, whose stream/event overhead grows when many modules are
loaded. The module cache key and manifest record this linkage; existing static
modules remain historical artifacts. See the
[runtime investigation](jit_shared_runtime_20261010.md) for the measured scope.

When a research campaign uses an immutable compilation broker, bind fresh
calibration to that same broker by setting `CUBUTTERFLY_CALIBRATION_BROKER`.
Also set `CUBUTTERFLY_TEMPLATE_ROOT`, `CUBUTTERFLY_MATHDX_ROOT`,
`CUBUTTERFLY_NVCC` and `CUBUTTERFLY_JIT_CACHE` to its exact configuration.
Calibration then records the generator, templates, compiler, runtime link
inputs and loader search environment in its manifest and stage/composition
checkpoints. The async timing runner rejects shared-runtime calibration reuse
if this binding is absent or different. Use a fresh calibration directory;
adding identity fields to old measurements does not recalibrate them.
This binding covers the broker's compilation inputs; it does not claim complete
ELF loader tracing or exhaustive hardware-profile coverage.

During explicitly authorized development, `--resume-search --resume-build-priors`
allows earlier measurements from the same hardware and timing protocol to inform
continued calibration and search. It preserves their source identities, refreshes
current descriptors, and measures missing candidates and current finalists.
This opt-in workflow does not turn earlier timings into current-build observations.
The local continuing campaign exposes adoption through
`results/install_compare_repair_20261014/run.py --resume --adopt-current-build`.
Its CPU inputs and already measured data remain reusable; a build change alone
does not require restarting every experiment.

If every search cell has already completed and this repair only needs current-build
confirmation, add `--reuse-completed-search` to that adoption command. The driver
requires the same workload scope and a complete prior stage/search checkpoint,
archives the prior attempt, and records its file hashes. The retained search is
labelled `retained-as-mapping-seeds`, not current calibration. Finalization reruns
whole-plan confirmation, independent stage measurements, composition checks and
selector replay on the rebuilt library before comparison can begin. An interrupted
adopted run uses `--resume --resume-after-gpu-interruption`; it must not adopt again
or attach a CPU ranking package from another build.

The October 16 stage-adapter repair exposes compact-stage NTT's three original
physical launches and queries FP64 cuFFTDx online-reorder resources from the same
kernel dispatch used for execution. Unknown resource values must not be reported
as measured zero. Serial/overlap resource identity checks remain strict. These
fixes close adapter gaps; they do not establish full concurrency-model accuracy.

When initialization or explicit refresh is requested, the wrapper runs the
following loop. The examples above disable
the total screening cutoff so each configured workload receives its candidate
budget; they can take longer than a bounded installation. Fresh installs use
independent stage calibration by default (`--stage-calibration full`) and set
the full-mode search-time budget to `0` (unrestricted). The `bounded` mode uses
the default 300-second budget, while `skip` records that stage calibration was
not requested.

1. build the configured tree;
2. probe the device, run operator and independent-stage calibration, then measure the declared serial/overlap pairs for staged calibration;
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

Full mode reserves a separate finite sampling budget for each curve. It includes
boundary seeds and 31 additional slots per nontrivial seed interval, enough for
four rounds of midpoint promotion followed by child-interval holdouts. The floor
is 33 slots, capped by the number of legal integer loads. Coverage records the
actual cap per curve; a later curve's larger budget cannot relabel an earlier
curve's cap. Explicit `stage_service_calibration.py --max-points N` caps remain
binding; bounded mode retains the 33-slot default. Further nonlinear refinement
can still produce an explicit incomplete status, never automatic validation.
Compatible recovery retains measured loads and fills the remaining intervals.
The count limits
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

For a finite incremental confirmation after stage calibration, the lower-level
`calibrate_local_hardware.py` also accepts `--seed-only --seed-budget N`.
It tests only matching proposals from the frozen seed snapshot, using the same
resolution, screening and repeated finalist confirmation as ordinary tuning.
It skips full-inventory enumeration, ordinary exploration and non-seed
incumbents. A workload without a matching seed is an error; budget omissions
remain explicit. This mode does not certify full-space search or replace
hardware migration. The unified installer retains its ordinary search policy.
Accepted provider aliases are normalized before seed comparison, while actual
thread, tile, partition, I/O and arithmetic choices remain checked.

Native low-precision FFT confirmation can explicitly request
`--low-fft-error-report-only`. It records same-precision CPU-reference errors
without applying a floating tolerance; only finite outputs qualify for timing
promotion. Integer checks and other floating operators keep their existing
contracts. This flag neither changes input scaling nor certifies numerical
accuracy, and its identity is preserved across resume. Public Graph comparisons
remain separate from the calibrator's direct-launch timings.

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
and runtime fingerprint. On a registry miss, the rank-one selector uses compatible
static model estimates or hardware-rate priors as described above. If neither
can rank the executable candidates, it reports `unmeasured-feasible`. These
sources remain distinct from a confirmed measured mapping.
Unsupported semantics still fail explicitly. Archived V100 timings remain
historical target-specific evidence. Compatibility mappings may be offered
as unmeasured platform priors; they do not qualify as current calibration.

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

[Independent CPU preparation](calibration_cpu_preparation.md) can export pending
candidate inventories once, score them without a visible GPU, precompile priority
modules, and persist exact score epochs. For an interrupted refresh,
`python scripts/prepare_refresh_calibration.py --campaign CAMPAIGN --output OUTPUT
--jobs 4 --compile-top 32` restores the campaign environment and checks checkpoint
identity after the owned measurement driver stops. It first compiles saved
`needs-compilation` configurations, including those in completed cells, then
captures full pending inventories and performs offline scoring and priority
precompilation. Capture may query GPU properties; compilation and scoring hide
CUDA devices and perform no timed measurements.

The GPU watcher accepts `--cpu-preparation-status OUTPUT/prepared/status.json`
and requires both `scoring_complete` and `compile_ready` before checking GPU
idleness. For research recovery, `calibrate_local_hardware.py --resume-search
--retry-compilation-gaps` retries selected, unmeasured gaps while retaining
successful measurements and selection order. Pass the matching
`--cpu-preparation-input OUTPUT/capture/input.json` and
`--cpu-preparation-dir OUTPUT/prepared` to reuse preparation. The refresh driver
exposes these as `--calibration-cpu-input` and `--calibration-cpu-dir` and enables
gap retries with an unrestricted research compilation budget when supplied.
Later adaptive feedback can still require CPU scoring and new compilation;
these preparations do not replace GPU calibration or verification.

For an explicitly requested automatic continuation of a finite campaign,
`scripts/watch_idle_campaign.py` observes one GPU UUID every 10 seconds. It
requires 30 continuous seconds with no compute or graphics processes, zero GPU
and memory utilization, and at most 64 MiB of idle driver memory. Query failures
reset this window. These limits are configurable; the watcher does not reserve
the device or stop other users' jobs.

A headless display server can retain a few MiB even when computation is idle.
Only when that baseline has been checked, pass its exact NVIDIA-reported name
through `--idle-graphics-name /usr/lib/xorg/Xorg`. This explicit exception admits
only a graphics-only process; zero utilization and the total memory limit still
apply. Compute and mixed compute/graphics processes always prevent launch.

```bash
python scripts/watch_idle_campaign.py --campaign-dir RESULTS --gpu-uuid GPU-UUID \
  -- python RESULTS/run.py --resume --resume-after-gpu-interruption
```

The child must use `driver.lock`, guard exclusivity during measurement, clean up
its children on SIGTERM, and write `status.json`. Only a fresh explicit
`retryable: true` GPU interruption is retried. Build mismatches, compilation
failures and numerical failures stop for inspection. A finished campaign must
report a `complete*` status and `remaining: 0`. The watcher uses `monitor.lock`
to prevent duplicate monitors and exposes `monitor_status.json`,
`monitor_events.jsonl` and `monitor_driver.log`. It continues until completion,
a real failure or an explicit stop; SIGTERM to its recorded `monitor_pid`
stops the owned driver cleanly. Same-build interruption recovery retains phase
checkpoints and compile caches; it does not readopt the build or erase progress.

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

## CPU preflight for runtime artifacts

When the install prefix contains `conda-meta`, the installer also installs
`etc/conda/activate.d/cubutterfly_profile.sh` and its matching `deactivate.d`
hook. On the next Conda activation, an unset `CUBUTTERFLY_PROFILE_ROOT` defaults
to `$CONDA_PREFIX/share/cuButterfly/hardware`. Explicit values, including an
empty value, take precedence. Deactivation removes the automatic value only if
it is still unchanged; a user override made during the session survives.
These hooks select the profile root for an installed library without requiring
a kernel rebuild. They do not publish, relabel or calibrate profile artifacts.
When installed template and MathDx directories exist, the hooks also supply
unset `CUBUTTERFLY_TEMPLATE_ROOT` and `CUBUTTERFLY_MATHDX_ROOT`, with the same
override-preservation rule. CPU precompilation must invoke that template tree's
`scripts/compile_module.py`; invoking the source-tree helper with an installed
`--root` can produce a different cache identity. Prepared modules can be placed
in the default `~/.cache/cubutterfly/modules` directory before GPU measurement.

Before launching a runner that requires existing tuned mappings, check its
explicit profile root, hardware identity and runtime fingerprint:

```bash
python scripts/static_hardware_profile.py preflight \
  --profile-root PROFILES --expected-identity identity.json \
  --registry registry.json --runtime-fingerprint BUILD_FINGERPRINT
```

`identity.json` supplies `device`, `compute_capability` and exact
`global_memory_bytes`. The root must contain
`<device>-sm<cc-without-dot>-<memory-bytes>B/hardware_model.json`, using the
runtime's sanitized device tag. A legacy leaf calibration directory does not
satisfy this layout. The check requires exact hardware capacity identity and at
least one registry record with `status: measured` and the supplied fingerprint;
an 80 GB target cannot stand in for a 40 GB target. Failures exit with code 1.
Optional `--workload-keys keys.json` requires each semantic object in the JSON
array to match a measured canonical registry key exactly. Without that option,
success establishes no workload coverage beyond the matching record count.

The importable interface is
`preflight(profile_root, expected_identity, registry_path, runtime_fingerprint,
workload_keys=None)` (the final argument is keyword-only). It reads JSON on CPU
and does not initialize CUDA. A complete capacity file describes resource limits;
compatible measured records establish only their recorded tuning scope. Neither
file presence nor this preflight certifies completed performance calibration,
stage validation, composition validation or comprehensive workload tuning.

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

The exported mapping inventory contains many policy combinations that share the
same physical execution group. The default `--stage-point-budget 0` preserves
the complete inventory behavior. A positive budget enables a deterministic,
stratified sample anchored at every `(operator, precision, logN)` stratum; it
is a sampling shortcut, not physically equivalent coverage of omitted policy
combinations. The selected and omitted counts, `scope`, and
`inventory_complete` state are recorded in `stage_point_sampling.json` and in
the stage profile/manifest. A sampled profile may retain
`selected_scope_validated`, but is reported as `complete-sampled` and cannot
satisfy the full migration completion gate.

`search_coverage.json` records the stage-point budget and sampling summary. With
`--search-stage-projection plan`, search descriptors explicitly carry an
omitted stage-service projection and the model keeps the conservative
unresolved-batch fallback; this mode must not be read as full sub-stage
composition coverage. Measurement identities include `libcuntt.so` when the
build is shared, so a relink invalidates reusable timing checkpoints.

The stage probe checks target-GPU exclusivity before and after each invocation;
it refuses to launch when another compute process is active and does not wait or
mix timing rows. This is a reservation check rather than a race-free system
lock, so the target must remain reserved for the entire calibration.

### Serial and overlap composition validation

Starting with calibration-tool revision 0.9.3.2, staged migration also emits
`composition_calibration/composition_calibration.json`. Independent stage
measurement, observed composition timing, model validation and complete
inventory coverage are separate states. Legacy manifests without composition
evidence report unknown coverage rather than a successful full migration.

The calibrator combines verified operator samples with independently measured
stage descriptors from the same device and runtime. It constructs bounded
serial/overlap pairs using their actual physical groups. Core, stage partition,
precision, direction and layout stay fixed within each pair. Default tile or
slice counts are 2, 4 and 8; at most two feasible seeds per semantic family are
selected deterministically without consulting latency. The default whole-request
bound is `N * batch <= 2^20`. Original batches are retained; an additional,
separately labelled `batch - 1` request tests a partial final tile when applicable.
Omitted families and requests remain visible in the inventory.

Predictions use the frozen independent-stage model before observing the paired
whole-plan timing. All batches are checked, and three trials use the configured
operator warmup/repeat settings. The report retains raw calls, negative speedups,
top-one ranking, latency regret and prediction errors. Compilation, planning,
input transfer and CPU references are outside kernel timing. Existing floating
benchmark checks are diagnostic, not substitutes for the campaign's pending
error-distribution acceptance decision.

`complete-observed` means the declared requests were measured. It does not mean
the model passed, all supported families were covered, or concurrent resource
parameters were identified. This finite sampler records `inventory_complete:
false` and `concurrency_parameters_identified: false`. Full migration therefore
remains incomplete even if its finite holdout succeeds. Factor-sliced service
projection is not yet independently calibrated; those predictions stay uncovered
instead of scaling a whole-group time into a purported slice measurement.

Normal staged installation and `finalize_recalibration.py` invoke this check.
Legacy/skip modes record it as skipped. The standalone diagnostic can also
prepare or resume an explicitly supplied, compatible seed set:

```bash
python scripts/calibrate_composition.py \
  --build-dir build-local --profile TARGET/calibration_device.json \
  --operator-calibration TARGET/composition_seeds.json \
  --output-dir TARGET/composition_check --prepare-only
```

Remove `--prepare-only` to measure on an exclusively reserved GPU. Reuse the same
arguments with `--resume` only for compatible interrupted measurements. Source,
library, device, compile policy, protocol and saved-evidence mismatches reject
resume. The [October 8 record](composition_calibration_20261008.md) documents
the current finite population and remaining coverage gaps.

The current `StageProbeAdapter` exposes the original launches for the supported
NTT `baseline`, `tile256`, `hybrid2d`, `stage-pipeline`, and
`shared-iterative` paths where a physical group split is available, plus the
three launches of `compact-stage`. It does not substitute an alternate kernel.
Native low-precision FWHT stage recomposition is also checked by
`tests/check_stage_probe.py`. Other lowerings require their own physical group
contracts; probe gaps and omitted scope remain explicit in the calibration
artifacts. A passing adapter test is not a completed hardware migration. Inspect
the target's measured coverage, independent validation, composition report and
migration manifest before declaring it calibrated.

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

### CPU ranking and validated scope

Candidate ranking prepares fixed features once per workload and reuses only
identical event schedules across model refits. Fitted coefficients and coverage
checks still use the current model. An optional C++ CPU event simulator preserves
the Python event order and double arithmetic; it compiles before scoring, outside
GPU timing. Its source ships beside the installed Python helper. If compilation
is unavailable, the Python implementation remains available; set
`CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE=1` to select it explicitly. The search
audit records the active backend and per-workload CPU scoring counters.

An incomplete stage profile may contribute only curves independently certified
in its `ranking_scope`; missing or unvalidated keys remain uncovered. This is
not permission to publish an incomplete calibration. Campaign confirmation and
registry publication require the declared stage scope, independent validation
and composition checks to complete. A finite sampled scope is still distinct
from the full inventory. Changes to the stage probe require new stage samples;
compatible whole-plan measurements can remain reusable after reranking.
See the [October 14 repair record](calibration_repair_20261014.md) for evidence.

When relocating a local-calibrator recovery directory, retain the original seed
source with `calibrate_local_hardware.py --resume-search --seed-registry PATH`
alongside the usual build, profile and output arguments. `PATH` must equal the
source recorded by `mapping_seed_snapshot.json`; the output registry remains
controlled separately by `CUBUTTERFLY_REGISTRY`. The seed snapshot and explicit
seed-file hashes are checked before GPU work and are not silently rebound.
If stage sampling is interrupted, the calibrator saves its profiles and exits
with code 130 before operator search or registry publication. On compatible
resume, clean measurements are reused and contaminated loads are retried while
their original records remain in the checkpoint.
