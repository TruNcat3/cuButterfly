# Full Benchmark Refresh, 2026-10-25

This refresh freezes all seven tracks in the [canonical A100 scope](../benchmark/scopes/a100_80gb_research_v1.json).
The [machine-readable plan](../configs/experiments/full_refresh_20261025.json) records populations,
protocols, entry points, dependencies and explicit gaps. Every new artifact belongs under
`results/full_current_20261025/`. Historical results supply priors and controls with their
original build and device identities; this campaign adopts zero historical timing rows.

## Readiness

### Cyclic follow-up continuation, 2026-10-29

The operator comparison has 66 recorded requests, including 37 floating-point
error-report-only results. Public-default composition has completed all 138
cells. The cyclic mapping follow-up stopped after 18 accepted requests on a
CPU process-pool failure, leaving 72 of 90 requests.

The continuation lives in
`results/full_current_20261025/recovery_followups_20261029/`. It reuses the
validated CPU preparation and 18 row checkpoints, while keeping the original
stopped report unchanged. New results go to its own `cyclic_mapping/measurement`
directory. The recovery restores the recorded research environment and CUDA
runtime library search path; calibration and identity guards remain unchanged.

The `cubutterfly-cyclic-recovery-1029` tmux session runs the idle watcher on A100
80GB UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`. Consult its
`monitor_status.json` and measurement `results.json` for live progress. The
dependent searched-composition run still requires the completed, verified
90-mapping source; the completed public-default run is not repeated.

### Follow-up repair and retest, 2026-10-28

The 66-request operator comparison is complete. Three follow-ups stopped on
stale preparation identities and a pinned GPU-NTT arithmetic boundary. The
[repair record](../results/full_current_20261025/recovery_followups_20261028/REPORT.md)
preserves their failures and documents explicit source/build rebinding.
The six mechanism cases and all twelve NCU/Nsight variant captures now complete.
GPU-NTT correctness checks pass for 60/61-bit moduli in both directions; its
broken 62-bit range is explicitly unavailable, while cuButterfly requests remain
unchanged. The cyclic retry preserves two valid rows and remeasures the failed
request. Public composition retains 46 completed cells at recovery. The idle
watcher resumes the remaining finite jobs; consult the live ledgers for counts.
These repairs do not establish full-paper completion or a universal speedup.

### Current execution recovery, 2026-10-28 morning

CPU preparation reached 5,823 ready modules with no outstanding compilation
requests. The comparison's first request exposed a missing event-log directory
in the preceding proxy repair and stopped before producing a usable comparison.
The [repair and archived evidence](../results/full_current_20261025/recovery_event_log_20261028/REPORT.md)
cover directory initialization before child spawn and child cleanup after a log
write failure. The 97 affected CPU tests and preparation-only validation pass.
The original 66 requests and 35 follow-up jobs resume under the same idle policy,
with completed calibration, model finalization and xD preparation preserved.
Use the live ledgers for measured progress; the following entries are dated
checkpoints.

### Current execution recovery, 2026-10-27 evening

Calibration, model finalization and xD preparation are now complete. The
following dated readiness paragraphs describe earlier checkpoints. The current
comparison was recovered with its 66 requests and 35 follow-up jobs unchanged;
its previous zero-row attempt remains archived. See the
[signal/CPU recovery report](../results/full_current_20261025/recovery_signal_20261027/REPORT.md)
and live `measurement/results.json` for the distinction between candidate/module
progress and committed comparisons.

The current driver uses `--external-compile-producer`. Start the CPU service
independently before a new queue's GPU driver, using the same broker and compiler
configuration. On this host the service is already running; do not launch a
duplicate. The entry for a fresh supervisor is:

```bash
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python \
  scripts/semantic_compile_producer.py \
  --broker results/full_current_20261025/compile --jobs 4 --watch \
  --priority-report results/full_current_20261025/measurement/results.json
```

The GPU driver verifies the producer's broker, process identity, heartbeat and
lock, and leaves it running during GPU waits and interrupted measurements. Normal
compiler/numerical failures remain failures. A reviewed signal-cancellation
receipt permits retry of unfinished ranking work only; it never accepts a timing.
After the finite queue completes, stop the independent producer by signaling its
leader, allowing active compilations to drain. Do not terminate its process group.

### Initial preparation, 2026-10-25

The current 66-request manifest has 66 CPU preparation records, all `ready`, checked on
2026-10-25. Reuse `results/full_current_20261025/prepared/`. The
[semantic refresh driver](../scripts/run_full_semantic_refresh.py) combines current
calibration with `run_async_semantic_campaign.py` and exposes status suitable for
`watch_idle_campaign.py`. Its `--followup-plan` option accepts the
[followup queue](../configs/experiments/full_refresh_followups_20261025.json), including
prepared application dependencies.
The mechanism journal `results/full_current_20261025/mechanism/timing.json` has been
prepared on CPU for all six requests and twelve variants. CPU imports passed for
CuPy 13.6.0, Torch 2.5.1+cu124 and NumPy 1.26.4. Two filter cases have persisted
CPU FP32 references for original and changed inputs. Two isolated CKKS consumers
and the crypto composition consumer link the current library without rebuilding it.
The selected stage calibration has 570 correct records across its 24 sampled curves;
whole-plan calibration and comparison measurements remain pending. Do not repeat completed CPU preparation
or attach the watcher directly to `run_current_recalibration.py`: that status lacks the
watcher's required `remaining` contract.

The latest measured count is recorded in `status.json`; watcher ownership and GPU
occupancy are in `monitor_status.json`. The initial 66-request preparation has no GPU
samples. Queue edits made while the watcher is idle require a new binding before
resumption; preserve the previous pre-measurement status as an archive.

The 2026-10-25 profile-identity repair preserves the completed stage, schedule and
pipeline checkpoints and two screened search records. All 23 initial incumbent
attempts had failed before timing because `hardware_profile.json` lacked exact
memory capacity. The benchmark had supplied that field in memory, but the runtime
ranker read the separate hardware directory. The initializer now records exact
CUDA identity, and calibration persists the verified profile beside the capacity
model. Conflicting identities remain errors. A CPU replay of the two actual search
records now ranks both using hardware-rate priors; this does not validate the
unfinished whole-plan cost model or supply new GPU performance results.

The old coordinator status, failed incumbents and preparation receipts are archived
under `recovery_profile_identity_20261025/`. The repaired queue retains the same
binaries, GPU, timing protocols and inner measurement files, uses `--resume-search`,
and binds 258 input files, including the profile writer and ranking tools. The idle
watcher resumes the finite queue when the target A100 80GB becomes exclusive.

### CPU Preparation Recovery, 2026-10-26

The profile repair passed GPU replay: all 23 incumbents completed three correct
trials. The next bottleneck was repeated CPU scoring and the install-style
1,200-second compilation budget. At the reviewed stop, 112 selected candidates
were `needs-compilation`. Their selection attempts had consumed search slots;
a cell's `complete` flag therefore meant its attempt budget ended, not that every
selected candidate had a valid measurement.

The owned GPU queue was stopped with its checkpoints intact. This entry prepares
that interrupted queue without repeating successful measurements:

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/prepare_refresh_calibration.py \
  --campaign results/full_current_20261025 \
  --output results/full_current_20261025/cpu_preparation_20261026 \
  --jobs 4 --compile-top 32
```

It first freezes and precompiles the known missing candidates, then captures the
remaining full inventories, scores them and precompiles a bounded priority set.
Capture can query GPU properties through the existing enumerator; it launches no
timed kernels. Scoring and compilation hide CUDA devices. The CPU worker removes
the cumulative installation compilation cap without changing compiler inputs or
the per-module timeout. Projection or compilation failures remain explicit gaps.

Measured curve matching, stage features and resource facts are prepared once per
candidate and reused across coefficient updates. A 200-mapping CPU check across
three coefficient tables preserved complete prediction explanations exactly;
repeated scoring took 0.004–0.035 seconds versus about 0.12 seconds before the
change. This bounded diagnostic is not a whole-inventory speedup or GPU result.
The preparer releases in-memory features at each workload boundary and persists
exact scores with model, source and descriptor identities. Later GPU feedback
still requires model updates and rescoring; precomputed scores are not reused
under a changed model.

The active refresh uses `--calibration-cpu-input` with
`cpu_preparation_20261026/capture/input-scoring-io.json` and the explicit
`--calibration-cpu-inventory-only` flag, without `--calibration-cpu-dir`. The
manifest was rebound with `--rebind-score-context`, retaining frozen-input
checksums, measurement/partition/projection identities and the original source
manifest. This mode reuses the frozen candidates and existing compilation cache,
then recomputes exact scores for each new model in memory. It reads no old score
database and claims no old-model score hits. The research continuation retains
unrestricted cumulative compilation and `--retry-compilation-gaps`, successful
trials, prior failure records and finalist reconsideration.

The first recovery pass compiled all 112 missing candidate records into 109
distinct modules. Capture then published all 23 workload inventories, with
1,111,051 candidates left to score across the 16 unfinished workloads. The seven
completed inventories need no new enumeration; their selected compilation gaps
will be measured on resume. Full offline scoring and priority compilation are now
complete, with their receipt at `cpu_preparation_20261026/prepared/status.json`.
The inventory-only watcher therefore omits the old `--cpu-preparation-status`
gate. Watcher PID `2821397` restarted this mode with all existing measurements
preserved, resuming near the partial 8/32 search checkpoint. GPU exclusivity still
controls launch; CPU readiness does not establish GPU correctness or speedup.

Online search retains exact full-inventory scores in memory and writes only
shortlisted and selected scores when a score database is configured. Offline
preparation still persists the full inventory. The cache now sorts batches of at
most 65,536 scores while retaining the one-second write-flush limit and SQLite
`synchronous=FULL`. The bounded three-epoch CPU/I/O comparison reduced cache workload
time from 6.584 to 1.208 seconds and process writes including close-time
checkpointing from 298,545,152 to 21,356,544 bytes; all 49,152 scores survived
reopening bit-for-bit. See [CPU preparation](calibration_cpu_preparation.md) for
the rerunnable check. This is CPU/I/O evidence, not a GPU speedup. Coverage
checkpoint serialization also removes whitespace only: the same schema and all
fields remain, with atomic rename, reducing the local full checkpoint from about
575 MB to 390 MB.

The build is `build-e13-cufftdx`, compile mode is `research`, and the environment is
`/home/wt/yes/envs/cubutterfly`. Execution requires an explicit A100 80GB PCIe UUID,
exclusive ownership, current library/JIT fingerprints and finalized measured calibration
with selector replay. Fixed explicit mechanism controls need current binary legality
checks rather than a new automatic selection. Application public plans must record
their actual current selection and fresh cache identity.

`planned` means frozen work; `ready` identifies available preparation or an entry point;
`pending` names an unresolved dependency; `unsupported` identifies a missing execution
contract. None of these states claims a new measurement. GPU exclusivity remains a
launch prerequisite even for a ready entry point.

## Frozen Tracks

| Canonical track | Population and protocol | Current execution boundary |
| --- | --- | --- |
| `operator_comparison_66` | 66 exact semantic requests; Graph 10 warmups, 7 replays, 20 calls/graph | CPU preparation and guarded driver ready; current GPU calibration and timing pending |
| `ntt_search_repair_5` | Five fixed word64 rank-one requests; budget 512; three independent Graph rounds per request | Current CLI binding ready; queued after the 66-request receipt, with all five points |
| `mechanism_profile_controlled` | Six requests, twelve fixed-core stage-partition variants; 10 warmups, 20 repeats, 3 trials | CPU journal and all 12 NCU/NSYS command pairs ready; L2, DRAM, stalls and occupancy included |
| `end_to_end_butterfly_applications` | 138 crypto composition contracts | Current consumer built; actual Graph capture added, including twists, all channels and changed/restored input checks. Default characterization and calibrated mapping replay remain distinct |
| `fft_xd_grouped_baseline` | Two FP32 requests, six implementation rows; rank4 `[8,8,8,8]`/B4 and rank8 `[2,2,2,2,2,2,2,2]`/B16 | Current CLI binding ready; queued after the 66-request receipt |
| `end_to_end_fft_poisson_current` | Three cases: 1024 squared/B4, 2048 squared/B4, 2048 squared/B16; eager and Graph | Three jobs queued; CPU imports passed, current GPU plans/execution pending |
| `end_to_end_fft_poisson_adaptive_revision` | One 2048 squared/B16 revision with complete-call trace | Existing `--profile` path; diagnostic timing stays separate from ordinary performance |

The 66-request population contains FFT17, FWHT13, NTT13, subset-zeta8,
superset-zeta8 and structured-2x2 seven. Ranks 1/2/3/4/8 contribute 42/6/6/6/6
requests; directions contribute 40 forward and 26 inverse. All are out-of-place.
It contains neither FFT-L nor xor-zeta nor in-place coverage. Baselines are cuFFT,
VkFFT, GPU-NTT and Dao FHT only where the complete contract applies, with CPU
fallback cohorts labelled separately. Historical gaps include 15 missing baseline
rows and 37 floating requests awaiting accuracy review; refresh each status from
new evidence rather than carrying a historical pass forward.

The Poisson population manifest overrides the common Graph contract with three
warmups, seven replays and twenty calls per graph. Preserve this explicit per-track
protocol. The Torch-only helper uses different input provenance from the CuPy-inclusive
runner; publish it as a separate two-backend cohort if CuPy remains unavailable.
The adaptive witness is a revision of the same application point, not another
independent application sample.

## Existing Commands

Bind the operator track on CPU first. The initial profile has a declared finite
sampling budget of 24 stage points and 32 searched candidates per calibration cell;
the 66-request comparison has a separate 512-candidate budget. Neither is an
exhaustive search claim. CPU references are precomputed; compilation discovered
during adaptive search uses the existing cache/producer and stays outside timing.
Installation calibration itself still contains CPU ranking and preparation work.

Run these commands from the repository root. Publication includes the small population
configuration, not this host's prepared binaries or build-bound current manifest.
On a fresh checkout, rebuild the current library first, then create local inputs:

```bash
export PATH=/home/wt/yes/envs/cubutterfly/bin:$PATH
python "$PWD/scripts/semantic_campaign_manifest.py" \
  --config "$PWD/results/install_compare_20261013/campaign_config.json" \
  --library "$PWD/build-e13-cufftdx/libcuntt.so" \
  --output "$PWD/results/full_current_20261025/manifest.json"
python "$PWD/scripts/semantic_preparation_queue.py" \
  --manifest "$PWD/results/full_current_20261025/manifest.json" \
  --directory "$PWD/results/full_current_20261025/prepared" --jobs 4
```

Skip input regeneration on this host: all 66 preparations are already ready. A new
machine needs its own build, preparations and measured calibration; the guarded driver
below performs calibration before measurements. Prepare the finite driver without GPU work:

```bash
export PATH=/home/wt/yes/envs/cubutterfly/bin:$PATH
export GPU_UUID=GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02
python "$PWD/scripts/run_full_semantic_refresh.py" \
  --build-dir "$PWD/build-e13-cufftdx" --output-dir "$PWD/results/full_current_20261025" \
  --manifest "$PWD/results/full_current_20261025/manifest.json" \
  --preparation "$PWD/results/full_current_20261025/prepared" \
  --gpu-uuid "$GPU_UUID" \
  --vkfft-library "$PWD/build-a100-vkfft/libvkfft_xd_bridge.so" \
  --gpuntt-library "$PWD/build-e13-cufftdx/libgpuntt_semantic_bridge.so" \
  --followup-plan "$PWD/configs/experiments/full_refresh_followups_20261025.json" \
  --calibration-cpu-input "$PWD/results/full_current_20261025/cpu_preparation_20261026/capture/input-scoring-io.json" \
  --calibration-cpu-inventory-only --external-compile-producer \
  --prepare-only --resume
```

The UUID above selects the named A100 80GB target; use the UUID assigned to this
campaign consistently. `--resume` verifies an existing preparation identity and also
works for a fresh status file. Changed source/build inputs require explicit review;
the driver rejects an incompatible status rather than silently resetting it.

Run the prepared finite queue through the watcher:

```bash
python "$PWD/scripts/watch_idle_campaign.py" \
  --campaign-dir "$PWD/results/full_current_20261025" --gpu-uuid "$GPU_UUID" \
  --max-idle-memory-mib 64 --idle-graphics-name /usr/lib/xorg/Xorg -- \
  /home/wt/yes/envs/cubutterfly/bin/python "$PWD/scripts/run_full_semantic_refresh.py" \
  --build-dir "$PWD/build-e13-cufftdx" --output-dir "$PWD/results/full_current_20261025" \
  --manifest "$PWD/results/full_current_20261025/manifest.json" \
  --preparation "$PWD/results/full_current_20261025/prepared" \
  --gpu-uuid "$GPU_UUID" \
  --vkfft-library "$PWD/build-a100-vkfft/libvkfft_xd_bridge.so" \
  --gpuntt-library "$PWD/build-e13-cufftdx/libgpuntt_semantic_bridge.so" \
  --followup-plan "$PWD/configs/experiments/full_refresh_followups_20261025.json" \
  --calibration-cpu-input "$PWD/results/full_current_20261025/cpu_preparation_20261026/capture/input-scoring-io.json" \
  --calibration-cpu-inventory-only --external-compile-producer \
  --resume
```

The shell expands every `"$PWD/..."` before the watcher changes the child working
directory to the campaign root. On this host, the sole process exception is the named
`/usr/lib/xorg/Xorg` G-only display process, observed at 4 MiB: GPU and memory
utilization must both be zero and total used device memory must not exceed 64 MiB.
Any compute or C+G process still prevents launch. The watcher resumes guarded
interruptions after this idle check. Calibration, compilation or operator-comparison failures stop for
diagnosis. After calibration and all 66 requests, the driver runs the original eight jobs:
mechanism timing, NTT five-point, FFT grouped two-point, three ordinary Poisson pairs,
mechanism profile, and adaptive Poisson trace. Prepared filter, composition and CKKS
jobs follow. The CKKS include file supplies 23 ordered GPU phases. A failed followup stays in
`followup_failures`; its dependent jobs enter `followup_blocked`, while independent
jobs continue. Failed jobs are not silently retried. The driver freezes the included
queue and its static consumer/configuration/helper inputs as part of the campaign identity.
The crypto include adds 90 derived cyclic-request confirmations and one searched
composition job after CKKS. Its asynchronous worker uses the existing CPU compile
producer and an explicit exit-75 recycling contract. The producer stops before other
application jobs. These 90 contracts derive from the 138 fixed compositions; they do
not expand the application population.
`complete-runnable-refresh-with-gaps` records these failures. Even
`complete-runnable-refresh` covers only this finite queue and retains
`full_refresh_complete=false` because the unresolved tracks below remain open.

The plan stores argument arrays. Prefix them with the existing environment's Python,
resolve `{gpu_uuid}`, and run GPU calls through the coordinator's ownership guard.
These commands prepare and rerun the existing mechanism study without changing its cases:

```bash
python scripts/run_controlled_experiment.py \
  --study results/mechanism_current_20261017_v2/study.json \
  --build-dir build-e13-cufftdx --mode prepare \
  --output results/full_current_20261025/mechanism/timing.json
python scripts/run_controlled_experiment.py \
  --study results/mechanism_current_20261017_v2/study.json \
  --build-dir build-e13-cufftdx --mode run --resume --gpu-uuid "$GPU_UUID" \
  --output results/full_current_20261025/mechanism/timing.json
python scripts/profile_current_mechanism.py \
  --journal results/full_current_20261025/mechanism/timing.json \
  --output-dir results/full_current_20261025/mechanism/profile \
  --gpu-uuid "$GPU_UUID" --run
```

For each of the three frozen Poisson pairs, use the existing runner with its actual
flags; the JSON plan supplies the size, batch and destination substitutions:

```bash
python experiments/tools/run_periodic_poisson_graph_20261013.py \
  --library build-e13-cufftdx/libcuntt.so --size 2048 --batch 16 \
  --norm backward --calls 20 --warmup 3 \
  --output results/full_current_20261025/poisson/poisson-2048-b16/report.json
```

The adaptive trace uses the same entry point and pair with `--profile` and output
`results/full_current_20261025/poisson_adaptive_trace/report.json`. Its CuPy/cuFFT,
Torch/cuFFT and cuButterfly paths retain full input refresh, transform, spectral
multiply, inverse and independent output handling. Setup/search/capture stays outside
ordinary timing. NCU and NSYS output is diagnostic.

The reusable [NTT supplement](../scripts/run_ntt_comparison_supplement.py) and
[FFT grouped supplement](../scripts/run_fft_grouped_supplement.py) accept explicit
checkpoint, manifest, library, preparation, output and GPU UUID. The queued arguments
bind `measurement/results.json`, the current library and the ready CPU artifacts under
this refresh. Both reject mismatched library SHA, GPU or preparation semantics before
CUDA. NTT selects all five fixed IDs regardless of baseline status. Its accepted report
requires `complete`, `full_population=true`, no remaining cases and three passing
public-measure rounds per point. FFT requires `complete-error-report-only`, both fixed
IDs, and cuFFT/VkFFT timing plus error evidence. Floating accuracy review remains visible.
Historical receipts retain their original four-point/external-library bindings as priors.

## Application And Coverage Gaps

The 138 composed requests are modular single-word/RNS witnesses, including twists.
They do not implement complete CKKS key operations, bootstrap or FFT filtering.
The isolated consumer is now `composed_138/consumer-build/crypto_ntt_bench` under the
refresh root. Its new Graph cohort captures 20 complete calls, warms up with 10 graph
replays and measures seven replays, for both random and boundary inputs. This is an
explicit protocol revision; historical event samples are not reused. Compilation,
plan construction, CPU checking and changed-input validation stay outside timing.
The reference implementation still computes full composition references during each
consumer run. These are not claimed to be precomputed CPU artifacts.

The fixed `public-default` job is a diagnostic. The 138 requests contain 90 distinct
cyclic channel contracts, recorded with their full semantics and composition-to-channel
mapping in `composed_138/calibration_workloads.json`. A calibrated composition claim
requires current measured mappings for those exact contracts, followed by complete
composition verification and timing. Old mapping files cannot fill that dependency.
All 90 CPU input/reference records are ready. Their descriptor manifest is produced
by the existing semantic manifest generator. The GPU runner retains public MEASURE
selection and confirms its output with the existing Graph protocol. The format adapter
requires all 90 exact integer validations, explicit mapping replay, current build/UUID
and complete timing evidence before exporting a composition source. Internal candidate
selection still uses host event timing; Graph confirmation does not retroactively make
it a Graph search. Both protocols are retained in the exported provenance.

CPU validation limits for this derived population are explicitly 16,777,216 elements
and 167,772,160 radix-two butterflies, the finite maxima of the existing contracts.
The core 66-request protocol is unchanged. No batch, word width or arithmetic semantics
are reduced to fit the smaller default CPU validation budget.

Existing application supplements remain separate frozen populations: the two
matched-fast-padding filter cases at image1024/2048, filter31, B16; the 36 CKKS
automatic-schedule configurations; and bootstrap B1/B8 with 118 distinct exported RNS
requests. Filter preparation binds the archived inputs, matched padding, original and
changed CPU references, current library and exact historical timing knobs. Its
`--run-prepared` entry point validates that binding before CUDA.

CKKS preparation records two successful consumer builds and validates all 118 request
contracts. The GPU queue separates four adapter correctness tests, mapping tuning,
cache-only replay, five configuration cohorts' request discovery/preparation/independent
measurement, and bootstrap preparation/measurement. The formal jobs require current
cache hits, exact ciphertexts, changed/restored inputs and complete result matrices.
Discovery timings are excluded from final performance results. Count complete configurations independently from their paths, stages and
RNS request counts. QuaRot Graph remains unsupported by a validated comparison because
the native baseline replay produced nonfinite outputs; preserve its eager negative row
as historical evidence.

Preparation receipts are separate from GPU results:

| Artifact under the refresh root | CPU evidence | GPU dependency |
| --- | --- | --- |
| `prepared/` | 66 input/reference records | Current calibration and comparison |
| `mechanism/{timing.json,profile/}` | Six requests, twelve fixed mappings and profiler command pairs | Correctness, ordinary timing, NCU/NSYS |
| `fft_filters/prepared.json` | Two archived inputs plus original/changed CPU FP32 references | Three-backend Graph measurement and error reports |
| `ckks/{cpu_build_receipt.json,request_validation.json}` | Two consumer builds, 118 exact request contracts | 23 dependent correctness, tuning and application phases |
| `composed_138/{preparation.json,cyclic_mapping/prepared/}` | Current consumer, 138 contracts, 90 derived CPU references | 90 selected-mapping confirmations and full compositions |

Filter and CKKS wrappers preserve successful work during guarded interruptions. They
archive only partial outputs tied to the same prepared identity before rerunning a
whole interrupted application attempt. Unknown or mismatched output identities are
rejected. Integer and Graph replay checks remain mandatory; finite floating error
reports do not introduce a new accuracy threshold.

`config/benchmark_full.json` expands 12,552 request rows: 11,424 operator, 360 e2e and
768 profile. It is an inventory, with unavailable operator/provider/precision/scene
combinations, rather than a universal executable matrix. The 37-case A100 comprehensive
suite is a separate implementation comparison with its own event protocol. Neither
population replaces the 66 semantic requests or completes the seven-track scope.

## Follow-up Queue

1. Use the prepare-only and watcher commands above to run current calibration and the
   66-request async comparison, then the explicit followups and prepared application jobs.
2. Keep ordinary timing separate from all six paired mechanism and adaptive Poisson traces.
   NCU now captures the full process, rather than only the first launch. Setup and
   teardown kernels may be present, so per-kernel diagnostics need attribution and
   must not be summed into a claimed operator latency.
3. Check the queued supplements' output contracts against the new current checkpoint,
   library and preparation. Require all five NTT Graph confirmations before
   aggregating their speedups.
4. Verify calibrated composition mappings and the prepared filter/CKKS cohorts on GPU.
   Their CPU receipts establish readiness and provenance, not numerical correctness or speedup.
5. Summarize every track with case counts, failures, numerical review and baseline gaps.
   Update paper figures from current result paths and explicitly labelled historical
   priors. A pending track keeps the full refresh unfinished; a negative measured point
   completes that experiment without triggering an expanded optimization matrix.
