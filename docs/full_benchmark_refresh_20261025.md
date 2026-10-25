# Full Benchmark Refresh, 2026-10-25

This refresh freezes all seven tracks in the [canonical A100 scope](../benchmark/scopes/a100_80gb_research_v1.json).
The [machine-readable plan](../configs/experiments/full_refresh_20261025.json) records populations,
protocols, entry points, dependencies and explicit gaps. Every new artifact belongs under
`results/full_current_20261025/`. Historical results supply priors and controls with their
original build and device identities; this campaign adopts zero historical timing rows.

## Readiness

The current 66-request manifest has 66 CPU preparation records, all `ready`, checked on
2026-10-25. Reuse `results/full_current_20261025/prepared/`. The
[semantic refresh driver](../scripts/run_full_semantic_refresh.py) combines current
calibration with `run_async_semantic_campaign.py` and exposes status suitable for
`watch_idle_campaign.py`. Its `--followup-plan` option accepts the
[eight-job queue](../configs/experiments/full_refresh_followups_20261025.json).
The mechanism journal `results/full_current_20261025/mechanism/timing.json` has been
prepared on CPU for all six requests and twelve variants. CPU imports passed for
CuPy 13.6.0, Torch 2.5.1+cu124 and NumPy 1.26.4. Current GPU calibration and
measurements remain pending. Do not run serial CPU reference generation
or attach the watcher directly to `run_current_recalibration.py`: that status lacks the
watcher's required `remaining` contract.

The monitor is running as of 2026-10-25 and is `waiting-for-idle`, with 0/66 new
measurements. A foreign `xdit` workload occupies about 72 GB at 100% GPU utilization;
this is not a completed calibration or measurement receipt.

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
| `mechanism_profile_controlled` | Six requests, twelve fixed-core stage-partition variants; 10 warmups, 20 repeats, 3 trials | CPU journal ready; timing and paired FWHT logN18/B1 NCU/NSYS jobs queued |
| `end_to_end_butterfly_applications` | Declared 138 composed contracts | Actual JSON contains crypto compositions. Current consumer binary, measured mapping sources and Graph-protocol alignment remain pending |
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
  --resume
```

The shell expands every `"$PWD/..."` before the watcher changes the child working
directory to the campaign root. On this host, the sole process exception is the named
`/usr/lib/xorg/Xorg` G-only display process, observed at 4 MiB: GPU and memory
utilization must both be zero and total used device memory must not exceed 64 MiB.
Any compute or C+G process still prevents launch. The watcher resumes guarded
interruptions after this idle check. Calibration, compilation or operator-comparison failures stop for
diagnosis. After calibration and all 66 requests, the driver runs eight jobs in order:
mechanism timing, NTT five-point, FFT grouped two-point, three ordinary Poisson pairs,
mechanism profile, and adaptive Poisson trace. A failed followup stays in
`followup_failures`; independent jobs continue and failed jobs are not silently retried.
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
python paper/tools/run_periodic_poisson_graph_20261013.py \
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

The 138 composed requests are modular single-word/RNS composition witnesses, including
twists. They do not implement complete CKKS key operations, bootstrap or FFT filtering.
`run_crypto_composition.py` exists, but its expected current consumer path
`build-e13-cufftdx/crypto-ntt-sm80/crypto_ntt_bench` was absent during this audit. Its
existing 3-trial, 100-warmup, 100-repeat event protocol also needs an explicit cohort
revision before it can satisfy the canonical Graph claim. Keep the population file
unchanged and record this mismatch.

Existing application supplements remain separate frozen populations: the two
matched-fast-padding filter cases at image1024/2048, filter31, B16; the 36 CKKS
automatic-schedule configurations; and bootstrap B1/B8 with 118 distinct exported RNS
requests. The plan identifies their existing runners and receipt sources. Rebind
current consumer binaries, RNS mappings and exact historical protocol knobs before
execution. Count complete configurations independently from their paths, stages and
RNS request counts. QuaRot Graph remains unsupported by a validated comparison because
the native baseline replay produced nonfinite outputs; preserve its eager negative row
as historical evidence.

`config/benchmark_full.json` expands 12,552 request rows: 11,424 operator, 360 e2e and
768 profile. It is an inventory, with unavailable operator/provider/precision/scene
combinations, rather than a universal executable matrix. The 37-case A100 comprehensive
suite is a separate implementation comparison with its own event protocol. Neither
population replaces the 66 semantic requests or completes the seven-track scope.

## Follow-up Queue

1. Use the prepare-only and watcher commands above to run current calibration and the
   66-request async comparison, then the eight explicit followup jobs.
2. Keep ordinary timing separate from the paired mechanism and adaptive Poisson traces.
   FFT/NTT mechanism profiles and L2 counters remain explicit gaps because
   the current profile helper collects only the paired FWHT case and its named counters.
3. Check the queued supplements' output contracts against the new current checkpoint,
   library and preparation. Require all five NTT Graph confirmations before
   aggregating their speedups.
4. Resolve the 138-composition binary/mapping/protocol gap, and freeze the existing
   filter/CKKS consumer receipts under the new output root. Report unavailable cohorts.
5. Summarize every track with case counts, failures, numerical review and baseline gaps.
   Update paper figures from current result paths and explicitly labelled historical
   priors. A pending track keeps the full refresh unfinished; a negative measured point
   completes that experiment without triggering an expanded optimization matrix.
