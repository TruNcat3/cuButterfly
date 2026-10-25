# Engineering review and comparison revision 0.9.1-review.1

This campaign was subsequently cancelled by the user in favor of the
[0.9.2 unified-semantic repair](unified_semantics.md). Its frozen files are
historical preparation records. A persistent cancellation marker blocks
calibration/run/resume; the commands below document the cancelled workflow.

This review precedes the next A100 comparison. It does not certify every
possible butterfly workload or a globally optimal search. GPU timing is held
until the reviewed build and its calibration pass the execution gates.

## Contracts under review

| Layer | Required contract | Evidence boundary |
| --- | --- | --- |
| Semantics | Operator, precision, accumulation, direction, normalization, order, strides, modulus and matrices survive lowering and replay | CPU contracts plus later GPU value checks |
| Mapping | Platform preferences propose mappings without changing semantics or deleting portable candidates | Platform rule and registry tests |
| Search | Executable inventory, screened candidates, rejected candidates and budget omissions remain distinct | Search journals and measured finalist replay |
| Calibration | Measurements belong to the same library, executables, hardware and compile policy | Content identities and selector replay |
| Comparison | Baselines have matching semantics and timing protocols; compile and initialization are excluded | Frozen manifest, runtime metadata, raw trials |
| Statistics | Failed, duplicate, aliased, unverified or mismatched rows cannot become independent wins | Adversarial CPU reporting tests |

## Confirmed campaign defects and repairs

* The historical prior used SM70 as a proxy for V100. It now applies only to
  the recorded `Tesla V100-SXM2-16GB` model/capacity; Titan V, GV100 and other
  V100 variants use portable defaults or their own profiles. Local NTT points
  are considered before this compatibility prior. Historical priors no longer
  report calibrated status.
* Compile policies previously shared a `:default` identity. Precompiled,
  auto, on-demand and research now have distinct identities, and invalid policy
  strings are rejected consistently. Generated kernel contents and build
  options contribute to the runtime hash. The generated local selector has a
  separate selection-cache hash, so updating it does not invalidate the kernel
  measurements that produced the table.
* One CUDA event timing helper did not check API results. It now rejects
  event errors and nonpositive/non-finite timings and cleans up event handles
  on exceptions. Candidate-specific failures are recorded or skipped; memory
  exhaustion and a failed CUDA context still terminate tuning. xD candidate
  timing failures cannot silently select the initial candidate.
* Stage-model primitive keys omitted semantic and physical policy axes. The
  model is now `physical-stage-composition-v7` and rejects stale physical-model
  coefficient tables. The separately identified legacy v3 compatibility path
  remains available for legacy controls.
  Canonical defaults and CSV/JSON scalar types agree. Existing raw samples can
  be refitted only when their runtime identity remains compatible. Validated
  local curve coverage is usable without claiming complete inventory coverage.
* A successful calibration label alone did not bind the calibration to the
  comparison library. Reviewed campaigns now compare the shared-library and
  benchmark hashes, compile policy, hardware identity, platform-rule identity
  and successful selector replay before executing a step.
* Correctness failures previously permitted later diagnostic timings. Reviewed
  campaigns stop after a failed correctness or profile gate. Historical
  manifests keep their recorded behavior and are not rewritten.
* A GPU conflict could leave partial CSVs for a later `--resume`. Reviewed
  campaigns preserve incomplete-step outputs in `quarantine/` and repeat that
  step. Successful steps remain reusable. The quarantine is recoverable.
* A source/build change during a queue step could evade the startup check.
  Reviewed campaigns verify frozen sources and artifacts before and after
  every step, including available external benchmark binaries/modules.
* Historical cuFFT cases using the ButterflyPlan adapter were accidentally
  regenerated as cuButterfly AUTO cases. The generator now retains their
  explicit cuFFT identity for FP64, normalized inverse and strided FFT.
* CPU preparation formerly described a hardware label as runtime-confirmed.
  The manifest now distinguishes the intended target from the identity checked
  on the selected UUID at execution time.
* Reporting now rejects duplicate trials, mismatched contracts and unverified
  timings. Alias rows and vendor fallbacks cannot create independent wins.
  Missing references are separate from unfinished execution. Geometric means
  span the declared sizes and batches within each named cohort; repeated
  observations of the same semantic workload receive only one vote.
* xD timing previously omitted the selected mapping from its result, while
  the separate checker created a default plan. The reviewed protocol exports
  and replays the exact timed mapping, verifies its returned mapping identity,
  and joins value checks by workload, library identity and compile policy.
  Legacy semantic checks alone do not promote timing-only rows. A strict
  coverage summary must pass before the baseline comparison starts.

The previous comparison waiter was stopped before starting any comparison
phase. The preceding calibration was also stopped. Its 64 mapping confirmations
and physical-stage checkpoint are retained under
`results/local_hardware_profile_a100_gpu1_repaired_20261007`. They are partial
calibration evidence, not a completed calibration release.

## Selection interpretation

An exact calibrated rank-one registry hit differs from an unmeasured first
feasible fallback. The rank-one AUTO benchmark does not itself run a complete
per-workload search. The xD MEASURE selector times a bounded executable
inventory. Neither policy proves global optimality. Generated
`rank1_search_workloads.json` is an input plan, not evidence that those searches
have run. Missing stage services and warning-only models remain coverage gaps.

## Reviewed entry point

Use the existing conda environment. Finish CPU compilation before preparation
and calibration; the reviewed GPU queue performs no rebuild.

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/reviewed_comparison.py prepare \
  --output results/reviewed_campaign_20261007_r1 \
  --gpu-uuid GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
  --hardware-profile results/local_hardware_profile_a100_gpu1_current_20261006/calibration_device.json

/home/wt/yes/envs/cubutterfly/bin/python scripts/reviewed_comparison.py validate \
  --manifest results/reviewed_campaign_20261007_r1/manifest.json
```

`prepare --build` optionally compiles first. Preparation and validation never
query CUDA, initialize a GPU, or start a waiting process. Preparation refuses
to overwrite a frozen experiment. This review changes the runtime, so the old
partial calibration is preserved as seed evidence. After GPU release, confirm
the seeds into a fresh calibration directory:

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/reviewed_comparison.py calibrate \
  --manifest results/reviewed_campaign_20261007_r1/manifest.json \
  --input-profile results/local_hardware_profile_a100_gpu1_current_20261006 \
  --output-profile results/local_hardware_profile_a100_gpu1_reviewed_20261007_r1 \
  --additional-mappings-csv results/ntt_order_repair_20261007/postchange/auto.csv \
  --additional-mappings-csv results/ntt_order_repair_20261007/prechange/shared_10_10.csv \
  --additional-mappings-csv results/ntt_order_repair_20261007/postchange/log18_forward.csv \
  --additional-mappings-csv results/ntt_order_repair_20261007/postchange/word32_inverse.csv \
  --additional-mappings-csv results/ntt_order_repair_20261007/postchange/word64_inverse.csv
```

These CSVs supply mapping descriptions only. No old timing is imported. The
entry confirms three trials per mapping, rebuilds physical service evidence and
replays selector choices. Its finite scope still excludes unsupported stage
adapters and does not qualify the entire design space. Once it succeeds:

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/reviewed_comparison.py run \
  --manifest results/reviewed_campaign_20261007_r1/manifest.json \
  --calibration-manifest results/local_hardware_profile_a100_gpu1_reviewed_20261007_r1/calibration_manifest.json
```

Add `--resume` to resume completed steps. A changed build needs fresh
calibration and a new campaign directory. An unavailable or mismatched
calibration is rejected before comparison. No waiting deadline is renewed
implicitly.

The preserved finite populations are 135 rank-one cases, 480 xD coverage
arithmetic rows plus 48 alias rows, 240 xD scale arithmetic rows plus 24 alias
rows, and 113 public-contract/large-load probes. These cohorts overlap and must
not be added into one independent sample count. Negative tests, unavailable
references and resource limits retain separate statuses. This is an A100
campaign, not evidence of a completed V100 migration or a Cartesian product
of all sizes, moduli and layouts.

## Verification status

The reviewed runtime and the mapping-export benchmark compiled successfully.
The final affected CPU regression suite passed **299 tests and 22 subtests**
with CUDA disabled. The CPU-only C++ platform matching test also passed.
Build, test and preparation logs are retained in
[`results/engineering_review_20261007`](../results/engineering_review_20261007/README.md).

The frozen campaign has been created at
[`results/reviewed_campaign_20261007_r1/manifest.json`](../results/reviewed_campaign_20261007_r1/manifest.json).
Its 49 commands passed CPU-only input/artifact/command validation. The initial
report contains zero measured pairs, as required for a new experiment. There
is no automatic GPU waiter for this revision. Explicit calibration and run
commands above are ready for the next exclusive GPU window.

The kernel runtime fingerprint is
`bc32390f2052b26f1d2bdece57180e4f9b3560e79eeab691c0630aa20b959789:research`.
The campaign additionally freezes benchmark binaries and Python orchestration,
which are not represented by the kernel hash alone.

GPU correctness, selector replay and repeated baseline timing for the reviewed
revision remain pending. CPU verification checks engineering contracts and
reporting behavior; it cannot replace those execution gates.
