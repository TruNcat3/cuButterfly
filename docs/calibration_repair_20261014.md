# Calibration repair and recovery, 2026-10-14

Revision `0.9.3.11-calibration.1` repaired CPU calibration scoring and stage-probe
validation. Revision `0.9.3.11-calibration.2` repairs seed-source binding and
interruption handling, then resumes the finite installation campaign in
[`results/install_compare_repair_20261014`](../results/install_compare_repair_20261014).
Revision `0.9.3.11-calibration.3` repairs the finite curve refinement budget.
Driver PID `1775735` completed the frozen stage-scope supplement. After a reviewed
18:46 interruption, successor PID `1831324` continues plan search with unchanged
inputs. Final confirmation, selector release and the 66-request GPU
comparison have not yet been established.

## Isolated provider fix and current recovery

### Current development continuation, 20:21 CST

The user explicitly authorized replacing the interrupted build and continuing
with development priors across versions. `provider.3` is now merged into the main
tree, rebuilt in `build-contract-20261013` with the existing xD capabilities, and
installed into `/home/wt/yes/envs/cubutterfly`. The installed benchmark resolves
that environment's library. FP64 N65536 B64 with the imported cuFFTDx EPT2 suffix
passed verification for all 64 batches. These smoke timings are not new formal
comparison results.

The driver was restarted as PID `1916768`, calibration as PID `1916771`, and the
CPU/GPU join follower as PID `1917488`. It uses the new CPU compilation queue at
`results/install_compare_repair_20261014/compile`. Existing CPU inputs and 83
external CPU timing rows are retained. The sequence remains finite calibration,
current-build finalist confirmation, then 66 semantic comparison requests.

`--adopt-current-build` records the previous state and checkpoints in
`attempts/adopt-current-build-1791980445704138291`, retires incompatible scheduling
caches and delegates the current stage scope to the calibrator. Explicit
`--resume-search --resume-build-priors` preserves clean prior observations and
their original identities; current descriptions and new candidates are acquired
again, while finalists/incumbents are verified and timed with the current build.
The current status file remains authoritative as the background run advances.
This is a development continuation, not a claim that historical measurements were
collected with the new binary or that the full comparison has finished.

Affected validation passed before continuation: 50 main-tree provider/JIT/interface
checks, 55 search-resume checks, the stage-resume tests, and 5 adoption-driver
checks. A previously reproduced stratification-coverage test remains excluded
from the search-resume subset and was not changed by this continuation.

The following sections preserve the earlier isolated work and interruption history.

Revision `0.9.3.11-provider.1` implements the missing common capability check in
the separate worktree `/home/wt/git/cuButterfly-capability-IPwAsa`. Candidate
enumeration, explicit validation and Python JIT share the installed cuFFTDx
database contract. Missing implementations return `NOT_SUPPORTED`; invalid
parameters return `INVALID_VALUE`. Unknown capabilities defer to actual
compilation. Native cores and supported alternatives remain available, and
upgrading MathDx at the same path invalidates the old capability table.

The isolated revision passed 90 affected CPU tests, compiled five affected
C++/CUDA objects, rejected both missing EPT32 mappings before NVCC, and compiled
their EPT16 alternatives with the same transform semantics and two-column
geometry. Its `docs/provider_capability_contract_20261014.md` and
`results/provider_contract_20261014/handoff.json` record validation and source
hashes; the accompanying patch passes `git apply --check` against this tree.
It has not been installed or mixed into this campaign's measurements. Adoption
requires a separate build/cache identity and compatible calibration.

The same isolated worktree now also contains `0.9.3.11-provider.2`, adding an
explicit `suffix_core="native"` implementation to the public register FFT
template. Both formerly missing EPT32 geometries can execute with unchanged
precision, EPT, thread shape and workload. The vendor-specific requests retain
their correct rejection; there is no silent replacement. The core choice reaches
candidate enumeration, mapping replay, JIT and group-local calibration identities.
Its record is `docs/native_suffix_core_20261014.md` in the isolated worktree.
Validation includes 209 affected CPU tests, 432 final standalone GPU checks,
four public API forward/inverse replays and four stage-resource descriptions.
The final GPU run used the separately identified A100 40GB because the second
80GB card became occupied. Earlier 80GB evidence is retained separately.
The native core is slower than the available cuFFTDx EPT16 controls at both
diagnostic points and is not promoted as a winner. Its full shared reservation,
255 registers/thread and remaining compiler local allocation are recorded.
No new native timings or sources are mixed into this frozen campaign.

Revision `0.9.3.11-provider.3` corrects that isolated experiment's direction:
reuse the existing mature cuFFTDx suffix and enumerate its legal EPT choices,
including previously omitted EPT2. The newly written native suffix is retained
only for explicit experiments and replay, not automatic enumeration. A missing
vendor EPT32 configuration did not make the FFT semantic request unsupported.
The isolated `docs/provider_reuse_20261014.md` records 79 affected CPU checks,
four prepared cuFFTDx modules and 144 clean GPU checks on the separate A100 40GB.
This revision is not installed into the active environment. The shared research
principles now explicitly prioritize mature-core reuse for every butterfly family.

At 19:49:47 CST the original calibration was interrupted again by foreign GPU
PID `1892053`; the driver and calibration worker exited. The status is
`quarantined-pending-review`, after the log reached FP64 N65536 B64 screen 22/32.
The prior interruption review applies to the earlier checkpoint, not automatically
to this newer interruption. Preserve the saved records and review the new boundary
before resuming; the isolated 40GB checks completed separately at 19:47:29 CST.

At 18:46:09 CST, the original campaign stopped on foreign GPU PID `1822880`.
The new [v3 interruption review](../results/install_compare_repair_20261014/interruption_review_calibration3.json)
preserves 562 correct/exclusive stage rows, 329 search records (328 correct and
one unavailable), and 23 correct incumbents. All 368 saved search samples carry
the existing exclusive-measurement guard. The last search checkpoint predates
the observed interruption by 32.69 seconds; no unsaved in-flight timings were
adopted. Per-trial wall-clock timestamps are unavailable, so the review does
not claim continuous observation between guard polls. Prior status and
checkpoints are retained under `attempts/calibration3_interrupted_184609`.
Frozen inputs match, the target UUID was free at restart, and driver PID
`1831324` plus join follower PID `1831325` resume the original finite campaign.
The first search cell had saved 21/32 screens at interruption; GPU comparison
remains pending behind calibration and finalization gates.

## V3 refinement and support classification

V2 inspected all 24 selected static templates and retained 457 clean stage
measurements, one excluded contaminated record and 322 candidate measurements.
Two FWHT FP16 curves (N32/N64, B4096, FP32 accumulation) exhausted their old
41-load cap with 15/10 unvalidated intervals. That default reserved only one
refinement after the first failed midpoint. Later curves increased the global
reported cap to 46 without revisiting earlier curves, obscuring the actual limit.

V3 uses a finite cap per curve: boundary seeds plus 31 slots per nontrivial
seed interval, with a 33-slot floor and a legal-load-span ceiling. This reserves
four promotion rounds and child-interval holdouts. Explicit limits and numerical
validation criteria are unchanged; exhausting the new limit still reports
incomplete validation. The stage driver and installed conda copy are updated.
The affected CPU suites passed 61 stage/recovery and 41 calibration/gate tests.

The [prior attempt](../results/install_compare_repair_20261014/attempts/calibration2_incomplete_scope)
preserves the original status, source, logs and measurement journals. The
[revision review](../results/install_compare_repair_20261014/calibration3_review.json)
checks that only CPU orchestration changed. GPU binaries, compilation broker,
generator, input semantics and measurement identities remain unchanged. A
[stage preflight](../results/install_compare_repair_20261014/complete_stage_scope.py)
resumes only the frozen inventory and requires validated scope before expensive
plan search. It verifies preservation of every earlier raw measurement payload.

The supplement passed that gate. There are now 562 clean measurements, up from
457, plus the retained excluded contaminated row. All previous raw payloads and
the measurement identity are unchanged. The N32 and N64 FP16 curves closed at
109 and 78 measured loads, within their respective finite caps of 203 and 237.
All 578 independent heldout group checks were covered; cost prediction relative
error has median 0.49% and p90 6.60%, passing the unchanged 10%/20% criteria.
These are latency-model errors, not arithmetic accuracy tolerances. The scope
contains 24 inspected templates, 22 executable service configurations and two
audited provider rejections. Its status is `complete-sampled`, with
`selected_scope_validated=true`; this does not validate the full design inventory.
The [recovery receipt](../results/install_compare_repair_20261014/stage_scope_recovery.json)
records the preservation checks and full validation report. CPU/GPU join follower
PID `1780548` tracks the active driver.

Two further candidates failed to compile. Their classification is narrower than
an unsupported user request:

| Layer | Finding | Consequence |
| --- | --- | --- |
| User semantics | FFT FP64 N65536 B64 is legal | Keep the request in calibration and comparison |
| Current provider | cuFFTDx SM800 local N1024 permits EPT 2/4/8/16; N4096 permits 4/8/16; neither has EPT32 | These two provider/parameter combinations are not executable in this version |
| Parameter connection | Generated size, precision, EPT32 and FFTsPerBlock2 match the request | This failure is not an incorrectly forwarded field; FPB2 is not its cause |
| Common capability check | Enumeration and geometric validation do not query this provider table | Engineering gap remains in early rejection and capability reporting |
| Same-build alternatives | One correct incumbent and 32 correct screened mappings for FP64 N65536 B64 | The user request has executable alternatives; no precision or size substitution is needed |

The current-build incumbent is `incumbent-search-bfc423a3b487a753d4f1` in
`initial/operator_calibration.json`, with three correct trials, online-reorder,
cuFFTDx-block, local stages 7, suffix EPT4 and suffix threads 128. This establishes
implementation coverage, not completion of the new comparison or superiority.
The local MathDx `records/800/database_fp64_fwd.hpp.inc` and inverse table list
the relevant support entries at lines 283 and 311. EPT32 is legal for some other
sizes, so a blanket FP64/EPT32 ban would be wrong.

Both rejected candidates remain in `coverage.non_executable`. They must not be
reported as calibrated kernels or as mathematical exclusions. A shared provider
capability predicate should eventually be consumed by enumeration, public
validation and JIT. Changing those GPU/generator inputs requires a new build
identity; this recovery does not silently mix that change into existing timings.

## Repair and measured CPU cost

[`PreparedPredictor`](../scripts/stage_cost_model.py) prepares fixed features
once per candidate and stable profile, reusing them across model refits. The
[native CPU event simulator](../scripts/stage_schedule_native.cpp)
preserves the event model's arithmetic and ordering; it does not approximate the
schedule or execute GPU kernels. On the measured 19,539-candidate `xor-zeta`
population, every score was bitwise identical and the entire stable sorted order
was identical to the original path.

| CPU operation | Measured time |
| --- | ---: |
| Original scoring pass | 167.65655 s |
| Prepared/native scoring pass | 2.20227 s |
| Predictor preparation, separately measured | 9.115 s |
| Candidate projection, separately measured | 2.628 s |

The scoring pass improved by 76.1x for this population. The two setup costs above
are excluded from that ratio and remain part of calibration work. This result
does not establish a 76.1x improvement in whole calibration time or GPU throughput.

The [stage microbenchmark](../apps/cubutterfly_stage_microbench.cpp) now matches
FWHT FP16/BF16 reference rounding to actual storage boundaries. All 38 targeted
GPU probe tests passed, including the two formerly unsupported stage cases.
The GPU library and whole-plan benchmark binaries were preserved; only the stage
probe was replaced. Its prior executable is retained in
[`previous_stage_probe`](../results/install_compare_repair_20261014/previous_stage_probe).

## Scope and release checks

Zero independently executable groups are recorded as a scope/adapter issue, not
as a one-point interpolation curve. Certified curves may contribute to candidate
ranking while other keys remain unavailable. That ranking eligibility does not
complete a declared calibration population or authorize registry publication.

The [driver gates](../scripts/run_current_recalibration.py) require validation of
the declared acquisition scope. A finite 24-mapping sample may complete without
claiming global inventory coverage, but an arbitrary smaller certified subset
cannot replace those 24 requested mappings. Missing groups, one-point curves,
pending descriptions and failed validation remain explicit gaps. The
[finalizer](../scripts/finalize_recalibration.py) checks complete validated
services for its confirmed mappings, no service gaps, and composition status
`complete-observed` before registry promotion. Selector replay remains required.

The model worker reported 103 passing tests; main integration reported 54 passing
tests. These collections overlap and must not be added together. The 38 GPU
probe tests above are a separate bounded validation report, not evidence that
the installation comparison matrix has completed.

The v2 integration run passed 116 of 117 tests. The new seed preflight and
interruption regressions passed. The remaining failure is the existing
`test_compiled_product_preserves_asymmetric_axes_and_budget_coverage` assertion:
bounded `stratified_candidates(points, 24)` covers six of twelve expected
`local_stages` / `cross_twiddle` / layout combinations. The current
`calibration_space.py` and the
[pre-repair copy](../results/install_compare_repair_20261014/source_before/calibration_space.py)
have identical SHA256
`b168b7351868bd1d40916566d32f4d48964b1ce3636f644c2816eee45ea9be7f`,
and that copy reproduces the failure. This independent coverage issue was not
changed. The current `model` search uses the complete frontier with budget zero
for ordering, then scores remaining candidates; its periodic exploration still
uses that order. The failed 24-point assertion does not truncate its frontier
or govern the historical seed pool, and does not establish complete axis coverage.

## First recovery interruption and v2 repair

The v1 recovery ran on 2026-10-14 from 14:17 to 14:41 CST, about 24 minutes.
It described 21/24 templates and retained 412 correct `measured` stage records
plus one `contaminated` record. Foreign NCU PID `1588829` triggered the
post-measurement exclusivity check. The process subsequently entered search
and stopped with `mapping seed inputs changed`: its frozen seed registry source
was under `install_compare_20261013`, while the new output registry path was
under `install_compare_repair_20261014`. The previous status, logs, stage
checkpoints and tools are retained in
[`attempts/calibration1_failed`](../results/install_compare_repair_20261014/attempts/calibration1_failed).

V2 supplies `--seed-registry` explicitly as
`results/install_compare_20261013/initial/registry.json`, independently of the
destination registry. Seed preflight freezes or validates the snapshot before
GPU work. Source-path and explicit-input hash checks remain strict; the existing
[seed snapshot](../results/install_compare_repair_20261014/initial/mapping_seed_snapshot.json)
keeps SHA256
`a4942b858d38c9a529f1dc120b456c5763ab531404dda8ed50d4be726ba5d87d`.
An interrupted stage calibration now saves its checkpoint and exits with code
130 before search; the driver records `interrupted-calibration`.

The v2 restart preserves the repaired stage probe and GPU library. Compatibility
checks passed for the 412 clean stage measurements and 321 candidate records;
the contaminated record remains retained and is scheduled for remeasurement.
The conda copies of `calibrate_local_hardware.py` and `calibration_seeds.py` were
updated, and isolated installed imports and the new parameter check passed.

The v2 run progressed to 439 correct, exclusive stage records and 22/24 described
templates. All 412 earlier clean timing payloads were preserved, and the earlier
contaminated batch was successfully remeasured. At 16:51:48 CST, another process
(`1698471`, `tdflow/bin/python`) occupied the target GPU and the driver stopped
its own worker. The last saved checkpoint predates that interruption, all saved
measured rows carry successful correctness and exclusivity checks, and all frozen
campaign input hashes are unchanged. The original contaminated row remains
excluded and retained. The [review receipt](../results/install_compare_repair_20261014/interruption_review_calibration2.json)
records these checks. State is `paused-after-interruption-review`; an exclusive
GPU and the same driver with `--resume` are required to continue. No new waiting
window has been started, and the GPU comparison remains 0/66.

## Recovery binding

The recovery preserves 321 candidate measurement records and 23 incumbent
records after checking their whole-plan measurement identities. It reuses actual
timings and reranks under the repaired protocol; it does not claim the old search
selection trajectory remains unchanged. The v1 stage-probe change required fresh
stage measurements; v2 reuses compatible clean measurements from that probe.
The earlier stage checkpoint is retained under
[`old_stage_artifacts`](../results/install_compare_repair_20261014/old_stage_artifacts).
Its prior sidecars and all measurement records remain available in the
[original campaign](../results/install_compare_20261013).

The public conda `bin` copies of `calibrate_local_hardware.py`,
`stage_cost_model.py`, `stage_service_model.py`, `stage_service_calibration.py`,
`stage_schedule_native.py` and its C++ source, plus the repaired stage probe,
were updated. Their predecessors are preserved in
[`previous_installed_calibration.tar.gz`](../results/install_compare_repair_20261014/previous_installed_calibration.tar.gz).
The installed probe uses `$ORIGIN/../lib` and CUDA runtime paths; `ldd` resolved
the installed library, and isolated Python imports selected `native-cpp-double`.
The build library SHA256 remains
`6f590d064e9167f7ff9285ca4b91a9d0788dd73a6d018b7e0fd0948346d7e2d1`;
the installed library SHA256 remains
`a461c7d1bfa3e282b346b99cb990d842eb27dff91b0b470107ddc513f5c05a51`.

Conda registry activation now targets the recovery directory's
`finalized/registry.json` only after that file is published. The old hook is
preserved as
[`previous_registry_activation.sh`](../results/install_compare_repair_20261014/previous_registry_activation.sh).
The v1 CPU/GPU join follower was started as PID `1567316`; its successor was PID
`1691634`, following driver PID `1690709`. It ended without a GPU checkpoint
after the interruption. Follower PID `1749223` ended when the reviewed V2 search
was stopped to repair the incomplete stage scope. CPU data is
unchanged, and earlier join records remain under
[`previous_join`](../results/install_compare_repair_20261014/previous_join).

The [recovery driver](../results/install_compare_repair_20261014/run.py) was
first started in the background as PID `1566575`; the v2 driver was PID `1690709`.
After the 2026-10-14 16:51 CST interruption and checkpoint review, PID `1749222`
resumed with matching input hashes and the reviewed checkpoint, then was stopped
after two incomplete curves were identified. V3 driver PID `1775735` now resumes
through a stage-scope preflight. Its
[status](../results/install_compare_repair_20261014/status.json)
records the authoritative phase and status as execution
advances. The target is A100 80GB PCIe UUID
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`.

| Deliverable | Status at this snapshot |
| --- | --- |
| External CPU comparison | 83/83 timing rows retained |
| Finite stage scope | Passed: 562 clean records, 22 executable configurations, two audited provider rejections |
| Plan search | Running after stage validation |
| Final confirmation and selector release | Pending |
| GPU semantic comparison | 0/66 completed requests |

The remaining sequence is the declared finite calibration, final confirmation
and strict release checks, then the same 66 GPU requests. Calibration search
budget remains 32 and comparison public MEASURE budget remains 512. The
[population and timing contract](install_compare_20261013.md) and
[CPU supplement](cpu_external_baselines.md) retain their existing scope. No full
research-matrix completion or completed GPU performance comparison is claimed.

When the target GPU is exclusive, the compatible resume command is:

```bash
/home/wt/yes/envs/cubutterfly/bin/python -u \
  results/install_compare_repair_20261014/run.py --resume
```

Do not launch another copy while this driver is running. Recovery phase commands
and immutable artifact/source identities are recorded in its status file;
[`calibrate.log`](../results/install_compare_repair_20261014/calibrate.log) records
the calibration worker's progress.
