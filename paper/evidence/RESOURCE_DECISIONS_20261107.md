# Resource characterization, selected organization and independent confirmation

This CPU audit connects the frozen November 3 resource descriptors, independently
measured service curves, complete-plan confirmations and public registry replay.
It adds no GPU measurements, model fitting, search ranking competition or global
optimality claim. The [exported JSON](../generated/resource_decisions_20261107.json)
contains source SHA256 hashes, complete requests/configurations, group resources,
service-key links and training/validation samples. The compact
[LaTeX table](../generated/resource_decisions_20261107.tex) can be placed in a
single or compact two-column layout; `Retro.` means retrospective composition
validation, not a prediction certified as frozen before execution.

## Scope correction

The 34-key / 117-clean-probe / 68-group-holdout / 15-plan-holdout cohort has
**two arithmetic families and three requests**: FFT FP64 N=2^20/B16, FWHT FP32
N=2^16/B16 and FWHT FP32 N=2^15/B32. It contains no NTT request. The model's
3/3 configuration-winner agreement must not be described as FFT/FWHT/NTT
three-family prediction accuracy.

The source boundary is
`results/a100_closure_20261103/final_acceptance/`. Core identities, hardware,
compiler and binary/JIT identities are retained in the exported
`measurement_identity`; `serialized_finalist_mapping` and `frozen_confirmation_command` retain replay inputs. Before a new confirmation, verify binary/library/JIT identity; changing the build requires a separate provenance label. This is the A100 80GB PCIe direct-launch cohort, not the
A100 40GB M1 control experiment or the later FP16/QuaRot build. All three requests
are forward, unnormalized, native accumulation, natural order, unit element
stride and out of place. FFT stores/operates on complex FP64 (16 bytes/value),
FWHT on real FP32 (4 bytes/value). Compilation, reference checks, allocation and
selection are outside ordinary timing. Confirmations use three independent
processes, 30 warmups and 50 repeats; CUDA events enclose host launch loops.
These are not CUDA Graph latency predictions.

## What each evidence layer supplies

| Layer | Retained evidence and interpretation |
| --- | --- |
| Dependency derivation | For G physical execution groups, semantic I/O is 2BNs bytes and logical group-interface reads/writes are 2GBNs bytes. A group spanning k radix-2 stages has BN/2^k closed regions, each with 2^k values and 2^k s value bytes. This abstract region count differs from CTA packing. These are logical value footprints, not measured HBM traffic or aggregate live storage. |
| Compiled resources and launch resolution | `tuning/plan_descriptors.json` and confirmed `execution_groups_json` retain actual kernels, threads, grid CTAs, registers/thread, dynamic/shared bytes and known/unknown flags. These are separate from the dependency footprint. Registers alone do not establish occupancy or spills. |
| Independently measured service | `stage/final_scope/service_profile.json` has 34 exact resource/semantic keys, derived from 117 clean probes; contaminated acquisition is excluded by the frozen scope. The exporter links each selected physical group by arithmetic, precision, size, stage range, core, local partition, threads and compiled resources to exactly one service curve. Curve training and independent validation samples stay separate. |
| Model composition validation | `refit/cost_model.json` supplies 68 independent group holdout observations and 15 complete-plan comparisons. CPU recomputation gives median relative errors 0.0056499408 and 0.0042936008. These validate bounded service/composition consistency, not all dependency-derived mechanisms or arbitrary hardware. |
| Independent complete-plan confirmation | `tuning/operator_result.json` retains three process samples per candidate. The exporter recomputes medians and checks them against the model's observed values, rather than substituting service-probe timing for complete plans. |
| Public use | `public/auto/selections.json` records DEFAULT replay of measured registry mappings. A measured registry and a model are distinct selection mechanisms; replay does not establish that the final refit selected the winner before its measurement. |

## Three concrete decisions

| Request | Selected core / actual execution organization | Dependency footprint / actual compiled resources | Retrospective model / independently confirmed ms |
| --- | --- | --- | --- |
| FFT FP64 [1048576], B16 | register-tile prefix with local [4,6] and table roots, cuFFTDx-block suffix; physical [10,10], threads256/128 | Two sets of16384 logical 1024-value regions, 16KiB values/region; actual grid4096/8192, shared64/32KiB, registers180/96 | 1.065062433 / 1.072701 |
| FWHT FP32 [65536], B16 | hierarchical scalar radix2, physical [10,1,1,1,1,1,1], threads256 | First group1024 logical regions×1024 values (4KiB); six groups each524288 regions×2 values. Actual grid1024 then2048 each, shared4KiB then0, registers14 then21 | 0.044953599 / 0.044872 |
| FWHT FP32 [32768], B32 | temporal-tile scalar radix2, physical [15], threads256 | 32 closed regions×32768 values (128KiB values/region); actual grid32, shared32KiB, registers255. The value footprint is not all shared storage. | 0.018585600 / 0.018391 |

The device descriptor limits per-block registers to 65,536 slots and shared memory to 166,912 bytes. Unrounded register demand is 46,080/12,288 slots for FFT, 3,584/5,376 for the FWHT N16 kernels, and 65,280 for FWHT N15. These necessary checks pass, but allocation granularity, other occupancy limits and unknown component fields prevent treating them as complete legality or residency proofs. Dependency counts alone do not choose a winner; the independently measured service curves supply timing information. The JSON also retains original hardware-rate priors and their provenance separately from the new services.

The precise model values and mappings are authoritative in the JSON; rounded
values above are presentation only. Logical interface traffic is 1GiB, 56MiB
and 8MiB, respectively, excluding coefficient/internal accesses. No measured
DRAM-byte figure or bandwidth bottleneck is inferred from those values. The
large FFT's register prefix and cuFFTDx suffix have different core/resource
contracts; these selected organizations are not a same-core causal ablation.

The final FWHT N15 model record predicts an incumbent name while the measured
winner has a search name. The producer reports a match based on identical full
configuration, not record-name equality. Their measured latencies differ by
about 0.78%; this does not prove zero noise or a globally optimal choice.

## Prediction chronology and what remains unproven

The final refit file is a persisted prediction artifact, but persistence alone
is not proof of execution-before-confirmation chronology. The frozen harness
`run.py` calls `run_operator_calibration`, writes `operator_result.json`, and
then invokes `stage_cost_model.py --operator-calibration operator_calibration.json`.
The final model's `static_ranking_identity` explicitly cites 15 confirmed records
and 45 samples. Its final-scope services supersede an initially incomplete
service model. No immutable receipt proving that these exact final-refit
winner predictions were committed before the three independent confirmation
processes was identified in this bounded chain.

The fitting and prediction paths must be distinguished. In
`scripts/stage_cost_model.py:415–508`, complete-plan totals fit nonnegative
legacy/fallback overhead and service factors; `report` refits those factors
while excluding each held workload. But `_predict_measured:675–690` uses the
independently measured curve duration directly for covered groups, with zero
extra overhead. Only missing/out-of-range groups consume the plan-fitted
parameters. `report:940` excludes fallback predictions from covered validation.
All 15 complete-plan validation rows here are covered. For each displayed
request, this exporter recomputes the frozen prediction and repeats it with
`parameters={}`: all three remain exactly unchanged, and
`whole_plan_fallback_used=false`. Thus these times are not complete-plan-target
fits. The operator results also supply validation/ranking diagnostics and
measured registry choices. This CPU check performs prediction only, with no
model refitting or service-curve updates.

Accordingly this export labels the final-model results **retrospectively evaluated
independent-service predictions**. The absence of an earlier receipt is a
chronology gap, not evidence that the held target latency fitted these covered
predictions.
Independently acquired group service training/holdouts and independently timed
complete plans remain valid observations, but they do not by themselves certify
a prospective resource-characterization→choice→confirmation experiment. The
3/3 statement retains the producer's finite configuration-agreement meaning;
it is not used as a search-method ranking result. No holdout observation is
refit here, and no post hoc winner is inserted into a missing prediction field.

## NTT extension boundary

The existing M1 historical request NTT word64 [1048576]/B16, scalar radix2/Shoup,
q=576460756061519873, has internal organization measurements on A10040. Its
`NARRATIVE_SCALING_RESULTS_20261027.md` explicitly describes hand-selected
publication/residency controls, not searched conventional baselines or prospective
selection. It cannot fill a missing Nov3 A10080 service/choice/confirmation row.
The JSON `extensions` entry retains the absent NTT stage-service prediction chain. The completed separate resource-constrained measured-selection experiment below now supplies a prospective NTT organization decision example; it supplies no numeric cost prediction and does not alter the Nov3 scope.

Reuse is limited to compatible exact semantics, arithmetic/core, compiled
resource signature, device/build/timing identity and measured service ranges.
Overlap/concurrency composition, Graph predictions, arbitrary sizes and another
platform remain unvalidated. Unknown compiler/shared-component fields retain
their original unknown flags. The characterization closure contract and CLAIMS
P3 can cite this bounded chain with those qualifications; it does not complete
the broader cross-family prospective-selection claim.

## Reproduction and checks

Run `python paper/tools/export_resource_decisions.py` for a CPU-only rebuild.
`tests/test_resource_decisions.py` checks formula separation, exact service links,
68+15 error recomputations, three-process medians, retained chronology/NTT gaps
fallback-parameter invariance, and generated-output consistency. Source hashes are regenerated from frozen
files and stored in the JSON. No original result or production runner is edited.

## Completed prospective resource-constrained measured-selection extension

A separate frozen experiment has now completed at
`results/nonfft_endpoint_control_20261107`. Its accepted freeze SHA256 is
`711953e9327fbccbc5a65750de98c1c809efe5d72552053202e6023a1b32104c`;
screen `1690b26194f4a4a6bc700e5ca434a5cdbf9bab7124dfd51d34288e2f8e3fe927`;
winners `026da323e8e23bbea4f7f067b0e0c7ebdf84c0805fcdd86b3f8ee58139de1a7e`.
The exporter checks the frozen evidence index for byte counts/SHA256, root and
clean-release copies, three distinct process IDs, each holdout's freeze/winner
hash references, and ordered guard completion receipts. The frozen runner hash
matches its stored source identity: it writes the resource/candidate freeze
before the screen, publishes winners after the screen, then launches the three
holdouts. The second holdout reverses unique winner order. This supports the
prospective resource→candidate→screen→confirmation sequence; it does not turn
screen times into independent performance evidence or numeric forecasts.

The source-derived resource expectations were frozen with the candidate pool.
FWHT FP32 [65536]/B16 uses two stage bands with 16MiB logical value-path
traffic, two launches and column grouping 1/2/4. The prefix has 4096 CTAs and the
suffix 4096/2048/1024; source shared-memory allocation lower bounds are
1/2/4KiB. Each radix2/radix4 leaf has two thread settings 128/256, yielding six
accepted candidates. The conventional side retains both column 1 settings;
extended includes the same endpoints and all six. These are shared pools,
not six independent attempts per side or a claim of globally equal search spaces.

NTT word64 [1048576]/B16, q=576460756061519873, natural forward/unnormalized,
uses two stage bands with 512MiB logical value-path traffic and one
cooperative persistent producer/consumer launch. Both leaves retain data_time
1/2/4 at 256 threads, three accepted candidates each; the conventional side
has dt1 and the extended side includes it. The radix4 shared lower bound is
32800 bytes; warp-radix2 has a recorded lower bound of 0, which does not prove
zero total compiler storage. Occupancy and target_ctas_per_sm cap launch grids:
`batch/data_time` is not a valid grid formula. These are lower bounds and logical
counts, not full compiler legality proofs, measured shared residency or HBM
bytes. Accepted execution and correctness records supply separate checks.

Terminology correction (November 8): the frozen `logical_regions=2` field counts
stage bands, not dependency-closed regions. Each FWHT band contains 4096
256-value regions; each NTT band contains 16384 1024-value regions. In these
two-band factorizations, one transform is one dependency boundary packet.
NTT `data_time` groups independent transforms for traversal and does not
merge their dependency packets or imply simultaneous residency. The frozen
source fields and measurements remain unchanged. See the
[dependency-to-decision walkthrough](DECISION_WALKTHROUGH_20261108.md).

The compact [new table](../generated/resource_measured_selection_20261107.tex)
uses only independent holdouts. Values below are medians of three process
times; ratios are medians of process-paired ratios.

| Fixed leaf | Conventional winner | Extended winner | Conventional / extended ms | Paired ratio |
| --- | --- | --- | ---: | ---: |
| FWHT radix2 | T128/cols1 | T256/cols4 | 0.060447 / 0.045875 | 1.317646 |
| FWHT radix4 | T128/cols1 | T256/cols4 | 0.059894 / 0.043878 | 1.365126 |
| NTT homogeneous-radix4 | T256/dt1 | identical dt1 | 1.585398 / 1.585398 | 1 by identity |
| NTT homogeneous-warp-radix2 | T256/dt1 | identical dt1 | 3.706778 / 3.706778 | 1 by identity |

For both NTT leaves, screen selected the original dt1 endpoint on both sides.
The identical winner is deduplicated and measured once per process: two equal
columns are not two independent timing arms, and 1× is no claimed gain. Radix4
dt1 uses the mature specialized 10×10 wrapper while dt>1 uses the homogeneous
packet-loop wrapper; same arithmetic leaf does not isolate traversal alone.
The warp leaf retains the same wrapper family across dt. FWHT uses fixed leaves
within each row; the radix2/radix4 cross-row difference is a core comparison,
not pure organization attribution.

All holdouts use 100 warmups/100 repeats and full execution CUDA-event timing,
excluding setup, H2D/D2H, CPU correctness and public Plan oracle. Host submission
gaps may remain; this is neither Graph replay nor arithmetic-only time. The
A10080 UUID here is `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`, different from
the Nov3 device UUID. Library SHA256 is `775a85c8f76e44bbfee886fd90489e7c309bd097b60e968183e446aa006177b2`.
The full-batch integer checks are exact; FWHT retains existing C++ correctness
semantics and separately saved full-output FP32 CPU-oracle error arrays without
adding an accuracy threshold. The original numerical policy stays in the JSON.

This extension closes **resource bounds→bounded accepted candidates→measured
selection→three independent confirmations** for FWHT and NTT requests. It does
not close an NTT independent stage-service cost-prediction chain. Every new
`cost_prediction_ms` is null. The Nov3 retrospective independent-service table
remains FFT plus two FWHT requests, with its original identities and chronology
limits. No new fitting, GPU work, sample expansion or global-optimum claim was
introduced by this export.

## Portable artifact validation

The test suite separates raw-evidence recomputation from frozen-artifact checks.
When any raw source is absent, the recomputation test explicitly skips and states
that the skip is not measurement completion. A portable test still checks the
frozen JSON's request scope, sample medians, relative errors, dependency formulas,
NTT null forecasts/identical mappings, paired ratios and both generated tables,
without importing the production predictor or accessing results. A temporary
checkout containing only these artifacts was tested: 2 passed, 1 skipped.
The source host with full evidence reports 3 passed. Portable consistency checks
do not replace raw source-hash verification or independent measurement.

The resource table now uses FFT64/FWHT32 with N/B, G, maximum SMEM in KiB,
and four decimal places for millisecond predictions/confirmations. A standalone
LaTeX `small` box measured 206.27pt, below a 252pt column. The prospective table
uses four columns with microsecond times to three decimals and `same` for NTT's
identical extended mapping. Ratios remain medians of process-paired ratios,
not ratios of median times; full precision stays in the JSON.
