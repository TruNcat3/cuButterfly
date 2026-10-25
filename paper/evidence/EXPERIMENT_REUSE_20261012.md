# Experiment evidence reuse (2026-10-12)

Machine-audited draft; human scientific acceptance is not recorded. All numerical evidence, raw file SHA256, field locators, build identities and transformations are in [the JSON ledger](EXPERIMENT_REUSE_20261012.json). Recompute with `CUDA_VISIBLE_DEVICES='' python3 paper/evidence/recompute_experiment_reuse_20261012.py`. No GPU measurement is performed.

## Reuse boundaries

Historical method evidence remains usable under its named build. Recorded latest-campaign binaries are not the current working-tree build. Each raw identity is retained; binary compatibility must be checked before calling a measurement current. Do not merge E1 and NTT-extension observations, even where semantic requests coincide. Float error distributions/reference errors remain descriptive with accuracy review pending; integer/NTT exact-pass requires exact equality. No new threshold is imposed.

| Evidence group | Historical reuse | Recorded campaign | Remeasure / missing |
| --- | --- | --- | --- |
| operator comparison | RC2 named build; retain each competitor independently and losses | 48 rows in two cohorts; floating accuracy review pending | If claiming latest working-tree build, incompatible binary/semantic/hardware/timing identity requires affected rerun; matched GPU baselines for zeta/structured; word32 GPU-NTT; FWHT FP64 DAO; broader ranks/layouts/precision matrix |
| ablation 1: fixed component and disabled-component re-search | fixed layout/depth/pipeline controls and current normal-timing/profile diagnostics | component diagnostics retained with actual_changed_axes and underfilled status | equal-budget full versus disabled-component actual search and independent winner timing; cross-operator re-search envelope, actual avoidance of harmful mechanism, occupancy/L2 hit rate |
| ablation 2: core x organization | FFT native/cuFFTDx four cells on A10080 and A10040; E3 one-core cross-arithmetic local organization witness | FFT core x overlap only | same-core original/matched well-filled versus template organizations, controlled plus equal-budget searched comparisons; second replaceable core for FWHT/NTT/zeta/structured and complete cross table; retained generated historical E3 CUDA source |
| applications | FNO complete-call loss; QuaRot random-weight B1 near parity; separately pinned builds | filter v2 nine matched real-batch points; exploratory tolerance unassessed | affected application build/contract changes only; repaired QuaRot pending full calls; CKKS complete RNS bootstrap; accepted floating policy for filter; this-round FNO batch campaign |
| selection and costs | finite-pool E9 offline replay and planning/first-use costs with separate protocol | cache receipts and selected mapping identities in filter; do not imply rank1 calibration consumption | actual online equal-budget selection and consumed profile/cache compatible with current binary; E6 eligible prediction priors; calibrated E9 stage-service (zero covered priors); full install/tuning/amortization evidence |

## Operator groups

All 48 rows retained. Ratios below are descriptive for floating rows, not accepted accuracy-qualified wins. Cohort, operator and precision stay separate; competitors never share a mean.

| Cohort | Operator / precision | Rows / process trials | Comparator | Pairs | Wins | GM | Range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1-20261011-v4-complete | fft / fp32 | 6 / [1] | cufft | 6 | 4 | 1.102918 | 0.916667--1.314286 |
| E1-20261011-v4-complete | fft / fp32 | 6 / [1] | vkfft | 6 | 2 | 1.028800 | 0.736111--1.526012 |
| E1-20261011-v4-complete | fft / fp64 | 6 / [1] | cufft | 6 | 6 | 1.353343 | 1.079710--1.780303 |
| E1-20261011-v4-complete | fft / fp64 | 6 / [1] | vkfft | 6 | 6 | 1.324918 | 1.100478--1.522727 |
| E1-20261011-v4-complete | fwht / fp32 | 6 / [1] | daofht | 6 | 6 | 1.667277 | 1.239583--2.634328 |
| E1-20261011-v4-complete | fwht / fp64 | 2 / [1] | no matched external time | NA | NA | NA | NA |
| E1-20261011-v4-complete | ntt / word32 | 4 / [1] | no matched external time | NA | NA | NA | NA |
| E1-20261011-v4-complete | ntt / word64 | 4 / [1] | gpu-ntt | 4 | 4 | 1.197216 | 1.033088--1.368421 |
| E1-20261011-v4-complete | structured-2x2 / fp32 | 2 / [1] | no matched external time | NA | NA | NA | NA |
| E1-20261011-v4-complete | subset-zeta / uint32 | 3 / [1] | no matched external time | NA | NA | NA | NA |
| E1-20261011-v4-complete | superset-zeta / uint32 | 3 / [1] | no matched external time | NA | NA | NA | NA |
| NTT-20261012-extension-snapshot | ntt / word64 | 12 / [3] | gpu-ntt | 12 | 12 | 1.285470 | 1.029412--1.614771 |

## Batch FFT filtering: new nine-point cohort

V2 is a new application cohort, not nine additional operator rows. Each point uses independent images in a real batch and a shared filter recomputed once per complete call. Native competitor padding differs as explicitly recorded. Planning/selection are outside warm call timing. Tolerance remains unassessed, so retain exploratory status and losses.

| Input | B | cuB ms/batch | CuPy ms/batch | CuPy/cuB | cuB ms/image |
| --- | --- | --- | --- | --- | --- |
| image1024_filter31 | 1 | 0.335872 | 0.379648 | 1.130335 | 0.335872 |
| image1024_filter31_batch16 | 16 | 4.120166 | 1.673574 | 0.406191 | 0.257510 |
| image1024_filter31_batch4 | 4 | 1.151181 | 0.429978 | 0.373510 | 0.287795 |
| image2048_filter31 | 1 | 1.453978 | 0.556544 | 0.382773 | 1.453978 |
| image2048_filter31_batch16 | 16 | 16.573491 | 6.483559 | 0.391201 | 1.035843 |
| image2048_filter31_batch4 | 4 | 4.438016 | 1.804134 | 0.406518 | 1.109504 |
| image512_filter31 | 1 | 0.246630 | 0.357222 | 1.448412 | 0.246630 |
| image512_filter31_batch16 | 16 | 0.883558 | 0.392038 | 0.443704 | 0.055222 |
| image512_filter31_batch4 | 4 | 0.250573 | 0.379750 | 1.515529 | 0.062643 |

Batch 1: 2/3 descriptive wins; GM 0.855750, range 0.382773--1.448412.

Batch 4: 1/3 descriptive wins; GM 0.612795, range 0.373510--1.515529.

Batch 16: 0/3 descriptive wins; GM 0.413118, range 0.391201--0.443704.

## Core and organization audit

Raw factorial CSV medians reproduce the organization gains. A10040 summary core-speedup fields use cuFFTDx/native, opposite to A10080 and the manuscript definition. Consequently the manuscript's claim that imported-core execution slows on A10040 is reversed. Use the recomputed native/cuFFTDx ratios below, not the inverse summary values.

| Historical GPU | Native stage/resident GM | cuFFTDx stage/resident GM | Native/cuFFTDx stage GM | Native/cuFFTDx resident GM |
| --- | --- | --- | --- | --- |
| A10080 | 1.217016 | 1.220723 | 1.042327 | 1.045502 |
| A10040 | 1.225558 | 1.231062 | 1.038308 | 1.042970 |

E3 fixes the shared-iterative backend, physical [8,4] groups, 128 threads, writer-aligned mapping and paired user semantics. The requested change is local_stage_partitions from absent/empty to [[2,2,2,2],[2,2]], which changes local execution schedule and compiled module. It is one core in each arithmetic family and a local organization witness, not a two-core transfer cross table or a re-search experiment. Source temporary CUDA files were removed; compiled modules and manifests remain hash-bound. Avoid claiming isolated HBM traffic or a strictly identical compiled kernel from these records.

- Historical E3 fwht N=4096 B=1: leaf0/leaf2 1.136713 (0.007577600/0.006666240 ms); JSON /e3_audit/results/rows.
- Historical E3 fwht N=4096 B=1024: leaf0/leaf2 1.301985 (0.349306881/0.268288016 ms); JSON /e3_audit/results/rows.
- Historical E3 ntt N=4096 B=1: leaf0/leaf2 1.072614 (0.010588160/0.009871360 ms); JSON /e3_audit/results/rows.
- Historical E3 ntt N=4096 B=1024: leaf0/leaf2 1.090576 (0.600934386/0.551024616 ms); JSON /e3_audit/results/rows.

## Existing manuscript number sources

- RC2 operator main figure: frozen sibling repository integrated_repairs_20261001_replay_fix1/a100-80gb suite summaries; generated rc2_unified_absolute_throughput.csv and renderer are presentation derivatives. Existing speedup_summary mixes competitors within FFT-FP32 and FWHT groups; use per-competitor RC2 snapshot/raw suites for numerical claims. Old RC2 speed is not the latest library speed.
- FFT fixed-core table and GM: core_dataflow_ablation/measurements.csv; A10040 replication has its own CSV and protocol. Correct the ratio direction before reusing its core claim.
- E3 cross-arithmetic ratios: e3_cross_operator/summary.json and a100-80gb/journal.json, with compilation.json/module manifests for actual axes.
- Layout, depth, pipeline and finite-pool selection claims: paper/generated/current_results_summary.json, its provenance control report and named frozen source. E9 stage-service has zero covered priors and is structural ordering; offline replay is not online tuning cost or calibrated prediction validation.
- Historical FNO/QuaRot and first-use costs: application_snapshot_20261009.json and APPLICATION_RECORDS_20261009.md. FNO slowdown and QuaRot random-weight near-parity must remain. CKKS compatibility probe is not complete bootstrap timing.

## Safe prose conclusions

The historical FFT factorial demonstrates organization gains with both tested cores, with replication on the two named A100 devices. The bounded FWHT/word64 NTT control extends local organization evidence to two arithmetic families, while a second core per family and equal-budget organization search remain open. Current operator observations cover all six families and retain unsupported external contracts and negative cells; floating correctness remains under review. The new batch-filter records show strong workload dependence and losses at larger batches, so operator gains cannot establish end-to-end benefit. Fixed-component diagnostics and finite-pool replay do not replace a cross-operator disabled-component re-search experiment.

## Protocol and older model evidence

The primary out-of-place matrix records the same graph protocol for every measured backend: ten warmup executes, twenty complete executes per captured graph, seven CUDA-event replays normalized by twenty, then the sample median. Required baseline scaling and order/layout adapters are inside execute. E1 has one process observation; the NTT extension has three. This statement is bounded to the actual measured rows; in-place reset timing, old RC2 and profiler studies have their own protocols.

The historical sixteen-candidate E9 replay remains reusable as finite-pool policy evidence. Its stage-service policy has zero covered priors and uses structural fallback; invocation wall time is not online tuning cost. The historical twenty-seven-plan composition study remains reusable as a diagnostic with all correctness trials retained, including the large FWHT prediction error and unqualified overlap assumption. The RC2 local holdout study retains 561 covered checks, median relative error 1.1733% and P90 6.0294%. These numbers are /model/validation fields in the hash-bound RC2 calibration checkpoint. Local interpolation and full-plan composition are distinct prediction tasks; neither result is a current-build blanket model claim.

E3 lowering audit: inspected current render_shared selects shared_iterative_kernel for leaf0 and shared_resident_kernel for leaf2. The latter keeps four values per two-stage leaf in registers, changing shared access, barriers, thread work and wrapper code. Both retain one launch per physical group, the group grid, threads and inter-group HBM boundaries. Do not claim per-stage HBM writeback in leaf0. Current source explains the implementation pattern but is not asserted to be the missing historical generated CUDA source.
