# Parameterization and recalibration retest

The first campaign completed with harness and selection failures. The
[platform-boundary repair](platform_boundary_repair_20261007.md) records the
confirmed causes, focused regression results and the fresh repair campaign.
Use its new build/calibration identities for current measurements; do not
merge them with the first campaign's timings.

This campaign uses the current `build-e13-cufftdx` build, with both cuFFTDx
options enabled, on A100 80GB PCIe
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`. It is a new measurement campaign,
not a merge of historical timing results. The package version remains 0.9.1;
the campaign manifest records the exact dirty-source and shared-library hashes.

## Engineering changes

- Rank-2 FFT shortcuts are explicit policies with configurable candidates and
  length bounds. Default selection consults the hardware mapping registry.
  Measurement mode actually measures candidates, validates legality, and
  caches the winner under the effective policy.
- xD promotion margins apply consistently to real and complex paths and their
  cache identities. Local-core eligibility uses the core capability query.
- Stage-cost prediction reuses bounded indices without changing the ranking
  equation. The focused replay compared 249 stage records with the previous
  implementation and obtained identical predictions.
- Stage acquisition and search have separate, explicit budgets. Sampling
  omissions and plan-only projection must not be described as full migration
  qualification. Shared-library hashes belong to measurement identity.

Physical core length/resource restrictions still apply. A configurable search
policy does not create an unavailable kernel or establish global optimality.

## Execution order

The initial search is saved under
`results/local_hardware_profile_a100_gpu1_current_20261006/`. Its 14 workload
cells use 16 stage-acquisition seeds, one mapping seed and two additional
search candidates per cell. These are exploratory screening budgets, not the
comprehensive comparison population. Interrupted earlier, wider searches were
preserved separately and are not current final performance results.

After search, confirm the measured candidates with repeated timing and fill
their actual physical-stage services through the existing acceptance engine:

```bash
export CUDA_VISIBLE_DEVICES=GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02
export CUBUTTERFLY_COMPILE_MODE=research
/home/wt/yes/envs/cubutterfly/bin/python scripts/finalize_recalibration.py \
  --build-dir build-e13-cufftdx \
  --input-profile results/local_hardware_profile_a100_gpu1_current_20261006 \
  --output-dir results/local_hardware_profile_a100_gpu1_confirmed_20261007 \
  --trials 3 --warmup 100 --repeat 100 --resume
```

This creates a separate registry, model and selector-replay report. The raw
screening profile is preserved. Actual mappings and runtime fingerprints must
match during confirmation; building or relinking while measuring is forbidden.
The finalizer qualifies only this finite confirmed population and explicitly
retains missing services and cost-model warnings.
It releases the final registry only after selector replay succeeds. A journal
without its matching measurement identity cannot be resumed.

Regenerate the campaign manifest after the final build and script changes,
then execute it with the confirmed profile:

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/prepare_recalibrated_campaign.py
/home/wt/yes/envs/cubutterfly/bin/python scripts/run_recalibrated_campaign.py \
  --manifest results/recalibrated_campaign_20261006/manifest.json \
  --gpu-uuid GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
  --calibration-manifest results/local_hardware_profile_a100_gpu1_confirmed_20261007/calibration_manifest.json
/home/wt/yes/envs/cubutterfly/bin/python scripts/summarize_recalibrated_campaign.py \
  --campaign results/recalibrated_campaign_20261006
```

Use `--resume` on the queue after an interruption. Its journal validates
manifest, library, calibration and target identities before retaining earlier
steps. It must not fall back to the global registry. GPU contention invalidates
the affected step; no foreign process is stopped.

## Evidence and reporting

The final-build CTest run passed 20/20 checks in 82 seconds, including C-API,
NTT, xD FFT lowerings, mapping and public planner checks. The log is
`results/recalibrated_campaign_20261006/correctness_ctest.log`. This verifies
the affected implementation; it does not replace the declared performance or
public semantic matrix.

The executable population and exact runner commands are in the campaign
manifest and [campaign document](current_retest_campaign_20261006.md). They
include public coverage gaps, xD shapes, old large-N/stride contracts rerun
on the current build, and available external baseline adapters. Alias rows
do not count as independent arithmetic operators.

| Population | Requested scope |
| --- | ---: |
| Rank-1 comparison cases | 135 |
| xD coverage rows, including compatibility aliases | 528 |
| xD scale rows, including compatibility aliases | 264 |
| Public contract gap rows | 113 |

These populations overlap and must not be added into a count of unique
workloads. The four unsupported low-precision real-FFT contracts remain
explicit negative tests. The large-memory phase has a 64 GiB IO/workspace
allowance on the target A100 80GB; actual allocation failures remain visible.
Rank-1 NTT comparisons use the public automatic selector, preserving modulus,
word width, direction and output-order semantics. An exact registry miss is
still a miss; it does not establish that a workload was searched.
The comprehensive runner also checks up to two transforms within each actual
timed cuButterfly/cuNTT batch. This verifies the measured mapping rather than
only the potentially different plan selected for a small preflight batch.
Compilation, reference evaluation and verification stay outside kernel timing.

`tables/performance.md`, `tables/performance_summary.csv` and
`tables/performance_cases.csv` contain fresh paired results. Keep population,
precision, direction and reference library separate. Speedup is reference
latency divided by cuButterfly latency. Internal controls, external libraries,
vendor fallbacks, unsupported contracts, memory limits and unverified rows
must remain distinguishable. Missing measurements are not zero-latency values
and are not wins. A selected finite population is not proof that every
possible butterfly workload or hardware mapping has been optimized.
