# Platform boundary and retest repair

This is the preceding repair record. Its calibration and comparison waiter
were paused for the [engineering review](engineering_review_20261007.md).
The diagnostic timings below belong to the named earlier build, not the
reviewed revision.

Target: NVIDIA A100 80GB PCIe, UUID
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`.

## Confirmed causes

The old bit-reversed NTT comparison was not a controlled output-order
ablation. Its natural-order case used N=65536, B=64, whereas bit-reversed
used N=1048576, B=4. Both used uint64 storage and modulus
576460756061519873.

For the latter workload, the old AUTO path missed the exact measured registry
and selected the first legal portable candidate: twenty single-stage groups,
32 threads per CTA. The native CompactStage implementation already supports
bit-reversed terminal stores without a separate naturalization pass, but was
absent from the portable inventory and reached through the legacy V100 table
only. This is a candidate/selection integration defect.

A diagnostic screening on the same old binary measured AUTO at 31.136869 ms,
explicit CompactStage at 0.209039 ms, and explicit shared 10+10 at 0.636334 ms.
All passed reference checking. Raw records are under
`results/ntt_order_repair_20261007/prechange/`. These single-screening numbers
are diagnostic evidence, not the final repeated external comparison.

The 18 unavailable uint32 xD NTT timing rows had a separate harness defect.
On the unchanged old binary, 128x128, B=1, modulus 257, MEASURE+XD succeeds
with ordinary loading or `LD_LIBRARY_PATH` alone, and fails when the runner
additionally forces `LD_PRELOAD=libcuntt.so`. Removing this injection fixes the
reproducer. This loading-order effect must not be reported as missing NTT
mathematics or a demonstrated kernel resource limitation. Oversized resident
candidates may still be legitimately rejected; composite lowering remains a
candidate.

## Boundaries now enforced

- `src/mapping_selector.cpp` contains the common exact-registry/portable
  selection flow. Legacy model-specific tables live in
  `src/platforms/legacy_selector.cpp` and supply unmeasured mapping priors.
- Optional JSON platform rules enter the same public inventory through
  `src/platform_tuning.hpp`. Exact device/SM/memory matching, semantic filters,
  ordinary mapping validation and cache identities apply. Rules do not delete
  candidates, replace requested semantics, or confer measured status.
- Legacy NTT default launch preferences live in
  `src/platforms/ntt_defaults.hpp`; kernel/resource legality remains in the
  implementation. Public C API axis construction no longer chooses a backend
  from a size-only heuristic.
- Native CompactStage eligibility is shared between validation and inventory.
  Generic unmeasured NTT fallback prioritizes balanced groups derived from
  available shared memory, retaining small groups for measured search. This
  fallback is a heuristic, not a calibrated prediction or optimality proof.
- Installation inputs now include representative inverse and bit-reversed
  cells. The finite manifest does not cover every precision/direction/layout/
  length/batch combination. Exact misses remain explicit.

See [platform rule interface](platform_tuning.md) for extension instructions.
Historical V100-only offline analysis scripts are not the modern installation
stage model. ISA gates and actual resource limits remain legitimate hardware
capabilities; they must not be removed as if they were performance preferences.

## Retest integrity

Report-directory creation, low-precision lowering choices, per-case failure
continuation, and external timing metadata have been repaired. Large analytic
impulse checks now validate every element using bounded temporary storage;
the old full-output subtraction/absolute-value allocations caused the
80GB large-FFT checker failure even though the native benchmark ran.

Changing the runtime invalidates old exact-build calibration records. The
finalizer supports `--revalidate-mappings` to import mapping descriptions only,
then remeasure plans and physical services in a fresh output directory. Old
timings and correctness claims are not transferred. Additional verified CSVs
can supply descriptions via `--additional-mappings-csv`. Stage-service gaps
and model generalization warnings remain visible.

Build and focused-validation artifacts are stored in
`results/ntt_order_repair_20261007/`. Final timing and coverage results must be
reported with their matching build and fresh calibration identity.

## Focused validation

The repaired runtime fingerprint is
`2de9f8963871edd3a36abdc6989ee01c00e8d83c1b1f69ffea22d637fa08dc33:research`.
The final incremental build passed. CTest passed 21/21 tests in 69.85 seconds,
including the 18-case uint32 xD MEASURE plan-creation regression and the NTT
word-width/direction/order checks. Focused Python tests passed 63 tests and
22 subtests; the final loader-precedence check adds a passing regression to
the xD harness tests.

Same-target diagnostic screenings, without a measured registry, gave:

| NTT workload, bit-reversed output | Old AUTO ms | Repaired AUTO ms | Repaired mapping |
| --- | ---: | ---: | --- |
| N=2^20, B=4, uint64, forward | 31.136869 | 0.208957 | CompactStage native terminal order |
| N=2^18, B=16, uint64, forward | 28.023346 | 0.543898 | SharedIterative 9+9 |
| N=2^12, B=128, uint32, inverse | 2.379673 | 0.047821 | SharedIterative 12 |

All three passed reference checks of two transforms in the actual timed
batch. These before/after screenings are not repeated external-library
comparisons. The 64-bit modulus is 576460756061519873; the 32-bit modulus is
998244353. CSVs record their complete mappings, timing counts and identities.
The repaired selector correctly reports these uncalibrated defaults as
`unmeasured-feasible`.

The repaired FP16-storage/FP32-accumulation logN=12 preflight passed. The
previously failing FFT FP32 N=2^23, B=256 large checker also passed, validating
all 2,147,483,648 complex outputs in 2048 bounded blocks with zero impulse
reference error. This is analytic-input correctness coverage, not a random
oracle for that entire allocation.

## Fresh campaign

The preceding finite mapping confirmation was started under
`results/local_hardware_profile_a100_gpu1_repaired_20261007/`: 59 prior mapping
descriptions plus five additional NTT descriptions, each measured anew with
three trials, 100 warmups and 100 repeats. The 64 mapping confirmations were
retained when the job was paused during physical-stage calibration. It has
not released a completed registry for the reviewed revision.

`results/recalibrated_campaign_20261007_repair/manifest.json` freezes the new
135-case rank-1, 528-row xD coverage, 264-row xD scale and 113-row contract-gap
campaign. These overlapping populations must not be summed as independent
workloads. Its waiter was paused before comparison. The queue state and logs,
not this note, are authoritative for live progress.
The runner stops on foreign GPU contention and does not renew the expired
idle-GPU waiting budget. Old campaign files remain unchanged.
