# Integrated repair acceptance: proposed 0.9.1-rc.2

Plan added on 2026-09-30 at the user's request. **Status: planned, awaiting
source integration and a new build.** This is not a release or a GPU pass.
The machine-readable inventory is [integrated_acceptance_rc2.json](integrated_acceptance_rc2.json).
The local input audit and reserved phase queue live in
`results/integrated_acceptance_20260930/`.

## Why a new campaign is needed

The existing rc.1 acceptance predates the FFT pipeline dev.5.1, NTT boundary,
and FWHT full-state fixes. The September 30 recovery controller runs an older
hybrid-order campaign and a separate NTT boundary comparison. Neither is
acceptance of all these fixes in one library. The new campaign must build
one integrated source revision, then use its binaries and JIT templates for
every operator. Historical timings remain evidence of their named builds.

The proposed release identifier is `0.9.1-rc.2`. Do not change any existing
worktree's VERSION or relabel old measurements. Create a separate integration
worktree from rc.1 commit `796a152b809cdc1bbdbab0eaa97d11677c483ea2` and record
the actual patch provenance, resolved source hashes and final build hashes.

## Required repair inventory

| Input | Required integration | Existing evidence; still required in rc.2 |
| --- | --- | --- |
| rc.1 integrated source | CT/GS, output order, canonical layouts, FP64 direct paths, stage/ranking and descriptor fixes | Keep existing tests; rerun affected correctness and same-build acceptance |
| FFT `pipeline-dev.5.1` | Native/heterogeneous factor cores, grouped scheduling and overlap, byte-aware shared exchange, legal EPT/thread projection, public selection and the ten mapping-only weak-region seeds | dev.5 has 36/36 per A100; later focused 80GB evidence does not establish dev.5.1 full-matrix or 40GB acceptance |
| NTT `ntt-boundary.1` | Coset/negacyclic input/output factors, inverse scaling, both input-fold choices in candidate enumeration, public API and JIT identity | 21-cell focused study and separate full external comparison; remeasure through the integrated public path |
| FWHT `fwht-full.1` | FP16/BF16 endpoints with FP32 retained state, register/shared/resident paths, FP32 scratch and ring slots, inverse normalization, C API and typed calibration | 36-cell standalone comparison complete on its recorded GPU; both integrated targets remain pending |

Shared files (`butterfly.cu`, `shared_iterative.cuh`, `shared_resident.cuh`,
public headers/C API, JIT compiler/identity, search and stage-model scripts,
and CMake) need a reviewed merge. Do not copy an older branch wholesale over
rc.1. Preserve numerical-contract distinctions and both sides' regression
tests. A missing lowering is an explicit gap, not a calibrated design point.

## Frozen test populations

Counts below describe separate cohorts per GPU, **not their deduplicated sum**.
Reuse workload contracts and mapping proposals; obtain fresh measurements.

| Cohort | Declared population | Purpose and baseline |
| --- | ---:| --- |
| General acceptance | 749 entries: 74 core + 284 precision/batch + 208 crypto + 138 composed + 29 extensions + 16 planning | FFT/cuFFT and eligible VkFFT; FWHT/Dao; NTT/GPU-NTT; other operators use explicitly labelled in-tree controls |
| FFT repair matrix | 36 = original31 + supplement5 | FP32/FP64, forward/inverse, placement and large-load contracts; repeat all36 and both named subcohorts on each card |
| FFT weak regions | 10 semantic contracts with recorded mapping proposals | N=2^12 and 2^16, B=1/4/16/64 forward and B=8 normalized inverse; compare public auto selection, confirmed candidate and cuFFT |
| NTT external application matrix | 346 declared; currently 214 comparable, 78 baseline-unsupported, 54 baseline-correctness exclusions | Cyclic/negacyclic/coset, forward/inverse, 32/64-bit words, modulus bits and RNS; retain exclusions and recheck the baseline identity |
| FWHT full-state matrix | 36 = 3 lengths × 3 batches × 2 endpoint types × 2 directions | `fp32-full` register, same-contract shared path and Dao; never pool with legacy per-stage narrowing |
| FP64 direct/I/O regression | 68 = 48 correctness + 20 performance | Forward/inverse, normalization, placement and padded/strided I/O |
| Stage and mechanism controls | Existing 56 CT/GS basis points; 24 workloads × 4 variants | Add the integrated FFT core/exchange, NTT boundary and FWHT first/middle/last stage services before claiming coverage |

Retain the old structured-2x2 semantics-mismatch rows as unresolved until
the new contract check passes; an old `complete` journal alone is insufficient.
Across cohorts, deduplicate only identical operator semantics, N/B, precision,
accumulation/retained-state width, direction, normalization, layout, modulus/RNS,
target UUID, build, candidate population and timing protocol. Keep all suite
memberships. Overlapping cohorts may share an identical accepted trial, but
must not receive duplicate weight in an aggregate. Any deduplication count
is established when the final campaign is generated, not guessed in this plan.

## Execution order and ownership

1. **Integrate and freeze.** Resolve the four source inputs in an isolated
   rc.2 checkout. Review the public paths and run affected CPU tests. Freeze
   source, workload, seed and protocol manifests; adapt the existing rc.1
   preparation entry's hardcoded version, date and parent dependencies.
2. **Build and precompile on CPU.** Build all public tests/benchmarks and the
   external-comparison consumer from this source, then compile the union of
   required modules with `CUDA_VISIBLE_DEVICES=`. Do this before reserving a
   GPU. Use one new cache and verify no stale source modules are accepted.
3. **Per-GPU correctness.** Test all batches in the finite repair matrices,
   numerical tolerances and exact modular results, public C/C++ APIs,
   forward/inverse normalization, in-place/strided I/O, workspace guards,
   multi-group overlap and batch tails. Also check FWHT full-state retained
   precision across HBM boundaries and NTT factor folding through public plans.
4. **Per-GPU calibration.** Collect/reuse only identity-compatible primitive
   services. Audit new cores, exchanges, state widths and boundary conversions.
   Missing services stay explicit. This is an engineering prerequisite, not
   an additional paper speedup claim.
5. **Fixed comparisons first.** Run the FFT36/weak10, NTT external and FWHT36
   cohorts with finite mapping proposals and fresh public-selector replay,
   plus controlled variants and fixed baseline/plan measurements for the
   general matrix. Separate preparation/search time from GPU operator time.
6. **Search last.** Run the remaining general-matrix searches, stage-ranking
   checks and public automatic-generation/selection replay. Fix a budget
   before launch; cache/share exact valid fixed-phase trials. A finite seed
   comparison does not count as successful full search acceptance.
7. **Audit and report.** Join only compatible rows; produce per-operator,
   precision, direction, N/B and memory-capacity reports, distributions,
   geometric mean, median, win/tie/loss and worst-case tables. Keep correctness
   failures, unsupported baselines and unmeasured work visible.

The phase dependencies are recorded in the JSON plan. They are reservations,
not commands secretly pointing to historical binaries. Runtime queue creation
must follow the integration/build gates. Reuse the existing supervisors,
acceptance, precompile and comparison runners after their integration; do not
create another independent scheduling framework.

## Targets and scheduling

| Target | Physical UUID | Same-GPU predecessor |
| --- | --- | --- |
| A100 40GB | `GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e` | September 30 `followup-a100-40gb` |
| A100 80GB | `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02` | September 30 `ntt-boundary-full-a100-80gb` |

Use the 80GB card on which the recent FFT/NTT repair studies ran. The other
80GB card (`GPU-0eae...`) retains its existing hybrid-order experiment; do not
merge its timing/calibration identity with this target. New target jobs may
run concurrently on their distinct UUIDs, after their respective predecessors
complete and release their locks. A blocked predecessor is reported as such;
GPU utilization of zero is not permission to bypass its scheduling ownership.
An explicit later handoff can supersede an old campaign with recorded state.

The user's current checker deadline is **2026-10-07 09:00 Asia/Shanghai**
(epoch `1791334800`). Inherit it when activating this plan, and record a
pending/not-started outcome if integration or predecessors are not ready by
then. Do not silently renew it. Adding this plan does not start a new waiter
or change the existing three jobs. The host experiment-plan index links both.

On contention, stop accepting the affected measurements, retain earlier
verified cells and resume only an identical build/contract. Recovery must
audit quarantined cells and keep unconfirmed observations out of aggregates;
do not quarantine-and-restart an entire long campaign without exposing the
retained progress. Numerical/build failures are terminal, not busy retries.

## Measurement and completion contract

Use research JIT with CPU precompilation; exclude compilation, plan creation,
transfers and reference checking from kernel latency. Record those costs
separately for setup/search cost analysis. Each performance pair gets at least
three trials with 100 warmup and 100 timed repeats; focused weak-region
confirmation uses five alternating process rounds. Baselines use matching
semantics and the same GPU. Profiler replay time is not ordinary-launch time.

Public selector tests must show which integrated candidate ran, its mapping
and source identity, and whether a fallback occurred. Do not substitute a
private best-of-pool latency for an automatic-selection result. Use held-out
legal N/B or layout/precision combinations within the declared matrix for
ranking/generalization checks; report missing coverage and search budgets.

Any >5% matched regression versus the frozen predecessor triggers one bounded
five-round confirmation and attribution. Report confirmed losses as losses;
do not modify the matrix or keep optimizing until its geometric mean improves.
Completion requires the declared coverage and correct semantics, not that
every case beats every library. Near-1x averages do not establish statistical
superiority. A combined all-operator number cannot mix external baselines and
in-tree controls without identifying their distinct populations.

V100 remains the user's separate-machine stage. FNO/QuaRot/HEonGPU end-to-end
work remains a separate paper milestone; this repair-acceptance plan does not
mark it complete or add it to the present A100 operator queue.

## Inspect this plan without GPU work

From `/home/wt/git/cuButterfly`:

```sh
/home/wt/yes/envs/cubutterfly/bin/python \
  results/integrated_acceptance_20260930/plan.py check
```

The checker verifies source/input hashes, cohort counts, target UUIDs and
acyclic phase dependencies. It reports pending integration/build status
separately from a valid plan. `plan.py snapshot` creates the immutable local
input lock and reserved phase queue; a changed input cannot overwrite that
lock. It neither invokes CUDA nor launches a supervisor. The concrete runtime
commands are generated after integration, not inferred from an old build.
