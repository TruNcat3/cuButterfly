# Framework alignment audit, 2026-10-12

This audit checks the five gaps found in the recent performance review against
the public planning path.  It separates semantic expression, executable
lowering, candidate registration, and measured selection.  A generated
candidate or a CPU manifest is not counted as a measured performance result.

## Findings

| Area | Current implementation | Evidence and boundary |
| --- | --- | --- |
| Public candidate registration | The public registry now exposes and prioritizes the compiled short FP16 FWHT kernel for `N=32/64`, forward, out-of-place, contiguous input, FP32 accumulation. A matching `AUTO` plan reaches the `warp-register`/radix-2 mapping; other low-precision FWHT requests use the ordinary legal candidates. | `src/plan_registry.cpp`, `src/butterfly.cu`, `tests/test_fused_short_fwht_candidate.py`, and the A100 `cubutterfly_bench` correctness run. The standalone QuaRot provider still accepts an application scale that the public FWHT descriptor does not carry. That provider therefore remains an explicit adapter/fallback and is not claimed as public-plan integration. |
| xD cost accounting | Real FFT lowerings emit physical cost metadata for prefix loops, permutations, scale kernels, C2R copies, real boundaries, total launches, workspace, and compact-boundary bytes. Batch tiles account for both body and tail. The Python fitter consumes this metadata first and keeps legacy inference only for old rows. | `src/xd_real_fft.cu`, `scripts/fit_local_cost_model.py`, `docs/xd_lowering.md`, and `tests/test_xd_cost_features.py`. The metadata describes the executed lowering; its coefficients still require target-specific calibration. |
| Stage dependency and wait cost | Stage-overlap candidates are represented as dependency graphs. The stage model composes ready nodes under shared SM, HBM, and dispatch capacity and models the full batch tile plus tail. Serial measurements are used for fitting; overlap is validated separately and is marked uncalibrated when no overlap profile exists. | `scripts/stage_cost_model.py` (`schedule_nodes`, `compose_stages`, `_predict_batch_composition`) and the stage-overlap/dependency tests. NCU scoreboard counters remain diagnostic evidence, not an input that can be invented for an unprofiled GPU. A high scoreboard result is therefore not silently converted into a new performance claim. |
| Application attribution | FNO and QuaRot reports now use `cubutterfly-application-breakdown-v1` with `compile_plan`, `adapter_conversion`, `operator_gpu`, and `application_rest`. Eager headline time excludes compilation and plan creation; nested component timings are marked independent and are not added to reconstruct a composed call. | `experiments/tools/profile_application_boundaries_20261012.py`, `docs/application_component_breakdown_audit_20261012.md`, and `tests/test_profile_application_boundaries_20261012.py`. The existing filter report has not yet migrated to this schema and remains a separate result. |
| Semantic coverage | The CPU manifest enumerates the supported semantic axes across FFT, NTT, FWHT, both zeta directions, and structured 2x2: precision, rank, batch, direction, placement, logical order, compact real boundary, and composite/xD mode. xD is an execution choice on the common descriptor, not a second API. | `scripts/build_semantic_coverage_matrix.py`, `docs/semantic_coverage_audit_20261012.md`, and `tests/test_semantic_coverage_matrix.py`. The manifest is an inventory. Each GPU run must query capabilities and record `missing-lowering`; it does not turn an unimplemented lowering into a pass. |

## Design check

The current flow is consistent with the research decomposition:

```text
operator semantics + workload shape
              |
              v
       canonical descriptor
              |
              +--> capability check and explicit lowering
              |
              +--> executable candidate registry
              |          |
              |          +--> measured mapping for this hardware/workload
              |          +--> bounded portable fallback when no record exists
              |
              +--> complete-plan cost and boundary accounting
```

The selector does not treat a larger candidate inventory as proof of an
improvement.  A measured registry row must match the semantic key and runtime
fingerprint.  A cache miss is reported as a static/portable selection and is
not described as calibrated.  Explicit mappings still undergo legality and
resource validation.

The audit also exposes three remaining, bounded follow-ups rather than hiding
them behind defaults:

1. Recalibrate the new xD physical-cost terms on each target GPU before using
   them for quantitative ranking.
2. Add the filter application profiler to the same component schema, then
   collect independent adapter-only and application-only controls where they
   are needed for attribution.
3. Extend the public FWHT semantic descriptor only if an application requires
   arbitrary output scaling. The current short public kernel is deliberately
   limited to scale one; the standalone scaled provider remains explicit.

These are implementation or calibration boundaries. They do not change the
method's semantic claim: a butterfly request is separated from its physical
organization and hardware mapping, and the organization is selected only from
legal, measured candidates when evidence exists.

## Verification performed

The focused CPU and public-planner checks for this audit pass:

```text
16 Python tests passed (semantic, xD cost, and application breakdown)
8 Python provider tests passed
57 stage-service/composition tests passed
5 CTest checks passed (descriptor, logical order, mapping replay,
public planner consistency, and uint64 zeta reference), including the
public FP16 FWHT AUTO-candidate assertion
```

The C++ planner comparison ignores the top-level public `contract` envelope
when comparing physical mappings.  The envelope is semantic metadata added by
the public API; all physical fields still compare exactly with the legacy C++
planner.  This keeps replay and semantic identity separate without reporting a
false planner mismatch.
