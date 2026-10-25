# NTT physical launch descriptor audit

Source audit dated 2026-10-24. This records the existing NTT implementations,
their descriptor gaps, and a proposed common interface. It contains no new GPU
measurement or claim that the proposed interface is implemented.

## Interpretation

A logical stage describes a dependency level. A physical launch describes a
kernel invocation, its grid, threads, resource allocation, and launch kind.
Several logical stages or scheduler roles can execute inside one kernel;
copy and scale kernels can execute without adding a butterfly stage.

The counts below cover one complete `Plan::Impl::launch_once` on a supported,
resolved configuration. They exclude construction, allocation, transfers, and
other CUDA API events. Let `L=log_n`, `I=1` for inverse and `0` otherwise.
Unknown compiler resources do not mean the operator or backend is unsupported.
An inaccurate physical descriptor means that this descriptor cannot support a
cost estimate. Such a candidate can remain available for backend construction
and an explicitly unmeasured fallback.

## Backend inventory

The production projection is `make_dataflow_plan(const PlanConfig&)` in
[mixed_dataflow.cpp](../src/mixed_dataflow.cpp). The probe descriptions below
come from `StageProbeAdapter::StageProbeAdapter(Plan&)` in
[ntt.cu](../src/ntt.cu). "Prior" describes what the existing evidence could
support after a faithful descriptor is exported; it does not claim that the
current ranker consumes the probe or that a model has validated coverage.

| Backend | Actual launches | Production descriptor | Probe descriptor and resources | Static prior boundary |
| --- | --- | --- | --- | --- |
| Baseline | `1+L+I`: copy, `L` stage kernels, optional scale | Usually one whole-transform group; misses launches and auxiliary work | Exposes every launch; compiler resources remain unknown | Keep unranked until copy/scale have a cost contract |
| Tile256 | `2+L-min(L,8)+I`: copy, tile, remaining stages, optional scale | Usually one whole-transform group; misses launches | Exposes actual groups and grids; tile has 128 threads, other kernels 256; compiler resources unknown | Same auxiliary-work gap as Baseline |
| Hybrid2D | Two passes | Usually one group; two physical passes are absent | Splits two passes and their grids; shared remains inherited unknown/zero, compiler resources unknown | Possible after exporting corrected groups and shared demand |
| StagePipeline | `8/stage_space`, in its `L=8` domain | Usually one group, even when several kernels run | Splits launches and grids; compiler resources unknown; named-barrier dynamic shared is absent | Possible after exporting groups and actual dynamic shared |
| CompactStage | Three kernels, stages `5+6+9`, in its `L=20` domain | Usually one whole-transform group | Correct three groups, 256 threads, `2048*batch` CTAs each; reads registers, static shared, and local bytes for each kernel | Structurally usable from probe; model coverage still required |
| HybridDataflow | One kernel containing the local role schedule | Whole-transform group has correct launch count and `grid=batch`; thread configuration must be resolved first | Copies production descriptor; no compiler-resource query | Needs actual dynamic shared and resolved launch metadata |
| HierarchicalBarrier | One cooperative kernel | Logical grid can differ from the resident cooperative grid; cooperative launch is not faithfully described | Copies production descriptor; no compiler-resource query | Keep unranked until cooperative grid/resources are described |
| HierarchicalDataflow | One cooperative kernel containing multiple segment roles, including the inspected specialized/APPT dispatch paths | Multiple logical segment groups can appear as multiple physical launches | Retains logical groups; multiple groups have no independent launcher; no compiler-resource query | Keep unranked until roles and their enclosing launch are separated |

The non-shared production projection leaves `live_shared_bytes=0` and
`compiler_resources_known=false`. Here zero is an unresolved footprint, not
evidence of zero actual shared allocation. The unified resource check consumes
the group fields and leaves the estimate incomplete; it does not substitute
the old top-level whole-transform shared estimate. This avoids a false capacity
rejection but does not make the projected grids or launch counts accurate.

## Source evidence

All launch implementations below are in [src/ntt.cu](../src/ntt.cu):

| Source function or branch | Evidence |
| --- | --- |
| `Plan::Impl::launch_once`, Baseline/Tile256 tail | Launches `bit_reverse_copy_kernel`, optional `tile_kernel`, a loop of `stage_kernel`, and optional `scale_kernel` |
| `launch_hybrid2d_pass` and Hybrid2D branch of `launch_once` | Two pass dispatches; each uses `rows_per_block * (2^LocalLog + 1) * sizeof(Word)` dynamic shared |
| `launch_stage_pipeline_ntt256` | Loops over stage bases in increments of `StageSpace`; launches 256 threads with an additional 64 dynamic-shared bytes for named barriers |
| `launch_compact_stage_log20` | Three explicit kernel invocations; the last kernel selects natural-order or in-place writeback |
| `launch_hybrid_dataflow_point` | One kernel; Radix4 uses 256 threads, other role schedules use `roles * RoleStages * 32`; dynamic shared comes from `hybrid_dataflow_shared_bytes` |
| `launch_hierarchical_dataflow_point_implementation` | One cooperative launch; grid is bounded by work groups and `SM_count * target_ctas_per_sm`; shared includes the padded local tile |
| `launch_hierarchical_streaming` | Builds one role descriptor, distributes CTAs among segment roles, and performs one cooperative launch; specialized/APPT dispatch precedes the generic branch |
| `StageProbeAdapter` CompactStage branch | Queries three `cudaFuncAttributes` objects and exports `numRegs`, `sharedSizeBytes`, and `localSizeBytes` per kernel |
| `StageProbeAdapter` Baseline/Tile256 branch | Adds zero-stage `bit-reverse-copy` and optional `inverse-scale` groups alongside butterfly groups |
| `StageProbeAdapter` Hybrid2D/StagePipeline branch | Replaces inherited groups with the actual pass/launch partition and grids; does not query compiled allocations |

For comparison, shared-iterative JIT resources are already attached to production
groups by `attach_module_resources` in [jit_module.cpp](../src/jit_module.cpp).
For the precompiled shared-iterative path, `StageProbeAdapter` queries the kernel
attributes and adds static shared to the known dynamic footprint; that feedback
is currently local to the probe. The probe returns early for stage overlap, so
its independent bulk-group export is not an overlap description.

## Reuse and next interface

The existing [stage_probe.hpp](../src/stage_probe.hpp) accepts an existing
`Plan&` and exposes `groups()`. Constructing that view neither constructs another
Plan nor benchmarks or launches a kernel. Its constructor builds launch
closures and, in the CompactStage and precompiled shared branches, queries CUDA
function attributes. The originating Plan must outlive its launch closures.

Copying these groups into a separate ranking sample is possible. Copying all of
them into the production stage IR is not a complete solution: the current
`validate_plan` requires positive stage counts and ordered stage coverage, while
the probe represents copy/scale as zero-stage groups. In addition,
`stage_terms` in [stage_cost_model.py](../scripts/stage_cost_model.py) uses
`max(1, stage_count)`, which would assign butterfly computation to those
auxiliary kernels. Probe `independent=true` establishes a callable diagnostic
entry, not calibrated performance-model coverage.

The proposed next interface is `describe_physical_launches`, shared by production
construction, the probe, and the ranker. It is a plan, not an implemented API.
It should retain logical stage/role ownership separately from physical launch
identity and describe launch kind, grid/block shape, cooperative mode, launch
multiplicity, dynamic plus static shared, compiler registers/local allocation,
and explicit unknown fields. Kernel selection and resource description should
use the same resolved template dispatch so they cannot drift.

Until that interface and auxiliary-work model exist, a ranking sample should
explicitly mark an unavailable or inaccurate physical descriptor and remain
`unranked`. A known capacity violation remains a resource rejection. Unknown
resources on a faithful descriptor can support a labeled hardware-rate prior;
they do not establish execution feasibility, validated model coverage, or a
measured workload result. This preserves executable candidates without assigning
scores to invented physical launches.
