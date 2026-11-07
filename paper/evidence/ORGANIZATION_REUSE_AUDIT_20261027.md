# M2: implementation reuse audit, 2026-10-27

## Finding and scope

FFT, NTT and FWHT share executable organization code for **contiguous closed stage groups with explicit materialized handoff** on the `shared-iterative` path. The reuse includes candidate seeds, physical group construction, resource constraints, tile indexing, stage exchange and the device kernel template. Arithmetic and parts of request projection, launch orchestration and backend validation remain operator-specific. This supports a scoped implementation-reuse claim beyond API uniformity; it does not establish cross-operator performance, lower development cost or a new-operator integration result.

This is a CPU source audit of the current working tree, whose HEAD is `45de6b5d0382c8ad050f344aeda33df5c1ba1fcd`. The tree already contains extensive modifications, so HEAD alone does not identify these sources; SHA256 values below bind the inspected files. No runtime source, existing experiment queue, result or paper TeX was changed. No GPU execution, parameter-holdout experiment or performance measurement was performed. `/sdb` was not accessed.

The audit follows `AGENTS.md`, `docs/research_principles.md` sections 1-3 and 4's fixed-core reuse rules, and `paper/NARRATIVE_EVIDENCE_PLAN.md` M2. Documentation was written using the writing-clearly-and-concisely skill; this affects wording, not the evidence standard.

## One rule and its actual path

For a group covering binary stages `[f,f+s)`, one tile owns indices differing only in those `s` bit positions. `iterative_global_index(tile,local,f,s)` inserts the local bits into those positions (`include/cuntt/iterative_layout.hpp:18`). Every butterfly in the group changes a bit within that set, so the group is closed. A partition must cover all `log_n` stages in order. A materialized boundary publishes the producer's completed values before the consumer group reads them. This rule is arithmetic-independent; coefficient lookup, input permutation and normalization are separate contracts.

| Path obligation | Shared implementation actually consumed | FFT / FWHT adaptation | NTT adaptation and boundary |
| --- | --- | --- | --- |
| Public rank-1 request | One public plan API branches into two internal configuration types, rather than one identical implementation. | `src/c_api.cpp:4633` through `4683` projects operator, precision, stride, inverse, normalization and scale into `ButterflyConfig`; FFT/FWHT share this branch. | `src/c_api.cpp:4571` through `4628` projects modulus, word width, direction and NTT order into `PlanConfig`; direct batches must be packed. |
| Automatic path reaches common inventory | `src/mapping_selector.cpp:7` and `:15` first try calibrated registry mappings, then call portable selection. `src/plan_registry.cpp:1171` and `:1174` instantiate the same `select_static_inventory` template. | `portable_butterfly_candidates` calls the common organization generator at `src/plan_registry.cpp:833`. | `portable_ntt_candidates` calls the same generator at `src/plan_registry.cpp:907`. A registry/cache/explicit mapping can bypass enumeration; automatic selection does not promise to select this backend. |
| Capacity-bounded stage grouping and seeds | `capacity_partitions` at `src/plan_registry.cpp:50` bounds two-buffer tile storage by shared capacity and calls `shared_partitions`, with default partition budget 128. The latter round-robins bounded positive stage compositions (`src/partition_space.hpp:11`). `src/shared_organization_candidates.hpp:34` generates layout, threads, local-stage, bulk-batch and fusion variants. | Scalar bytes include complex factor 2 for FFT (`src/plan_registry.cpp:830`). Thread priors start 64/128/256. Adapter writes `stage_partition`, `SharedLayout` and `FftBoundaryMapping` (`:837`). | Word bytes determine capacity (`src/plan_registry.cpp:908`); `balanced_first=true`, thread priors start 128/256/64. Adapter writes `DataflowLayout` and `NttBoundaryMapping` (`:911`). These are different inventory priorities, not different capacity formulas. |
| Logical groups become physical execution groups | Both `make_dataflow_plan` overloads call `project_execution_groups` (`src/mixed_dataflow.cpp:480`, `:631`). That function accumulates resident-fused logical edges, sets ordered `[first_stage,stage_count]`, derives batch work/grid and creates explicit producer/consumer boundaries (`:148`). | `src/mixed_dataflow.cpp:358` projects `ButterflyConfig`; scalar shared storage is `2 * 2^s * value_bytes` (`:493`). | `src/mixed_dataflow.cpp:558` projects `PlanConfig`; shared storage is the same formula in word bytes (`:649`). Core identity becomes `radix2-shoup`. |
| Common structural/resource legality | `validate_plan` checks stage coverage, edge counts, ordered execution-group coverage and explicit physical exchange (`src/mixed_dataflow.cpp:893`). It calls common lowering and per-group resource estimates (`:946`, `:951`, `:740`). Shared/register limits and CTA residency use one implementation (`:695`). | Numeric support is deferred to `ButterflyPlan`; scalar radix2/shared exchange and permitted boundary layouts are checked at `src/butterfly.cu:635`. | Numeric support is deferred to `Plan`; radix2/Shoup, modulus, order, thread/grid and boundary support are checked at `src/ntt.cu:9180` and `:9203`. |
| Candidate selection validates actual backend | Shared `select_static_inventory` projects and rejects before ranking (`src/plan_registry.cpp:1072`), then constructs the typed backend probe and reranks resolved resource samples (`:1113`). | Instantiated with `ButterflyPlan`; actual constructor may reject a projected candidate. | Instantiated with `Plan`; same control algorithm, different backend constructor. Static legality alone does not certify execution (`src/mixed_dataflow.cpp:956`). |
| Actual lowering/dispatch, ordinary materialized path | Both wrappers launch **the same** `detail::shared_iterative_kernel<Operator>` from `src/shared_iterative.cuh:49`; common indexing, stage loop, barriers, buffer swap and final stores live at `:83`, `:97`, `:118`, `:123`. | FFT and FWHT select traits then call `launch_operator` (`src/butterfly.cu:1886`, `:2051`). The shared branch selects ping-pong destination (`:2268`) and launches the common template (`:2299`). | `launch_shared` builds `NttStageOperatorT` (`src/ntt.cu:9873`), selects ping-pong destination (`:9908`) and launches that same template (`:9915`); `launch_once` reaches it at `:9930`. Host wrappers remain separate. |
| Operator-owned numerical semantics | Kernel delegates arithmetic to `op.apply(stage,offset,a,b)` and terminal scaling to `op.finalize` (`src/shared_iterative.cuh:111`, `:135`); `kBitReverseInput` controls first-group permutation (`:89`). | FWHT has add/subtract, no bit reversal, optional inverse division (`src/operator_traits.cuh:87`). FFT has complex twiddle arithmetic, bit reversal and optional normalization (`:222`, `:274`). | NTT has modular Shoup multiplication, modular add/subtract, bit reversal and modular scale (`src/ntt_traits.cuh:25`). These arithmetic implementations are not claimed to be shared. |

For a concrete structural instance, `log_n=8`, partition `[4,4]`, 128 threads and writer-aligned layout use physical groups `(first=0,count=4)` and `(first=4,count=4)` and one materialized boundary. Each group has sixteen independent closed 16-value tiles per transform and launches `B * 2^(8-4)` CTAs at bulk width 1. Two shared buffers require 256 bytes for complex FP32 FFT and 128 bytes for FP32 FWHT or word32 NTT. These are source-derived counts, not measured allocations or occupancy. Fusion seeds may represent the same physical `[4,4]` as logical `[2,2,2,2]` with alternating fused/materialized/fused edges; the common projector recovers the physical groups.

## Exceptions and limits that must remain visible

| Boundary | Evidence and implication |
| --- | --- |
| A shared API does not imply every execution shares the rule | Registry/explicit/cache paths can bypass portable inventory (`src/mapping_selector.cpp:11`, `src/c_api.cpp:4603`, `:4659`). FFT has external/factor/register paths; `check_lowering` limits factor-streamed and cuFFT to FFT (`src/mixed_dataflow.cpp:877`). This audit covers the shared scalar radix2 family only. |
| Candidate capabilities and priors differ | Shared generator filters invalid thread widths and unavailable residency (`src/shared_organization_candidates.hpp:42`); resident seeds require on-demand compilation in both adapters (`src/plan_registry.cpp:836`, `:910`). Its common capability explicitly disables exchange chunking (`src/shared_organization_candidates.hpp:29`). Bulk seeds are 2/4, with domain/capability bounds (`:71`); this finite seed set is not the theoretical space. |
| Backend validation remains necessary | Common lowering returns `RequiresBackendValidation` for these operators (`src/mixed_dataflow.cpp:879`). Common static estimation records unknown compiler allocations and says launch remains unverified (`:731`). Public portable selection probes typed plans; this CPU audit did not execute those device-dependent probes. |
| Shared scalar family excludes some core combinations | FFT/FWHT shared backend rejects non-scalar FFT core, warp-register exchange, non-radix2 and per-group specialized mappings (`src/butterfly.cu:635`). Shared NTT requires radix2/Shoup and rejects specialized streaming mappings (`src/ntt.cu:9209`, `:9220`). A missing combination here is a lowering boundary, not evidence that the organization rule cannot express the arithmetic. |
| NTT has extra semantic and boundary restrictions | `N` must divide `modulus-1`; word32 requires modulus below `2^31` (`src/ntt.cu:9180`). Shared NTT boundaries allow full-scratch/resident-fused, one buffer; overlap owns ring scheduling (`:9224`). APPT-static public requests require APPT register-tail (`src/c_api.cpp:4594`) and are outside this audited shared path. |
| Public layouts differ | Rank-1 NTT requires packed input/output batches (`src/c_api.cpp:4572`); FFT/FWHT common plans require matching input/output layouts (`:4646`). These are real frontend constraints, not removed by common organization. |
| Handoff options are not identical | Butterfly shared execution supports explicit tiled-transpose boundaries and extra workspace (`src/butterfly.cu:648`, `:2282`); ordinary NTT shared launch uses packed canonical handoff (`src/ntt.cu:9910`). Both use the common kernel, but the complete boundary wrappers are not one shared implementation. |
| Hardware is shared input, not measured portability evidence | Both capacity adapters and common selector call `query_hardware_resource_model` (`src/plan_registry.cpp:52`, `:1062`). It resolves explicit/local/bundled hardware descriptions against live identity (`src/mixed_dataflow.cpp:219`; `src/hardware_model.hpp:103`). This audit loaded no target profile or timing model and makes no cross-GPU claim. |
| Shared scheduling is not arithmetic-sample transfer | Projected ranking samples preserve operator/precision/core identity. Common generator and selector reuse does not authorize FFT cost samples to predict NTT/FWHT without calibration/validation. No such transfer was tested here. |

## Bounded CPU validation actually executed

On 2026-10-27, Python 3.10.6 from `/home/wt/yes/envs/cubutterfly` ran:

```sh
/home/wt/yes/envs/cubutterfly/bin/python -m pytest -q -p no:cacheprovider tests/test_shared_organization_projection.py tests/test_iterative_layout.py
```

Result: **21 passed in 0.25 s**. Projection tests exercise FFT/NTT/FWHT plus three other operators, logical fusion to physical groups, threads, bulk/pipeline tile identity and rejected NTT boundary attributes. The layout test checks that a Python transcription of writer-aligned indexing is a permutation for stage counts 2-12. It does not invoke the C++ header or check global tile closure/handoff. These tests validate host-side contracts and that permutation property; they do not launch kernels or numerically validate transforms.

With g++ 9.4.0:

```sh
g++ -std=c++17 -Isrc tests/test_shared_organization_candidates.cpp -o /tmp/cubutterfly_m2_shared_candidates_20261027
/tmp/cubutterfly_m2_shared_candidates_20261027
```

Result: exit 0, `Shared organization candidate contract passed`. The test checks capability filtering, preserved seeds, resident shapes, bulk widths, fusion preserving physical groups, bounded seed count and overlap tile endpoints. It is operator-independent and cannot replace a three-operator public planner execution check. No full suite was run.

## Finite next experiment and residual M2 work

A useful next CPU experiment can freeze these source hashes and a synthetic hardware descriptor, then choose two parameter tuples after checking that they were not used in manual tuning records. Use existing scalar FFT/FWHT and Shoup NTT cores; introduce no new leaf. The audit does not certify any tuple as held out. Candidate examples for that provenance check are `(log_n=13,B=7)` and `(log_n=17,B=19)`, with NTT modulus `998244353` and word32, FFT/FWHT FP32, natural order and forward direction.

For each tuple and operator, enumerate with fixed partition/seed budgets and record the complete inventory, projections and rejection reasons. Apply a fixed 128-thread/partition cap and compare canonical organization signatures after removing semantic/core fields; differences must be explained by value width, capability or declared priority. Check exhaustive tile closure, pair coverage and handoff index bijections using the shared index functions. Apply the same mutations (missing edge, wrong stage sum, oversized shared state and invalid thread width) and require consistent common rejections. At a small bounded `log_n`, a CPU schedule oracle can apply the existing arithmetic traits' semantics to verify the same partitioned dataflow against direct CPU references, retaining exact NTT comparison and floating error distributions. CPU oracle correctness would still not certify GPU code generation or compiled resources.

This would validate parameter generalization and organization invariants at a finite cost. It would not supply GPU compilation, independent numerical execution, finite-budget selected-plan performance, or cross-arithmetic performance reuse. Those remain the execution part of M2. Preparation wall time, number of model/search inputs, compilation outcomes and measurement count must be logged prospectively; no historical developer-effort comparison or saved-person-days estimate is available. The actual audit changes **zero public organization rules, zero semantic adapters and zero platform rules**; that describes this documentation task, not an operator-integration cost.

## Inspected source identity

SHA256 values captured from the audited working tree:

```text
f081ebaa414a720ab990aa15684cebac77fdee675ed277ca9c875e42abec8e2e  src/c_api.cpp
50085b467aff3a03350fac1abcafbd27e33ed22e04ef1cfff22737a7df56056b  src/mapping_selector.cpp
001fec88efadffee271380496b5bd5e43ddb64b16964163f4b6aa73a3bced548  src/plan_registry.cpp
2b6244db60cb62031fd004e1549a457ff055b9ad85c7ccd7806981582eb18a61  src/shared_organization_candidates.hpp
39866e8b8dcc08964da144e51b856d2f5a0126906ffaec5ab6a1d76e92b78847  src/partition_space.hpp
8fe79fbf714c20d1246da5c86a911efca1aedb19a726786bd058db0411ecd3fd  src/resident_mapping.hpp
44f143599732501e6b2375f82c2ecb5b575a6332104991242f1798b8ba74f03b  src/mixed_dataflow.cpp
c666060bcf732fabc6054f89757ca15defb6995bb46b0ccfcc14be1340f4b800  include/cuntt/iterative_layout.hpp
5c437b5de77756bd4a9246eed3d704cce5ee516ec814340909c4a6d36aec4880  src/shared_iterative.cuh
5112558da05d0f08c759b3b09d6d0120c2ecbf9220ad73d05057abeece812c21  src/butterfly.cu
db136b3c3d19ccfde7129fd8a73b964602bef37182698a52111dda8c256e2547  src/ntt.cu
0b92b82e374ac06f429f5cc328341bf6323278ba6c06b6f46468ba4ba6b764d6  src/operator_traits.cuh
bc278ca03c9c49d4d230418ba2385b40ad2b1a604d7aa44d48932d4bc3b3344a  src/ntt_traits.cuh
831379192b9a8037eec56dbe1e7e7ca591688b03547c7ee6ea4fcafc131f83e6  src/hardware_model.hpp
0b577d1af11025548e9f24874fc80f825f082f0d0d4c7288e4c1319435a2beab  tests/test_shared_organization_candidates.cpp
cd9c1f74a43d80084e085e8268a50ad67aa0eeebd1877f4ae98259f9a05ab856  tests/test_shared_organization_projection.py
a93e882e1aada6eb14bc4e74789680e2b6579e7974306a1070cc0f0720692c18  tests/test_iterative_layout.py
665e44d2fcc0844ce85e0f0b52c3571cb4fb53cd387370c29b63777badeb0034  scripts/research_compile_requests.py
```
