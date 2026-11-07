# Unified planner and specialization modules

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

The main C++ interface is `<cubutterfly/plan.hpp>`. `Transform` owns mathematical
semantics; `PlanOptions` owns selection policy or an explicit execution mapping.
The C++ wrapper calls the public C API. Rank-one selection shares registry and
candidate facilities with the benchmark programs; xD planning has a separate
bounded selector. The [0.9.2 semantic contract](unified_semantics.md) now joins
public C/C++/Python descriptors, workload identity and replay. The earlier
[dimension audit](public_frontend_dimension_audit_20261007.md) records the
defects that motivated this repair. Some arithmetic lowerings remain missing;
capability queries distinguish them from invalid semantic requests.

Rank-one construction shares a static-capacity legality gate and per-execution-
group compiler feedback. On a registry miss, AUTO uses the existing stage cost
model to rank candidates, then compiles and reranks bounded finalists. Selection
provenance distinguishes a validated estimate, a hardware-rate prior and an
unmeasured executable fallback. These paths do not acquire new measurements.
See the [static profile policy](hardware_profile_install.md#static-profile-policy)
for compatibility and exploration-budget controls; a missing physical cost
descriptor is a modeling gap, not an unsupported operator.

The [A100 acceptance](../results/a100_closure_20261103/final_acceptance/README.md)
now confirms this path for the new local-root and per-group thread options.
Current measured records are consumed by public DEFAULT and seeded into public
MEASURE; the independently validated serial service model covers the finite
competing cohort. Registry selection, model prediction and direct measurement
remain distinct evidence. The fit does not imply coverage outside its primitive
keys, load range or direct-launch protocol.

Dynamic cuFFTDx candidates additionally share an installed-provider capability
contract across enumeration, explicit mapping validation and JIT. A missing
local core implementation does not invalidate the user's transform; unknown
provider versions remain subject to actual compilation. See the
[provider capability contract](provider_capability_contract_20261014.md) for
the support dimensions, candidate audit and cache identities.

Online register FFT mappings additionally accept explicit `suffix_core="native"`.
The default and automatically enumerated suffix core is `cufftdx-block`; its
legal EPT choices come from the installed provider capability checks and the
organization's geometry constraints. The new native suffix remains an explicit
experimental mapping with a separate compiled and calibrated identity. Missing
provider implementations never silently change an explicitly requested core.
The [reuse policy](provider_reuse_20261014.md) explains the search correction;
the [native suffix contract](native_suffix_core_20261014.md) preserves its measured
geometry, shared-memory requirement and bounded validation.

Register FFT prefixes expose `prefix_twiddle="runtime"` (default) or `"table"`
independently of the local codelet and layout. This controls only the diagonal
coefficients between the prefix's two local factors; `cross_twiddle` continues
to describe the global boundary. The table policy requires research/auto
compilation. It embeds read-only roots computed before compilation at 80 decimal
digits and rounded to the requested FP32/FP64 storage; inverse transforms use
their conjugates. It adds neither a per-call upload nor a precision conversion
of the transform. Square, rectangular and cooperative prefixes use the same
coefficient-provider interface. Public candidates contain a bounded table basis,
and evolutionary neighbors can vary the policy independently. Compilation and
prefix service identities distinguish the policies; suffix service identities
remain reusable. Candidate availability does not imply a measured winner or a
calibrated service curve for the new policy.

Shared-iterative mappings accept `execution_group_mappings` with one entry per
physical group after fused boundaries. Butterfly entries use
`{"core":"scalar","exchange":"shared","threads":64,"ept":0}`; NTT entries use
`{"core":"radix2","threads_per_block":64}` with default scheduler fields.
Threads must be warp multiples in `[32,1024]` and satisfy device resource limits.
Logical `segment_mappings`/`subgraph_mappings` are not substitutes for this physical
axis. Fields that the shared lowering does not consume are rejected.

With no explicit group entries, the existing global thread field applies to all
groups. Precompiled ordinary shared kernels consume the vector dynamically;
research/auto specialization records `group_threads` in its module request and
cache identity. Resident leaves retain their existing compilation requirement.
Probes, batch overlap, cost projection and service keys use the same group widths.
Public enumeration supplies a bounded basis and GA changes one group at a time.
This enables mixed widths without exhaustively enumerating their Cartesian product.
The [A100 validation](../results/a100_closure_20261103/stage_threads/README.md)
reports the measured scope and remaining internal tile-packing gap.

```cpp
#include <cubutterfly/plan.hpp>

cubutterfly::Context context;
cubutterfly::Transform transform;
transform.extents = {1u << 18};
transform.batch = 4;
transform.storage = CUBUTTERFLY_DATA_COMPLEX_FP64;
cubutterfly::Plan plan(context, transform);
plan.execute(device_input, device_output); // asynchronous on the context stream

cubutterfly::PlanOptions replay;
replay.mapping = plan.mapping_json();
cubutterfly::Plan same_mapping(context, transform, replay);
```

`Plan` allocates its workspace by default. With `allocate_workspace=false`, bind
an allocation of at least `workspace_size()` before execution. Inputs, outputs,
and external workspace must remain valid until the stream completes. Set the
context stream before constructing plans. NTT requires matching storage and
`word_bits` (32 or 64); in-place requests use the common boundary adapter.

## xD planning contract

For rank-two through rank-eight transforms, `AUTO` and `XD` use the bounded xD
selector. `COMPOSITE` forces the legacy per-axis composition and its generic
rank-N layout boundary. The selector compares complete executable candidates
when the device-memory budget permits it. It is a finite candidate search, not
an exhaustive xD-space calibration or a global-optimality claim.

For exact tensor-product flattening (FWHT, subset/superset zeta, and structured
2x2), a parent `MEASURE` request also tunes the rank-one child through the same
handle and cache. `DEFAULT` retains ordinary child selection, while explicit
replay takes precedence and preserves the supplied physical mapping. Child
tuning and complete xD selection remain separate steps; timings from different
output-lifetime or launch protocols are not interchangeable.

The generated FP32 register FWHT provider specializes aligned unit-stride
accesses at launch without changing its arithmetic or public mapping. Pointer
alignment and batch distances select this lowering; strided, unaligned, and
zero-extended inputs retain their general paths. This is an implementation
choice under the existing semantic contract, not a shape-specific winner table.
The [A100 layout follow-up](fwht_native_repair_20261107.md) records its measured scope.

The rank-two composite path consumes a packed, standard-length input directly
in its first axis, without an initial full-array copy. The first butterfly axis
is out-of-place internally, including when the public request is in-place; the
user buffer is not overwritten before its input is consumed. The physical
composition mapping records `direct_input_boundary=true`. Explicit older
composition mappings without this field retain their pack operation. Requesting
the direct boundary for a strided or embedded input is rejected; those requests
continue to use the generic adaptation path. The first axis's placement also
participates in its registry and cache key. This boundary choice is shared by
FFT, NTT and the other rank-two butterfly operators.

Scalar temporal tiles use one compiled-domain and byte-capacity predicate in
both candidate enumeration and plan validation. The currently generated domain
ends at logN=12, with shared storage bounded by both the hardware profile and the
48 KiB default launch contract. A larger legal tile is added alongside existing
multi-stage candidates, not selected unconditionally. New legal candidates need
measurement before treating their static prior as a confirmed performance rank.
BF16 native primitives use CUDA's explicit round-to-nearest intrinsics, including
the CUDA-provided fallback on targets without native BF16 arithmetic. Explicit
FP32 accumulation retains its separate arithmetic contract.

The candidate inventory includes the legacy composite path, exact flattening
only for FWHT, subset zeta, superset zeta, and structured 2x2 tensor products,
resident shared subgraphs with device-valid warp-multiple thread counts and
axis-order choices for supported small packed power-of-two
cases, and FFT axis candidates based on imported rank-one cuFFT cores. The FFT
axis candidates are `strided-axis-cufft`, `packed-axis-cufft` and the optional
`axis-tiled-cufftdx` with EPT 8/16 variants. These combine axis access and
local FFT in one kernel on covered axis lengths, retaining packed fallback
for other axes. `pair-tiled-cufftdx` additionally fuses two short, physically
trailing axes through a shared-memory plane; EPT variants affect its remaining
unfused axes. The legacy Thread-pair fusion covers FP32 axes 8–64 and FP64
axes 8–32 with its fixed 48 KiB shared-memory guard. The separate cooperative
pair inventory adds finite 64/128-axis shapes with four-lane FFT cooperation;
FP64 pair CTAs use 256 threads and fold 128-line passes into two rounds. Shape
availability is filtered by the compiled SM, register/occupancy resources and
the executing device's actual opt-in dynamic shared-memory budget. On A100,
FP64 128x128 is unavailable under that budget. Full-plan timing chooses between
these mappings and the earlier candidates.
None delegates the whole xD transform to a vendor multidimensional routine. FFT and
NTT are not eligible for the strict flattened candidate.

The portable rank-one inventory also contains a compiled short FP16 FWHT
provider for N=32 and N=64. Its semantic contract is forward, out-of-place,
contiguous storage, FP32 accumulation and explicit
`output_rounding=FINAL_STORAGE`. It uses a warp-register exchange, applies
`output_scale`, and narrows once. Matching generic candidates keep FP32 working
storage until the same scaled final store, recording both storage conversions.
Default `STAGE_STORAGE` requests cannot select this final-only provider.
The output contract is part of the public descriptor, registry and cache key;
unsupported combinations report missing lowerings. QuaRot now requests it
through public plans rather than the earlier standalone provider.

The long-axis inventory adds 2048/4096 points in both precisions and directions
with EPT 8/16 and device opt-in shared-memory checks. The
`axis-packed-cufftdx`/`axis-packed-cufftdx-ept16` choices explicitly permute
noncontiguous axes into contiguous core inputs, retain that layout across
compatible axes, and restore the public output at the end. Their complete-plan
cost includes these permutations. The v18 xD cache namespace separates this
inventory from prior bounded searches; explicit mappings still record the
actual per-axis core and access policy. See
[application integration evidence](e2e_unified_coverage_20261013.md).

Compact C2R mappings record `c2r_input_boundary` and
`c2r_input_copy_elision`. With one or more complex-axis plans, the first axis
reads the compact input directly and writes owned scratch. Rank-one C2R keeps
the scratch-copy fallback required to preserve input from the destructive
real boundary. These fields describe the executed lowering and are not
unchecked requests to skip a required copy.

The v9 cooperative lowerings are `axis-tiled-register`,
`pair-tiled-cooperative`, and `pair-tiled-cooperative-register`. The first is a
four-lane register-only core for non-contiguous axes of length 64 and 128. The second combines
the cooperative pair with the existing cuFFTDx axis core; the third uses the
register core for eligible remaining axes. Mapping JSON records
`axis_core`, `pair_core`, `cooperative_lanes`, and the actual physical core
names. Their finite inventory is not a complete subblock cross-axis pipeline
or a global-optimality claim.

The v10 inventory additionally exposes `axis-tiled-register-l8`/`-l16` and
`pair-tiled-cooperative-register-l8`/`-l16`. The lane count changes the
non-contiguous axis core; a cooperative pair still uses four lanes per FFT.
`pair-tiled-cooperative-layout1` and
`pair-tiled-cooperative-register-layout1` use an XOR permutation of shared
rows with the same padded allocation. The baseline layout remains available.
More lanes or a different layout are candidates, not unconditional upgrades.
The original cooperative cores use a shared 128-root constant table;
their per-axis mapping records `twiddle_policy=constant-roots-128`. This avoids
inlined transcendental polynomial work in the previous phase construction.

The v26 pair inventory includes `xd:pair-tiled-radix4` for 32×32 complex
planes in FP32 and FP64. FP64 uses double arithmetic and double root constants.
The pair dispatcher records the core, `[4,4,2]` axis radices, precision-specific
padded shared exchange, and separable-axis semantics. Packed plans can move
two scheduled complex axes together to the trailing positions, retain that
layout, and restore the public layout at the final boundary. Compact R2C/C2R
uses the same path; its Hermitian axis remains a plane index and is never
treated as a full complex FFT axis. Permutations, real boundaries and scaling
remain in complete-call timing. An explicit pair request with no eligible pair
is rejected. Automatic selection may choose an existing axis core or whole
cuFFT instead. Cache namespaces are `xd-v26-packed-pair-fp64` and
`xd-real-v26-packed-pair-fp64`; exact replay checks the actual physical mapping.

The v14 inventory extends `axis-tiled-register-l8`/`-l16` to non-contiguous
256-point axes in both precisions and directions. It composes the existing
Thread FFT32/16 codelets with the same warp butterfly, using
`twiddle_policy=constant-roots-256` for that axis. Four lanes remain unavailable
at length 256 in this inventory because the imported FP64 Thread64 codelet is
absent. Availability is checked for the length/lane configuration. Unsupported
axes retain their explicit Block fallback; a forced register family with no
actual register axis is rejected. Compact-real axes and batch-tile children use
the same configuration check. Pair-core length support is unchanged.
The v14.1 lane helper replaces small-radix DIF table lookups with analytic
unity/quarter/eighth turns and uniform literal coefficients. Nontrivial
cross-sub-FFT phases still use the recorded 128/256 root table. This affects
the eight/sixteen-lane cores; the dedicated four-lane codelet is unchanged.

Packed C2C plans can finish all axes of a batch tile before advancing to the
next tile. The bounded rank-three selector proposes tiles near the device's
L2 capacity for two live complex buffers, retaining whole-batch execution.
Exact replay admits any legal positive tile smaller than the batch, including
a tail; it is not limited to the proposal set. Mapping fields `batch_tile`,
`batch_chunks`, `tile_body`, and optional `tile_tail` describe this schedule.
Top-level axes describe a full tile when `axes_scope` is `per-batch-tile`;
the top-level batch and semantic contract still describe the complete call.
This schedule improves cross-axis cache locality for independent transforms;
it does not implement overlapping kernels or a general within-transform
subblock pipeline. Complete-call timing includes every tile and tail.

The v11 inventory additionally exposes `axis-tiled-cufftdx-cap8`/`-cap16` and
`axis-tiled-cufftdx-ept16-cap8`/`-cap16`. Their independent
`axis_ffts_per_block_cap` bounds the existing FFTs-per-CTA policy without changing
EPT. It applies to contiguous and staged axes, both precisions and directions;
per-axis `ffts_per_block` records the actual value. Default 0 retains v10.1's
policy and serialization, including exact replay of existing mappings. Automatic
search omits cap candidates that would instantiate only identical default cores.
This finite cap inventory is a resource-granularity experiment, not the complete
theoretical parameter space. Compact real plans use the same cap on complex axes.

The v12 axis inventory adds `-direct` and `-unrolled` suffixes to each Block
EPT/cap family. `axis_exchange_policy=1` loads a non-contiguous axis directly
into the imported core's registers; policy 2 preserves shared staging with
compile-time transfer loops and hoisted address calculations. Policy 0 retains
the original staged implementation. These are measured tradeoffs: fewer shared
exchanges can increase global transactions or register pressure. The per-axis
`io_exchange` records the actual path; contiguous axes always reuse the existing
contiguous-register kernel. The policy reaches compact-real complex axes and
batch-tile children. Default-policy mappings keep their previous serialization.
These named families apply policy/EPT/cap across eligible axes in a plan. The
v15 mixed resource mapping below additionally supports independent axis choices.

The v13 resource inventory adds EPT4 families and `-unrolled32` exchange
variants. Policy 3 uses 32-bit element indices for a staged strided axis only
when its complete local span fits `UINT32_MAX`; larger explicit requests are
rejected, and automatic C2C search records the unavailable candidate. Buffer
pointers remain full width. Batch-tile children check their own local spans,
so tiling may make a large complete call eligible. Contiguous axes retain the
old path. `io_exchange=unrolled-staged-u32` identifies the actual narrow path.
EPT4 changes the local FFT mapping, while policy 3 changes address arithmetic;
both remain candidates alongside the old implementations.

The v15 `axis-tiled-mixed` physical mapping stores `axis_resources` by logical
complex-axis index, independently of execution order or retained physical
layout. Each entry specifies `core`, `ept`, `register_lanes`, `block_cap` and
`exchange`. C2C has one entry per axis; compact R2C/C2R excludes the final real
axis. Empty overrides preserve prior mapping serialization. Batch body and tail
receive the same vector and validate their own local index spans. Nonempty
overrides require packed fused axes without paired-axis fusion. A requested
register core must be executable on that particular strided axis; an unavailable
Block core retains the explicit vendor fallback recorded in `axes[].core`.
Thus `axis_resources` describes requests, while `axes` describes execution.

After measuring the existing uniform candidates, the selector refines the
fastest eligible unpaired axis plan with one coordinate sweep. The current v19
inventory includes 36 legacy EPT/cap/exchange combinations, 12 exact-group
combinations, three register lane widths, and four local-core combinations.
It removes physical duplicates and records unavailable proposals. All timing
covers the complete call; the default promotion factor is 0.99, and its effective
configured value participates in cache identity. The sweep preserves the seed's
axis order and batch tile, so it is bounded by 55 proposals per axis and is not
a Cartesian-product search. Explicit complete
mapping replay restores per-axis resources without search. The short diagnostic
name `xd:axis-tiled-mixed` alone is insufficient to specify a resource vector.
Compact real plans use the same refinement, while their real boundary is fixed.

The historical v15 retained-layout mappings used the cache namespaces
`xd-v15-axis-resources` and `xd-real-v15-axis-resources`. Compact real R2C/C2R
plans also support cooperative-register cores for eligible complex axes, but do not fuse a
cooperative pair or apply the C2C batch-tile wrapper at the compact real boundary.

The v19 inventory adds optional `axis_resources[].ffts_per_block` as an exact
CTA line count, independent of the older upper-bound `block_cap`. Values 2/4
are compiled for 2048/4096-point Block cores with EPT8/16 and exchange 0/2/3.
An exact request requires `block_cap=0`; device thread/shared-memory limits and
compiled core availability still apply. An unavailable exact request is rejected
instead of silently falling back. Zero or an absent field retains the previous
policy and serialization. Public compact-real selection measures these uniform
candidates; the shared per-axis refinement also makes them available to C2C.
Both directions and small outer grids are supported. This is a finite compiled
inventory, not a bound on the theoretical space.

Inverse compact-real plans can additionally set `c2r_prefix_scale=true` to absorb
the requested normalization/output scale into the final complex-axis core's
store. `c2r_scale_placement=last-complex-axis` records this choice and
`cost.scale_kernel_launches=0` records the removed real-output pass. It requires
a fused final complex axis and is invalid for forward or C2C requests. Automatic
selection measures this candidate alongside the previous post-C2R scale pass;
old mappings without the field keep the old execution. The scalar commutes with
the remaining linear C2R operation, but floating-point rounding may differ, so
matching-precision numerical errors are reported for both paths.

The v19 cache namespaces were `xd-v19-grouped-axis` and
`xd-real-v19-grouped-scale`. Per-axis actual grouping, scaling placement,
candidate times and omission reasons remain in the public mapping/search report.

The v20 inventory adds `axis_resources[].core="vkfft"` and the uniform
`xd:axis-vkfft` family for both C2C and compact-real complex axes. This imports
one unnormalized C2C axis from VkFFT, not a whole multidimensional transform.
The public axis scheduler continues to own axis order, retained layout, real
boundaries, scale placement and complete-plan selection. Different axes may
use different providers in the same resource vector. Both uniform selection
and the shared coordinate sweep measure the new provider; explicit mapping
replay retains the actual provider and its resource metadata.

The current adapter accepts power-of-two axis lengths at least two, both
FP32/FP64 and both directions. A logical axis is presented as
`[prefix, length, suffix]`; for a strided axis, VkFFT omits the suffix dimension
and executes all prefixes in a batched invocation without an external packing
pass. The suffix need not be a power of two. Unsupported lengths or disabled
builds are recorded as unavailable candidates; other existing providers remain
eligible. This is the adapter's validated support domain, not a claim that
VkFFT itself only supports power-of-two transforms.

VkFFT does not expose arbitrary fused output scaling through this adapter.
The lowering keeps the complete final scale pass when the last core requires
it, and rejects a `c2r_prefix_scale=true` request with such a final core.
Provider-owned scratch/LUT storage is reported separately from caller workspace;
an axis invocation may contain several provider kernels, so it is not a device
kernel-count measurement. The provider compiles and allocates while creating
the plan, before steady-state timing or Graph capture.

The v21 core contract now defines `XdAxisResources` and its compiled candidate
inventory in one CPU-only header, `src/xd_axis_resources.hpp`. Public JSON
decoding, lowering validation and per-axis refinement consume the same validator
and candidate factory. This prevents a parameter from being accepted by one
layer while being silently ignored by another. The inventory includes the
existing block/register/local choices, optional VkFFT, packed exchange policy 4,
and the local length-64 32-line variants. These are finite compiled candidates;
shape and device resource checks remain lowering responsibilities, and the
theoretical space is wider. The cache namespace is `xd-v22-common-organization` (and
`xd-real-v21-core-contract`) so old mappings cannot be replayed under the new
parameter contract. Other butterfly families retain their own operator-specific
core predicates, documented in the [template audit](template_parameter_audit_20261013.md);
they are not claimed to share unsupported lowerings merely because their
semantic traits are compatible.

Enable the optional provider with `-DCUBUTTERFLY_ENABLE_XD_VKFFT=ON`; its default
follows `CUBUTTERFLY_ENABLE_VKFFT`. It uses the existing vendored VkFFT checkout
and CUDA driver/NVRTC libraries. Disabling it builds a stub and preserves the
existing providers. Current cache namespaces are `xd-v22-common-organization` and
`xd-real-v21-core-contract`, with provider availability in the key; the CMake
build fingerprint also includes the provider enable option.

The compact real API supports FP32/FP64 R2C and C2R for ranks one through eight.
The core uses packed out-of-place storage; the public semantic boundary handles
independent layouts, orders and in-place requests. R2C uses the full logical
shape and emits `N_last / 2 + 1` on the last axis. C2R accepts that compact
shape and emits the full shape. Odd and even lengths, inverse normalization,
and `fft_output_scale` are part of the contract. FP16/BF16 power-of-two real
requests have an explicit same-precision full-spectrum adapter. FP32 storage
with FP64 compute has an explicit cast/FP64-child/cast adapter. Mapping and
whole-call timing include these boundaries. Arbitrary-length C2C requests beyond
rank two use exact-length rank-one children and the generic permutation boundary.
For more than three complex axes, default order candidates are cyclic rotations
of forward/reverse traversal; explicit replay still accepts any legal order.

Research compilation is completed before steady-state timing. If tuning fits
`CUBUTTERFLY_XD_TUNE_MAX_BYTES`, complete candidates use three trials and
median timing; a replacement must improve the retained baseline by more than
one percent by default, subject to the configured promotion margin. When
unset, the bound has a `512 MiB` floor, a `2 GiB` cap, and
uses at most one quarter of currently free VRAM, with a second free-memory fit
check before temporary allocations. This avoids silently bypassing large-batch
candidate selection while keeping installation-time search bounded. Setting
the variable to `0` skips empirical promotion. This policy provides a
reproducible selector, not a performance guarantee. Mapping JSON contains the
semantic contract, device name/VRAM/SM/fingerprint, physical candidate, and
replay information. Explicit diagnostic mappings such as `xd:resident-t128`
are useful for controlled experiments and still undergo contract and resource
validation.

For cooperative candidates, codelet preparation, dynamic-shared-memory
attribute setup and occupancy checks finish before candidate execution or
complete-plan timing. These resource steps are not steady-state kernel timing.
The bounded implementation record is
[xD v10.1 scheduling and cores](../results/paper_profile_rc2/end_to_end/xd_schedule_v10/README.md),
with [v9 controls](../results/paper_profile_rc2/end_to_end/xd_register_v9/README.md);
it records correctness and resource gating without asserting a complete
cross-axis pipeline or global optimum.

The six mathematical operators are FFT, NTT, FWHT, subset zeta, superset zeta,
and structured 2x2 butterflies. The legacy `xor-zeta` name retains its subset-zeta
semantics. Floating low-precision traits retain their explicit rounding and
accumulation contract. Structured coefficients are semantic inputs, not tuning
parameters. Rank-2 and embedding plans serialize every physical axis; Bluestein
serializes both internal convolution transforms. External cuFFT remains an
explicit baseline option.

In implementation revision 0.9.3.4, normalized FP16/BF16 FFT and FWHT inverses
apply a factor of 1/2 at each radix-two butterfly. The factor follows the
butterfly when all input components have magnitude at most one quarter of the
storage maximum, and precedes it otherwise. The bound leaves room for a complex
product and addition; deferring the factor on small operands avoids premature
subnormal rounding. Across an axis the total factor is the requested 1/N.
The same trait serves real-FFT adapters, complex FFT, shared subgraphs and
generated stage partitions. This changes the rounding schedule and requires
new error evidence; it neither promotes the requested precision nor rescales
the user's input. It cannot make mathematically out-of-range outputs representable.
Forward and unnormalized requests retain their original scaling semantics.
The [repair record](numeric_plan_repair_20261009_v0934.md) documents the original
failures, intermediate regression, final diagnostics and performance scope.

## Compile policy

Set one process-wide policy before constructing plans:

```bash
export CUBUTTERFLY_COMPILE_MODE=research
```

- `precompiled`: use linked implementations and report mappings requiring code
  generation as unavailable.
- `auto`: allow a standalone module for register FFT and factor-streamed FFT
  mappings. Existing generic shared templates remain available.
- `research`: additionally specialize the common shared-group template for its
  operator, numeric regime, partition, launch width, and writer layout.

Research specialization uses NVCC to build one shared object. It never rebuilds
the main library. The first plan pays compilation; repeated plans reuse the
module cache. Plan construction, coefficient initialization, compilation and
resource queries are outside kernel timing. Research compilation has no timeout
unless `CUBUTTERFLY_COMPILE_TIMEOUT` is set.

`CUBUTTERFLY_JIT_CACHE` overrides the module cache directory. The cache includes
template contents, compiler version, SM, ABI, and mapping. V1 modules implement
register-prefix/grouped-suffix and factor-streamed FFT in FP32/64. V2 adds semantic coefficient data
for the shared template used by all six operators. Compiled register and shared
resource limits are checked before execution. The shared template supports
arbitrary positive stage partitions whose individual live states fit a CTA;
different launch boundaries are not inferred from the number of logical stages.

`stage_overlap=true` with `batch_tile_count` streams batches through any number
of physical shared groups for all six operators, including NTT word32/64.
Each group has a private CUDA stream; each adjacent edge owns two tile slots.
Ready/consumed events protect every slot, and the last consumer joins the
caller's stream. Shared workspace is therefore
`2 * (groups - 1) * batch_tile_count * batch_stride * sizeof(value)`.
The two-group cuFFTDx and register-prefix FFT adapters use the same scheduler.
Register FFT supports both FP32 and FP64; its precompiled and on-demand
candidates expose bulk execution, power-of-two interior tiles, and the full
batch tile. Arbitrary positive tile counts can also be replayed explicitly.

Shared V2 modules export `cubutterfly_module_launch_group_v2`; register FFT V1
modules export `cubutterfly_module_launch_group_v1`. These optional versioned
entries let the same specialized group kernel participate in bulk and pipelined
execution. The caller owns inter-group buffers and dependencies; the original
whole-plan descriptors are unchanged. Both precompiled and research paths
support the stream schedule. This is a pipeline between kernels with global
ring storage, separate from the existing warp-role schedules inside a CTA.

Installed packages include the template sources and enabled MathDx headers.
`CUBUTTERFLY_TEMPLATE_ROOT`, `CUBUTTERFLY_MATHDX_ROOT`, `CUBUTTERFLY_NVCC`, and
`CUBUTTERFLY_JIT_PYTHON` override discovery for relocated or cross-compiled setups.

## Resident processing units and output exchange

The common organization implementation and its cross-operator validation are
tracked in [the organization extension](common_organization_20261013.md).
For shared bulk execution, `batch_tile_count=1..4` folds independent batches
inside a CTA using the same shared storage. With `stage_overlap=true`, the field
instead sizes pipeline chunks and per-edge double buffers. Shared logical
resident-boundary fusion is available to NTT as well as the other five operators;
local partitions are indexed by the resulting physical groups.

The public mapping has two axes indexed by **physical execution group**:

| Field | Meaning | Current executable lowerings |
| --- | --- | --- |
| `local_stage_partitions` | Nested stage partitions inside an existing resident group | Shared-iterative, for all six operators; register-prefix FFT |
| `exchange_chunks` | Register slots published in each output-ownership exchange | Grouped FFT suffix |

An empty outer list, or an empty local partition entry, preserves the previous
lowering. Each nonempty partition must cover its physical group's stage count.
These local partitions introduce no HBM boundary. Logical `stage_partition`,
resident boundary fusion and the group-local register partition remain separate
choices. After fusion, the outer list follows the resulting physical groups.

For all six shared-iterative operators, each local unit covers 1--5 stages and
keeps 2--32 values in registers. The operator's existing butterfly primitive
executes those stages, then writes to the next unit's shared-memory read layout.
This reduces shared publication/barriers from one per stage to one per local
unit. It can increase registers and reduce active threads for a small tile;
search must retain the original lowering as a competitor. For example:

```json
{"schema_version":1,"backend":"shared-iterative","stage_partition":[8,4],
 "local_stage_partitions":[[4,4],[2,2]],"exchange_chunks":[0,0]}
```

The FFT register prefix additionally supports unequal local factors. A local
partition `[a,b]` describes factors `A=2^a`, `B=2^b`, with common register EPT `K`.
The cooperative widths are `A/K` and `B/K`; `prefix_codelet_lanes=A/K` and
`prefix_threads=A*B*C/K`, where `C` is the independent column count. Both
cooperative groups must satisfy the warp and shared-layout constraints.
For example, `local_stage_partitions=[[6,5],[]]` permits a 64-by-32 prefix
without forcing the old square factorization. An empty suffix entry records
that its BlockFFT core owns an opaque internal decomposition.

In that suffix, `exchange_chunks=[0,8]` publishes eight register slots at a
time. A nonzero chunk must be a power of two dividing suffix EPT, with more
than one column. Allocated shared memory is the maximum of the core's scratch
space and the exchange chunk's space; chunking cannot shrink the core itself.
Generic shared-iterative output already has its final ownership and rejects
nonzero chunks. Other lowerings reject these unsupported axes explicitly.

Nondefault choices require `auto` or `research` compilation. Public replay,
precompilation, search mutation and stage-service identities carry both axes.
The finite candidate inventory adds representative local partitions and legal
rectangular factors; it is not an exhaustive compile of every partition. Old
measurements do not qualify a new local partition or exchange chunk. The
promotion experiment (local artifact: `../results/paper_completion_20260918/public_resident_promotion/README.md`; not included in this source release)
defines focused correctness and same-dataflow performance checks.

## FFT leaf, shared-layout and per-stage I/O strategies

Research and automatic compilation expose these independent execution choices
through the same public mapping JSON used by explicit replay and search:

| Mapping field | Values | Applicable lowering |
| --- | --- | --- |
| `prefix_codelet` | `native` (default), `radix4`, `cufftdx-thread` | OnlineReorder register-prefix FFT |
| `prefix_codelet_lanes` | Power of two, default `1` | Threads cooperating on one register-prefix codelet |
| `prefix_shared_layout` | `linear` (default), `xor` | OnlineReorder register-prefix FFT |
| `factor_io_policies` | Empty/default, or one `dynamic`/`static-unrolled` entry per physical factor | FactorStreamed FFT |

`prefix_shared_layout` controls the internal register-codelet shared-memory
handoff. It is separate from the existing `shared_layout=writer-aligned`
contract for the lowering. The XOR choice changes the matching write/read
addresses without adding shared storage. `cufftdx-thread` replaces the FP32
prefix's local thread FFT, retaining the surrounding dataflow and suffix.
The FP64 native and radix4 cores support both layouts; an explicit FP64
`cufftdx-thread` request is rejected rather than silently substituted.
The linked radix4 inventory covers the FP32 linear, one-lane, local-stage-8
provider. Other legal radix4 configurations use on-demand compilation;
`precompiled` mode reports uncovered configurations unavailable.

The radix4 codelet combines pairs of FFT stages and retains a radix2 remainder
for odd local logarithms. It supports FP32/FP64 and the same rectangular,
cooperative-lane and shared-layout geometry as the register-prefix template,
subject to the existing resource checks. FP64 invokes the selected codelet
instead of bypassing it with native arithmetic. JIT projection and candidate
enumeration preserve this identity. Long 1D requests retain their cross-factor
twiddles and global ordering; they are not replaced by separable 2D FFTs.
The shared codelet header is installed and participates in the JIT source hash.
Portable calibration seeds accept this core for FP32/FP64 FFT, and bounded
post-install tuning can propose both native-to-radix4 and radix4-to-native
neighbors. A core change resets dependent physical group metadata; it does not
reuse the previous core's measured time as a calibrated result.

For a prefix of size `R*R`, `prefix_codelet_lanes=G` splits each length-`R`
FFT among `G` lanes. Each lane runs a length-`R/G` register FFT, applies its
phase, then exchanges results through a warp FFT. This changes the compute
mapping without reducing the independent contiguous I/O columns `C`:
`prefix_threads=R*G*C`, `prefix_ept=R/G`, and shared storage remains
`R*R*C*sizeof(Complex)`. Static execution descriptors and captured launch
checks both derive `C=threads*EPT/(R*R)`; deriving it from `threads/R`
would misreport cooperative grids and data folding. `G` must divide `R`;
for `G>1`, `G*C<=32` and the
CTA must fit the device thread and register limits. XOR shared addressing
also incorporates the lane digit through matched write/read offsets.
For example, local stages `12`, `G=2`, and `C=4` use `512` threads,
EPT `32`, and 128 KiB of shared storage in FP32. More lanes can raise
occupancy but add shuffle/phase work, so this is a search choice rather
than a guaranteed improvement. Cooperative variants require JIT;
the default `G=1` preserves the prior precompiled inventory.

The `G=2/4` lowering uses a small DIF exchange. Radix-4 leaves the lane
frequency digit in bit-reversed order; the existing shared/global write
addresses consume that digit directly, with the shared read using the matched
physical map. It therefore avoids a shuffle solely to restore natural lane
order. Twiddle selection precedes one complex multiply, and the radix-4
quarter-turn uses a component swap and sign change. This changes the lowering,
not the logical partition, public output order, or independent column count.
The matched lowering study (local artifact: `../results/paper_completion_20260918/fft_leaf_lowering/REPORT.md`; not included in this source release)
measures its effect; more lanes still need measured selection.

For example, the following extends any otherwise valid register-prefix mapping:

```json
{"prefix_codelet":"cufftdx-thread","prefix_shared_layout":"xor"}
```

A factor mapping with `factor_partition=[11,12]` can request
`factor_io_policies=["static-unrolled","dynamic"]`. Entries correspond to
physical factors, independently of logical macro-stage grouping. Static I/O
exposes a fixed thread count and iteration count to the compiler. It can
improve address scheduling, but can also increase live registers and spills;
each stage can therefore choose its own policy. The default remains dynamic.

These fields survive serialization, JIT projection, candidate mutation and
precompilation. Physical group descriptors report the selected codelet and
I/O policy, and stage-service identities distinguish affected implementations.
A new strategy needs matching stage measurements; an old strategy's timing
does not become calibrated evidence merely because dimensions match. Search
retains alternatives and measures winners within its declared budget; no
strategy is assumed universally faster.

## FFT factors and data prefetch

The `factor-streamed` lowering separates these mapping axes:

| Mapping field | Meaning |
| --- | --- |
| `stage_partition` | Logical macro-stage partition of logN |
| `factor_partition` | Physical FFT factor partition; each factor uses one whole launch or several partial launches |
| `fft_core` | Imported `cufftdx-block` or local `register-tile` processing core |
| `factor_ept`, `factor_columns` | Local core shape and independent FFTs per CTA |
| `data_tiles_per_cta` | Consecutive independent data tiles processed by one CTA, including at batch=1 |
| `prefetch_depth` | Number of future input **groups** to prefetch; a group contains `factor_columns` complete local FFTs; zero selects synchronous loading |
| `factor_slices`, `factor_overlap` | Power-of-two dependency-closed slices of the unprocessed digit; optional cross-factor streams |

For example, one logical macro stage can contain three physical factors:

```json
{"backend":"factor-streamed","fft_core":"cufftdx-block",
 "shared_layout":"writer-aligned","cross_twiddle":"recurrence",
 "stage_partition":[24],"factor_partition":[8,8,8],
 "factor_ept":16,"factor_columns":8,
 "data_tiles_per_cta":4,"prefetch_depth":1,
 "factor_slices":2,"factor_overlap":true}
```

Pass this object through `PlanOptions.mapping` or `--mapping-json`. Macro
endpoints must coincide with factor endpoints. Each factor uses a shared
writer layout and writes an explicit global digit rotation. Its successor
consumes that layout directly, without a separate full transpose. Even within
one macro, physical factor boundaries materialize in global scratch and remain
visible in `execution_boundaries`; macro grouping alone does not fuse launches
or make their state resident on chip.

On SM80+, positive prefetch depth uses `cp.async` to load a later independent
data tile while the current tile executes. Input buffers and the local core's
shared workspace are accounted separately. Depth zero reuses one shared tile;
positive depth adds the input ring. This creates lookahead along the data
dimension at batch=1. With `factor_slices>1`, each non-final factor can publish a
dependency-closed periodic slice of the unprocessed digit. `factor_overlap=true`
uses one private stream per physical factor and a ready event per slice; the
final factor waits for all slices because each of its local transforms consumes
as many producer transforms as the final factor's size. `factor_overlap=false` is the serialized control
using the same partial ranges. The existing batch-level `stage_overlap` scheduler
is a separate lowering and cannot be combined with this mapping.

The ring unit is currently coupled to the core group, not an independent
transfer chunk. For local size `F`, logical columns `C`, value bytes `W` and
depth `D`, let `Q=F*C*W`. Effective depth `E` is zero when
`data_tiles_per_cta=1`, and otherwise equals `D`. At `E=0`, the dynamic shared
allocation is `max(Q, core_shared_bytes)`. At positive depth it is
`(E+1)*Q + max(Q, core_shared_bytes)`: the ring holds the current input group
and `E` future groups, while the separate workspace serves the core and output
layout. Thus depth one overlaps a future input group with the current group's
computation; it does not mean two shared buffers in total. Neither `N`
nor batch multiplies this per-CTA allocation when the local geometry is fixed.
Search must constrain the complete shape/buffer combination. A rejected large
group does not invalidate smaller groups with the same prefetch mechanism.
Independent transfer chunking inside this provider call remains unimplemented.

The slice contract is derived from address matching, not from a contiguous CTA
tile assumption. For factor `g`, `P=2^Done` and last-factor size `K`, a tile row
advances by `K*P/factor_columns`; a slice selects a contiguous subrange of that
row. The CPU dependency oracle in `scripts/factor_dependency_oracle.py` checks
the producer/consumer fan-in, packet closure, and no cross-batch overlap for
small exact instances. It does not claim that every abstract stage schedule is
lowerable.

The compiler resource query reports each factor's actual registers, static and
dynamic shared memory. Optional `cubutterfly_module_local_bytes_v1` reports
static per-thread local allocation. `compiler_local_resources_known=false`
means unavailable; it must not be interpreted as zero. Static allocation is
not a count of dynamic spill transactions or HBM bytes; those require profiling.
Resource estimates use the physical groups even when the macro count is one.

The current template supports native FP32/FP64 complex FFT, both directions,
in-place/out-of-place execution and positive nonoverlapping strides. The imported
cuFFTDx core also supports native FP16 forward and unnormalized inverse. Its
`half2` lanes represent two independent complex transforms: `factor_columns`
counts logical transforms, while physical `blockDim.y` is `factor_columns/2`
and total threads are `(F/factor_ept)*(factor_columns/2)`. Logical columns must
be an even power of two from 2 through 32 and fit the independent digit domain
at every factor. Shared-memory accounting uses four bytes per logical complex
value, not eight bytes per packed pair multiplied by the logical column count.
FP16 normalized inverse, BF16-native and explicit FP32 accumulation are not
implemented by this factor provider; requests are rejected rather than silently
changing arithmetic. The register-tile provider remains FP32/FP64.
Factor
sizes are powers of two, bounded by the imported core and device resources.
Search enumerates resource-feasible factor counts independently of data folding
and prefetch. It visits balanced representatives before other compositions;
this is a bounded exploration order, not a fixed factorization. The partition
budget applies per core/buffer configuration. Explicit replay permits more
configurations than the default EPT/column/tile/depth search grid.

The local `register-tile` core applies two existing register FFT codelets with
a shared handoff whose write offsets align the next codelet's read layout.
Internal cross factors use recurrence, with a direct phase reset every eight
values. This core currently requires even-log factors and
`factor_ept = 2^(factor_log/2)`; the common EPT means its factors currently have
equal size. The imported core supports additional, unequal shapes. Both cores
use the same global factor layout, data loop, prefetch ring and module ABI;
neither is forced as the runtime winner.

Calibration continues round-robin exploration across lowering/core families.
Within the factor family, it stratifies by physical factor count, input-buffer
depth and data tiles per CTA. A single seed no longer exhausts a new family's
exploration while legacy prefix/suffix variants consume the remaining budget.

`tests/check_factor_streamed.py` checks the public execution contracts.
`scripts/benchmark_factor_streamed.py` journals a controlled factor/prefetch
ablation against cuFFT. These explicit-mapping experiments are separate from
recalibration and the full automatic-selector acceptance matrix.
Native FP16 coverage is checked by `tests/check_fp16_factor.py`; the bounded
study in `config/fft_l_factor_study.json` is prepared by
`scripts/prepare_fp16_factor_study.py` and executed by
`scripts/run_fp16_local_core_study.py`. Its CPU preparation and cache-only GPU
execution remain separate. See [the dated integration report](fft_l_local_core_20261105.md)
for measured configurations and the outstanding calibration scope.

## Installation and calibration

```bash
PYTHON="$CONDA_PREFIX/bin/python" scripts/install_with_hardware_profile.sh \
  --prefix "$CONDA_PREFIX" --enable-cufftdx \
  --search-seconds 300 --compile-seconds 600
```

Architecture detection uses the first visible GPU; cross compilation supplies
`--cuda-architectures`. Profiles are stored below
`share/cuButterfly/hardware/<model>-sm<SM>-<memory-bytes>B`. Different capacities
remain different targets. Calibration updates the cumulative registry in
`$XDG_CACHE_HOME/cubutterfly/registry.json` (normally `~/.cache/cubutterfly`), or
`CUBUTTERFLY_REGISTRY`. It does not rewrite the frozen legacy selector or require
a second library build. Registry and user measurement-cache writers use locks.

Optional [platform tuning rules](platform_tuning.md) contribute mapping priors
to the same exported candidate inventory. The generic selector contains no
GPU-model dispatch table. Exact measured registry records take precedence;
DEFAULT otherwise takes the first legal unmeasured candidate, while MEASURE
and installation search can time the candidates. Rule-file contents participate
in selection-cache and calibration identities.

The search and additional compilation budgets are separate; zero disables each
limit. The search budget is distributed over workload cells. These budgets cover
additional search/specialization, not the initial library build, capability
probes, regression tests or confirmation of an already started finalist.

Both benchmarks export `--list-design-points` and accept `--mapping-json`.
Installation reads these mappings directly. A local model is fitted after valid
bootstrap measurements and then ranks candidates, with periodic exploration.
Only correctly verified, repeatedly confirmed records are promoted; source or
compile-policy changes invalidate measurement fingerprints. Omitted candidates
remain recorded. A registry miss chooses an executable fallback and labels it
unmeasured; it does not claim the fallback is optimal.

`CUBUTTERFLY_PARTITION_BUDGET` controls the shared-partition enumeration effort
(128 by default). The cursor cycles over every resource-feasible group count;
it is not a fixed list of two- or three-stage splits. Set it to zero to enumerate
all compositions. Exhaustive enumeration grows exponentially with stage count
and is separate from the budget for compiling or measuring exported candidates.

Performance promotion requires an exclusive GPU. If another job occupies it,
calibration stops with the conflicting process information and retains completed
trial checkpoints. CPU builds and host checks can continue. GPU correctness
runs follow the agreed device-sharing arrangement; their timings are not
calibration evidence. See the
[shared GPU recovery procedure](hardware_profile_install.md#shared-gpu-waiting-and-recovery)
for the separate resume flags and explicit waiting policy.

## Method alignment and acceptance

The shared graph/IR records per-group stage, data and batch unfolding, storage,
writer layout, live state, and compiler resources when known. Bench CSV exports
these separately as `execution_groups_json`; a mapping record does not mix in
timings. The model distinguishes unequal partitions with equal group counts.

The common lowering is a correctness-complete fallback within its resource and
numeric contracts. Specialized APPT streaming cores and imported local FFT
cores remain additional implementations. Some abstract stage-role schedules
still require a new lowering; they are not certified by the presence of an IR
record or by enumerating a candidate. Arbitrary global optimality is not an
acceptance claim. The expanded two-capacity performance matrix is tracked in
[framework_refactor_progress.md](framework_refactor_progress.md).

The FFT acceptance runner prepares the exact GPU-capacity workload matrix,
then compares the automatic selection with cuFFT using alternating repeated
trials. For example, with the research compile policy already set:

```bash
python scripts/run_fft_acceptance.py --build-dir build-a100-cufftdx \
  --output-dir results/fft-acceptance --prepare-only
python scripts/calibrate_local_hardware.py --build-dir build-a100-cufftdx \
  --profile-dir results/current-hardware --output-dir results/fft-calibration \
  --search-only --search-workloads results/fft-acceptance/workloads.json \
  --verify-batches 2
python scripts/run_fft_acceptance.py --build-dir build-a100-cufftdx \
  --output-dir results/fft-acceptance --resume
```

Defaults cover FP32/64, logN 3..24 and batch 1/4/16/64. Memory-clamped duplicates
are counted once and retain their requested batches. Resume requires identical
binary, GPU, compile policy and timing protocol. Uncalibrated fallback cells
are reported and cannot satisfy the acceptance gate. The full matrix can need
more search time than the default installation budget.

`--verify-batches 2` still executes the actual full batch, then checks its first
and last transforms against the CPU reference. CSV records `verified_batches`;
zero checks every transform, which remains the benchmark/calibrator default.
Only the acceptance runner opts into bounded checking by default. Compilation,
initialization and CPU verification are outside CUDA-event kernel time.

## Explicit-root RNS NTT integration

`prepare_rns_plan()` adds persistent, request-specific mapping preparation to
the C++ RNS entry point. Its `Prefer`, `Require`, and `Refresh` policies use the
complete semantics, placement, hardware capabilities, build/runtime identity,
and measurement protocol. `Require` never tunes on a miss. Both HEonGPU adapter
paths consume this cache; the additive `cubutterflyRns*` C ABI and
`cubutterfly.rns` Python module expose the same preparation and execution path.
Actual application requests can be exported and prepared offline with
`scripts/prepare_rns_requests.py`. See [prepared RNS integration](rns_prepared_integration.md)
for commands, ownership, timing, and Python callback contracts. This measured
cache does not imply integration with the general descriptor or installation
cost model. Records distinguish timing evidence from independent exact checks.

Call `plan.tune(RnsPlacement::InPlace, stream)` for aliased input/output, or
`RnsPlacement::Separate` for distinct buffers. The original `tune(stream)`
overload still measures separate buffers. Selection records the measured
placement; the HEonGPU adapter includes placement in its plan-cache key.
Tuning uses private canonical data and excludes initialization from event
timing. Aliased layouts that move rows between domains are reset between
single-transform timing samples, so replay cannot feed residues to the wrong
modulus.

Equal-offset, disjoint rows can execute fused CT/GS directly in place at its
native boundaries (forward natural to bit-reversed, inverse bit-reversed to
natural). Other boundary orders retain the two-pass scratch path. Global bit
reversal can cross CTA ownership and cannot safely overwrite unread input.
Explicit offsets also support sparse and permuted physical row lists.

Independent requests whose dependencies have completed may share a packed
arena and one `RnsPlanConfig.polynomials` list. This already provides an explicit
stage batch. An application must arrange the arena and lifetimes; the adapter
cannot postpone synchronous transform dependencies to collect future calls.
The stage-only diagnostic `benchmarks/rns_stage_batch_compare.cu` compares
looped and packed requests for both cuButterfly and pinned GPU-NTT, with CPU
exact checks. `--include-packing 1` adds identical gather/scatter kernels for
both libraries and measures their full cost on independently allocated request
buffers. These remain stage controls, not complete applications.

`cuntt::RnsPlan` (`include/cuntt/rns_ntt.hpp`) is an additive C++ entry point for
word64 polynomial batches with explicit per-domain roots, cyclic or negacyclic
rings, independent polynomial offsets, and natural/bit-reversed boundaries.
Inputs are canonical residues. It shares the existing Hybrid2D arithmetic
template with the single-domain plan and fuses twists, inverse scaling and
permutations into the two pass boundaries. All RNS domains participate in each
launch; the HEonGPU adapter no longer needs separate pack/NTT/unpack calls per
modulus when this mode is selected.

Its current compiled families cover logN 12..20. `tune()` measures a bounded
set of split, row, thread, radix, cross-twiddle and boundary-traversal configurations using private input
and CUDA Graphs before application capture. `measured-rns-request` reports this
request-local evidence; `calibrated=false` explicitly means the installation
cost model was not used. This interface has not yet been added to the general
C descriptor, global planner candidate inventory, or installation calibration
schema. Those remain integration work. Cross-ciphertext batching is now an
explicit HEonGPU Graph lowering option, described below; eager calls retain
their original dependency contract. See [CKKS comparison](../paper/experiments/README_ckks.md) for the
controlled legacy path, semantics and reproduction commands.

`adapters/heongpu/rns_stage_batch.hpp` binds independent RNS requests to their
actual source/destination pointers. It deduplicates domains by modulus and root,
gathers explicit polynomial offsets, executes a packed `RnsPlan`, and scatters
the results. Its native GPU-NTT control uses the identical packing layout and
deduplicated domains. Packing and scattering are part of the complete call.
Overlapping outputs or dependencies between different requests are rejected.
Within-request aliasing is safe because gathering snapshots every input before
any output is written. This adapter currently requires canonical word64,
negacyclic native boundary orders and logN 12..20. The pinned native control
further bounds polynomial count to `min(65535, INT_MAX/N)`.

`ntt_capture.hpp` and `ntt_graph_batch.hpp` preserve whole-application dependencies.
Each ciphertext is first captured in a private, persistent allocation arena.
Explicit NTT markers carry the prepared semantics and buffer bindings; these
graphs cannot run directly. `BatchedNttGraph` checks the entire stage sequence,
replaces each corresponding stage with one gather/NTT/scatter child graph, and
clones all remaining nodes and edges. Unsupported graph node types or incompatible
stage signatures fail explicitly. Each stage owns its workspace. The caller
keeps the source graphs, arenas, context and keys alive until execution ends.
`adapters/heongpu/graph_schedule.hpp` adds reusable complete-Graph selection.
Named launch callbacks supply equivalent, prevalidated candidates. `tune()`
measures alternating candidates on an explicit stream, stores every sample and
selects the minimum median. `launch()` reuses that instance's choice without
events, synchronization or new search. Callbacks borrow their graph/buffer
lifetimes; capture-time tuning and uninitialized selection are rejected.

The CKKS runner's `--auto-schedule --stage-batch --graph` supplies its existing
loop and stage-batched implementations. It validates both before selection,
then checks the selected path with changed/restored inputs and independently
measures it alongside the fixed controls. The `.selection.csv` and `.samples.csv`
sidecars distinguish selection from holdout timing. Selection is
`measured-full-graph-request`, not an installation-model prediction or a
cross-process cache hit. This does not close global RNS descriptor/model
integration. No shape, batch threshold or device-name rule fixes the winner.

This is a bounded one-device Graph lowering, not a transparent delay or merge of
ordinary eager API calls. `--stage-batch` compares this candidate against the
scalar-call loop; selection between these application schedules is still explicit.

The RNS boundary-traversal option changes lane-to-address mapping, not transform
order. Bit-reversed input/output were already directly supported without a
standalone permutation kernel. Physical traversal coalesces that boundary's
global addresses while preserving logical coefficient indices. Both traversals
remain searchable, as shared-memory costs can differ. Rows per block now range
over 1/2/4/8/16/32 for Hybrid2D, subject to the actual device's shared-memory and grid limits.

`RnsCoreFamily::FusedCTGS` adds the mature negacyclic fused-root CT/GS strategy
to the same request tuner. Its template binds stage intervals, independent
lines, threads, explicit roots, polynomial addresses and boundary orders. It
uses existing Shoup arithmetic and explicit CTA synchronization. Forward and
inverse are compiled separately; power-of-two indexing uses shifts and masks.
Radix2 and register-fused radix4 are executable candidates. Radix4 keeps two
adjacent stages in four registers before shared writeback and synchronization;
an odd local stage count has a radix2 remainder. Both variants enter the same
resource validation, dispatch and request-local measurements. Neither is forced
by application or device name. The family has no separate twist/cross operation;
incompatible mapping fields are rejected. It is an additional executable family,
not a claim that the global planner or installation model has been integrated.

`RnsCoreFamily::GpuNttMerge` exposes the pinned GPU-NTT Merge contract as a
separate searchable provider. It is legal only for negacyclic radix2 Barrett
execution with moduli below 2^62 and independently addressable, N-aligned
input/output rows with an explicit per-polynomial domain sequence. Logical
polynomial order may be a modulus-grouped permutation or sparse selection; the
compact stage uses each polynomial's physical offset rather than assuming
`index*N`.
The provider keeps the same stage descriptor and CUDA Graph timing protocol as
FusedCTGS. Equal input/output offsets and native CT/GS boundary orders preserve
the direct in-place path; other layouts or boundary orders use scratch. This constraint is checked by
the plan and is never inferred from a library or application name.
Both compiled addressing variants consume explicit polynomial offsets; their
names do not imply that physical-offset loads have been removed. The tuner also measures a
provider-specific 512-thread candidate for `N >= 2^16`; it is eligible only
when the device accepts that launch and is selected only when timing confirms
it.

FusedCTGS local stage logs range from 2 through 10; its power-of-two row groups
extend through 128, bounded by the independent tile count and device resources.
`second_rows_per_block` and `second_threads_per_block` independently configure
the second executed pass; zero inherits the first pass. Inverse transforms
reverse stage intervals, not these execution-order fields. Resource validation
and launch dispatch use the effective parameters separately for each pass.
The tuner retains coupled mappings and adds equal tile-word capacities with
independent threads per pass. For example, a 5+9 split with 16/1 rows represents
512 words in either tile without forcing the same row count. This is a bounded
search subset, not the full Cartesian product of independently legal mappings.
Explicit mappings can replay additional combinations. Changing the C++ mapping
structure requires rebuilding consumers against the matching header and library.


### Scalar online subgraph grouping

For the typed scalar `online-reorder` lowering, `data_tiles_per_cta` groups
independent prefix subgraphs and `reorder_columns` groups independent suffix
columns. The axes are independent; changing one does not silently change the
other or the arithmetic core. The complete descriptor and mapping retain both
values for replay and stage-service keys. The generic mechanism applies to FFT,
FWHT, zeta and structured operators supported by these traits; it does not add a
new grouping API to the separate NTT executor.

`online_reorder_contract.hpp` separates each local tile's compiled exponent range
from the grouped live-state byte requirement. Both counts are powers of two
bounded by their graph dimensions, with shared memory no greater than the
hardware budget or the current 48 KiB non-opt-in limit. Integer zeta uses its
normalized UINT32/UINT64 storage width for this check. Resource projection uses
the actual grouped CTA grids and per-stage shared bytes. Public candidates expose
single-axis and coupled representatives; local tuning proposes adjacent powers
of two and leaves byte-capacity decisions to the runtime contract.

Grouped writers directly materialize the existing edge layout with contiguous
lane ownership. The shared XOR permutation is internal and distinct from the
`writer-aligned` global address permutation. Grouping does not itself enable
asynchronous prefetch or stage overlap. Calibration must measure the resulting
core/geometry/service keys before claiming that automatic selection benefits.
See the [bounded evidence](../results/a100_fft_l_boundary_20261104/REPORT.md),
including the retained single-group regressions and small-batch limits.
