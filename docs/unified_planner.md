# Unified planner and specialization modules

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

The main C++ interface is `<cubutterfly/plan.hpp>`. `Transform` owns mathematical
semantics; `PlanOptions` owns selection policy or an explicit execution mapping.
C, C++, and the benchmark programs use the same registry and candidate projection.

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
context stream before constructing or executing plans. NTT uses distinct input
and output allocations and requires matching storage and `word_bits` (32 or 64).

The six mathematical operators are FFT, NTT, FWHT, subset zeta, superset zeta,
and structured 2x2 butterflies. The legacy `xor-zeta` name retains its subset-zeta
semantics. Floating low-precision traits retain their explicit rounding and
accumulation contract. Structured coefficients are semantic inputs, not tuning
parameters. Rank-2 and embedding plans serialize every physical axis; Bluestein
serializes both internal convolution transforms. External cuFFT remains an
explicit baseline option.

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
candidates expose bulk execution and power-of-two batch tile choices. Arbitrary
positive tile counts can also be replayed explicitly.

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
| `prefix_codelet` | `native` (default), `cufftdx-thread` | OnlineReorder register-prefix FFT |
| `prefix_codelet_lanes` | Power of two, default `1` | Threads cooperating on one register-prefix codelet |
| `prefix_shared_layout` | `linear` (default), `xor` | OnlineReorder register-prefix FFT |
| `factor_io_policies` | Empty/default, or one `dynamic`/`static-unrolled` entry per physical factor | FactorStreamed FFT |

`prefix_shared_layout` controls the internal register-codelet shared-memory
handoff. It is separate from the existing `shared_layout=writer-aligned`
contract for the lowering. The XOR choice changes the matching write/read
addresses without adding shared storage. `cufftdx-thread` replaces the FP32
prefix's local thread FFT, retaining the surrounding dataflow and suffix.
The FP64 native core supports both layouts; an explicit FP64
`cufftdx-thread` request is rejected rather than silently substituted.
Nondefault prefix strategies require on-demand compilation; `precompiled`
mode reports them unavailable.

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
| `prefetch_depth` | Number of input tiles buffered in shared memory; zero selects synchronous loading |
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
in-place/out-of-place execution and positive nonoverlapping strides. Factor
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
