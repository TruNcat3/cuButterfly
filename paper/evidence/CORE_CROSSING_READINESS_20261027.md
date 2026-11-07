# M3 core crossing readiness, 2026-10-27

## Decision and scope

The current source can express a finite FWHT local-radix x column-grouping
crossing without a new arithmetic implementation. The attached
[study](../../configs/experiments/core_crossing_20261027.json) now contains four
requests and sixteen cells across FWHT and NTT. CPU structural validation and command lowering pass;
runtime legality, compiled-source compatibility, numerical acceptance and timing
remain untested. This is preparation, not completed M3 evidence.

NTT has two identifiable leaves with a common runtime organization control.
Its benchmark can express all eight cells. The controlled-study runner now accepts
explicit nested dot-path controls and checks the entire controlled mapping array;
the missing audit interface found in the first inspection has been repaired and
CPU-tested. The runner source identity changed, so existing frozen campaigns must
rebind that identity before reuse. No GPU was queried or launched, and no
historical queue/result was changed.

These are bounded crossed interventions. FWHT changes suffix column grouping;
NTT changes batch traversal inside resident producer/consumer roles. Neither
constitutes a searched conventional-versus-hybrid comparison or an automatic
library performance result. M3's stronger comparison still needs matched
organization search and independent finalist confirmation. Historical FFT four-cell
evidence is retained; it needs no rerun here. E3 remains a one-core organization
control for each arithmetic family.

## FWHT: executable source-level crossing

Semantics: forward, unnormalized, out-of-place FP32 storage/arithmetic, native
accumulation, natural layout, N=65536, B=4 and 16. Both requests use stage
partition 8+8, 128 threads, shared local exchange, linear boundary layout, and
the scalar backend implementation. Batch points test a fourfold work-domain
change with the same leaves; they were selected before timing.

| Factor | First level | Second level |
| --- | --- | --- |
| Local algorithm | radix-2, `compute_unit=radix2` | paired-stage radix-4, `compute_unit=radix4` |
| Organization | `reorder_columns=1` | `reorder_columns=4` |

`src/operator_traits.cuh:87` supplies `FwhtOperator<float>` and its exact add/sub
pair. The two distinct local algorithms are the `Radix4=false/true` branches of
`hierarchical_prefix_kernel<...,8,...>` and
`online_reorder_suffix_kernel<...,8,...>` in `src/stage_pipeline.cuh:19` and
`:174`. The radix-4 branch keeps four values in registers through two stages;
the radix-2 branch materializes each local stage to shared storage. These are
existing scalar local algorithms, not the Dao-inspired register FWHT provider.

For each row, changing columns keeps both 256-point subtransform widths, stage
order, radix, precision and source branches fixed. It changes the suffix work
assignment and shared footprint. Prefix launch: B*256 CTAs and 1024 shared bytes
per CTA in both columns. Suffix launch: B*256 versus B*64 CTAs; shared storage
1024 versus 4096 bytes per CTA. Both use 128 threads. Thus B=4 gives 1024/256
suffix CTAs and B=16 gives 4096/1024. Counts are launches, not occupancy.

The scalar constraint in `src/butterfly.cu:893` permits both settings because
columns are powers of two and columns*256 <= 1024. The largest grouping is the
existing automatic default for this shape; c1 is a native one-column scheduling
choice. Neither column is declared an optimized strong baseline. No deliberately
stagewise `[1,...,1]` comparator is included. Changing radix is the core factor;
it intentionally changes local fusion and synchronization and must not be
attributed to organization.

Codelet invariance here means identical source algorithm and local width within
each core row, not identical generated machine code or an opaque callable leaf.
The wrapper is templated and inlines the arithmetic. Record compiler/module
identity and resource reports before claiming stronger binary invariance.

The apparently stronger alternative, shared-iterative versus register FWHT,
cannot currently supply a valid crossing: `register_fwht.cuh` is a whole-transform
temporal-tile kernel (generated logN 3..15), while `butterfly.cu:855` confines
warp-register exchange to temporal-tile. It is not a callable per-segment leaf
in shared-iterative. Radix-4/8 are also explicitly rejected for shared-iterative
at `butterfly.cu:1167`. A second core name alone does not fill those cells.

## NTT: executable source-level crossing

Common semantics: N=2^20, B=4 and 16, word64, forward, natural input/output,
out-of-place, q=576460756061519873, Shoup multiplication, stage partition 10+10,
full-scratch handoff. `ntt_template_candidates.hpp:15` checks prime q, N|(q-1),
word width and order. Both cores are in the existing executable inventory.

| Core | Local identity | Fixed physical mapping |
| --- | --- | --- |
| `homogeneous-radix4` | `hierarchical_resident_radix4<false/true,uint64_t,10,4>` in `src/hybrid_dataflow.cuh:540`, called by `hierarchical_homogeneous_two_level_kernel` | 256 threads, 4 units, data_space=1 |
| `homogeneous-warp-radix2` | `hierarchical_warp_register_subgraph<false/true,uint64_t,10>` at `src/ntt.cu:3344`, calling `appt_apply_cached_register_stage` | 256 threads, 8 units, data_space=1 |

Each row has `subgraph_mappings[0].data_time` and `[1].data_time` jointly set to
1 or 2. `ntt.cu:3232` and `:3420` place this traversal outside the unchanged
local leaf. The control changes packet ordering across transforms; it does not
add intra-tile prefetch or remove the complete-transform producer barrier. Both
settings use the same core-specific launcher instantiation. Shared-radix4 uses
4*(1024+1)*8 = 32800 dynamic shared bytes/CTA; warp-radix2 requests zero dynamic
shared bytes, with 32 values/lane and register usage to be obtained from the
build. Per-core resource differences are part of the core factor.

Keep `cta_weight=1` for both roles, `role_stages=1`, `token_interleave=1`,
`coefficient_reuse_stages=0`, full-scratch buffers=1, and target residency policy
fixed within each row. The launchers (`ntt.cu:6567` and `:6628`) cap the persistent
grid by measured kernel occupancy and `target_ctas_per_sm`; resolved grid,
weights and occupancy need recording. Do not force target residency=1 merely
to generate a large apparent benefit. B>=4 makes both data-time levels nontrivial.

The existing direct CLI can express one cell as follows; expand the two core
names, B=4/16 and DT=1/2 into eight cells, using UNITS=4 for shared-radix4 and
UNITS=8 for warp-radix2. Variables below are explicit run inputs, not frozen
hardware identities. This is a command template, not a measurement launched by
this preparation:

```bash
CUDA_VISIBLE_DEVICES="$M3_GPU_UUID" CUBUTTERFLY_COMPILE_MODE=research \
  build-contract-20261017/cuntt_bench \
  --backend hierarchical-dataflow --logN 20 --batch "$M3_BATCH" \
  --word-bits 64 --modulus 576460756061519873 \
  --input-order natural --output-order natural --mod-multiply shoup \
  --stage-partition 10,10 --segment-cores "$M3_CORE" \
  --segment-threads 256 --segment-units "$M3_UNITS" \
  --segment-data-space 1 --segment-data-time "$M3_DT" \
  --segment-role-stages 1 --segment-token-interleave 1 \
  --segment-coefficient-reuse-stages 0 --segment-cta-weights 1 \
  --boundary-storage full-scratch --boundary-buffers 1 \
  --warmup 0 --repeat 1 --verify --verify-batches 0 --csv
```

Timing follows only after exact full-batch preflight, with verify outside timing,
three independent trial processes, 100 warmups and 100 repetitions per cell.
The checked-in study expresses these same cells through mapping JSON and the
controlled runner, which also supplies trial journaling and mapping validation.

The original missing interface was recursive comparison of requested and resolved
`subgraph_mappings` fields. Those arrays were discarded as derived, and a CPU
reproduction returned an empty difference for nested data_time=1/2. The fix uses
`changed_axes=["subgraph_mappings.0.data_time", "subgraph_mappings.1.data_time"]`.
Each path must name an explicitly requested scalar in both variants. Declaring
one nested path promotes the entire corresponding mapping array to recursive
comparison; a changed core, thread count, list shape or other field then needs
its own declaration. A blanket array-root declaration does not authorize changes.

Preflight also checks every explicitly requested leaf in a promoted array against
the resolved plan before resource probing or timing, so both variants silently
switching cores is rejected. Unrequested resolved fields remain visible in the
recursive comparison. The existing exact preflight-to-timing mapping comparison
continues to protect each trial. Arrays without a declared nested path keep the
legacy derived-field behavior. Supported nested roots are `subgraph_mappings`,
`segment_mappings`, and `execution_group_mappings`; paths have the form
`root.index.field`. Explicit controls require exact replay; automatic sentinel
values should be resolved and frozen before using this control mode.

NTT organization comparisons declare only the two data_time paths. Cross-core
comparisons additionally declare the two `.core` and two `.units_per_cta` paths,
because the mature leaves have different spatial-unit contracts. They measure a
core replacement including its required mapping; they do not isolate radix alone.
No new arithmetic core was introduced.

## Build, preparation and execution boundary

Existing build `build-contract-20261017` has both benchmark binaries, CUDA
architecture 80, `/usr/local/cuda/bin/nvcc`, and source root matching this checkout.
Those facts do not prove its binaries match today's dirty source. Before GPU use,
build in the authorized build workflow and freeze its library, binaries, compiler,
compile policy and JIT module/source identities. Relevant build commands are:

```bash
CUDA_VISIBLE_DEVICES='' cmake --build build-contract-20261017 \
  --target cubutterfly_bench cuntt_bench cubutterfly_stage_microbench -j 2
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python \
  scripts/run_controlled_experiment.py --mode prepare \
  --study configs/experiments/core_crossing_20261027.json \
  --build-dir build-contract-20261017 \
  --output results/core_crossing_20261027/prepared.json
```

These commands are provided, not executed; rebuilding a shared build changes its
identity. The CPU verification below used `prepare_document` in memory and wrote
no preparation journal. Future guarded execution uses a fresh attempt directory,
a selected UUID/model/capacity and an explicit deadline:

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/run_isolated_gpu.py \
  --gpu-uuid "$M3_GPU_UUID" --until "$M3_DEADLINE" \
  --report-dir results/core_crossing_20261027/guard -- \
  env CUBUTTERFLY_COMPILE_MODE=research \
  /home/wt/yes/envs/cubutterfly/bin/python scripts/run_controlled_experiment.py \
  --mode run --gpu-uuid "$M3_GPU_UUID" \
  --study configs/experiments/core_crossing_20261027.json \
  --build-dir build-contract-20261017 \
  --output results/core_crossing_20261027/timing.json
```

The runner records seed-derived benchmark inputs: FWHT uses
`0x43554246 + B`; NTT uses `0x43554e5454 + logN + B`. Save actual inputs or their
hashes, all requested/resolved mappings, trial order, and complete numerical
error distributions. Existing floating pass thresholds do not replace the user's
unresolved floating acceptance policy. Integer results require exact equality.

Budget: eight cells/operator, eight full-batch preflights and 24 timed processes
under the current controlled runner; at 100+100 calls/process, 4800 timed-process
invocations plus eight preflight calls/operator. FWHT launches two kernels/call,
so this is 9616 kernel launches excluding runtime setup/probes. NTT launches one
cooperative kernel/call, with initialization inside it. Root tables, allocations,
compilation and CPU references stay outside kernel timing. These are work counts,
not wall-time predictions. Build duration, first-call compilation, CPU-reference
cost and GPU completion time have not been measured for this cohort. Do not infer
an elapsed-time promise from historical latencies. The combined study totals
16 cells and 48 timed processes; no further matrix is implied by a negative
result.

## Source identity and CPU verification

Checkout HEAD: `45de6b5d0382c8ad050f344aeda33df5c1ba1fcd`; the worktree is dirty.
The following SHA256 values identify inspected source, not compiled binaries:

| File | SHA256 |
| --- | --- |
| `src/operator_traits.cuh` | `0b92b82e374ac06f429f5cc328341bf6323278ba6c06b6f46468ba4ba6b764d6` |
| `src/stage_pipeline.cuh` | `3b2262d3fb776047b3c470d270f8c20916f08ff2676311854835d39eb21d17cc` |
| `src/register_fwht.cuh` | `476f0fe1d635e25e5210bb9d9984f7aa62a7ef8abbe061ff71ec667c65f0b09c` |
| `src/butterfly.cu` | `5112558da05d0f08c759b3b09d6d0120c2ecbf9220ad73d05057abeece812c21` |
| `src/ntt.cu` | `db136b3c3d19ccfde7129fd8a73b964602bef37182698a52111dda8c256e2547` |
| `src/hybrid_dataflow.cuh` | `01519f54333adbe6c76c9e39d5392dbb20af16e3e09455c6dd4ae3b5d309d98e` |
| `src/ntt_traits.cuh` | `bc278ca03c9c49d4d230418ba2385b40ad2b1a604d7aa44d48932d4bc3b3344a` |
| `src/ntt_template_candidates.hpp` | `79ae28ef355c3ac643774b5236db965454669aca14ce41aa903df70a8d668002` |
| `scripts/run_controlled_experiment.py` before nested-control repair | `42f486006e4411721facb5e23dfd43b76ce4630124e31f46920c8f379fb32039` |
| `scripts/run_controlled_experiment.py` after nested-control repair | `210fcec35b4fd36372a0fe849c30caf4cc564b68ce1959ef0a9a310a6e41a6ea` |

Validation performed with `CUDA_VISIBLE_DEVICES=''` and the existing Python env:
`prepare_document` accepted 4 cases, 16 cells and 16 comparisons, reported
`gpu_queried=false` and `executable_claim=false`. The runner and nested-control
tests cover accepted data_time changes, aliases, undeclared core/thread changes,
unrequested resolved-field drift, array-shape drift, exact preflight replay and
legacy controls. The following CPU-only command passed all 26 tests:

```bash
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python -m pytest -q \
  tests/test_controlled_experiment.py tests/test_controlled_nested_mappings.py
```

No correctness/performance result or fixed-core organization speedup was produced.
