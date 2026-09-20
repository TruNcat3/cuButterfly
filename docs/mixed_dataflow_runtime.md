# Mixed-Dataflow Runtime Migration

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

The unified plan is currently a projection of a resolved `ButterflyConfig`,
not a replacement for every operator-specific execution plan.

## Execution Status

| Path | Integration |
| --- | --- |
| Generic temporal, hierarchical, online-reorder, warp and stage-pipeline | `ButterflyPlan` launches through the typed `MixedDataflowPlan::dispatch` route |
| FWHT, scalar FFT, subset/superset-zeta, legacy XOR-zeta, structured-2x2 | Existing kernels retain their numeric, layout, stream and workspace contracts |
| Imported FFT codelets and cuFFT | Projected as `ExternalFft`; existing specialization checks and launch paths remain authoritative |
| NTT resident/dataflow | `Plan` exposes a unified NTT projection; the existing NTT runtime remains the execution authority |
| Two-group FFT batch stage overlap | Plan-owned producer/consumer CUDA streams, event dependencies and two bounded global scratch slots |
| Large FFT persistent CTA pipeline with on-chip cross-group exchange | Not implemented by this adapter |

`xor-zeta` retains its established subset-zeta/Mobius compatibility semantics.
This migration does not introduce an XOR-convolution transform.

## Validation Contract

- `make_dataflow_plan(config)` describes a legacy Butterfly or NTT configuration. It does not
  compile a kernel or assert executable support.
- `check_lowering(plan)` rejects unknown backends. Known legacy routes return
  `RequiresBackendValidation`: precision, compiled codelets and launch legality
  are still checked in the original plan constructor.
- `estimate_resources(plan, hardware)` supplies resource upper bounds. Missing
  compiler register counts are not evidence of zero register use, and this
  estimate does not certify occupancy or supply a calibrated latency.
- `validate_plan()` never promotes a description to executable. Only successful
  `ButterflyPlan` construction marks its owned projection executable.
- `ButterflyPlan::dataflow_plan()` returns that immutable projection. A caller
  cannot use an arbitrary edited IR to launch a kernel through this interface.
- `cuntt::Plan::dataflow_plan()` exposes the same graph/partition/boundary view
  for NTT. Its resident/dataflow CUDA kernels still use the validated NTT
  lowering; the projection is used for inspection and future shared lowering.

The graph keeps binary stages independent of the physical radix. Storage and
accumulator widths, complex values and strides are recorded separately. Each
logical edge retains its boundary policy; inspecting only the legacy singular
`boundary` field is insufficient for mixed fused/materialized decompositions.

For the generic hierarchical path, the projection records a resident local
prefix followed by one launch per remaining stage. This is not a persistent
CTA pipeline. Warp count is not interpreted as data-time unfolding.

## Verification

The host-only `mixed_dataflow` CTest covers projection, all graph operator
names, unsupported routes, resource bounds and invalid partitions. It does
not require an available GPU, and its checks remain enabled in Release.

`cubutterfly_correctness` executes the real kernels and checks the resolved
dataflow contract along with reference output comparisons. Run it on an idle
CUDA device after building `cubutterfly_tests`.

No new latency or cross-library speedup is claimed by this migration.

## Two-stage batch scheduling

Set `ButterflyConfig::stage_overlap = true` and `batch_tile_count` to choose
the number of independent transforms per tile. The CLI equivalents are
`--stage-overlap --batch-tile-count N`. The supported lowerings are two physical
execution groups of scalar `shared-iterative` FFT and `online-reorder` with
`cufftdx-block`, subject to the existing precision/unit availability checks.
Unsupported configurations fail construction. Logical partition, local FFT
unit, and boundary layout remain separate choices.

The producer writes tile `i` into slot `i % 2`; the consumer waits for that
tile, executes the suffix and any output transpose, then releases its slot.
The producer can execute tile `i+1` while the consumer executes tile `i`.
Reusing a slot waits for its consumer, including the suffix's in-place writes
to scratch. A final event joins both stages back to the caller's stream.
Plan resources are created once, and execute performs no stream creation or
host synchronization. Submissions through one plan must be host-serialized;
sequential submissions may change caller streams. Buffers must remain alive
until the caller stream completes. CUDA graph capture is not validated here.

Workspace per intermediate buffer is
`2 * batch_tile_count * batch_stride * sizeof(value)`, independent of total
batch. Prefix transpose needs two such buffers. Input/output and twiddle
storage are separate. This schedule still communicates through global memory:
`inter_stage_global_bytes` describes one logical intermediate footprint, not
the ring allocation or total read-plus-write traffic. It does not assert
cross-CTA shared-memory residency or persistent CTAs.

Installation search enumerates bulk and power-of-two batch tile schedules for
each compiled two-group cuFFTDx mapping. Runtime accepts any positive tile
count; enumeration is a screening policy, not a restriction of the theoretical
space. A tile at least as large as batch has no inter-tile overlap opportunity.
The selector serializes the measured schedule. Cost-model features include
schedule/tile count; older fitted models require refitting. No speedup or
optimality follows from enabling this flag alone.

Run `cubutterfly_stage_overlap_tests` for stream joins, slot reuse, tail tiles,
strided padding, external-workspace guards and in-place/forward/inverse FFTs.
`scripts/benchmark_stage_overlap.py` compares identical bulk/overlap mappings
against cuFFT with full-batch preflight verification and repeated CUDA-event
timing. Compilation and plan construction are outside the timed interval.
Use Nsight Systems with CUDA tracing, export SQLite, and run
`scripts/analyze_stage_overlap.py trace.sqlite --require-overlap` to check
actual intersection of stage-1 and stage-2 GPU timestamps. NCU replay is not
used as evidence of concurrency.

See the A100 implementation and paired measurement report (local artifact: `../results/stage_overlap_a100_20260911/report.md`; not included in this source release)
for verified overlap, correctness checks, and cases where bulk remains faster.

## Register subgraphs and grouped natural-order output

`--fft-core register-tile --backend online-reorder` composes a native register
prefix with grouped cuFFTDx suffix codelets. The prefix keeps two small FFT
codelets' intermediates in registers/shared memory. The suffix computes several
independent local FFTs in one CTA and exchanges their outputs in shared memory
before writing natural order. There is one global intermediate boundary and
no separate global transpose. This is local subgraph residency, not a promise
that the entire large FFT fits in one CTA or that kernels run persistently.

Use `--list-register-tile-mappings` to inspect the linked library's compiled
and resource-validated points on the current GPU. The initial FP32 projection
has prefix logN 6/8/10 and suffix logN 10/12 (including total logN=22). Prefix EPT is the
square root of its local FFT length; prefix and suffix CTA widths are independent.
Forward/inverse, optional inverse normalization, in-place/out-of-place, strided
data, nonblocking caller streams, and external workspace are supported. This
family uses `writer-aligned` layout, recurrence roots and two physical groups.
Bulk and `--stage-overlap --batch-tile-count T` execution use the same cores;
the latter allocates two global tile slots. FP32/64 on-demand specializations
support the same scheduling contract. This does not add CTA warp-role schedules.

For example, this is one measured explicit mapping, not a universal default:

```sh
build-a100-cufftdx/cubutterfly_bench --operator fft --precision fp32 \
  --backend online-reorder --fft-core register-tile --logN 20 --batch 16 \
  --stage-partition 10,10 --prefix-threads 256 --prefix-ept 32 \
  --suffix-threads 256 --suffix-ept 16 --shared-layout writer-aligned \
  --cross-twiddle recurrence --normalization none --verify
```

The installer reads the new inventory and measures these points alongside the
existing cuFFTDx adapters. Its serialized mappings preserve separate segment and
group core identities. Whole-library cuFFT remains an external reference.
See the accuracy, NCU, and paired performance report (local artifact: `../results/fft_register_tile_20260912/report.md`; not included in this source release).

To compare schedules without hard-coding a split or launch shape, first calibrate
the requested cells, then run `scripts/benchmark_stage_overlap.py --auto-mapping`
with `--binary`, `--output`, `--precision`, `--logs`, `--batches` and `--tiles`.
It freezes the selected full mapping per cell and changes only the batch schedule.
Compilation and numerical preflight are excluded from its timing trials.
