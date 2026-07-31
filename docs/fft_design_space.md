# FFT Architecture Design Space

## Scope

The canonical specification is `config/fft_architecture_space.json`. It defines
the architecture independently of the CUDA specializations currently compiled.
The model has five layers:

1. Semantic: length, precision, direction, placement, normalization, and layout.
2. Decomposition: single-unit, two-dimensional, or multi-dimensional stage partition.
3. Per-dimension mapping: spatial scope, stage fusion, temporal reuse, core,
   threads, elements/thread, units/CTA, exchange, padding, and pipeline buffers.
4. Boundary: residency, permutation, cross-twiddle policy, and global/shared handoff.
5. Hardware: SM count, thread/shared/register limits, and scheduling capacity.

`units_per_cta` is recorded as a derived quantity when the codelet uses a fixed
elements/thread contract:

```text
units_per_cta = threads * elements_per_thread / dimension_size
tile_shared_bytes = threads * elements_per_thread * sizeof(complex<float>)
```

The relation must produce an integer and cover at least one unit.

The architectural terms remain independent of a particular codelet:

- `spatial_scope` states whether concurrent work is owned by a thread, warp,
  CTA, or cooperating CTAs;
- `stage_fusion` states how many equivalent butterfly stages form one operation;
- `temporal_reuse` states whether values survive one stage, a fused group, a
  resident codelet, or the full dimension;
- threads, EPT, and units/CTA are the physical realization selected for that
  architecture point.

## Candidate States

The enumerator never treats the current binary as the definition of the space.
Every point is assigned one state:

| State | Meaning |
|:--|:--|
| `compiled` | A matching specialization exists in the current build. |
| `awaiting-codegen` | Hardware/resource constraints pass, but the specialization is not emitted. |
| `requires-new-kernel` | The point needs a new layout, residency, or mixed-core implementation. |
| `hardware-infeasible` | Coverage, CTA-thread, or shared-memory constraints fail. |

For V100 and `logN=16,18,20`, enumerating both twiddle modes, six thread
budgets, seven EPT values, every split, and both dimensions produces:

| State | Points |
|:--|--:|
| compiled | 2,048 |
| awaiting code generation | 10,878 |
| requires a new kernel/layout | 10,344 |
| hardware infeasible | 156,658 |
| total | 179,928 |

These counts describe coverage, not a performance sweep. Generate them with:

```bash
./scripts/fft_design_space.py --logNs 16 18 20 \
  --summary results/fft_design_space_v100_summary.json
```

Emit only a selected class for inspection or code generation:

```bash
./scripts/fft_design_space.py --logNs 18 --status awaiting-codegen \
  --threads 128 256 512 --ept 4 8 16 \
  --output results/fft_design_space_logN18_codegen.csv \
  --summary results/fft_design_space_logN18_codegen_summary.json
```

## Build And Runtime Contract

CMake validates and copies the canonical specification into the generated build
directory. `config/v100_fft_codegen.json` separately selects the specializations
compiled into this build. The generator validates every selected thread/EPT pair
against the canonical axes and V100 shared-memory/thread limits, then emits
availability, dispatch, and a resolved build manifest.

The current selection contains EPT 4/8/16 and 128/256/512/1024-thread points;
1024/16 is excluded because its tiled storage exceeds the V100 per-CTA shared
memory limit. The public FFT configuration records independent prefix/suffix
thread and EPT values plus normalized units/CTA. Feasible but unselected values
fail explicitly as `awaiting code generation`; they do not fall back to EPT=8.

Regenerate an isolated manifest with:

```bash
./scripts/generate_fft_codegen.py \
  --space-spec config/fft_architecture_space.json \
  --selection config/v100_fft_codegen.json \
  --output-dir /tmp/cubutterfly-fft-codegen
```

The hierarchical performance explorer reads the same family definition to get
compiled thread options and dimension ranges. Hardware profiles remain separate:
V100 is measured, while A100/H100/RTX 4090 entries in
`configs/hardware/cross_gpu_matrix.csv` remain placeholders until captured.

## Remaining Implementation Families

Completing the model does not imply every family is implemented. The major
implementation queues are now explicit:

- optionally emit the remaining cuFFTDx online EPT 1/2/32/64 points selected by a target profile;
- fuse the implemented directional tiled transposes into adjacent work or retain
  their layouts across multiple dimensions; standalone prefix and suffix
  transpose kernels are measured but do not win at saturated batch;
- allow different cores on the two dimensions;
- add shared-resident and cooperative-grid boundaries beyond selected points;
- generate padding and multi-buffer pipeline variants;
- extend the same schema to three or more dimensions for transforms beyond a
  profitable two-dimensional factorization.

Fine-grained performance search should operate only on `compiled` points after
the desired `awaiting-codegen` subset has been emitted.

The bounded candidate search and measured runtime selection built on this
space are documented in [FFT Pipeline Generator](fft_pipeline_generator.md).
