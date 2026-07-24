# cuButterfly Cross-Operator Results

## Scope

The controlled first experiment fixed `N=256` and replaced only the local
operator. The current runtime extends FFT, FWHT, and XOR to `logN=1..20`
through local and hierarchical paths, two floating-point precisions, both
directions, and explicit placement/layout semantics. Four operators establish
the common layered graph:

| Operator | Value | Local update | Twiddle | Input permutation |
|:--|:--|:--|:--:|:--:|
| NTT64 | 64-bit residue | Shoup modular butterfly | yes | bit reverse |
| FFT | FP32/FP64 complex | complex multiply plus add/subtract | yes | bit reverse |
| FWHT | FP32/FP64 | add/subtract | no | identity |
| XOR-zeta | uint32 | `(a,b) -> (a,a+b)` | no | identity |

Operator traits own value type, permutation, stage coefficients, and the local
or lane-selective update. The runtime provides three self-implemented kernel
forms: a shared-memory temporal tile, a warp/shared hybrid, and an explicit
stage-role pipeline. FFT also has a cuFFT baseline. Batch-17 tests verify tail,
strided, and in-place handling against CPU references.

## Controlled Stage Pipeline

This experiment fixes an eight-warp CTA, two tokens per independent pipeline,
and V100 named-barrier handoff. It changes only the allocation between
stage-space and data-space roles. Each number is the median of five independent
process trials at batch 16,384, with 20 warmups and 100 timed executions.

| Operator | `Us=1` ms | Best `Us` | Best pipeline ms | Speedup vs `Us=1` | Baseline | Baseline ms | Pipeline/baseline throughput |
|:--|--:|--:|--:|--:|:--|--:|--:|
| NTT64 | 0.778 | 4 | 0.404 | 1.928x | tile256 | 0.277 | 68.7% |
| FWHT | 0.354 | 4 | 0.187 | 1.895x | temporal tile | 0.055 | 29.6% |
| XOR-zeta | 0.353 | 4 | 0.186 | 1.895x | temporal tile | 0.052 | 27.7% |
| FFT | 0.754 | 4 | 0.314 | 2.399x | cuFFT | 0.084 | 26.7% |

The interior `Us=4` point wins for all four operators inside this controlled
family. It retains two independent data pipelines while halving the number of
stage groups. `Us=8` removes the group boundary but leaves only one pipeline.
This supports the two-dimensional factorization argument, while the low
absolute efficiency shows that explicit queues and global spill between stage
groups are not a universally efficient realization.

## Expanded Kernel-Form Search

The expanded V100 search uses the same batch, warmup, repeat, and five-trial
methodology. It evaluates 21 self-kernel configurations per operator:

```text
temporal-tile:  compute_unit = radix2 or fused radix4,
                tile_threads = 32, 64, 128, 256
warp-hybrid:    warp_stages = 0, 1, 2, 3, 4, 5
stage-pipeline: pipeline_warps = 4 or 8, stage_space <= pipeline_warps
FFT baseline:   cuFFT
```

| Operator | Overall best | Median ms | Gbutterfly/s | Best comparison | Result |
|:--|:--|--:|--:|:--|--:|
| FWHT | radix-4 temporal-tile, 128 threads | 0.051292 | 327.091 | warp-hybrid radix-2 | 1.065x faster |
| XOR-zeta | radix-4 temporal-tile, 128 threads | 0.050156 | 334.504 | radix-2 temporal-tile | 1.047x faster |
| FFT | radix-4 temporal-tile, 128 threads | 0.110346 | 152.042 | cuFFT, 0.083794 ms | 75.9% of cuFFT throughput |

The fused radix-4 unit executes two dependent stages per unit and halves the
temporal tile's CTA barriers. It is best for all three operators in this
contiguous, forward `N=256` run, although its XOR-zeta advantage over radix-2 is
only 4.7%. Broader length, layout, and direction sweeps also select radix-2 and
warp-hybrid points. Thus even a reusable core is a candidate, not a universal
replacement. Core choice, thread count, transport, precision, length,
direction, placement, and layout must be searched jointly.

There is no universally winning CUDA kernel. The operator changes the balance
among arithmetic, shuffle transport, shared-memory transport, synchronization,
and state residency. It therefore changes the best hardware realization while
leaving the architecture-level dimensions intact.

## Mapping Descriptor

The four unfolding factors do not fully determine a hardware mapping. Two
implementations can share `(Us,Ts,Ud,Td)` and differ radically in whether a
temporal boundary stays in registers/shared memory or crosses a kernel and
global memory. The expanded descriptor is:

```text
M = (Us, Ts, Ud, Td, Ub, Tb, Hs, Rs, Rd, Rb, L, F, Q)

Hs = physical service for a spatial stage edge
Rs = residence level across stage-time folds
Rd = residence level across data-time folds
Ub/Tb = batch-space and batch-time unfolding
Rb = residence and ownership across batch-time folds
L  = input/intermediate/output layout and permutation policy
F  = kernel realization family
Q  = realization parameters such as compute_unit, tile_threads, warp_stages,
     and pipeline_warps
```

For example, temporal-tile and stage-pipeline `Us=1` both temporalize all eight
stages. The former uses `Rs=shared, one kernel`; the latter uses
`Rs=global, eight kernels`. Treating them as the same mapping hides the main
performance cause. `Hs`, `Rs`, `Rd`, and legal `F` choices must be calibrated
for each hardware generation.

## Reproduction

```bash
cmake --build build -j
cmake --build build --target test

python3 scripts/sweep_cubutterfly_designs.py \
  --trials 5 --output results/cubutterfly_design_sweep_v100_raw.csv
python3 scripts/summarize_cubutterfly_designs.py \
  results/cubutterfly_design_sweep_v100_raw.csv \
  --output results/cubutterfly_design_sweep_v100_summary.csv
```

The raw and ranked data are stored in
`results/cubutterfly_design_sweep_v100_raw.csv` and
`results/cubutterfly_design_sweep_v100_summary.csv`. Compute Sanitizer memcheck
reports zero errors for the best FWHT, FFT, and XOR-zeta mappings; racecheck
reports zero hazards. These checks use batch 17. The NTT named-barrier path was
validated separately with the same tools.
