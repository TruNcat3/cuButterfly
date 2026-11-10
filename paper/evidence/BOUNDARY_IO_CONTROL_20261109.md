# Boundary input/output controls — 2026-11-09

This bounded study tests the boundary argument in Section IV without changing
an arithmetic implementation or searching for a favorable workload. It separates
external boundary placement from local shared staging. Their observations must
not be concatenated into a three-step ablation or pooled into one speedup.

## Questions and controls

1. **External boundary.** Does publishing a consumer-oriented layout remove
   enough read or explicit-permutation cost to offset the producer's stores?
   The existing `cubutterfly_boundary_bench` compares canonical strided access,
   a separately materialized transpose, and mapped producer stores. Scalar
   radix-2 arithmetic, two stage bands, one tile per compute group, 256 threads,
   coefficients, internal writer-aligned shared layout, inputs and outputs are
   fixed. The three complete Graphs contain two, three and two kernels.
2. **Local bank mapping.** Does changing only the prefix shared-staging
   permutation improve service with the same cuFFTDx core and global I/O?
   `boundary_bank_control.cpp` uses the existing Linear/XorSwizzle paths.
   The global boundary, recurrence twiddles, 256-thread/EPT16 prefix and suffix,
   partition, and absence of overlap stay fixed. The local core does not see
   the staging permutation. This is an independent experiment with another
   core and another executable, not the third external-boundary variant.

The existing factor-streamed path hard-codes its writer-aligned/XOR placement;
changing a configuration label would not create a valid disabled-mechanism
control. No production kernel or default was modified for this study.

## Frozen scope and numerical checks

The config is `paper/experiments/boundary_io_control_20261109.json`. External
controls use FFT/FWHT FP32 at `(log2 N,B)=(16,16),(20,4)`, split equally,
with three independent process measurements per request. Their whole scalar
cores are diagnostic baselines, not the library's fastest available cores.
The local-bank study uses `N=2^20, B=4/16`, with partition 8+12. Static
preflight excludes logN 16 and 18 under this fixed core configuration:
`cufftdx_fft.cu` only dispatches tiled-transpose direct suffixes at log11/12,
and the fixed 256 threads × EPT16 requires log12. No failed request is replaced
by another shape and reported as the original. The support exclusions are
recorded in the config before GPU measurement. The earlier standalone layout
sweep instead varied the prefix as logN−12; this study does not use that sweep.

Both drivers check every batch against a same-precision CPU reference. The
external controls must match their unchanged scalar-core result exactly;
the local-bank pair must have byte-identical outputs. Floating CPU errors are
reported without introducing an acceptance threshold. Preparation, allocation,
reference evaluation and capture are excluded from ordinary execution time.
Ordinary runs rotate layouts within trials; three process medians and their
paired ratios are retained. Workload ranges are not confidence intervals.

Global timing repeats a complete-call Graph twenty times per event interval.
Local-bank timing captures twenty complete calls into one Graph. Both use seven
event intervals, but the timing protocols remain distinct. NCU captures are
diagnostic and never supply ordinary speedups. Their caches and clocks remain
uncontrolled; counter changes require interpretation at the recorded scope.

All runs use `scripts/run_isolated_gpu.py`; successful attempts alone become
`release`. Source, binary, library, config hashes, commands and GPU UUIDs are
retained under `results/boundary_io_control_20261109/`. A GPU conflict invalidates
the attempt, without stopping any other user's process.

## Reproduction

Use the existing `build-e13-cufftdx/cubutterfly_boundary_bench` with
`paper/tools/run_boundary_io_control.py --phase timing|profile`. For the local
control, build `paper/experiments/boundary_bank_control.cpp` against the same
headers and library, then use `paper/tools/run_boundary_bank_control.py`.
Every runner requires the config, binary and output paths; the bank runner also
requires the library path. Run inside the GPU guard with a chosen UUID and an
explicit finite deadline. The preparation receipt stores the complete C++ build
command; CPU checking and wrapper compilation precede GPU measurement.

## Interpretation boundary

An analytical conflict-free bank group establishes a mapping property under
its assumptions. It does not establish a complete-call speedup or conflict
freedom for arbitrary concurrent/vector accesses. Existing FP16 ownership plus
writeback results remain joint interventions. The new controls isolate the
named choices and retain near-parity and negative outcomes; they neither
replace the public-library comparison nor imply a globally optimal selection.

## Completed observations

All four guarded jobs completed without a detected foreign CUDA process.
External placement used A100 80GB UUID
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`; local bank placement used A100 80GB
UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`. There are 18 fresh timing
processes, 54 timing rows including six retained cuFFT rows, and ten complete
NCU captures. The external-library rows are retained as context; they do not
enter the internal organization ratios below.

Times are medians of three process medians in microseconds. Speedups are
medians of the three within-process ratios, not ratios of these displayed times.

| External control | N | B | Strided | Separate transpose | Producer stores | Strided / producer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFT FP32 | 65,536 | 16 | 100.301 | 104.192 | 101.427 | 0.989× |
| FFT FP32 | 1,048,576 | 4 | 498.483 | 487.117 | 491.418 | 1.015× |
| FWHT FP32 | 65,536 | 16 | 74.752 | 76.851 | 73.216 | 1.020× |
| FWHT FP32 | 1,048,576 | 4 | 274.842 | 280.064 | 282.522 | 0.973× |

For FWHT at N=65,536/B16, the NCU suffix global-load sectors/request decrease
from 31.94 to 4 with producer layout, while prefix store sectors/request
increase from 4 to 32. Materializing a transpose gives four sectors/request on
both those stage accesses, but adds a third kernel and its transfers. Thus
better consumer locality does not establish a whole-call gain. This concrete
observation matches Figure 5's two-sided boundary argument.

| Local shared-bank control | N | B | Linear (µs) | XOR (µs) | Linear / XOR | Three process ratios |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FFT FP32, cuFFTDx | 1,048,576 | 4 | 202.138 | 161.946 | 1.247× | 1.244–1.248× |
| FFT FP32, cuFFTDx | 1,048,576 | 16 | 722.022 | 578.560 | 1.248× | 1.244–1.248× |

The second control changes the existing prefix shared-staging map only. Both
variants retain the 8+12 partition, 256/4096-point local transforms, 256 threads
per compute group and EPT16. The prefix contains 16 local transforms per group;
the suffix contains one. Both calls include two compute kernels **and a final
transpose**, with identical physical launch shapes. The prefix uses 80
registers/thread and 32 KiB dynamic shared storage in both variants. Prefix
load bank conflicts decrease by 93.6–93.7%, and store conflicts by 83.3%.
The suffix and final transpose retain their accesses and nonzero conflicts.

Local-bank outputs are byte-identical for all batches and all six ordinary
processes. CPU FP32 relative L2 error is at most 3.253e-7, reported without a
new threshold. External internal controls also match the unchanged scalar
output exactly. These are complete-output checks, not sampled spot checks.

Plan `execution_groups` describes the two logical transform groups; it omits
the separate terminal transpose and its generic grid field is not an observed
physical prefix grid. The frozen data therefore separately records NCU
`actual_launch`: at B4 both compute grids contain 1024 blocks, prefix block
shape is (16,16,1), suffix is (256,1,1), and the final transpose has 4096 blocks.
B16 multiplies those grid sizes by four. These observed launches, rather than
logical descriptors, support the fixed-geometry comparison.

## Paper connection and deliverables

Section IV now points to the two independent controls in Section VI-D. The
new result supports a bounded causal statement: changing local bank placement
can improve an unchanged core's complete execution. It does not establish
that every boundary mapping, every arithmetic family, or every workload wins.
The earlier FP16 joint ownership/writeback result remains distinct.

- [Frozen observations and raw provenance](../generated/boundary_io_control_20261109.json)
- [Plotted values](../generated/boundary_io_control_20261109.csv)
- [Four-panel evidence figure](../figures/boundary_io_control_20261109.pdf)

The standalone evidence figure accompanies this audit; the manuscript retains
its existing figure count. `paper/tools/export_boundary_io_control.py
--frozen-only` regenerates the figure on CPU without the original host results.

The main and evidence manuscripts compile successfully; the main paper remains
15 pages including references after removing repeated exposition. The new
analysis appears on page 14 and links back from the boundary method. Rendered
text and the standalone figure were visually checked. Reference checks and
`git diff --check` pass, with no undefined references or overfull boxes. The
frozen archive contains 97 raw records, including the wrapper's build command
and floating-contract compiler flag. Their hashes and the exporter hash match;
pre-existing measured datasets are unchanged.
