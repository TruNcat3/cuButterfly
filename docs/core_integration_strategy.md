# Processing-Core Integration and Library Comparison

## Separation of Concerns

cuButterfly treats the layered parallel pattern as the contribution and keeps
the arithmetic core replaceable. Evaluation therefore separates five objects:

```text
G: mathematical graph and exact transform semantics
A: space-time mapping (Us, Ts, Ud, Td, handoff, residency)
P: processing unit (radix, arithmetic algorithm, coefficient representation)
F: hardware realization (kernel form, threads, warp boundary, pipeline warps)
B: complete external library baseline
```

An imported radix codelet, modular reduction, or matrix fragment belongs to
`P`. A complete cuFFT, GPU-NTT, or HadaCore execution path belongs to `B` unless
its processing unit can be isolated behind the common `P` contract while
cuButterfly still owns scheduling, storage, and layout.

## Candidate Core Space

| Operator | Implemented processing units | Candidate established units | Main hardware axis |
|:--|:--|:--|:--|
| FFT | radix-2/4/8; four-multiply and Gauss three-multiply; generated thread-register and WMMA DFT8 | split/mixed-radix long-transform composition | FP/Tensor issue, registers, shuffle/shared transport |
| NTT | radix-2/4/8 with independent Shoup/Barrett arithmetic | Montgomery, lazy reduction, specialized modular units | integer multiply, reduction temporaries, root bandwidth |
| FWHT | radix-2/4/8 add/subtract | warp-register and Tensor Core Hadamard units | communication and synchronization |
| subset transforms | radix-2/4/8 uint32 add/subtract | packed integer or semiring-specialized pair units | integer issue and transport |

The common radix-4/radix-8 units prove this interface across FFT, FWHT,
XOR-zeta, and NTT. The V100 selects radix-8 for FP32 FFT at `N=256`, radix-4
for FWHT/XOR at the same length, and radix-4 for all three at `logN=20`.
Radix-8 is substantially slower for 60-bit NTT at `logN=12`. Full data and the
factorized unit model are in `docs/processing_unit_design_space.md`. This is
direct evidence that an imported core must remain a searched option.

## Admission Rules for External Cores

An external core enters the design space only when all of these are recorded:

1. Source and version, license, and any local modifications.
2. Supported operator, length/radix, precision, input/output layout, and error
   or modular-correctness contract.
3. Required warp/block shape, register/shared-memory footprint, coefficient
   format, and supported GPU architectures.
4. A standalone correctness test against the same CPU reference.
5. An adapter in which the core owns only local processing and cuButterfly owns
   the selected mapping, residency, and scheduling.

If rule 5 cannot be met, the implementation remains an external baseline. Its
performance is still important, but it cannot demonstrate that cuButterfly's
mapping successfully absorbed the core.

## Fair Comparison Matrix

Every library comparison must match or explicitly label:

```text
operator and direction
length and batch
numeric precision and modulus width
normalization and convolution mode
native and requested output layout
in-place versus out-of-place storage
resident-data kernel time
layout-conversion time
plan/setup time
host-device transfer time
GPU, clocks, CUDA version, and library commit/version
```

Report three views rather than one headline number:

1. `core-only`: equivalent processing-unit cost in a controlled mapping.
2. `resident transform`: all kernels and required layout conversions with data
   already on the GPU.
3. `application end-to-end`: setup and transfers included where relevant.

The repository includes an optional Dao-AILab FWHT baseline adapter:

```bash
python3 scripts/benchmark_external_fht.py --sm70-patched \
  --logNs 8 10 12 15 --dtypes fp16 bf16 fp32 \
  --output results/external_fht_raw.csv
```

It fails explicitly when PyTorch, the extension, or CUDA is unavailable. V100
execution additionally requires `--sm70-patched`, acknowledging the recorded
gencode-only patch rather than silently presenting it as an upstream binary.

The primary comparison is measured on the same machine. Numbers quoted from a
paper are contextual evidence, not entries in the normalized performance
ranking.

## Current Evidence Boundary

- NTT has a same-V100 comparison against a fixed GPU-NTT revision and wins on
  several length/layout combinations.
- FFT has a same-process cuFFT baseline; the current best self kernel reaches
  75.9% of its throughput at FP32 `N=256`, batch 16,384. At `logN=12..20`,
  online-reorder reaches 32.7%-47.4% of cuFFT throughput. It removes the
  stage-by-stage global boundary, so the remaining gap is now attributed to
  local-core efficiency, permutation transactions, twiddle traffic, and
  occupancy rather than kernel count alone.
- FWHT now has both a same-V100 Dao-AILab baseline and a locally integrated
  `warp-register` unit using the same register/shuffle/XOR-swizzle hierarchy.
  It reaches 99.9%, 98.5%, 97.5%, 95.8%, and 92.4% of Dao FP32 throughput at
  `logN=8,10,12,14,15`, while preserving cuButterfly's strided, in-place, and
  inverse contracts. XOR-zeta still has only internal optimized baselines.

This boundary permits two distinct conclusions: processing-core reuse improves
the cuButterfly mapping, while comprehensive superiority over butterfly
libraries remains an experiment to be completed.
