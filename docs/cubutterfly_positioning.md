# cuButterfly: Scope and Research Positioning

## 1. Thesis

Hermes should be generalized from an NTT implementation technique to a mapping
method for regular layered butterfly networks. The invariant is not the
butterfly arithmetic. It is the two-dimensional dependent computation graph:

```text
stage dimension k = 0 .. K-1
data dimension  b = 0 .. B-1
x[k+1, dst(k,b)] = Op(k,b, x[k,src0(k,b)], x[k,src1(k,b)], parameters)
```

The method independently factorizes both dimensions into spatial and temporal
execution:

```text
Us * Ts >= K       stage-space roles and stage-time folds
Ud * Td >= B       data-space lanes and data-time folds
```

`Op`, its radix, coefficient representation, and reduction algorithm are
replaceable. The architecture problem is selecting the four unfolding factors,
state placement, inter-stage service, and permutation/layout for the target
hardware.

This claim is broader than NTT but narrower than "all computations containing a
butterfly drawing." It applies when the graph is layered, bounded-degree,
mostly regular, and has enough repeated data groups to amortize a stage
pipeline. Irregular graph transforms, strongly data-dependent pruning, and
unbounded or global reductions need additional dimensions or a different
model.

## 2. Operator coverage

| Family | Local operator | Stage parameters | Expected fit |
|:--|:--|:--|:--|
| FFT | complex add/subtract and twiddle multiply | complex roots, radix | direct |
| NTT | modular add/subtract and twiddle multiply | finite-field roots, reduction | direct; current evidence |
| FWHT | add/subtract | none | direct; isolates scheduling cost |
| DCT/DST | sparse rotations plus permutations | trigonometric constants | direct after factorization |
| XOR/OR/AND subset transforms | semiring-like pair update | operation by stage | direct |
| learned butterfly layers | learned 2-by-2 blocks | weights | direct if the sparsity pattern is regular |
| Monarch/block butterfly | block GEMMs plus permutations | dense blocks | compatible after lifting `Op` to a block |
| bitonic/sorting networks | compare-exchange | direction by stage | structurally compatible, but nonlinear |
| polar/LDPC decoding graphs | combine/select | code and control state | partial; divergence and pruning extend the model |

The strongest initial paper scope is FFT, NTT, and FWHT. They share the same
regular graph while stressing different hardware resources: complex floating
point, modular integer arithmetic, and communication/synchronization.

## 3. Relationship to prior work

The table below is the short navigation index for the work discussed here.
The linked papers and project pages are context and comparison references; a
link does not mean that the corresponding implementation is installed or
directly comparable on the V100. The exact runnable baseline scope is defined
in [FFT Library Comparison](fft_library_comparison.md) and [V100 External
Baselines](v100_external_baselines.md).

| Work | What it contributes | Primary link | Role in this repository |
|:--|:--|:--|:--|
| NVIDIA cuFFT | Complete vendor GPU FFT library | [CUDA documentation](https://docs.nvidia.com/cuda/cufft/) | External FFT baseline |
| NVIDIA cuFFTDx | Device-side FFT building blocks | [CUDA documentation](https://docs.nvidia.com/cuda/cufftdx/) | Imported local processing unit |
| VkFFT | Cross-platform complete GPU FFT | [GitHub](https://github.com/DTolm/VkFFT) | External FFT baseline |
| TurboFFT | Generated and fused GPU FFT kernels | [GitHub](https://github.com/shixun404/TurboFFT) | Imported/archived FFT evidence |
| Dao-AILab FHT | Tuned GPU FWHT implementation | [GitHub](https://github.com/Dao-AILab/fast-hadamard-transform) | External FWHT baseline |
| GPU-NTT | CUDA NTT implementation | [Local gap report](gpu_ntt_gap_analysis.md) | Pinned archived NTT baseline |
| FFTW | Adaptive FFT planning and codelets | [Paper](https://doi.org/10.1109/ICASSP.1998.681704), [project](http://www.fftw.org/) | Algorithm/planning context |
| SPIRAL | Symbolic transform search and generation | [Paper](https://doi.org/10.1109/JPROC.2004.840306), [project](https://www.spiral.net/) | Algorithm/generator context |
| MAESTRO | Analytical spatial/temporal dataflow mapping | [Paper](https://doi.org/10.1145/3352460.3358252) | Mapping prior art |
| Timeloop | Systematic accelerator mapping and cost modeling | [Paper](https://doi.org/10.1109/ISPASS.2019.00042), [GitHub](https://github.com/NVlabs/timeloop) | Mapping/modeling prior art |
| tcFFT | Tensor-Core half-precision FFT | [Paper](https://arxiv.org/abs/2104.11471) | FFT processing-unit prior art |
| FlashFFTConv | Fused long-sequence FFT convolution | [Paper](https://arxiv.org/abs/2311.05908), [GitHub](https://github.com/HazyResearch/flash-fft-conv) | Fused-operator context |
| HadaCore | Tensor-Core Hadamard transform | [Paper](https://arxiv.org/abs/2412.08832) | FWHT processing-unit prior art |
| Butterfly factorization | Learned sparse linear transforms | [Paper](https://proceedings.mlr.press/v97/dao19a.html), [GitHub](https://github.com/HazyResearch/butterfly) | Graph/operator context |
| Monarch | Block-structured matrices | [Paper](https://proceedings.mlr.press/v162/dao22a.html), [GitHub](https://github.com/HazyResearch/monarch) | Block-operator context |
| NTTFusion | GPU NTT fusion and modular arithmetic | [Paper](https://doi.org/10.1109/ICCD58817.2023.00061) | NTT processing-unit prior art |
| TensorFHE | Hardware/software FHE acceleration | [Paper](https://doi.org/10.1109/HPCA56546.2023.10071017) | NTT/FHE context |

### Transform generators: FFTW and SPIRAL

FFTW selects plans and generates small codelets for DFT-family transforms.
SPIRAL represents transform algorithms symbolically, searches decompositions,
and generates platform-tuned software and hardware for DFT, DCT, and other
linear transforms. These systems are broader on the algorithm side than the
current Hermes work.

cuButterfly should not claim to replace their factorization or code generation.
Its distinction is a compact architecture-level mapping space for an already
chosen layered graph: explicit `Us, Ts, Ud, Td`, the physical inter-stage
service, and the state/layout consequences of spatializing a dependent stage
dimension. A future generator can use SPIRAL-derived factorizations as input
and search the cuButterfly mapping space as the hardware schedule.

### Generic spatial mapping: MAESTRO, Timeloop, and loop schedulers

Generic accelerator frameworks already distinguish spatial and temporal loop
mapping. This is the closest conceptual prior art and must be acknowledged.
Their common target is affine tensor programs dominated by independent and
reduction loops. A butterfly stage is a loop-carried dependence with a
stage-dependent permutation: spatializing it creates a physical producer to
consumer network rather than merely distributing independent iterations.

The defensible novelty is therefore not "spatial and temporal mapping" alone.
It is the four-factor specialization to a dependent butterfly graph, together
with explicit models for inter-stage links, live state, changing stride,
permutation, utilization when `K mod Us != 0`, and hardware-calibrated handoff
services. The work becomes stronger if these quantities predict mappings on
more than one operator and GPU.

### Operator-specific GPU transforms

tcFFT maps half-precision FFT fragments to Tensor Cores and co-designs the data
layout around the fixed matrix unit. FlashFFTConv rewrites FFT convolution as
matrix products, uses matrix-multiply units, and fuses kernels to reduce memory
hierarchy traffic. HadaCore similarly changes FWHT decomposition to use Tensor
Cores. TurboFFT uses architecture-aware threadblock FFTs, kernel fusion, and
template generation, with fault tolerance as an additional objective.

These works demonstrate that the equivalent processing unit can and should be
operator- and hardware-specific. They generally present one optimized point or
a transform-specific search space. cuButterfly treats such a unit as `Op` and
asks the orthogonal architecture question: how many stages and data groups are
simultaneously physical, how many are time-multiplexed, and which hierarchy and
handoff implement each boundary?

### Learned and hardware-friendly structured matrices

Butterfly factorization work learns products of sparse factors capable of
representing FFT-like transforms. Monarch replaces fine-grained sparse
butterflies with products of block-diagonal matrices to improve accelerator
utilization. Their primary contribution changes or learns the mathematical
operator graph. cuButterfly maps a chosen graph to hardware. The two approaches
are complementary: learned butterfly or Monarch blocks can become local
operators, while the four unfolding factors schedule their repeated stages and
blocks.

### NTT and FHE accelerators

GPU NTT, NTTFusion, TensorFHE, and FPGA/ASIC FHE accelerators optimize modular
units, stage fusion, memory layout, and workload-specific data reuse. Hermes
currently differs by making stage-based and pipeline-based NTT the endpoints of
one four-factor mapping space. Generalizing to cuButterfly is credible only if
the same mapping abstraction survives replacement of modular arithmetic with
other operators; the NTT result alone is not sufficient evidence.

## 4. What the present evidence proves

The correct V100 `N=256` NTT experiment fixes the CTA at eight warps and varies
the stage/data role allocation. With equal 32,832-byte shared-memory residency,
the best named-barrier point is `Us=4`: 0.403 ms versus 0.778 ms for `Us=1`, a
1.93x speedup. Atomic polling instead selects `Us=2`. Thus:

1. The two endpoints are not sufficient; an interior hybrid point wins within
   the explicit pipeline family.
2. The optimal factorization depends on the hardware handoff service, not only
   on `N` or arithmetic count.
3. A mapping must model resource thresholds: a 64-byte shared-memory difference
   changes V100 CTA residency unless controlled.

This supports the reasonableness of the method. It does not by itself prove
general efficiency: the best explicit pipeline remains 1.45x slower than the
mature `tile256` NTT kernel. The subsequent FFT, FWHT, and XOR-zeta common
runtime experiments add cross-operator evidence, but do not replace the need
for another GPU generation and more tuned external baselines.

## 5. Initial cross-operator experiment

The first fixed-size implementation supports NTT64, FP32 FFT, FP32 FWHT, and
uint32 XOR-zeta through common operator traits. In the controlled explicit
stage-pipeline family, all four select the interior `Us=4` point on V100. That
experiment isolates the stage/data role allocation but is not the fastest
realization.

The expanded search adds temporal-tile and warp/shared hybrid kernels, then
introduces a reusable fused radix-4 processing unit. For contiguous, forward
`N=256`, the joint search selects radix-4 temporal tiles for FWHT, FFT, and
XOR-zeta; the XOR-zeta lead over radix-2 is only 4.7%, and broader shape/layout
sweeps select other units. The **initial fixed-size experiment** reached 75.9%
of cuFFT throughput; this is historical design-space evidence, not the current
long-FFT result. The current matched V100 library matrix is summarized in
[V100 Three-Way Comparison](v100_three_way_comparison.md). Thus a common graph
mapping admits multiple processing units and kernel realizations, and the joint
optimum depends on operator, shape, semantics, and layout. Full data and
interpretation are in [Comprehensive Butterfly Comparison](comprehensive_butterfly_comparison.md).

The broader experiment should extend this common runtime as follows.

Use one scheduler, buffer protocol, mapping descriptor, and measurement format
for three operators:

| Operator | Arithmetic stress | Primary baseline | Required sizes |
|:--|:--|:--|:--|
| FWHT FP32/FP16 | communication and add/sub throughput | tuned CUDA/HadaCore-style kernel | `2^8` through `2^20` |
| FFT FP32 | complex FMA, twiddles, numerical error | cuFFT and a fused CUDA kernel | `2^8` through `2^20` |
| NTT 32/64-bit | integer multiply/reduction, root traffic | tile256, GPU-NTT | `2^8` through `2^20` |

For every operator, sweep the same named dimensions:

```text
(Us, Ts, Ud, Td)
operator/radix
handoff service
register/shared/L2/HBM state placement
tile size and batch
kernel realization and warp-local stage boundary
native versus natural output layout
```

Report absolute time against the best relevant implementation, useful
butterflies/s, achieved bandwidth and instruction throughput, occupancy limits,
handoff stalls, numerical error where applicable, and model prediction error.
The strongest evidence is not that exhaustive search finds a fast point. It is
that hardware microbenchmarks plus the four-factor model predict a point near
the measured optimum across operators and GPUs.

## 6. Project upgrade decision

`cuButterfly` is the appropriate research and eventual repository name, with
`cuNTT` retained as an operator backend and compatibility target. A clean
structure is:

```text
cuButterfly/
  include/cubutterfly/graph.hpp
  include/cubutterfly/mapping.hpp
  src/runtime/               common scheduler, buffers, handoffs
  src/operators/ntt/
  src/operators/fft/
  src/operators/fwht/
  benchmarks/
  models/
```

Do not rename the current repository yet. Promote it after all of the following
are true:

1. NTT, FWHT, FFT, and XOR-zeta execute through one shared stage-pipeline
   implementation with operator traits rather than copied kernels. (Completed
   for the controlled `N=256` experiment; the local runtime is broader.)
2. FFT validates that coefficients, floating-point error, and a different
   processing unit do not break the abstraction. (Completed for FP32/FP64
   `logN=1..10` in the temporal path and FP32 through `logN=20` in the
   hierarchical path.)
3. The mapping model ranks `Us/Ud` candidates for at least two GPU generations.
4. At least two operators reach a defensible fraction of their tuned baseline;
   the target should be at least 80%, with wins on some nontrivial shapes.

Until then, the accurate description is: "cuNTT is the first operator instance
of the proposed cuButterfly mapping framework."

## 7. Primary references

- [FFTW, *An Adaptive Software Architecture for the FFT*](https://doi.org/10.1109/ICASSP.1998.681704)
- [SPIRAL, *Code Generation for DSP Transforms*](https://doi.org/10.1109/JPROC.2004.840306)
- [MAESTRO, *Understanding Reuse, Performance, and Hardware Cost of DNN Dataflow*](https://doi.org/10.1145/3352460.3358252)
- [Timeloop, *A Systematic Approach to DNN Accelerator Evaluation*](https://doi.org/10.1109/ISPASS.2019.00042)
- [tcFFT, *Accelerating Half-Precision FFT through Tensor Cores*](https://arxiv.org/abs/2104.11471)
- [FlashFFTConv, *Efficient Convolutions for Long Sequences with Tensor Cores*](https://arxiv.org/abs/2311.05908)
- [TurboFFT, *A High-Performance Fast Fourier Transform with Fault Tolerance on GPU*](https://arxiv.org/abs/2405.02520)
- [HadaCore, *Tensor Core Accelerated Hadamard Transform Kernel*](https://arxiv.org/abs/2412.08832)
- [*Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations*, ICML 2019](https://proceedings.mlr.press/v97/dao19a.html)
- [*Monarch: Expressive Structured Matrices for Efficient and Accurate Training*, ICML 2022](https://proceedings.mlr.press/v162/dao22a.html)
- [NTTFusion, *Efficient Number Theoretic Transform Acceleration on GPUs*](https://doi.org/10.1109/ICCD58817.2023.00061)
- [TensorFHE, HPCA 2023](https://doi.org/10.1109/HPCA56546.2023.10071017)
