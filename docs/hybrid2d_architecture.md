# Hybrid2D GPU Mapping

## Architecture boundary

APPT/Hermes defines a hybrid dataflow: both the stage dimension and the data
dimension have spatial and temporal parallelism. `Npart`, butterfly-unit
organization, and the amount of unfolding are implementation parameters chosen
for a target device. The U280 values `Npart=256` and `p=16` are one realization,
not the architecture definition.

`cuNTT` keeps the processing unit replaceable, but the current kernels are not
yet a complete physical realization of all four APPT unfolding modes. They are
a GPU baseline for determining which architectural factors still need an
explicit mapping.

## CUDA realization

| APPT dimension | CUDA realization | Current control |
|:---------------|:-----------------|:----------------|
| data spatial | lanes, warps, CTAs, and SMs operating on independent coefficients/rows | `threads_per_block`, grid size |
| data temporal | multiple butterflies and rows reused by a thread/CTA, plus device scheduling waves | `rows_per_block`, loop trip counts, grid size |
| stage temporal | dependent stages reuse the same execution lanes while data remains in shared memory | local stage loop counts |
| stage spatial | not explicitly implemented in the current kernels | future stage-specialized warp pipeline and explicit inter-stage queues |

Warp scheduling can incidentally overlap CTAs at different program positions,
but this is not a controlled stage-space unfolding factor and is not counted as
the APPT stage-spatial mechanism.

The complete hardware-constrained parameter method, including the distinction
between explicit FPGA stage pipelines and the current temporally folded GPU
baseline, is documented in
[`hardware_mapping_methodology.md`](hardware_mapping_methodology.md).

For `N=N1*N2`, Hybrid2D performs:

1. `N1` independent `N2`-point transforms, followed by the cross-dimension
   twiddle and a global-layout transpose.
2. `N2` independent `N1`-point transforms, followed by a transpose to natural
   output order and optional inverse scaling.

This retains intermediate values in shared memory within each local transform
and requires two global read/write passes independent of the selected factors.

## Hardware-dependent selection

The parameter selector must consider at least:

- warp size and available warp schedulers;
- shared memory and registers per CTA/SM;
- maximum resident threads, CTAs, and warps per SM;
- SM count and workload batch size;
- global-memory transaction efficiency and L2 capacity;
- native integer multiply throughput for the coefficient width;
- barrier frequency and occupancy loss from larger local transforms.

The V100 (`sm_70`) has 32-thread warps, 80 SMs, 2,048 resident threads per SM,
48 KiB shared memory per block in the current configuration, and a 4,096-bit HBM
interface. A measured search currently selects `N1=N2=256`, four rows per CTA,
and 256 threads per CTA at `logN=16`. The processing-unit selector uses radix-2
for the short 64-bit `logN=12` case and fused radix-4 for the remaining measured
cases. Other GPUs must rerun the search; they should not inherit these values
solely because they were optimal on V100.
