# Cross-Platform Ports

cuButterfly's portable object is the **mapping method**, not a CUDA binary or
a fixed kernel configuration. A cross-platform port should preserve the
stage/data space-time decomposition and rebuild its hardware lowering,
processing units, legality rules, and measured selection data for the target.

## Migration Instances

| Target | Project | Status | Architecture mapping | Evidence boundary |
|:--|:--|:--|:--|:--|
| NVIDIA CUDA | [cuButterfly](https://github.com/TruNcat3/cuButterfly) | Reference implementation | lanes/warps/CTAs, registers/shared memory, shuffle/global transport | V100 is the frozen v0.8 baseline; other GPUs require target-local calibration |
| Huawei Ascend | [Ascend-FFT](https://github.com/TruNcat3/ascend-fft) | Independent FFT port | AIV execution, UB residence, MTE transport, and GM boundaries | Ascend910_9382 C2C/R2C/C2R fp32 results; see that repository's published protocol and snapshot |

Ascend-FFT reuses the architecture vocabulary and search methodology. It does
not reuse cuButterfly CUDA kernels, and its measurements are not evidence for
CUDA performance. Conversely, cuButterfly's V100/A100 configurations are not
valid Ascend tuning tables.

## Port Acceptance Contract

A project is a method-level migration instance when it documents all of the
following:

1. The stage and data dimensions expose independent space/time factors, such
   as `(Us, Ts, Ud, Td)`, rather than one fixed tile recipe.
2. Hardware facts and resource limits are measured on the target, and illegal
   mappings are rejected before performance ranking.
3. Processing-unit selection is separate from architecture mapping, with an
   explicit lowering contract for each supported core.
4. Residence, boundary layout, synchronization, ownership, and materialized
   off-chip traffic are reported as physical realization properties.
5. Model-ranked candidates are validated by target-local correctness and
   timing measurements; results identify hardware and benchmark protocol.

This list is intentionally stricter than “implements an FFT.” It lets readers
distinguish a port of the architecture method from an unrelated implementation
or a direct translation of one CUDA kernel.

## Adding A Port

Open a documentation change that adds one row to the table and links to:

- the target repository and license;
- its architecture/lowering explanation;
- its supported operators and numeric contracts;
- a reproducible, target-local benchmark protocol;
- the cuButterfly citation or provenance statement used by the port.

Performance claims remain local to each project and hardware target.
