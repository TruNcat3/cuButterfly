# Semantic coverage audit, 2026-10-12

The recent end-to-end descriptor work is present.  The public path is

`Transform -> Descriptor -> canonical semantic JSON -> Plan -> lowering`.

The xD mode is an execution choice on a ranked request, rather than a second
operator API.  The descriptor carries rank, axes, strides, placement, order,
direction, precision, and FFT boundary type before either composite or xD
execution is selected.  `PlanGetDescriptorJson` and mapping replay preserve
that identity.

This audit adds a CPU-only manifest generator at
`scripts/build_semantic_coverage_matrix.py`.  It covers the boundary axes that
previous performance queues under-sampled:

| Axis | Values in the bounded manifest |
| --- | --- |
| Operator | FFT, NTT, FWHT, subset/superset zeta, structured 2x2 |
| Precision | FP32/FP64 for floating operators, word32/word64 for NTT |
| Rank | 1, 2, 3 |
| Batch | 1, 4, 32 |
| Direction | forward and inverse, with valid R2C/C2R pairing |
| Placement | out-of-place and in-place |
| Order | natural and bit-reversed per axis, with compact real axes fixed natural |
| Execution | composite for rank one, composite/xD for ranked requests |

The matrix is a request inventory, not a support or performance claim.  A GPU
runner must still query capabilities and record `missing-lowering` separately.
Rank-one NTT bit-reversal can use the legacy-compatible scalar spelling.  Rank-N
bit-reversal is represented by the generic per-axis logical order vector, so the
old scalar spelling is never silently applied to xD.  Compact real FFT axes are
fixed to natural order because the half-spectrum boundary has its own extent.

The CPU tests verify that the axes are present, IDs are unique, and xD does not
create a second semantic contract.  They do not allocate a device or measure a
kernel.  Existing public descriptor and high-rank real-FFT tests remain the
correct execution checks.

Known remaining boundary gaps are implementation coverage, not semantic
expressiveness: some low-precision real FFTs, arbitrary-rank requests beyond
the current execution limit, and complete public registration of newly measured
fused candidates.  These must remain explicit capability gaps in subsequent
GPU runs.
