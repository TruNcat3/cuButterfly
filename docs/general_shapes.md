# General Shape Mapping

General shapes do not add another architecture principle. They add a lowering
step before the same two-dimensional space-time mapping is selected.

```text
logical axes and batch
        |
        +-- power-of-two axis --------> native staged butterfly
        |
        +-- standard arbitrary FFT ---> direct vendor plan OR chirp convolution
        |
        +-- standard arbitrary NTT ---> root validation -> modular chirp/convolution
        |
        +-- embedding ----------------> zero pad to next power of two
        v
axis plan 0 -> online transpose -> axis plan 1 -> transpose back
        v
(Ud,Td) data mapping x (Us,Ts) stage mapping x selected processing unit
```

For an exact FFT axis of length `N`, direct cuFFT is a processing-unit candidate.
Bluestein converts the transform into a
convolution of length at least `2N-1`; the implementation chooses the next
power of two `M` so an existing high-performance FFT core remains a selectable
physical unit. The chirp pack, pointwise multiply, and chirp output are fixed
boundary work. They do not alter the FFT formula.

An arbitrary NTT uses the same composition over the finite field. Unlike a
floating FFT, its domain is conditional: the selected prime must contain the
needed roots of unity. Rejecting a missing root is a semantic requirement, not
a performance limitation. Embedding remains available when the application
actually wants a padded power-of-two transform rather than an exact-length NTT.

For rank two, rows provide independent data-space work for the first axis.
After the first axis completes, transpose changes ownership/layout so columns
become packed independent transforms for the second axis. Each axis retains
its own `(Us,Ts,Ud,Td)` and processing-unit choice. The current implementation
materializes this global boundary in caller workspace because ordinary CTAs
cannot perform a grid-wide synchronization inside a conventional kernel.

The physical extent query has operator-specific meaning:

| Mode | Returned physical axis extent |
|:--|:--|
| native power of two | `N` |
| standard arbitrary FFT | logical `N` for direct cuFFT; convolution length for Bluestein |
| standard arbitrary NTT | convolution length `next_pow2(2N-1)` |
| zero-extended embedding | embedded transform length `next_pow2(N)` |

Input/output byte queries always describe the logical API contract, including
the asymmetric forward/inverse embedding contract. Workspace size includes
packed matrices, transpose buffers, convolution buffers, and underlying axis
plan scratch.

When the physical output layout is contiguous, the composition does not
materialize a separate final scatter. A rank-one embedded butterfly packs and
zero-fills directly into the caller output before its in-place physical core;
an out-of-place NTT core writes there directly; and a rank-two plan makes its
final transpose target the caller output. Arbitrary output strides keep the
explicit scatter. The selected algorithm name carries `+direct-output` so the
boundary choice is visible in logs and result files.
For a rank-one in-place physical butterfly this also removes the packed
workspace buffer entirely; the workspace query then reports only underlying
core scratch and may return zero. A rank-one out-of-place NTT retains one pack
buffer, and rank two retains the two transpose buffers.

For forward FP32 FWHT embedding with a generated warp-register physical unit,
the input boundary is lowered into that unit as well. The kernel receives the
logical extent plus the independent input and output layouts, reads valid
logical values, synthesizes zero for the tail in registers, and writes the
complete physical result. This eliminates both composition kernels and is
reported as `+direct-output+fused-input`. Other processing units retain the
pack path unless the selector has a validated fused candidate. In particular,
Structured 2x2 currently rejects this candidate on performance grounds; a
legal fusion is not automatically an efficient mapping.
