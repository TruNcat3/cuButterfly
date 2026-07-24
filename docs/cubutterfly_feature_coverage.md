# cuButterfly Feature Coverage

## Implemented Semantics

| Operator | Precision | Forward | Inverse | Inverse normalization |
|:--|:--|:--:|:--:|:--|
| FFT | FP32, FP64 | yes | yes | selectable `none` or `inverse` |
| FWHT | FP32, FP64 | yes | yes | selectable `none` or `inverse` |
| XOR-zeta/Mobius | uint32 | yes | yes | arithmetic modulo `2^32`; no scaling |
| NTT | 32/64-bit physical paths | yes | yes | existing modular inverse convention |

FFT and FWHT inverse normalization is part of timed kernel execution. For
cuFFT it requires a separate scale kernel; `--normalization none` exposes the
native unnormalized inverse for a core-equivalent comparison.

## Length and Backend Capabilities

| Backend | Length | Processing unit | FP32 | FP64 | uint32 |
|:--|:--|:--|:--:|:--:|:--:|
| temporal-tile | shared: `logN=1..10`; register FWHT: `logN=3..15` | radix-2, fused radix-4/radix-8; FP32 FWHT register-vector unit | yes | yes | yes |
| hierarchical | `logN=6..20` | local stages 5..10, radix-2/radix-4/radix-8, global remainder | yes | yes | yes |
| online-reorder | `logN=6..20` | radix-2/radix-4/radix-8 groups with a fused permutation store; optional two-pass cuFFTDx local FFTs | yes | yes | yes |
| warp-hybrid | `logN=8` | radix-2 | yes | yes | yes |
| stage-pipeline | `logN=8` | radix-2 | yes | FWHT only | yes |
| cuFFT | `logN=1..20` | vendor plan | yes | yes | n/a |

Generated FFT codelets add FP32 `thread-dft8` and mixed `fp16-fp32`
`wmma-dft8` points at `logN=3`, plus an FP32 CTA-composed DFT8 point at
`logN=3..10`. The WMMA core keeps Complex32 external storage but rounds Tensor
Core operands to FP16 and accumulates into FP32. Generated availability is
controlled by `config/v100_design_points.json`.

When cuFFTDx is enabled, FP32 online-reorder can compose two local FFTs whose
individual sizes are `logN=3..10`. The first pass fuses the cross twiddle and a
transposed scratch store; the second pass restores natural order. Cross
twiddles are independently selectable as table lookup or register recurrence,
so this processing-unit choice remains subordinate to the architecture-level
space/time schedule.
An explicit `cufftdx-resident` core adds single-CTA `logN=12` 6+6 and
persistent `logN=14` 7+7 mappings with 256/512/1024-thread choices. The
`cufftdx-direct` points admit whole-transform `logN=11..14` units to measure the
processing-unit granularity ceiling independently of the composed schedule.
The V100-valid CTA choices are 256/512/1024 threads through `logN=13` and
512/1024 threads at `logN=14`.

Each temporal length is a compile-time CUDA specialization selected by the
runtime. This preserves unrolling at performance-critical fixed sizes while
presenting one plan interface. The FP64 complex stage pipeline is rejected
because its token buffers exceed the V100 per-CTA shared-memory limit.

All listed backends support GPU in-place and out-of-place execution, arbitrary
positive element stride, and a batch distance large enough to contain one
strided transform. Kernels address the requested layout directly; no hidden
pack/unpack kernel is inserted. Use `--list-capabilities` to query the compiled
matrix.

Warmup and timed repetitions remain pure kernel launches. Because successive
in-place launches consume the preceding result, `execute` restores the original
input and performs one untimed launch before download when repetition is used.
The returned output therefore always has single-transform semantics without
contaminating the reported kernel time.

## Search Dimensions Recorded in CSV

```text
operator
precision
direction
normalization
placement
element_stride and batch_stride
logN, N, batch
backend, compute_unit, FFT complex_multiply, local_exchange, and fft_core
tile_threads, local_stages, reorder_columns, warp_stages, stage_space, pipeline_warps, handoff
kernel/H2D/D2H time, transforms/s, points/s, butterflies/s, correctness
```

The summarizer ranks configurations independently for every semantic group, so
different lengths, precisions, directions, normalization policies, and batches
cannot be accidentally combined.

## Verification

The standalone test executables cover FP32/uint32 temporal radix-2, radix-4,
and radix-8 for every `logN=1..10`,
forward and inverse. FP64 temporal FWHT, temporal FFT, and cuFFT are checked at
`logN=1,5,8,10` in both directions. Existing `N=256` warp-hybrid,
stage-pipeline, tail handling, unnormalized inverse, and NTT tests remain
enabled.
The FP32 FWHT warp-register unit is checked forward and inverse for every
`logN=3..15`; a separate benchmark verification covers strided, padded,
in-place inverse normalization.
Strided in-place and out-of-place FFTs are also checked through temporal,
hierarchical, online-reorder, warp-hybrid, stage-pipeline, and cuFFT backends. Hierarchical
radix-2/radix-4/radix-8 FFT, FWHT, and XOR are checked in both directions at
`logN=11`. Online-reorder additionally covers unequal stage groups, multiple
columns per suffix CTA, FP64, strided/in-place execution, and verified
`logN=20` execution in the large-length protocol.
Optional cuFFTDx tests cover local `logN=3..10`, temporal and online-reorder
layouts, unequal long-FFT splits, table/recurrence cross twiddles, and forward
and normalized inverse execution. Online-reorder also covers independent
prefix/suffix EPT 4/8/16 and asymmetric thread/EPT mappings. Direct whole-transform tests cover every
valid `logN=11..14` CTA shape, normalized inverse, strided layout, and in-place
execution.
Gauss three-multiply FFT is checked through temporal, hierarchical, and
online-reorder mappings. NTT tests cover radix-8 at multiple lengths and on
both 32-bit and 64-bit physical paths. Hybrid2D additionally checks the full
product of first/second/fused cross-twiddle placement and Shoup/Barrett modular
multiplication; these are separate CSV fields.

## Deliberately Unsupported

The current public runtime does not yet claim:

- full-length FP16/BF16 Tensor Core FFT beyond the generated DFT8 point;
- non-power-of-two or multidimensional transforms;
- `logN>20` through the common runtime;
- fused application epilogues such as convolution or quantization.

These are separate semantic or processing-unit extensions. They should be
added with explicit layout/error contracts rather than accepted as ignored
flags.

The optional `scripts/benchmark_external_fht.py` adapter measures the
BSD-3-Clause Dao-AILab `fast-hadamard-transform` package and emits compatible
semantic fields. Upstream targets `sm_75+`; the recorded local patch adds only
an `sm_70` gencode target. V100 results are labeled
`dao-fast-hadamard-transform-sm70-local` rather than upstream-native.
The in-tree `warp-register` unit is independently exposed through the common
runtime; its attribution and redistribution terms are recorded in
`THIRD_PARTY_NOTICES.md`.
