# Operator Catalog

cuButterfly separates an operator's local pair update from the mapping that
moves pair operands through an ordered stage graph. An operator belongs in the
common runtime when its stages can use the same temporal, hierarchical,
online-reorder, warp-hybrid, or stage-pipeline scheduling contracts.

## Implemented Operators

| Operator | Value type | Local pair update | Inverse | Coefficients |
|:--|:--|:--|:--|:--|
| FFT | `Complex16`, `ComplexBf16`, `Complex32`, `Complex64` | weighted complex sum/difference | conjugate twiddle direction; optional `1/N` | stage twiddles |
| NTT | `uint32`, `uint64` words | modular weighted sum/difference | inverse root and modular scale | modular roots |
| FWHT | `Fp16`, `Bf16`, `float`, `double` | `(a+b, a-b)` | same graph; optional `1/N` | none |
| subset zeta/Mobius | `uint32` | forward `b += a`; inverse `b -= a` | Mobius subtraction | none |
| superset zeta/Mobius | `uint32` | forward `a += b`; inverse `a -= b` | Mobius subtraction | none |
| structured 2x2 | `Fp16`, `Bf16`, `float`, `double` | stage matrix times `(a,b)` | inverse of each local matrix | one broadcast or per-stage matrix |

All unsigned zeta arithmetic is modulo `2^32`. The pair-update direction is a
processing-unit property; the stage/data space-time mapping is unchanged.
This makes the subset/superset pair a direct test that the architecture layer
does not assume symmetric FFT/FWHT arithmetic.

FP16 and BF16 storage select either native-width or FP32 accumulation and
narrow each logical butterfly output back to storage width. On V100, the BF16
native-width contract is emulated and labeled as such because `sm_70` lacks a
native BF16 arithmetic path. Accumulation policy is part of the processing-unit
descriptor; it does not change the stage/data mapping abstraction.

## Subset And Superset Semantics

For an array indexed by bit masks, subset zeta computes:

```text
g[S] = sum of f[T] for every T subset of S
```

Superset zeta computes the dual:

```text
g[S] = sum of f[T] for every T containing S
```

Setting `inverse = true` applies the corresponding Mobius inversion. These are
also commonly called the OR-zeta and AND-zeta transforms. The parser therefore
accepts `or-zeta` as an alias for `subset-zeta` and `and-zeta` as an alias for
`superset-zeta`; canonical output names remain `subset-zeta` and
`superset-zeta`.

## Legacy xor-zeta Name

Earlier releases exposed `XorZeta`/`xor-zeta`, but the implemented update was
`b += a`, which is subset zeta rather than an XOR convolution transform. The
name remains available for source, CLI, CSV, and result compatibility and keeps
its established subset-zeta behavior. New code should use `SubsetZeta`.

An actual XOR convolution transform over a ring requires a Walsh-Hadamard-style
sum/difference operator and appropriate inverse scaling. Floating-point FWHT is
already available, but a separately named modular XOR transform is not yet part
of the public contract.

## Mapping Coverage

Subset and superset zeta use the scalar radix-2/4/8 processing units across:

- temporal tile;
- hierarchical decomposition;
- online reorder;
- warp hybrid;
- stage pipeline.

They support forward/inverse, in-place/out-of-place, positive element stride,
and valid padded batch stride under the same common-runtime length limits.
Automatic selection is not yet calibrated for the new canonical names; choose
an explicit backend until the new sweep is measured. The legacy `xor-zeta`
selector retains its archived V100 calibration.

## Initial V100 Mapping Check

The first expanded scan fixes `logN=8`, batch 16384, five trials, 20 warmups,
and 100 timed repetitions. It compares temporal tile with stage-pipeline
`Us=1/2/4/8`. These are fresh focused measurements, not an external-library
comparison.

| Operator | Temporal tile median | Best pipeline `Us` | Best pipeline median | Speedup over pipeline `Us=1` |
|:--|--:|--:|--:|--:|
| subset zeta | 0.052398 ms | 4 | 0.194232 ms | 1.829x |
| superset zeta | 0.052419 ms | 4 | 0.193311 ms | 1.838x |
| legacy `xor-zeta` name | 0.052429 ms | 4 | 0.194222 ms | 1.830x |

The mirrored subset/superset pair differs by less than 0.1% at the temporal
tile point and selects the same stage-space unfolding. This supports the narrow
claim that pair-update direction is subordinate to the mapping at this shape.
It does not yet establish length/batch scaling or performance against an
external subset-transform library.

Raw and reduced records are
[`results/butterfly_operators_expanded_v100_raw.csv`](../results/butterfly_operators_expanded_v100_raw.csv)
and
[`results/butterfly_operators_expanded_v100_summary.csv`](../results/butterfly_operators_expanded_v100_summary.csv).

## Structured 2x2

`Structured2x2` generalizes the real-valued pair operation without changing the
stage graph:

```text
[a']   [m00 m01] [a]
[b'] = [m10 m11] [b]
```

`ButterflyConfig::stage_matrices` accepts either one matrix broadcast to every
stage or exactly `log_n` matrices. FP32 plans convert the public double-valued
descriptor to FP32 coefficients during construction; FP64 plans retain FP64.
Coefficients are copied once to device memory and are only read during
steady-state execution.

With `inverse = true`, plan construction requires every matrix to be finite and
nonsingular and uploads its inverse. Different logical stages act on different
tensor-product axes, so their local factors commute; applying every local
inverse through the existing stage order gives the inverse structured
transform. No global matrix inversion is performed.

The CLI accepts one or repeated matrix descriptors:

```bash
# One matrix broadcast to every stage.
./build/cubutterfly_bench --operator structured-2x2 \
  --stage-matrix 1,0.25,-0.5,1 --precision fp32 \
  --backend temporal-tile --compute-unit radix4 \
  --logN 8 --batch 16384 --verify --csv

# Repeat --stage-matrix exactly logN times for stage-specific coefficients.
```

The general implementation supports scalar radix-2/4/8 with temporal tile,
hierarchical, online reorder, warp hybrid, and stage pipeline. FP32 additionally
supports a generated `warp-register` core for `logN=3..15`. That core keeps the
complete transform in one CTA, uses register vectors and warp shuffles for the
local stages, and uses the same XOR-swizzled cross-warp exchange as the FWHT
core. One matrix selects a broadcast-register coefficient policy; `logN`
matrices select the general per-stage-table policy. CUDA Graph and automatic selector status follow the common runtime
contract; no structured mapping is labeled calibrated yet.

The first V100 length/precision scan and matched FWHT control are reported in
[Structured 2x2 V100 Results](structured_2x2_v100_results.md). The initial
shared-core scan shows a temporal-to-hierarchical-to-online mapping transition
as length grows. The follow-up generated register core improves the FP32
`logN=8/12/15` points by 1.21x/2.81x/3.11x over those shared-core controls,
while preserving the same public semantics.

## Candidate Expansion Order

The next operators should be admitted according to how cleanly they test the
architecture abstraction:

1. modular Walsh-Hadamard/XOR convolution transforms;
2. Haar wavelets, which require an active-domain and multiresolution layout
   contract;
3. DCT/DST families as FFT compositions with explicit pre/post permutations;
4. non-power-of-two and mixed-radix transforms after the graph descriptor can
   represent nonuniform stages.

Structured `2x2` is now the general real-valued processing-unit interface. A
later coefficient-policy generator can specialize constant matrices into
immediates or codelets without changing the public workload semantics.
