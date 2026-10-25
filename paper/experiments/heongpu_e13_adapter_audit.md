# HEonGPU E13 CKKS Adapter Audit

Date: 2026-10-03

This is a source-contract audit for the pinned HEonGPU tree at commit
`1928a14bb002c5d7f1cb5534eca9fa19b555b4f2`. It is not a GPU run, a bootstrap
result, a security claim, or an application-completion claim. The separate
library build record is
`paper/experiments/heongpu_e13_build_status.md`.

## Executive conclusion

The current cuButterfly NTT cannot replace the HEonGPU GPU-NTT call by changing
only the function name or pointer arguments. Output order can be made to match,
but three independent contracts remain different:

1. HEonGPU uses the negacyclic `X^N+1` ring with a per-prime `2N`-th root
   (`psi`), while the public cuButterfly plan exposes a cyclic N-point NTT and
   no `reduction_poly`, `psi`, or root-selection field.
2. HEonGPU's RNS call assigns a different modulus to each contiguous row. The
   cuButterfly plan has one scalar modulus for all rows in `batch`.
3. HEonGPU uses in-place NTT in the full raised ciphertext path. cuButterfly's
   public device execution is out-of-place and requires distinct pointers.

An adapter therefore needs explicit negacyclic pre/post factors (or a matching
API extension), one plan or dispatch path per RNS prime with gather/scatter as
needed, and ping-pong storage. A direct call would change arithmetic or memory
semantics.

## Official regular-bootstrap parameters

The v1 example
`results/e13_assets/sources/HEonGPU/example/bootstrapping/1_ckks_regular_bootstrapping.cpp`
uses:

- `GenHEContext(..., sec_level_type::none)` at `:11-14`.
- `N=4096` at `:15-16`.
- Q bit sizes of one 60-bit prime followed by thirty 50-bit primes, and three
  60-bit P primes, at `:18-21`.
- Scale `2^50` at `:25-27`.
- `StoC_piece=3` and `BootstrappingConfig(3, 3, 11, true)` at `:70-80`.
- Thirty modulus drops before `regular_bootstrapping` at `:94-104`.

The v2 example
`results/e13_assets/sources/HEonGPU/example/bootstrapping/5_ckks_regular_bootstrapping_v2.cpp`
also passes `sec_level_type::none` at `:11-13`, but uses `N=1<<16` at
`:14-15`, an explicit 25-prime Q list and five 61-bit P primes at `:17-51`,
`h=192`, ephemeral secret weight 32, and scale `2^40` at `:56-59`.
Its EvalMod and encoding-matrix configuration is at `:110-119`, and its
24-level drop plus v2 regular call is at `:127-143`.

`sec_level_type::none` is material: both the generated-bit-size and explicit-
prime paths skip the maximum coefficient-bit check for this value
(`results/e13_assets/sources/HEonGPU/src/lib/host/ckks/context.cu:94-119` and
`results/e13_assets/sources/HEonGPU/src/lib/host/ckks/context.cu:204-229`). The
pinned bootstrap README explicitly calls the v1 `N=4096` parameters insecure
for practical use and demonstration-only
(`results/e13_assets/sources/HEonGPU/example/bootstrapping/README.md:24-30`). The v2 size
is a larger functional workload, not evidence that the example is a security
parameter set. No result from either example should be labeled secure.

For a bounded upstream-main smoke run, preserve the v1 source parameters and
the `sec_level_type::none` setting exactly, and label it `N=4096 insecure
functional smoke`. Keep v2 as a separate `N=65536` workload; do not silently
substitute it for v1 or infer security from either. This parameter advice is
not an upstream-main build or run.

## HEonGPU NTT and RNS boundary

The GPU-NTT configuration explicitly carries `NTTLayout` and
`ReductionPolynomial`, including `PerPolynomial` and `X_N_plus`/`X_N_minus`
(`results/e13_assets/sources/HEonGPU/thirdparty/GPU-NTT/src/include/gpuntt/common/nttparameters.cuh:25-36`;
`results/e13_assets/sources/HEonGPU/thirdparty/GPU-NTT/src/include/gpuntt/ntt_merge/ntt.cuh:31-51`).
HEonGPU creates a primitive `2*N`-th root for each prime
(`results/e13_assets/sources/HEonGPU/src/lib/util/util.cu:382-395`), builds
forward and inverse tables in bit-reverse order
(`results/e13_assets/sources/HEonGPU/src/lib/util/util.cu:398-450`), and stores
a per-prime `N^-1`
(`results/e13_assets/sources/HEonGPU/src/lib/util/util.cu:453-463`). CKKS encoder and bootstrap configurations select
`PerPolynomial`, `X_N_plus`, and the per-prime inverse factors
(`results/e13_assets/sources/HEonGPU/src/lib/host/ckks/encoder.cu:145-156,453-472`;
the regular bootstrap setup is
`results/e13_assets/sources/HEonGPU/src/lib/host/ckks/operator.cu:7027-7042`).

For RNS mode, the GPU-NTT API documents row-major polynomials with a modulus
per row and accepts `batch_size, mod_count`
(`results/e13_assets/sources/HEonGPU/thirdparty/GPU-NTT/src/include/gpuntt/ntt_merge/ntt.cuh:342-421`). The kernel
selects `mod_index = block_z % mod_count` and addresses row `block_z` at
`block_z*N`
(`results/e13_assets/sources/HEonGPU/thirdparty/GPU-NTT/src/lib/ntt_merge/ntt.cu:615-634`);
root-table selection also includes that modulus index
(`results/e13_assets/sources/HEonGPU/thirdparty/GPU-NTT/src/lib/ntt_merge/ntt.cu:672-680`).
The inverse applies the supplied per-prime `N^-1` in its final store
(`results/e13_assets/sources/HEonGPU/thirdparty/GPU-NTT/src/lib/ntt_merge/ntt.cu:1148-1193`).

The regular q0 boundary is concrete in the v1 path:

- `GPU_INTT(..., batch=2, mod_count=1)` converts the two ciphertext components
  at q0 (`results/e13_assets/sources/HEonGPU/src/lib/host/ckks/operator.cu:7044-7051`).
- `mod_raise_kernel` writes row `idx + idy*N + (Q*idz)*N`, so the full buffer is
  `c0,q0 ... c0,q(Q-1), c1,q0 ... c1,q(Q-1)`
  (`results/e13_assets/sources/HEonGPU/src/lib/kernel/bootstrapping.cu:339-375`).
- `GPU_NTT_Inplace(..., batch=2*Q, mod_count=Q)` then transforms those rows
  (`results/e13_assets/sources/HEonGPU/src/lib/host/ckks/operator.cu:7055-7065`).

The v2 regular path reaches the same q0-to-full-Q helper through
`mod_up_from_q0`
(`results/e13_assets/sources/HEonGPU/src/lib/host/ckks/operator.cu:7148-7190`),
whose inverse, mod-raise and full RNS forward calls are at
`results/e13_assets/sources/HEonGPU/src/lib/host/ckks/operator.cu:3963-4022`.
It therefore does not make the RNS contract more interchangeable.

At representation level, GPU-NTT Merge's native forward result is
bit-reversed, as recorded by the local comparison (`docs/gpu_ntt_gap_analysis.md:163-183`)
and exercised by the pinned per-polynomial test
(`results/e13_assets/sources/HEonGPU/thirdparty/GPU-NTT/example/ntt_merge/test_merge_ntt.cu:136-174`). The
bit-reversed output is part of the surrounding HEonGPU dataflow, not an
invitation to reinterpret the rows or moduli.

## cuButterfly public contract

`cuntt::PlanConfig` has one scalar `modulus`, one `batch`, and selectable input
and output orders (`include/cuntt/ntt.hpp:308-388`). It has no RNS modulus array,
`ReductionPolynomial`, supplied `psi`, or supplied root field. The public device
execution contract is out-of-place with distinct input and output pointers
(`include/cuntt/ntt.hpp:461-464`; `docs/device_api.md:90-94`).

cuButterfly can request bit-reversed output, but its input remains natural and
an inverse requesting bit-reversed output still consumes natural input
(`docs/ntt_output_order.md:3-12,28-32`). This can match the HEonGPU forward
representation only after the adapter handles the inverse-side layout boundary.

The reference arithmetic finds a primitive N-th root satisfying `root^N=1`
(`src/reference.cpp:564-581`) and runs the cyclic N-point transform
(`src/reference.cpp:584-630`). The application-coverage note gives the required
composition for a negacyclic transform: multiply coefficient `j` by `psi^j`,
run a cyclic plan whose root is exactly `psi^2`, then apply `psi^-j` on inverse
(`docs/crypto_application_coverage.md:55-67`). That composition is not currently
encoded in a cuButterfly plan, and a round-trip alone does not prove that its
root equals the HEonGPU root.

| Contract | HEonGPU pinned path | cuButterfly public path | Adapter consequence |
| --- | --- | --- | --- |
| Ring/domain | `X^N+1`, per-prime `psi` of order `2N` | cyclic N-point root | explicit twist and exact root matching required |
| Moduli | row `r` uses `q_(r mod L)`; full ciphertext has `2*L` rows | one scalar modulus per plan | one plan/dispatch per prime plus row packing or gather/scatter |
| Forward order | native GPU-NTT Merge bit-reversed | selectable bit-reversed output | compatible only at output representation level |
| Inverse input | merged GPU-NTT dataflow consumes its native layout | inverse input remains natural | explicit reorder or a matching cuButterfly plan contract |
| Placement | q0 inverse out-of-place; raised forward in-place | device execute out-of-place | temporary output and pointer swaps required |
| Normalization | per-prime `N^-1` supplied to inverse | normalized inverse is configurable | set normalization and compare exact residues |

## Minimum next step

Do not replace the bootstrap call yet. The smallest useful future GPU probe is a
single-prime, single-component q0 boundary using the v1 `N=4096` demonstration
parameters:

1. Capture q0, HEonGPU's `psi`, `psi^2`, inverse root and `N^-1` from the pinned
   context. Verify the actual cuButterfly root; if it is not exactly `psi^2`,
   stop and expose root selection before claiming compatibility.
2. Compare a host reference and cuButterfly plan with explicit coefficient
   twists, `word_bits=64`, `batch=1`, `OutputOrder::BitReversed`, distinct
   buffers, and explicit inverse normalization. Check both residues and the
   forward-to-inverse layout transition.
3. Only after that passes, repeat one plan per Q prime and validate the row map
   `c0,q0 ... c1,q(Q-1)` with gather/scatter. Then test the full bootstrap and
   decrypted precision statistics.

The first step is a correctness gate, not a performance benchmark. No GPU
execution, example target, replacement, precision result, or security result
was performed by this audit.
