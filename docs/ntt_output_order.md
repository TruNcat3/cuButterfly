# NTT output order

The shared-iterative NTT lowering supports natural or bit-reversed output for
both word32 and word64, forward and inverse transforms. The input remains in
natural order. The existing modulus, stage partition and device capacity checks
still apply. APPT static layout remains a separate specialized contract.

For a natural-order result `y`, bit-reversed output means
`output[reverse_bits(k, logN)] = y[k]`, independently for each batch. Inverse
normalization is applied before this store. An inverse transform requesting
bit-reversed output still consumes natural-order input; it does not implicitly
undo the layout of a previous bit-reversed result.

The final physical stage owns this permutation. Earlier groups retain their
original intermediate layout, including groups launched through the batch
pipeline or the independent stage probe. No additional permutation kernel or
full-size output buffer is required. With `s` stages in the terminal group,
writer lane index `i` reads logical tile index `reverse_bits(i, s)` and writes
to `reverse_bits(tile, logN-s) * 2^s + i`. Thus consecutive writer indices retain
consecutive global stores. This does not eliminate the cost of shared-memory
reads or make every partition equally efficient.

The precompiled kernel receives the output order from the plan. Research mode
specializes it in the shared kernel template; the standalone request and cache
identity include `output_order`. The V2 module invocation ABI is unchanged.
Installation precompilation projects the same semantic field as runtime JIT.

The C API accepts `CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED` as the output argument of
`cubutterflySetNttLayouts`. It requires a standard, rank-1, power-of-two NTT and
natural input; arbitrary-length and rank-2 permutations are rejected explicitly.
Existing enum values are preserved. Default selection, explicit
`ntt-shared-iterative`, and measured selection retain the requested output order.
The portable candidate inventory omits legacy families lacking this writeback.

Stage-service keys already distinguish output order. Natural-order timings must
not be relabeled as bit-reversed measurements. After deploying a changed
lowering, collect the affected physical services with the matching runtime
fingerprint. Existing hardware facts remain usable, but old kernel timings are
not measurements of the new build.

The repair is validated in a separate source snapshot, build, module cache and
result directory. See `results/ntt_output_order_fix/` for its raw evidence and
the final report. It does not replace the original A100 acceptance cohort.

Reproduce the bounded check in an independent build (SM80 example):

```sh
python scripts/validate_ntt_output_order.py --phase prepare \
  --build-dir build-a100-ntt-order --output results/ntt-order-check --sm 80
python scripts/validate_ntt_output_order.py --phase run \
  --build-dir build-a100-ntt-order --output results/ntt-order-check \
  --gpu-uuid GPU-YOUR-DEVICE-UUID
```

Preparation runs without visible GPUs. Execution verifies every batch and keeps
compilation outside the reported kernel time. The report compares output-layout
variants and compile policies; it is not an external-library speedup claim.
