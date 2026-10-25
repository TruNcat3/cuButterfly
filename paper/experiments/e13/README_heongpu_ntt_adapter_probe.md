# HEonGPU NTT Adapter Probe

This is a bounded correctness probe for the pinned HEonGPU/GPU-NTT source at
commit `1928a14bb002c5d7f1cb5534eca9fa19b555b4f2`. It covers only the v1
regular-bootstrap q0 boundary: one polynomial, `N=4096`, one 60-bit prime,
`X^N+1`, forward bit-reversed output, inverse normalization, and a caller
CUDA stream. It does not execute CKKS bootstrap, use the full RNS matrix, or
make a performance/security claim.

The probe links the already built pinned GPU-NTT archive and the existing
cuButterfly shared library. It reproduces q0 from HEonGPU's
`generate_primes(4096, [60])` rule and canonicalizes the `2N` root using the
same odd-power minimum rule as `find_minimal_primitive_root`. The host reference
then independently evaluates

`y[k] = sum_j a[j] * psi^j * (psi^2)^(j*k) mod q0`.

The pinned GPU-NTT path must pass before the cuButterfly comparison is accepted.
The cuButterfly part runs a cyclic forward/inverse control and an attempted
`psi^j` pre-twist with bit-reversed output. It records both the direct
comparison and an explicit frequency-index adapter. If
`cu_omega = (psi^2)^r` for odd `r`, the forward result is aligned with
`cu_output[k*r^-1 mod N]`; the inverse test prepares cuButterfly input at
`reference[k*r mod N]`, then applies cyclic inverse normalization and
`psi^-j`. This separates an algebraically valid adapter from a direct
semantic replacement: even `adapter_frequency_permutation_compatible` still
requires that explicit permutation and twist at the caller boundary.
The twist, bit-reversal conversion, and frequency reindex in this probe are
host-assisted preparation/checks; the NTT kernels execute on the GPU, but this
does not implement a device-resident application adapter.

## Build and run

```bash
/home/wt/yes/envs/cubutterfly/bin/cmake \
  -S paper/experiments/e13 \
  -B paper/experiments/e13/build-heongpu-ntt-adapter-probe \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
/home/wt/yes/envs/cubutterfly/bin/cmake \
  --build paper/experiments/e13/build-heongpu-ntt-adapter-probe \
  --parallel 2
CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
  paper/experiments/e13/build-heongpu-ntt-adapter-probe/heongpu_ntt_adapter_probe \
  > results/paper_profile_rc2/end_to_end/heongpu_ntt_adapter_probe.json \
  2> results/paper_profile_rc2/end_to_end/heongpu_ntt_adapter_probe.run.log
```

The output JSON records q0, psi, `psi^2`, cuButterfly's selected root, exact
mismatch counts, inverse normalization, stream-ordered GPU checks, and the
odd frequency multiplier when the roots differ. `adapter_forward_compatible`
means the direct twist path matches. `adapter_frequency_permutation_compatible`
means the direct path differs but both explicit frequency-reindexed forward and
inverse paths match. `blocked_root_mismatch` means no exact odd-index adapter
was found (or it did not pass); `probe_failed` means the pinned correctness
checks failed. None of these statuses runs the full RNS/bootstrap workload or
makes a performance/security claim.
