# External CPU baseline audit (2026-10-12)

This is a supplemental CPU-only audit. CPU measurements are not pooled into GPU speedups or operator geometric means. Sources are downloaded from official project/author endpoints and hash-bound in the JSON ledger.

## Coverage

| Missing GPU comparator | Real external implementation | Status | Contract boundary |
| --- | --- | --- | --- |
| fwht / fp64 | FXT walsh_wak (Walsh-Kronecker) | measured | in-place FXT call adapted to out-of-place; self-inverse unnormalized; natural input/output for this call; exact basis must be checked against requested FWHT order |
| subset-zeta / uint32 | FXT arith_transform_plus | measured | arithmetic subset transform; integer addition is unbounded in this wrapper, so it is only exact for non-overflow test values and not a modular uint32 baseline |
| superset-zeta / uint32 | FXT rev_arith_transform_plus | measured | reverse arithmetic transform; same non-modular uint32 caveat |
| structured-2x2 / fp32 | HazyResearch butterfly CPU | path identified; not built this round | supports arbitrary 2x2 butterfly factors, but API packaging and exact stage order/stride adapter remain to be implemented |
| ntt / word32 | Intel HEXL / NTL | no valid direct match | HEXL is negacyclic and uint64 API; NTL FFT is polynomial multiplication machinery, not a matching cyclic natural-order word32 transform |

## FXT measurement

FXT release `2026.05.19` was compiled from the downloaded author release. The wrapper calls the library transforms, uses an out-of-place copy adapter, and checks all output values. It adds N=2^20, B=4 at one and four OpenMP outer-batch threads to the earlier N=2^12/2^16 rows (30 rows total).

| Operator | Precision | N | B | Threads | Median us/batch | us/transform | Mismatches | Error check |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fwht | fp64 | 1048576 | 4 | 1 | 17694.486 | 4423.622 | 0 | relative L2 2.274e-16 |
| fwht | fp64 | 1048576 | 4 | 4 | 8048.976 | 2012.244 | 0 | relative L2 2.274e-16 |
| subset-zeta | uint32 | 1048576 | 4 | 1 | 14529.659 | 3632.415 | 0 | exact integer |
| subset-zeta | uint32 | 1048576 | 4 | 4 | 7607.779 | 1901.945 | 0 | exact integer |
| superset-zeta | uint32 | 1048576 | 4 | 1 | 14908.113 | 3727.028 | 0 | exact integer |
| superset-zeta | uint32 | 1048576 | 4 | 4 | 6839.641 | 1709.910 | 0 | exact integer |

The N=2^20/B=4 results are host measurements on AMD EPYC 7542. FP64 FWHT full-output relative L2 is 2.274e-16 for both thread settings with zero mismatches against a long-double reference. Zeta runs have zero mismatches against the direct subset/superset enumeration check at N=256 and full integer transform reference; the wrapper uses uint32 addition without a modulus, so these are not modular NTT or uint32-wrap claims.

## Reuse and blockers

- FXT is a real, templated transform library and is usable as a CPU algorithm witness for Walsh and arithmetic subset/superset transforms. Basis/order and normalization must be explicitly reconciled before a same-contract comparison; no GPU ratio is inferred.
- Intel HEXL is a real optimized CPU NTT library, but its documented contract is negacyclic `Z_q[X]/(X^N+1)`, `uint64_t`, and `q == 1 (mod 2N)`. It is therefore not a direct baseline for the current cyclic natural-order word32 request. Its output is bit-reversed and would require a declared adapter.
- NTL exposes user FFT-prime initialization and FFT polynomial arithmetic. It is a construction component, not a drop-in current word32 cyclic NTT baseline.
- HazyResearch butterfly has an official CPU C++ implementation for floating/complex 2x2 factors. It is a credible structured-butterfly path, but exact stage matrices, order, strides and packaging were not built in this bounded round.
- SPIRAL provides generation methodology and platform-tuned transform code generation; no directly executable matching contract was obtained in this round.

Reproduction command:

```bash
g++ -O3 -march=native -std=c++17 -fopenmp -I results/external_cpu_baselines_20261012/fxt/src paper/evidence/benchmark_fxt_cpu_20261012.cpp -o results/external_cpu_baselines_20261012/benchmark_fxt_cpu
OMP_PROC_BIND=close OMP_PLACES=cores CUDA_VISIBLE_DEVICES="" results/external_cpu_baselines_20261012/benchmark_fxt_cpu > results/external_cpu_baselines_20261012/fxt_cpu_rows.jsonl
```

See `EXTERNAL_BASELINE_AUDIT_20261012.json` for URLs, raw source hashes, seed contract and every measured row.
