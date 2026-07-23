# GPU-NTT Gap Analysis on V100

## Conclusion

The original gap was primarily in the Hybrid2D boundary operation, not in
kernel count. The original implementation performed a separate
`root^(n1*k2)` Shoup multiplication for every coefficient and read two 64-bit
tables (`root_powers` and `root_powers_shoup`).

The new fused mode evaluates the second dimension as a coset NTT. For a stage
with butterfly width `m`, it replaces the normal butterfly twiddle with:

```text
omega^((N1/m) * (N2*offset + k2))
```

This absorbs the cross factor into the modular multiplication already required
by each butterfly. It preserves the two-dimensional algorithm factorization and
processing-unit independence while changing the equivalent butterfly
organization. The current CUDA kernels temporally fold their local stages; they
are not, by themselves, an explicit stage-space pipeline implementing all four
APPT unfolding modes.

With V100-specific Hybrid2D mapping, cuNTT is 5.8% lower latency than GPU-NTT
at `logN=16`, 3.7% lower at `logN=18`, and 8.3% higher at `logN=20`. The later
compact-stage experiment closes the `logN=20` native-layout gap as described
below.

## Fair comparison protocol

- GPU: Tesla V100-SXM2-16GB, `sm_70`.
- CUDA: 11.8, Release builds.
- GPU-NTT commit: `d03c5eaeadaa780d153496afcb3b6a9b79a13a63`.
- Forward, cyclic, GPU-resident NTTs using `PerPolynomial` layout.
- Same 60-bit prime: `576460756061519873`.
- The input and root tables are resident before timing; transfers are excluded.
- Each row processes about `2^22` coefficients per invocation.
- 1000 warmups and 200 timed repetitions, with the median of three invocations.

The longer warmup matters on this server. A 20-iteration warmup lasts only a few
milliseconds and produced transient `logN=16` GPU-NTT rates from 194k to 226k
NTT/s as the V100 clock changed. The results below use the longer steady-load
protocol.

The comparator harness is `benchmarks/gpuntt_merge_gap_bench.cu`. With a
GPU-NTT checkout built for `sm_70`, compile and run it as follows (replace the
checkout path if needed):

```bash
/usr/local/cuda-11.8/bin/nvcc -std=c++17 -O3 -arch=sm_70 -ccbin g++-9 \
  -I /tmp/GPU-NTT-compare/src/include \
  benchmarks/gpuntt_merge_gap_bench.cu \
  /tmp/GPU-NTT-compare/build-v100-118/src/libntt-1.0.a \
  -o /tmp/gpuntt_merge_gap_bench
/tmp/gpuntt_merge_gap_bench 16 64 1000 200
build/cuntt_bench --logN 16 --batch 64 --backend hybrid2d \
  --compute-unit radix4 --word-bits 64 --modulus 576460756061519873 \
  --warmup 1000 --repeat 200 --csv
```

| logN | Batch | cuNTT ms | GPU-NTT ms | cu/GPU latency | cu throughput deficit |
|-----:|------:|---------:|-----------:|---------------:|----------------------:|
| 12 | 1024 | 0.260797 | 0.240978 | 1.082x | 7.6% |
| 14 | 256 | 0.281503 | 0.256072 | 1.099x | 9.0% |
| 16 | 64 | 0.349553 | 0.293622 | 1.190x | 16.0% |
| 18 | 16 | 0.450289 | 0.355733 | 1.266x | 21.0% |
| 20 | 4 | 0.789770 | 0.395156 | 1.999x | 50.0% |

The first five rows use cuNTT's automatic balanced mapping. GPU-NTT uses two
kernels through `logN=16` and three at `logN=18` and `20`; cuNTT uses two for
all rows. The growing gap despite cuNTT's lower pass count rules out launch or
full-array pass count as the main cause.

## Fused result

The fused implementation was measured with the same steady-load protocol and
interleaved GPU-NTT/cuNTT invocations:

| logN | cuNTT mapping | cuNTT ms | GPU-NTT ms | cu/GPU latency | cu throughput ratio |
|-----:|:--------------|---------:|-----------:|---------------:|--------------------:|
| 16 | `8+8`, 4 rows, 256 threads | 0.274104 | 0.291133 | 0.942x | 1.062x |
| 18 | `9+9`, 4 rows, 512 threads | 0.342482 | 0.355820 | 0.963x | 1.039x |
| 20 | `10+10`, 2 rows, 512 threads | 0.428129 | 0.395305 | 1.083x | 0.923x |

Moving the unchanged cross multiplication to the second-pass load was also
tested and rejected: it regressed `logN=16`, `18`, and `20`. The gain comes
from removing the standalone multiplication, not merely moving work between
passes.

## Attribution evidence

1. Nsight Systems reports that the cuNTT first pass consumes about 62% of its
   traced kernel time. Its second pass has no cross twiddle. Tracing perturbs
   GPU-NTT more than cuNTT on this system, so cross-implementation traced times
   are not used as benchmark numbers.
2. A temporary diagnostic build removed only the final cross-twiddle multiply
   and its two table reads from the first pass. The resulting output is not a
   valid NTT and is used only for timing attribution.
3. The diagnostic reduced cuNTT by 0.087 ms at `logN=16`, 0.147 ms at
   `logN=18`, and 0.313 ms at `logN=20`. At `logN=16` and `18`, removing this
   operation reverses the gap; at `logN=20`, it explains about 79% of the
   observed latency difference.
4. The two cross-twiddle tables occupy `16*N` bytes: 1 MiB at `logN=16`, 4 MiB
   at `logN=18`, and 16 MiB at `logN=20`. This matches the sharp length
   sensitivity and makes table traffic/cache behavior the leading hypothesis
   for the remaining large-size penalty.
5. GPU-NTT's 64-bit `ForwardCore` uses 48 registers per thread, while cuNTT's
   selected radix-4 kernels use 32. Register pressure is therefore not the
   obvious source of cuNTT's deficit.

## Single-table Barrett diagnostic

`--cross-twiddle fused-barrett` keeps the fused coset organization but replaces
the root-plus-Shoup pair with one root table and Barrett multiplication. It is
an explicit diagnostic mode; the default remains fused Shoup. On the V100,
with about `2^22` points per invocation, 1000 warmups, 200 repeats, and three
process medians:

This table records the legacy diagnostic implementation, where Barrett applied
only to the fused second pass. The current interface separates placement and
arithmetic; `--cross-twiddle fused --mod-multiply barrett` applies Barrett to
both local passes, cross-twiddles, and inverse scaling. Therefore a new run is
an end-to-end processing-unit experiment and should not be compared directly
with the legacy timings below.

| logN | fused Shoup (ms) | fused Barrett (ms) | Barrett change | Shoup table | Barrett table |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.274565 | 0.293309 | +6.827% | 0.996 MiB | 0.498 MiB |
| 18 | 0.342303 | 0.344704 | +0.701% | 3.992 MiB | 1.996 MiB |
| 20 | 0.428867 | 0.435569 | +1.563% | 15.984 MiB | 7.992 MiB |

The second-pass kernel uses 40 registers per thread in the Barrett mode versus
32 for fused Shoup (`cuobjdump --dump-resource-usage`). With the selected
256/512-thread blocks this reduces the register-limited resident-thread ceiling
from 100% to approximately 75%. Halving the table therefore does not recover
the `logN=20` gap: additional wide-integer work and register pressure outweigh
the saved load. This does not prove the Shoup table is cache-benign; NCU must
separate the DRAM/L2 benefit from the instruction and occupancy penalties.

## Optimization priority

The cross-twiddle arithmetic gap is closed at medium lengths. A direct compact
table plus Barrett substitution is rejected on V100. The remaining `logN=20`
work should reduce or reuse fused-table traffic without increasing register
pressure: derive stage factors on chip, stage reusable roots in shared memory,
or reorganize the butterfly so reduction temporaries have shorter live ranges.
The best mapping changed after fusion, confirming that factorization, rows, and
threads must be selected for both the hardware and the processing-unit
organization.

Nsight Compute initially returned `ERR_NVGPUCTRPERM` for an unprivileged
process. An administrator-enabled run has now provided the required counters;
the detailed attribution is in `results/ncu_logN20_analysis.md`.

At `logN=20`, cuNTT's first pass is 23.07 us faster than GPU-NTT's first two
kernels combined, while its fused second pass is 55.20 us slower than
GPU-NTT's final kernel. The final phases have similar L2 hit rates (43.11% and
43.83%), but cuNTT reads 96.38 MiB versus 54.74 MiB and has 42.12% versus
24.68% long-scoreboard stalls. This confirms the expanded per-`k2` fused table,
not launch count, occupancy, or integer instruction count, as the primary
remaining large-size bottleneck.

## Compact-stage experiment and output semantics

The original comparison mixed output semantics: Hybrid2D returns natural
order, while GPU-NTT Merge's native forward output is bit-reversed. The
comparator now verifies both layouts and can include an explicit naturalization
kernel through its sixth positional argument.

`compact-stage` tests the NCU-driven alternative at `logN=20`: it uses the same
three-stage V100 mapping shape as GPU-NTT (`5+6+9` temporally fused butterfly
levels), but uses cuNTT arithmetic and a compact standard root-plus-Shoup table.
The architecture is the reusable part; this concrete mapping remains a V100
parameter choice. All rows below use the same 60-bit prime, batch 4, 1000
warmups, and 200 repeats.

| Implementation | Output order | Kernel ms | Relative to matching GPU-NTT |
|:--|:--|--:|--:|
| cuNTT compact-stage | bit-reversed | 0.341320 | 13.9% lower latency |
| GPU-NTT Merge | bit-reversed | 0.396186 | baseline |
| cuNTT compact-stage | natural | 0.760494 | 0.4% higher latency |
| GPU-NTT Merge + reorder | natural | 0.757181 | baseline |
| cuNTT Hybrid2D | natural | 0.429117 | 43.3% lower than GPU-NTT + reorder |

The bit-reversed numbers are medians of three interleaved process runs. The
natural rows are representative verified runs. The result supports the NCU
diagnosis: replacing the expanded per-`k2` table with a reusable standard table
removes the native-layout gap. It also shows that a full bit-reversal
permutation is too expensive to treat as incidental. Hybrid2D remains the
preferred natural-order backend; compact-stage is useful when downstream work
can consume bit-reversed data or fuse the permutation into a later operation.

The exact collection and CSV-processing commands are documented in
`docs/ncu_profiling.md`.

Raw medians are in `results/gpu_ntt_gap_same_modulus.csv`; the intentionally
invalid attribution experiment is in `results/gpu_ntt_gap_diagnostic.csv`, and
the fused comparison is in `results/fused_vs_gpuntt.csv`. The reduction A/B
runs and summary are in `results/fused_reduction_ab.csv` and
`results/fused_reduction_summary.csv`.
