# V100 Initial Results

Measured on 2026-07-19 with a Tesla V100-SXM2-16GB (`sm_70`), CUDA 11.8,
and a Release build. The workload is a forward NTT with `logN=16`, the default
60-bit modulus, 20 warmups, and 200 timed repetitions.

| Batch | Backend | Kernel time (ms) | NTT/s | Gpoint/s | Speedup |
|------:|:--------|-----------------:|------:|---------:|--------:|
| 1 | baseline | 0.060554 | 16,514 | 1.082 | 1.000x |
| 1 | tile256 | 0.039173 | 25,528 | 1.673 | 1.546x |
| 8 | baseline | 0.146263 | 54,696 | 3.585 | 1.000x |
| 8 | tile256 | 0.104694 | 76,413 | 5.008 | 1.397x |
| 64 | baseline | 1.518531 | 42,146 | 2.762 | 1.000x |
| 64 | tile256 | 0.946867 | 67,591 | 4.430 | 1.604x |

Reproduce one row with:

```bash
build/cuntt_bench --logN 16 --batch 64 --backend tile256 \
  --warmup 20 --repeat 200 --csv
```

The CUDA object reports the following `sm_70` resource use:

| Kernel | Registers/thread | Shared memory/block |
|:-------|-----------------:|--------------------:|
| bit-reversal copy | 14 | 0 B |
| global radix-2 stage | 24 | 0 B |
| tile256 | 30 | 2,048 B |
| inverse scale | 14 | 0 B |

Both global and tile kernels have enough resources for full theoretical
occupancy on V100. At batch 64, baseline moves approximately 272 bytes per
point across global memory (bit reversal plus 16 stage passes), while tile256
moves approximately 160 bytes per point (bit reversal, one tile pass, and eight
global stages). The measured point rates imply roughly 751 GB/s and 709 GB/s of
global-memory traffic respectively. This is consistent with a bandwidth-bound
implementation and explains why fusing the first eight stages helps.

This motivated a parameterized two-pass decomposition for `logN=16`, replacing
the eight remaining global stage passes with a fused twiddle/transpose and a
second shared-memory local-transform pass.

## Hybrid2D Result

The two-pass `hybrid2d` backend was implemented and measured on 2026-07-22 with
the same V100, CUDA, modulus, transform size, warmup count, and repetition count.
The table uses the median kernel time from three independent benchmark
invocations for each backend and batch size.

| Batch | Backend | Kernel time (ms) | NTT/s | Speedup vs tile256 |
|------:|:--------|-----------------:|------:|-------------------:|
| 1 | tile256 | 0.039214 | 25,501 | 1.000x |
| 1 | hybrid2d | 0.020111 | 49,723 | 1.950x |
| 8 | tile256 | 0.104776 | 76,354 | 1.000x |
| 8 | hybrid2d | 0.067686 | 118,192 | 1.548x |
| 64 | tile256 | 0.946739 | 67,600 | 1.000x |
| 64 | hybrid2d | 0.360499 | 177,532 | 2.626x |

This original radix-2 V100 instance uses two kernels with `N1=N2=256`. Each
block processes four rows, and each fixed-size kernel used 26 registers per
thread and 8,224 bytes of shared memory. The two
passes read and write every coefficient once, for a minimum of 32 data bytes per
point; root and stage twiddle traffic is additional but benefits from cache
reuse across rows and batches.

The later parameterized radix-4 implementation reaches a three-run median of
184.3k NTT/s at batch 64. The same-card GPU-NTT Merge comparison at
approximately 223--226k NTT/s is therefore about 1.21--1.23x faster instead of
3.3x faster than the initial cuNTT backend. The remaining gap is an optimization
gap within the two-pass design, not an architectural pass-count gap.

These figures predate coset-twiddle fusion and the longer steady-clock warmup
protocol. The current same-modulus result and bottleneck attribution are in
[`gpu_ntt_gap_analysis.md`](gpu_ntt_gap_analysis.md); fused cuNTT now slightly
outperforms GPU-NTT at `logN=16` and `18` on this V100.

## Validation

- CPU/GPU comparisons cover `baseline` and `tile256` for `logN=1..16`, plus
  targeted `hybrid2d` cases at `logN=16`.
- Forward, inverse, batched, custom-modulus, and round-trip cases pass.
- CUDA memcheck reports zero errors.
- CUDA racecheck reports zero hazards, errors, or warnings.

The `hybrid2d` validation additionally covers `n1_log=6..10`, batch sizes 1, 2, 3, and 5,
forward and inverse operation, the default 60-bit modulus, and `998244353`.

### V100 decomposition scan

The architecture does not prescribe `N1=N2=256`. Holding the CUDA mapping at
four rows and 256 threads per block, an exploratory batch-64 scan selected the
balanced factorization on this V100:

| N1 x N2 | Kernel time (ms) | NTT/s |
|:--------|-----------------:|------:|
| 64 x 1024 | 0.475064 | 134,719 |
| 128 x 512 | 0.383140 | 167,041 |
| 256 x 256 | 0.360463 | 177,549 |
| 512 x 128 | 0.361027 | 177,272 |
| 1024 x 64 | 0.423409 | 151,154 |

A second scan over `rows_per_block={1,2,4}` and
`threads_per_block={64,128,256,512}` retained four rows and 256 threads. These
are V100 mapping results rather than constants of the Hybrid2D architecture.
