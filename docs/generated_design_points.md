# Generated Processing Units

## Build Boundary

The common runtime owns execution semantics and the APPT space-time mapping.
CUDA processing units are selected in `config/v100_design_points.json` and
generated as separate translation units:

```text
design-point JSON
  -> scripts/generate_design_points.py
  -> generated availability registry
  -> one CUDA translation unit per processing-unit family
  -> common ButterflyPlan runtime
```

The default V100 specification emits:

| Operator | Core | Precision | Generated range |
|:--|:--|:--|:--|
| FWHT | `warp-register` | FP32 | `logN=3..15` |
| FFT | `thread-dft8` | FP32 | `logN=3` |
| FFT | `cta-dft8` | FP32 | `logN=3..10` |
| FFT | `wmma-dft8` | FP16 input, FP32 accumulation/output | `logN=3` |

Unsupported combinations fail during plan construction. They are not silently
redirected to another core. The scalar shared-memory implementation remains the
general fallback and is selected with `--fft-core scalar`.

Configure with another generated range using:

```bash
cmake -S . -B build \
  -DCUNTT_DESIGN_SPEC=$PWD/config/v100_design_points.json
cmake --build build -j
```

The generator can also emit an isolated experiment without changing the main
build:

```bash
python3 scripts/generate_design_points.py \
  --spec config/v100_design_points.json \
  --operators fft --cores wmma-dft8 --logNs 3 \
  --output-dir /tmp/cubutterfly-wmma
```

## FFT Codelets

`thread-dft8` assigns one complete complex DFT8 to a CUDA thread. Eight values,
the bit-reversal load, three butterfly stages, and normalization remain in the
thread register file. The compiled V100 kernel uses 34 registers/thread and no
shared memory.

`cta-dft8` retains the same per-thread register DFT8 but composes it inside a
generated CTA tile. With `U_s=tile_threads` spatial codelets, a block owns
`8*U_s` points and processes `8*U_s/N` independent transforms. It loads and
stores natural-order global addresses contiguously, performs bit-reversal reads
and groups of three stages in registers, then exchanges between codelets through
`64*U_s` bytes of shared memory. The V100 specification generates 32, 64, 128,
and 256-thread variants where the workset is large enough for one transform.

`wmma-dft8` expresses a complex DFT8 as one real `16x16` block matrix. Eight
warps per CTA independently process 16 transforms per warp. They share one
coefficient matrix while retaining separate WMMA input and accumulator tiles.
The compiled V100 kernel uses 38 registers/thread and 12.5 KiB shared memory.
This maps three temporally fused butterfly levels and 128 spatially independent
transforms into one CTA.

The WMMA core exposes `fp16-fp32` explicitly. External arrays use the existing
Complex32 layout, operands are rounded to FP16 at the codelet boundary, Tensor
Core accumulation and output are FP32. It is not numerically equivalent to the
FP32 cores.

## V100 Evidence

Contiguous forward transforms, approximately `2^22` points, 20 warmups, 100
repetitions, and five trials give:

| Core | Semantics | Median ms | Gbutterfly/s | Relative observation |
|:--|:--|--:|--:|:--|
| shared radix-8, 32 threads | FP32 | 0.743147 | 8.47 | general scalar control |
| `thread-dft8` | FP32 | 0.131000 | 48.03 | 5.67x faster than shared |
| `cta-dft8` | FP32 | 0.096645 | 65.10 | 1.36x faster than thread FP32 |
| `wmma-dft8` | FP16/FP32 mixed | 0.089590 | 70.23 | 1.46x faster than thread FP32 |
| cuFFT | FP32 | 0.086323 | 72.88 | vendor reference |

The CTA register point reaches 89.3% of cuFFT throughput with identical FP32
semantics. The mixed WMMA point is within 3.8% of the FP32 cuFFT time, but
the different precision contract prevents treating this as an FP32 ranking.
Forward random-input verification observed maximum absolute complex error
`0.001328`; inverse normalized strided/in-place verification observed
`0.000159`.

Raw and ranked records are in:

- `results/fft_generated_units_v100_fp32_raw.csv`
- `results/fft_generated_units_v100_mixed_raw.csv`
- `results/fft_generated_units_v100_cta_raw.csv`
- `results/fft_generated_units_v100_summary.csv`
- `results/fft_cta_space_time_sync_v100_raw.csv`
- `results/ncu_fft_units_analysis.md`

The composed `logN=3..10` sweep and its mapping interpretation are reported in
[`fft_cta_space_time_results.md`](fft_cta_space_time_results.md).

## NCU

The administrator profiling run is prepared as:

```bash
sudo -E ./scripts/profile_fft_units_ncu.sh
sudo chown -R "$USER:$USER" results/ncu_fft_units
```

It profiles scalar radix-8, thread-register DFT8, CTA-staged DFT8, WMMA DFT8, and cuFFT with
identical shape and batch. It collects resident time, DRAM/L2/L1 behavior,
FP32 and Tensor instructions, Tensor-pipe activity, active warps, barrier and
long-scoreboard stalls, registers, and shared memory. The script writes both
raw NCU CSV files and `results/ncu_fft_units/summary_logN3.csv`.

The next FFT step is to use this generated `logN<=10` CTA unit as the local
prefix of long transforms, then fuse its online-reorder output with the suffix
stages. That extension changes the inter-CTA mapping, not the DFT8 codelet.
