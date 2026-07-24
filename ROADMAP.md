# Roadmap

This roadmap separates implementation goals from evidence required for the
research claim. Ordering may change after profiling or access to new hardware.

## Near Term

- Run an orthogonal `(logN, batch)` sweep instead of coupling all comparisons
  through constant total points; identify launch-limited, saturation, and
  throughput regions for every operator.
- Refresh Dao-AILab FHT and GPU-NTT under the comprehensive-suite clock,
  correctness, modulus, output-order, and trial protocol.
- Profile the remaining FP32 `logN=20` and FP64 `logN=16` FFT gaps against
  cuFFT, separating local-core, permutation, coefficient, and launch costs.
- Convert the measured architecture space into a calibrated selector and
  evaluate top-k prediction accuracy before exhaustive timing.
- Add a device-pointer and CUDA-stream execution API without weakening the
  current typed semantic contract.
- Reduce template warning volume and record compiled resource envelopes as
  generated metadata.
- Add focused sanitizer jobs on an available self-hosted GPU runner.

## Cross-GPU Validation

- Capture A100/H100/RTX 4090 hardware service profiles using the checked-in
  schema.
- Predict a reduced `(Us,Ud,Ts,Td,F,Q)` candidate set before exhaustive timing.
- Regenerate legal processing units for each architecture.
- Compare predicted and measured rankings for FFT, NTT, FWHT, and XOR-zeta.
- Identify which mapping thresholds transfer and which require recalibration.

This milestone is required before claiming that the selection methodology is
portable across GPU generations.

## Processing Units

- FFT: Stockham/split-radix and established register/shared long-FFT units;
  Tensor Core blocks only where precision and composition costs are explicit.
- NTT: compressed or recurrent root generation, shorter-lived modular
  reduction temporaries, and carefully bounded lazy/Montgomery candidates.
- FWHT: multi-CTA composition beyond the current register-resident length.
- Layered operators: add another non-transform butterfly workload with an
  independently validated reference and external baseline where available.

## Paper Artifact

- Preserve the V100 comprehensive suite as the single-GPU baseline and keep
  focused historical protocols separate from its cross-workload tables.
- Freeze a versioned multi-GPU measurement matrix.
- Publish scripts and container/toolchain metadata for every main table.
- Separate architecture ablations, core-only comparisons, resident transforms,
  and application end-to-end results.
- Add statistical confidence and clock/power controls.
- Archive a release with a persistent identifier when the paper is submitted.

## Non-Goals For The Current Release

- claiming universal superiority over specialized libraries;
- treating V100-selected tile sizes as architecture constants;
- equating a Tensor Core instruction count with end-to-end FFT acceleration;
- mixing native bit-reversed and required natural-order timings;
- presenting placeholder GPU rows as measured results.
