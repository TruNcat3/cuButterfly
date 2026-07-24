# Changelog

All notable repository and research-artifact changes are recorded here.

## Unreleased

### Added

- An orthogonal 184-case V100 length/batch suite and generated manifest for
  FFT, NTT, FWHT, and XOR-zeta.
- Scaling summaries that identify timing quality, throughput saturation, and
  mapping crossovers without using sub-0.020-ms rows as stable claims.
- Targeted NCU collection for the FFT saturation gaps and FWHT processing-unit
  crossover.
- A V100-calibrated mapping selector with held-out-shape evaluation and
  explicit separation of mapping family from processing unit.
- A matching-protocol 120-sample Dao FHT and GPU-NTT baseline refresh.

### Changed

- The comprehensive runner now supports exact case selection, validates batch
  consistency within comparison groups, and caches identical correctness
  preflights.
- The next release is scoped around counter-calibrated mapping selection and
  refreshed same-protocol external baselines.
- Cross-GPU validation is deferred until the repository moves to a host with a
  second GPU generation; it is not a `v0.3.0` release gate.

### Current Evidence Boundary

- The new scaling evidence remains V100-only.
- FP32 FFT matches or exceeds cuFFT at selected saturated shapes, but cuFFT has
  a higher large-batch ceiling at `logN=18` and remains ahead at `logN=20`.
- Cross-GPU portability and selector accuracy have not yet been measured.

## 0.2.0 - 2026-07-24

### Added

- Operator-independent butterfly and FFT architecture-space specifications,
  validation, exploration, and generated-code selection tools.
- Optional cuFFTDx, TurboFFT, and VkFFT integration boundaries.
- A controlled 37-case V100 comprehensive suite with correctness preflight,
  randomized trials, stability classification, raw records, and detailed
  length/batch tables.
- FFT architecture sweeps, processing-unit comparisons, and resident-kernel
  Nsight Compute evidence.

### Changed

- Extended FFT, FWHT, NTT, and XOR-zeta configuration and capability coverage.
- Updated the architecture, hardware-mapping, processing-unit, experiment, and
  reproducibility documentation to distinguish fresh comprehensive results
  from focused historical protocols.
- Defined the next phase as an orthogonal length/batch scan, refreshed external
  baselines, and targeted profiling of long FP32 and FP64 FFT gaps.

### Current Evidence Boundary

- V100 remains the only comprehensively measured GPU.
- Measured FP32 FFT reaches cuFFT parity at selected lengths, while FP32
  `logN=20` and FP64 `logN=16` remain open performance gaps.
- Dao-AILab FHT and GPU-NTT results were not rerun under the comprehensive
  protocol and remain contextual evidence.

## 0.1.0 - 2026-07-23

- Initial public cuButterfly research artifact with the common space-time
  mapping model, NTT/FFT/FWHT/XOR-zeta kernels, V100 measurements, and
  reproducibility documentation.
