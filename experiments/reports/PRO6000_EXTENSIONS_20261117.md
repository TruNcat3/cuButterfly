# PRO6000 FFT17 and ALLOW_FP16_MMA9 complete-call comparisons

**Status: measured finite extension cohorts, with floating error review and declared provider gaps.** All original FFT17 and MMA9 requests were retained. These are separate from the [25 fixed-seed samples](PRO6000_FIXED_MAPPING_SAMPLES_20261117.md), the main Full66 acceptance and global model validation.

The measured build is source commit `a305d97769a0c005591a8bbe4e14ae1a238b7e83` with the archived [compatibility patch](../snapshots/pro6000_fixed_mapping_20261117/provenance/build-compatibility.patch.gz). Main library SHA256: `3b0668f2530a75d1401dcbf5decea720f8e47db0c1357aafdf1680fa1bba6adb`. CUDA 12.8, driver 580.159.04; NVIDIA RTX PRO 6000 Blackwell Server Edition, 101,973,950,464 bytes per device (SM120). The native SM120 HadaCore provider is a separate build from the preserved SM80/PTX illegal-instruction attempt.

FFT17 uses GPU UUID `a257f4db-8c62-f26b-9c08-a4045615a140`; MMA9 uses `1067ab39-e411-e141-1775-64426bca5701`. Each cohort records its own hardware, libraries, provider/profile hashes and source fingerprints; timings are not pooled across cards or with A100.

Protocol: complete public calls, out-of-place, same full shape/batch/precision/direction/normalization and CPU-prepared input/reference per comparison. Each implementation has 10 warmup calls, then 7 CUDA Graph replays of 20 complete calls; values are sample medians in µs. Required copies, permutations and scales are included; setup, compilation, planning, allocation and CPU checks excluded. Implementations are collected sequentially, not alternating independent process trials. Both outer and per-case guards reject foreign CUDA contexts.

Floats remain `accuracy-review-required`, with complete error distributions; no threshold is introduced or relaxed. FP16 MMA and native stage-rounded FWHT have different reduction/rounding contracts. Ratios are comparator/cuButterfly; above 1 is nominally faster for cuButterfly. Near-1 differences are not statistical evidence. Selection/calibration costs are excluded from steady-state execution and are not amortization claims.

## FFT17

All 17 requests use public DEFAULT replay from same-GPU exact-request MEASURE calibration (candidate budget 16), with complete mapping and semantic equality checked. This avoids using the cost model for unknown requests: its warning/unvalidated status was retained. No hardware-rate-prior attempt is admitted. There are 51 comparator records: 47 measured pairs and four explicit GPU-FFT multi-axis gaps.

| Precision | Shape | Batch | Direction | cuButterfly µs | cuFFT µs / ratio | VkFFT µs / ratio | GPU-FFT µs / ratio |
|---|---|---:|---|---:|---:|---:|---:|
| FP32 | 256 | 1 | forward | 1.797 | 1.669 / 0.929× | 1.656 / 0.922× | 3.962 / 2.205× |
| FP32 | 256 | 1 | inverse / 1/N | 1.979 | 2.598 / 1.313× | 1.512 / 0.764× | 3.618 / 1.828× |
| FP32 | 4096 | 32 | forward | 4.741 | 4.699 / 0.991× | 3.861 / 0.814× | 8.558 / 1.805× |
| FP32 | 4096 | 32 | inverse / 1/N | 5.371 | 6.051 / 1.127× | 3.982 / 0.741× | 8.694 / 1.619× |
| FP32 | 65536 | 1 | forward | 6.142 | 6.234 / 1.015× | 8.563 / 1.394× | 7.722 / 1.257× |
| FP32 | 65536 | 1 | inverse / 1/N | 6.298 | 7.491 / 1.190× | 8.883 / 1.411× | 7.805 / 1.239× |
| FP64 | 65536 | 1 | forward | 14.531 | 41.378 / 2.848× | 48.898 / 3.365× | 15.472 / 1.065× |
| FP64 | 4096 | 1 | forward | 19.112 | 50.128 / 2.623× | 49.432 / 2.586× | 11.834 / 0.619× |
| FP64 | 4096 | 1 | inverse / 1/N | 19.493 | 51.595 / 2.647× | 52.149 / 2.675× | 11.866 / 0.609× |
| FP64 | 65536 | 1 | inverse / 1/N | 14.520 | 43.046 / 2.965× | 51.309 / 3.534× | 15.691 / 1.081× |
| FP64 | 262144 | 1 | forward | 37.640 | 51.677 / 1.373× | 80.938 / 2.150× | 39.530 / 1.050× |
| FP64 | 262144 | 1 | inverse / 1/N | 37.077 | 54.438 / 1.468× | 83.490 / 2.252× | 40.706 / 1.098× |
| FP32 | 128×128 | 4 | forward | 4.115 | 5.611 / 1.364× | 4.051 / 0.984× | unsupported |
| FP32 | 32×32×32 | 4 | inverse / 1/N | 5.061 | 8.226 / 1.625× | 4.880 / 0.964× | unsupported |
| FP32 | 8×8×8×8 | 4 | forward | 4.744 | 10.614 / 2.237× | 9.258 / 1.951× | unsupported |
| FP32 | 2×2×2×2×2×2×2×2 | 16 | inverse / 1/N | 8.094 | 20.435 / 2.525× | 16.611 / 2.052× | unsupported |
| FP32 | 65536 | 4 | forward | 7.920 | 6.510 / 0.822× | 9.123 / 1.152× | 11.621 / 1.467× |

The FP32 N=4096 FFT samples lose to VkFFT. FP64 N=4096 loses to GPU-FFT while beating cuFFT/VkFFT. Every negative region is retained; GPU-FFT multi-axis gaps are not flattened or counted as wins.

## HadaCore local validation and MMA9

The same-GPU paired profile contains 68 STRICT native endpoints and 204 independently validated MMA tile organizations (groups=1/2/4). All 68 public AUTO requests reproduce the profile-selected candidate: one native and 67 MMA. Public selection is checked against the minimum same-protocol measured candidate, with native retained on exact ties. Every raw calibration/public record is bundled; full output arrays remain on the measurement host except for the bundled small-request samples.

All nine original FP16 FWHT requests use explicit ALLOW_FP16_MMA and public DEFAULT selection from this local profile. ALLOW permits either native or MMA; these nine selected the integrated HadaCore tile provider. The external comparator is the complete original HadaCore transform plus its required output scale. Public-policy validation ignores only the intended STRICT→ALLOW policy change, including its duplicated contract field; physical mapping and all other semantics remain checked.

| N | Batch | cuButterfly µs | HadaCore µs / ratio | Chunks | Groups/CTA | cuButterfly relative L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 1.541 | 2.254 / 1.463× | 1 | 2 | 0.000623125 |
| 256 | 16 | 1.608 | 2.426 / 1.508× | 1 | 1 | 0.000688602 |
| 256 | 256 | 2.019 | 2.851 / 1.412× | 1 | 1 | 0.000706229 |
| 4096 | 1 | 2.253 | 2.949 / 1.309× | 2 | 4 | 0.000878097 |
| 4096 | 16 | 2.638 | 3.491 / 1.323× | 2 | 1 | 0.000864917 |
| 4096 | 256 | 4.030 | 6.330 / 1.570× | 2 | 1 | 0.000859536 |
| 32768 | 1 | 8.170 | 8.174 / 1.001× | 8 | 4 | 0.000993446 |
| 32768 | 16 | 8.958 | 10.632 / 1.187× | 8 | 1 | 0.000986004 |
| 32768 | 256 | 19.798 | 36.598 / 1.849× | 8 | 2 | 0.00098533 |

These external complete-call comparisons do not isolate an organization mechanism. The profile retains all groups=1/2/4 candidates and the chosen actual mapping. Error distributions and provider/compiler identities are preserved; an ALLOW result is not a STRICT result.

## Frozen evidence and reproduction

The [snapshot manifest](../snapshots/pro6000_extensions_20261117/manifest.json) binds original raw envelopes, CPU preparation records, exact cache contents, 17 calibration/replay receipts, 272 native/tile records, 68 public AUTO records, six lossless smallest-request implementation arrays, and completion audits. Gzip preserves the source bytes and both compressed/uncompressed SHA256 values. Other lossless outputs and compiled binaries remain local; their original paths/hashes and full error distributions are retained. [All 60 comparator records and actual mappings](../snapshots/pro6000_extensions_20261117/comparisons.csv) are available as CSV.

```bash
python3 experiments/tools/report_pro6000_extensions.py --check
python3 experiments/tools/report_pro6000_extensions.py --write
```

The CPU verifier checks envelope identities, explicit cache provenance, full semantic/mapping replay, every median/protocol, native/MMA candidate choice, guards and array hashes, and deterministically regenerates this report. Original runtime paths are provenance, not downloadable dependencies. A fresh target run needs the recorded target-compatible libraries/profiles and the published comparator workflows; the snapshots do not replace rebuilding or recalibration.

This closes the finite FFT17/MMA9 extension measurements in issue #9. It does not claim complete Full66 acceptance, validated predictions for unseen workloads, all numerical acceptance, an A100 winner transfer, or same-installed-build application timing. The main benchmark’s original GPU-NTT correctness failures remain separate negative results.
