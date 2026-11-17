# PRO6000 fixed-mapping samples at A100 nominal-win workloads

**Status: measured target timings and derived comparisons; preliminary subset, not full acceptance.**

The four highlighted requests were chosen from the published A100 80GB main66 snapshot: cuButterfly was nominally faster than at least one external comparator, the complete semantics match, and every published mapping-summary field matches the PRO6000 execution. Selection uses A100 evidence, not PRO6000 speedups. All 25 measured fixed-seed requests remain in the appendix; the other 41 Full66 requests have no exact seed match and remain explicit gaps.

The A100 frozen operator cohort is complete at 66/66 requests across all six operator categories, with the original comparator gaps and floating error-report status retained. This target publication also covers all six categories, at a smaller sampling density. Completing operator execution does not imply an external comparator exists for every request.

| Operator | A100 frozen requests | PRO fixed-seed timings | Highlighted source samples | A100 measured external pairs |
|---|---:|---:|---:|---|
| FFT | 17 | 8 | 2 | cufft: 17, vkfft: 17 |
| NTT | 13 | 4 | 1 | gpu-ntt: 5 |
| FWHT | 13 | 5 | 1 | daofht: 10 |
| subset-zeta | 8 | 3 | 0 | no selected external comparator; CPU oracle |
| superset-zeta | 8 | 3 | 0 | no selected external comparator; CPU oracle |
| structured-2×2 | 7 | 2 | 0 | no selected external comparator; CPU oracle |

“Nominal win” means a source baseline/cuButterfly ratio greater than 1, without a statistical significance claim. The A100 snapshot publishes mapping summaries, not the complete original mappings. Matching those fields does **not** prove bit-for-bit transfer of a full A100 search winner. The repository seed catalog is a candidate portfolio, not a certified winner list. No A100 timings, binary, calibration or registry were imported into target selection.

A100 source: NVIDIA A100 80GB PCIe, UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`, 84974239744 bytes, library SHA256 `fac10905e8c7d88a1c42f669545df6182e988409a297ee99f55d00e6d5e57778`. Source snapshot: [current_refresh](../snapshots/current_refresh/snapshot.json).

Target: NVIDIA RTX PRO 6000 Blackwell Server Edition, UUID `GPU-1067ab39-e411-e141-1775-64426bca5701`, 101973950464 bytes, SM120; driver 580.159.04, CUDA 12.8. Library SHA256 `3b0668f2530a75d1401dcbf5decea720f8e47db0c1357aafdf1680fa1bba6adb`. Measured source commit `a305d97769a0c005591a8bbe4e14ae1a238b7e83` with the archived build compatibility patch. The PR base is a newer documentation/workflow commit; it does not relabel the measured build.

Each target request uses an explicit fixed mapping (policy 2), CPU-prepared full inputs and references, complete public execution, unchanged precision/shape/batch/direction/normalization, mapping replay, padding/input checks, and exclusive GPU guards before and after measurement. The controller also monitored foreign contexts during the attempt. No automatic search was performed. Each implementation has 10 warmup calls and 7 CUDA graph replays of 20 complete calls; the table uses sample medians. These are one-process steady-state samples, not seven independent processes or alternating paired trials. Planning, compilation, allocation and CPU checking are excluded; required order conversion and normalization are included.

Floating-point results retain full error distributions and require numerical review; a timing ratio is not numerical acceptance. Integer cuButterfly and replay results are exact. Failed baselines are excluded from ratios. Higher baseline/cuButterfly is faster for cuButterfly. A100 and PRO6000 ratios describe separate builds and devices; no pooled cross-platform geometric mean is reported.

| Request | Comparator | A100 comparator/cuButterfly | PRO cuButterfly µs | PRO comparator µs | PRO comparator/cuButterfly |
|---|---|---:|---:|---:|---:|
| fft fp32 N=4096 B=32 forward | cufft | 1.016× | 4.459 | 4.690 | 1.052× |
| fft fp32 N=4096 B=32 forward | vkfft | 0.969× | 4.459 | 3.941 | 0.884× |
| fft fp32 N=4096 B=32 forward | gpu-fft | not in source | 4.459 | 8.515 | 1.910× |
| fft fp32 N=4096 B=32 inverse | cufft | 1.313× | 4.536 | 6.046 | 1.333× |
| fft fp32 N=4096 B=32 inverse | vkfft | 0.969× | 4.536 | 3.962 | 0.873× |
| fft fp32 N=4096 B=32 inverse | gpu-fft | not in source | 4.536 | 8.773 | 1.934× |
| ntt word64 N=4096 B=1 forward | gpu-ntt | 1.257× | 5.042 | integer failure | excluded |
| fwht fp32 N=256 B=1 forward | daofht | 1.478× | 1.541 | 2.158 | 1.401× |

The four source nominal-win workloads include both PRO6000 wins and losses. In particular, the FP32 FFT samples are slower than VkFFT, and the NTT comparator fails. GPU-FFT was not present in the cited A100 cohort; its new target timings are additional external comparisons, not evidence about an A100 win.

Selected actual design fields (complete mappings, nested execution groups, batch/element strides, orders and moduli remain in the JSON/CSV):

| Case prefix | Backend | Stage partition | Core | Threads | Local exchange/layout |
|---|---|---|---|---:|---|
| `f9d6a78a86c6` | temporal-tile | `[12]` | cufftdx-direct | 256 | shared |
| `f87de0e4a2c8` | temporal-tile | `[12]` | cufftdx-direct | 256 | shared |
| `65a20b1e396e` | hybrid2d | `[]` | dataflow-radix4 | 128 | hermes-xor |
| `0a5b77638a78` | temporal-tile | `[8]` | scalar | 32 | warp-register |

Stage partition is a serialized mapping axis; empty NTT partitions are not interpreted as zero execution stages. Other inactive/default mapping fields are not inferred as measured hardware occupancy or pipeline behavior.

**Negative results**

- `65a20b1e396e8dedf43998cf17c5abc6e335e828604cf260440d8bda6fe7ca8f` gpu-ntt: integer mismatch at 1: 11845814219123592113 != 152474654567292722. cuButterfly and explicit replay are integer-exact; this failed comparator has no timing ratio.
- `eab91b52d5d9956d6834400a1cbb896cb39a4fe9322928dc8eab4f00eb2623fa` gpu-ntt: integer mismatch at 1: 9711579215757401972 != 366828837344226011. cuButterfly and explicit replay are integer-exact; this failed comparator has no timing ratio.

**All measured fixed-seed requests**

| Request | cuButterfly µs | Raw case status | Highlighted source sample |
|---|---:|---|---|
| fft fp32 N=256 B=1 forward | 1.870 | accuracy-review-required | no |
| fft fp32 N=4096 B=32 forward | 4.459 | accuracy-review-required | yes |
| fft fp32 N=4096 B=32 inverse | 4.536 | accuracy-review-required | yes |
| fft fp32 N=65536 B=1 forward | 5.347 | accuracy-review-required | no |
| fft fp32 N=65536 B=4 forward | 7.253 | accuracy-review-required | no |
| fft fp64 N=4096 B=1 forward | 21.226 | accuracy-review-required | no |
| fft fp64 N=65536 B=1 forward | 14.794 | accuracy-review-required | no |
| fft fp64 N=262144 B=1 forward | 47.805 | accuracy-review-required | no |
| ntt word32 N=4096 B=1 forward | 4.570 | pass | no |
| ntt word32 N=65536 B=1 forward | 7.478 | pass | no |
| ntt word64 N=4096 B=1 forward | 5.042 | correctness-failure | yes |
| ntt word64 N=65536 B=1 forward | 8.526 | correctness-failure | no |
| fwht fp32 N=256 B=1 forward | 1.541 | accuracy-review-required | yes |
| fwht fp32 N=4096 B=32 forward | 3.512 | accuracy-review-required | no |
| fwht fp32 N=4096 B=32 inverse | 3.390 | accuracy-review-required | no |
| fwht fp32 N=32768 B=1 forward | 5.019 | accuracy-review-required | no |
| fwht fp64 N=4096 B=1 forward | 6.115 | accuracy-review-required | no |
| subset-zeta uint32 N=256 B=1 forward | 2.083 | pass | no |
| subset-zeta uint32 N=4096 B=1 forward | 2.947 | pass | no |
| subset-zeta uint32 N=4096 B=256 forward | 9.186 | pass | no |
| superset-zeta uint32 N=256 B=1 forward | 2.075 | pass | no |
| superset-zeta uint32 N=4096 B=1 forward | 2.981 | pass | no |
| superset-zeta uint32 N=4096 B=256 forward | 9.218 | pass | no |
| structured-2x2 fp32 N=256 B=1 forward | 1.661 | accuracy-review-required | no |
| structured-2x2 fp32 N=4096 B=1 forward | 3.432 | accuracy-review-required | no |

The two raw case statuses `correctness-failure` originate in GPU-NTT after cuButterfly and mapping replay passed; they are preserved as failures. The frozen data also retain unsupported/missing baselines and plugins not reached after a preceding failure. Zeta and structured transforms have no selected external comparator. HadaCore supports a different FP16 cohort, so its absence here is not a completed HadaCore comparison.

Frozen artifacts: [provenance](../snapshots/pro6000_fixed_mapping_20261117/provenance.json), [raw rows](../snapshots/pro6000_fixed_mapping_20261117/measurements.json), [case/seed/gap table](../snapshots/pro6000_fixed_mapping_20261117/cases.json), [all timings and mappings](../snapshots/pro6000_fixed_mapping_20261117/performance.csv), [all comparator statuses](../snapshots/pro6000_fixed_mapping_20261117/baselines.csv). Lossless arrays for the four highlighted cuButterfly results and both failed GPU-NTT results are bundled and hashed. Other floating comparator error arrays and build binaries remain on the measurement host; their original paths, hashes and full error distributions are preserved in the raw rows.

Verify frozen evidence and regenerate tables without CUDA:

```bash
python3 experiments/tools/report_pro6000_fixed_samples.py --check
# Optional: requires NumPy, verifies full outputs and regenerates CPU inputs/oracles
python3 experiments/tools/report_pro6000_fixed_samples.py --check --arrays
python3 experiments/tools/report_pro6000_fixed_samples.py --write
```

The snapshot includes the original bounded measurement-controller source and build command/patch for reproduction on the recorded host. Its original absolute paths describe the measurement environment; they are not public-download dependencies. To remeasure elsewhere, rebuild for the target, prepare the frozen cases with the repository semantic CPU pipeline, use each recorded fixed seed with explicit policy 2, preserve the resolved mapping on replay, and run the same complete-call timing and baseline adapters on an exclusive non-default CUDA stream. Calibration and comprehensive public AUTO comparisons remain separate experiments.

This subset is deliberately biased toward source-platform nominal wins; it cannot establish overall competitiveness, global optimality, the complete A100 winner portfolio, all 66 target cases, or completion of issue #9. FFT17, HadaCore/public AUTO, MMA9 and the main acceptance gates retain their own status.
