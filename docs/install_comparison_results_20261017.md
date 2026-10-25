# Installation comparison results, 2026-10-17

The frozen 66-request comparison has finished measurement. cuButterfly has
37 floating requests with reported errors and 29 integer requests with exact
outputs. This is finite evidence for these requests on the recorded build and
GPU; it does not establish full operator coverage, floating acceptance, or a
globally optimal selection.

The GPU is NVIDIA A100 80GB PCIe,
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`; the measured library SHA256 is
`cc78cfad96c1cba5ea9b64afef24f9e4f062aedf4e41d18c0e9faa9b7c29a169`.
The [population and timing contract](install_compare_20261013.md) describes
the original 36 anchors and 30 rank1/2/3/4/8 additions. The checkpoint's status
is `complete-with-gaps-and-pending-accuracy`, not an all-pass verdict.

## Sources and reproducibility

Run from the repository root:

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/summarize_install_comparison.py
```

The script performs CPU-only analysis and leaves measurement and review files
unchanged. It reuses the checkpoint integrity loader, checks the review's
checkpoint SHA256 and row digest, checks manifest associations, and verifies
all retained same-precision output artifacts. It checks CPU/GPU input hashes
and complete semantics, and reconciles statuses with the unified table.

| Source | Role |
| --- | --- |
| [GPU checkpoint](../results/install_compare_repair_20261014/measurement/results.json) | Original timing, identity and 66 row indices |
| [Same-precision GPU review](../results/install_compare_20261013/cpu_external_v2/gpu_same_precision_review.json) | Retained GPU outputs compared with the CPU reference |
| [CPU results](../results/install_compare_20261013/cpu_external_v2/results.json) | 83 external CPU implementation records |
| [Unified comparison](../results/install_compare_20261013/cpu_external_v2/unified_comparison.csv) | Absolute latency and separate GPU/CPU comparison columns |
| [Analysis summary](../results/install_compare_repair_20261014/analysis/summary.json) | Input file hashes, complete identities, counts and grouped errors |
| [Classification table](../results/install_compare_repair_20261014/analysis/classifications.csv) | 413 records: 66 cuButterfly, 264 GPU baseline slots, 83 CPU records |
| [Gap coverage](../results/install_compare_repair_20261014/analysis/gpu_gaps_cpu_coverage.csv) | All 15 missing GPU slots with exact case IDs and CPU matches |
| [Floating errors](../results/install_compare_repair_20261014/analysis/floating_errors.csv) | All 131 measured floating records, per-row quantiles and artifact paths/hashes |
| [Grouped errors](../results/install_compare_repair_20261014/analysis/floating_error_groups.csv) | Backend/operator/precision summaries |

Validation checked 194 same-precision output artifacts: 131 floating and 63
integer. Actual and reference shapes and storage dtypes agree. All integer
artifacts have zero mismatches: cuButterfly 29, GPU-NTT 5, HEXL CPU 13 and FXT
CPU 16. All 66 cases have external CPU measurements with matching semantics
and source input hashes.

## Floating error evidence

FFT references use SciPy 1.15.3 PocketFFT in the requested complex64 or
complex128 precision. Real floating FWHT and structured references use the
independent stage implementation in the requested float32 or float64
precision, including requested inverse normalization. External GPU libraries
are compared with this reference; they are not numerical ground truth.
Historical wide-reference results remain in the measurement checkpoint and
are not used for this table.

All 37 cuButterfly floating outputs and their references are finite. Across
all 131 floating implementation records, actual and reference nonfinite
counts are both zero. `accuracy-review-required` means an error report without
an acceptance threshold; historical `atol`, `rtol` and mismatch fields do not
establish acceptance. No new threshold is assigned here.

The following statistics summarize per-request maximum absolute error, maximum
relative error over nonzero references, and relative L2 error. They are not
quantiles pooled over output elements.

| cuButterfly operator/precision | Requests | Median of max absolute errors | Largest absolute error | Largest relative error | Largest relative L2 error |
| --- | ---: | ---: | ---: | ---: | ---: |
| FFT complex64 | 11 | 1.706e-5 | 7.8733e-3 | 1.6413e-2 | 2.8580e-7 |
| FFT complex128 | 6 | 1.1722e-13 | 1.4552e-11 | 6.2242e-11 | 4.2778e-16 |
| FWHT float32 | 11 | 0 | 4.8828e-4 | 1.9676e-1 | 1.5026e-7 |
| FWHT float64 | 2 | 0 | 0 | 0 | 0 |
| Structured float32 | 2 | 0 | 0 | 0 | 0 |
| Structured float64 | 5 | 3.2196e-15 | 7.1054e-15 | 2.9646e-11 | 2.9736e-16 |

The largest cuButterfly absolute error occurs at `extension-fft-rank1`
(N=65536, B=4, complex64): absolute error median 1.9069e-5, p95 8.4781e-5,
p99 2.1624e-4, maximum 7.8733e-3. The largest relative error occurs at
`extension-fwht-rank3` (32x32x32, B=4, inverse float32): its worst relative
point has reference -9.429641e-9 and actual -7.574272e-9, an absolute error
of 1.8553692e-9. Its full-output absolute error median/p95/p99 are
2.4738e-10/1.2224e-9/2.3283e-9. These examples show why absolute, relative and
L2 distributions must be considered together; they do not waive any error.

External reference implementations have their own errors against the same
CPU oracle. The table retains them alongside cuButterfly:

| External implementation/precision | Requests | Largest absolute error | Largest relative error | Largest relative L2 error |
| --- | ---: | ---: | ---: | ---: |
| cuFFT complex64 | 9 | 4.3673e-3 | 1.8482e-2 | 2.6629e-7 |
| VkFFT complex64 | 9 | 1.2260e-2 | 3.0265e-2 | 4.2018e-7 |
| FFTW t1 complex64 | 11 | 3.9062e-3 | 1.5132e-2 | 1.9101e-7 |
| FFTW t8 complex64 | 11 | 2.9926e-3 | 1.0546e-2 | 1.8268e-7 |
| cuFFT complex128 | 6 | 2.0580e-11 | 1.1221e-10 | 4.8101e-16 |
| VkFFT complex128 | 6 | 1.6270e-11 | 6.4037e-11 | 4.0266e-16 |
| FFTW t1 complex128 | 6 | 1.6270e-11 | 9.8413e-11 | 3.9340e-16 |
| FFTW t8 complex128 | 6 | 1.6270e-11 | 1.6246e-10 | 3.9648e-16 |
| Dao FWHT float32 | 10 | 4.8828e-4 | 1.9676e-1 | 1.5026e-7 |
| FXT FWHT float32 | 11 | 1.9531e-3 | 1.1026 | 1.5400e-7 |
| FXT FWHT float64 | 2 | 2.2737e-13 | 2.8658e-12 | 2.5500e-16 |
| Author structured float32 | 2 | 0 | 0 | 0 |
| Author structured float64 | 5 | 0 | 0 | 0 |

Backend populations differ where GPU contracts are missing, so extrema across
these groups are descriptive, not paired accuracy rankings. The detailed CSV
preserves each case's absolute, nonzero-reference relative and zero-reference
absolute error quantiles at 0/25/50/75/90/95/99/99.9/100 percent, zero counts
and nonfinite counts. Complete element errors and reference index partitions
remain in the linked lossless NPZ files; histogram counts also remain in the
source review/results JSON. Exact agreement on an input is not a bound for
other inputs.

## GPU coverage and CPU supplementation

| Original GPU record class | cuButterfly | External baseline slots |
| --- | ---: | ---: |
| Floating errors reported | 37 | 40 |
| Exact integer pass | 29 | 5 |
| Missing baseline | 0 | 15 |
| Not applicable | 0 | 204 |

The 15 `missing-baseline` slots cover 13 distinct cases. They match the target
operator domain but lack a compatible audited implementation or bound adapter.
The four GPU baselines each have a slot for every case. The 204
`not-applicable` slots are operator-domain mismatches: cuFFT 49, VkFFT 49,
Dao 53 and GPU-NTT 53. They are not 204 failed or unfinished experiments.
For example, a zeta case has four such slots; its lack of a matching GPU zeta
provider remains a domain-coverage limitation outside these named providers.
The review's `validation-unavailable` rows for absent output artifacts must be
interpreted through these original statuses, not counted as new failures.

| Missing slots | Cases/populations | Specific cause | External CPU coverage |
| ---: | --- | --- | --- |
| 2 Dao | e1-18/19-fwht-fp64, N=4096 B=1 | Audited Dao supports FP32/FP16/BF16 storage, not FP64 | FXT float64, forward/inverse |
| 4 GPU-NTT | e1-20/21/22/23-ntt-word32, N=4096/65536 B=1 | Requested modulus 2013265921 exceeds bridge word32 requirement q < 2^30 | HEXL, exact uint32 contract with explicit uint64 core/storage adaptation |
| 4 cuFFT/VkFFT | extension-fft-rank4/rank8, B=4/16 | Current bound exact C2C adapter contract limits rank to <=3 | FFTW t1 and t8, matching full rank and batch |
| 4 GPU-NTT | extension-ntt-rank2/rank3/rank4/rank8, B=4/4/4/16 | Bridge supports rank1 only; flattening would change xD NTT semantics | HEXL, exact xD axis composition |
| 1 Dao | extension-fwht-rank1, N=65536 B=4 | Flattened length exceeds audited limit 32768 | FXT float32, same N and batch |

All 15 slots have measured external CPU counterparts with identical requested
semantics and input hashes. This fills the absence of an external *CPU*
comparison for those cases. It does not fill any of the 15 GPU slots, prove
that upstream libraries fundamentally lack an implementation, or turn them
into wins. The rank4/8 FFT limitation specifically belongs to the current
adapter contract.

Across the full matrix, CPU records comprise FFTW t1/t8 17 each, FXT 29,
HEXL 13, and the author structured implementation 7. Of these 83 records,
54 report floating errors and 29 pass exact integer validation. CPU times
are **shared-host provisional**, recorded on an AMD EPYC 7542 with affinity
CPUs 16-23 and without exclusive CPU ownership. They include complete calls
and required semantic adaptations under the CPU protocol, exclude planning
and reference generation, and remain separate from GPU/GPU speedup means.
The [CPU supplement](cpu_external_baselines.md) records adapter details.

## Remaining minimum work

For this frozen 66-case numerical report, no additional GPU execution is
needed: every measured implementation has a same-precision retained-output
review, and all integer outputs are exact. Floating acceptance remains
unassigned because the user requested error reports without an automatic
threshold.

The remaining experiment needed before firm CPU timing claims is an
uncontended repeat of the 83 CPU records on the same 66 semantics and inputs,
retaining separate one/eight-thread FFTW columns and complete adapter costs.
This does not require rerunning unchanged GPU measurements.

Completing missing GPU comparisons first requires implementation/support
work, followed by only the affected measurements: four rank4/8 FFT slots,
four word32-modulus NTT slots, four xD NTT slots, and three Dao FP64/large-N
slots if a matching implementation becomes available. Unsupported requests
must retain their original precision, modulus, shape and batch. These are
conditional gaps, not queued reruns that the current adapters can complete.
GPU zeta/structured providers are a separate coverage question; existing CPU
rows provide an external comparison but cannot establish GPU competitiveness.
The broader mechanism and application experiments remain governed by their
own research contracts.
