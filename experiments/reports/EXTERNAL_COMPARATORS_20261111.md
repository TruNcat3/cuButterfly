# External comparator extension checkpoint, 11 November 2026

Status: adapter integration and bounded GPU diagnostics; this is not a replacement for the calibrated main-study comparison. Authorship and software citation follow the [project policy](../../docs/about.md).

## Scope and provenance

- GPU: A100 80GB PCIe, UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`. The historical main66 study used a different device UUID. Do not pool their samples.
- Fresh paired FFT preparation covers the same 17 original requests, including four multidimensional requests outside the rank-one GPU-FFT adapter. The separate FP16 FWHT population has nine forward requests and two inverse/in-place controls.
- Out-of-place GPU calls use 10 warmups and 20 calls per CUDA Graph, with the median of seven replays. The in-place control uses reset-outside-timing single-call CUDA events for both backends, not Graph, and is reported separately. Compilation, plan construction, CPU ranking and verification are outside the timer. Required conversion/copy/scale kernels are inside.
- Floating results retain full error artifacts against same-dtype CPU references, without an acceptance threshold. HadaCore uses internally normalized FP16 Tensor Core arithmetic and a timed output rescale; its grouping and coefficient rounding differ from radix-2 CPU/native FWHT.
- FWHT integration probes use the November 5 FP16 acceptance library (SHA256 `775a85c8f76e44bbfee886fd90489e7c309bd097b60e968183e446aa006177b2`). The matching resource profile does not establish a graph-calibrated FWHT cost model. Its actual DEFAULT selections use hardware-rate priors. These probes do not establish the latest fully tuned library frontier.
- The initial FFT probes used the current local library with an incompatible model fingerprint. They were stopped, labeled exploratory, and excluded from formal aggregate speedups. No old manually selected mapping was injected.

## External implementations

| Comparator | Local scope | Current evidence |
| --- | --- | --- |
| GPU-FFT | FP32/FP64, rank-one complex FFT, both directions | CPU adapter and 17-request preparation complete; 4 FP32 paired diagnostic rows retained. Formal calibrated pairing and FP64 GPU coverage remain pending. |
| HadaCore | FP16 FWHT, supplied-output integration with timed normalization adaptation | 11/11 paired integration probes recorded. Floating acceptance remains error-report only. |
| tcFFT | Restricted forward FP16 FFT; only two of the nine original FP16 requests fit its native lengths | CPU compilation and two references ready. For N=262144, B=4, an independent A100-kernel probe passes eager/recreated-plan/Graph bitwise repeatability, with relative L2 error 0.005798. The combined three-backend runner still reports a direct/Graph mismatch. Its cause is not yet isolated; no tcFFT latency or speedup is accepted. Seven unsupported requests remain explicit. |

Pinned source commits and licenses are in the [adapter guide](../../docs/benchmarks/external-extensions.md) and [catalog](../../config/external_comparison_catalog.json).

## FP16 FWHT integration probes

These are per-request diagnostics for the named build and DEFAULT choices, not a calibrated aggregate. Speedup is HadaCore complete-call time divided by cuButterfly complete-call time. Values below one are losses. FP16 results must not be merged with the older FP32 FWHT cohort.

| N | B | Direction / placement | cuButterfly (µs) | HadaCore (µs) | Ratio | HadaCore relative L2 | Status |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 256 | 1 | forward / out-of-place, Graph | 3.1232 | 3.3792 | 1.082 | 0.000703 | error report only |
| 256 | 16 | forward / out-of-place, Graph | 3.3792 | 3.9424 | 1.167 | 0.000737 | error report only |
| 256 | 256 | forward / out-of-place, Graph | 3.7376 | 4.4032 | 1.178 | 0.000727 | error report only |
| 4096 | 1 | forward / out-of-place, Graph | 14.5408 | 4.5056 | 0.310 | 0.000921 | error report only |
| 4096 | 16 | forward / out-of-place, Graph | 15.3088 | 4.9152 | 0.321 | 0.000894 | error report only |
| 4096 | 256 | forward / out-of-place, Graph | 20.2240 | 10.3424 | 0.511 | 0.000888 | error report only |
| 32768 | 1 | forward / out-of-place, Graph | 111.6672 | 14.1312 | 0.127 | 0.00104 | error report only |
| 32768 | 16 | forward / out-of-place, Graph | 117.4016 | 16.7936 | 0.143 | 0.00101 | error report only |
| 32768 | 256 | forward / out-of-place, Graph | 393.0624 | 67.7888 | 0.172 | 0.00101 | error report only |
| 4096 | 16 | inverse, normalized / out-of-place, Graph | 16.3328 | 4.8640 | 0.298 | 0.000894 | error report only |
| 4096 | 256 | forward / in-place, events | 64.5120 | 98.3040 | 1.524 | 0.000888 | error report only |

The completed forward probes select one temporal stage, radix-4 scalar arithmetic and shared-memory exchange. For example, N=32768/B=1 selects stage partition `[15]`, one 32768-point tile per CTA, 128 threads and prefetch depth zero; HadaCore uses FP16 Tensor Core arithmetic. This comparison changes both core and organization, so it is external competition evidence, not a controlled dataflow ablation. No bottleneck or theoretical failure is inferred from the speedup alone.

## Remaining work

1. The eleven FWHT integration probes are complete. Carry the negative results into calibrated follow-up; distinguish core choice from organization and selection quality.
2. Bind a compatible main-study runtime, calibration protocol and public-selection cache before continuing the formal FFT pairs. A numerically valid adapter does not validate the library selection model.
3. Isolate the tcFFT combined-runner mismatch before timing its second supported request. Independent eager and Graph outputs agree, so the remaining failure cannot be attributed to the upstream core without further evidence. The original Volta/Ampere dispatch error was corrected; numerical acceptance remains error-report only.
4. Add validated paired records to a separately identified figure cohort; the existing paper/README headline plot is unchanged by these diagnostics.

## Reproduction and raw evidence

The [workflow](../../docs/benchmarks/external-extensions.md) describes build, preparation, execution, resume and table generation. The population is [versioned JSON](../workloads/external_extensions.json). Local raw roots are `results/comparator_extension_20261111/paired28`, `hadacore11`, and `tcfft_ampere_prepared`. Each retains prepared-array hashes, runtime identities, per-request mappings, error arrays and timing samples. Those raw host artifacts are not bundled by this checkpoint.

Use `python scripts/summarize_comparator_extension.py CAMPAIGN --output SUMMARY` to regenerate per-request CSV/Markdown. The summarizer validates identity and guards, preserves pending and unsupported requests, and withholds aggregate speedups for model-fallback or excluded cohorts.

The [FWHT comparison and bottleneck checkpoint](FWHT_BOTTLENECK_20261111.md) separates historical FP32 contracts, the observed FP16 losses, static resource evidence and pending profiling checks.
