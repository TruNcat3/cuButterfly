# Application coverage and public selection, 2026-10-13

This milestone addresses application losses that remain after isolated operator
improvements. It does not replace the ongoing full calibration or claim that
all applications are faster. The independent build is `build-e2e-20261013`;
`build-e13-cufftdx` was left unchanged for the calibration that was active when
this work began. Once that process exited, independent checks also used GPU1
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`, another A100 80GB PCIe.

## Public implementation

- The xD cuFFTDx inventory now includes power-of-two 2048/4096-point axes in
  FP32 and FP64, both directions, EPT 8/16. Device opt-in shared-memory limits
  replace the former fixed shared-memory assumption. Illegal configurations
  are filtered before measurement. EPT4 remains a compiled short-axis option.
- The public xD selector also compares `axis-packed-cufftdx` and its EPT16
  variant. These transpose a noncontiguous axis, execute the continuous core,
  retain its physical layout for the next axis, and restore the public layout
  at the boundary. All permutations and workspaces belong to complete-plan
  timing. Direct strided cores and cuFFT axis candidates remain available.
- FWHT gains explicit `output_rounding=FINAL_STORAGE` and `output_scale`.
  The implemented domain is rank-one standard FP16 storage and full FP32
  arithmetic, with scaling before a single final FP16 narrowing. It remains
  explicit even when scale is one. The short provider and the FP32 composed
  fallback compete through public plans. The latter records its two storage
  conversion kernels. Unimplemented combinations report a lowering gap.
  Cache/replay identity includes both fields; old registry records cannot
  silently satisfy this contract. QuaRot requests `public_fp32_final` through
  the public descriptor, without a standalone provider or forced winner.
- Optional `cubutterfly.torch_graph.InferenceGraph` captures an inference
  callable using its existing plans. Each invocation copies a new input and
  returns an independently owned output. Input copy, graph replay, output clone
  and application synchronization are included in the FNO comparison for both
  libraries. Setup, search and capture are excluded and separately recorded.

## Bounded validation and performance experiments

The primary GPU is A100 80GB PCIe
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`. Source/library identities and raw
errors accompany each report. Floating error distributions remain the user's
chosen reporting contract; no new application tolerance is inferred.
Following the user's clarification, the primary numerical reference is CPU
execution at the requested storage/compute precision and normalization. GPU
vendor outputs are comparators. Earlier FP64 diagnostics remain labeled as
such and are not an FP32 acceptance criterion. The CPU FFT oracle uses PyTorch
to preserve FP32/FP64, avoiding NumPy's implicit FFT precision promotion.

1. CPU descriptor, mapping, compatibility and application-adapter regression.
2. Public FWHT final-storage correctness, measured selection and cache replay.
3. Long xD axes, both precisions/directions, and packed-layout input preservation.
4. Public complete 2048x2048 real FFT at B=1/16, including cache replay.
5. Complete fixed-filter calls, QuaRot B=1/4 and changing-input FNO 64/128.
   Paired ordinary execution remains separate from graph execution. Different
   process trials rotate backend order; preliminary medians are not significance
   claims. Explicit candidate validation rows are not public automatic results.

The coverage driver is `scripts/validate_e2e_xd_coverage.py`; it accepts
`--matrix coverage` for explicit candidate validation or `--matrix filter` for
public measurement/replay. Application runners are
`experiments/workloads/e13/run_fft_filter_end_to_end.py`,
`experiments/workloads/e13/run_quarot_random_end_to_end.py` and
`experiments/tools/run_fno_graph_integration_20261013.py`.

Validation completed on the new library includes 20 public FWHT GPU tests
against CPU arithmetic, 64 complex xD candidate rows, eight compact-real
packed-core rows and 36 physical packed-layout cases. The latter check input
preservation, in-place operation, batch tails, mixed axis resources and
workspace rebinding. The primary CPU-reference reports are
`xd_public_cpu32/CPU_reference_coverage.json` and `packed_real_cpu_r3/results.json`
under `results/e2e_unified_20261013`. Packed-real relative L2 error ranges are
1.906e-7--2.856e-7 for FP32 and 3.631e-16--4.160e-16 for FP64. Both GPU
libraries' errors are recorded against the same-precision CPU result.

The older component-wise comparison against GPU cuFFT still records two
FP32 comparison failures; it is not the newly specified numerical oracle.
No tolerance was enlarged. The distribution mode completes the structural
checks without turning those numerical comparisons into passes.

The four complete 2048x2048 public real-FFT cases also completed measured
selection and identical cache replay in `xd_filter_cpu32_rerun/release`.
An earlier run (`xd_filter_cpu32`) omitted the requested 8 GiB tuning budget:
the library's default 512 MiB limit skipped B=16 measurement, and the driver
incorrectly expected a measured cache entry from a static selection. The
driver now records `--tune-max-bytes`, checks the selection source first and
retains the complete failed row. The successful rerun used 8 GiB explicitly;
this is a test-configuration correction, not a library cache repair.

## Isolated filter scale-compensation result

Before the new build completed, one application-only optimization was measured
on the unchanged library SHA `08db08db7af52cd7b6373d3d0a30b2841da33a16b2e7d126ea784dab28c10f82`.
Fixed-filter frequency-domain normalization is now compensated once in its
cached spectrum, eliminating one batch-sized scaling pass per invocation.
The power-of-two padding makes that scalar an exact binary exponent factor.

For 1024x1024 FP32 images, 31x31 shared filter, B=16 and 2048x2048 cuButterfly
padding, the new call measures 3458.15 us, versus 3373.47 us for Torch with the
same padding and 1441.48 us for CuPy using its native smaller padding. The
preceding same-library cuButterfly measurement was 3802.83 us. This is an
application-only cross-run improvement of about 9.1%, not a new-library speedup
or a repeated-process significance result. Relative L2 error is 4.98e-7 for
both cuButterfly and same-padding Torch, and 3.69e-7 for CuPy. No accepted
floating threshold has been added.

Raw source:
`results/e2e_unified_20261013/filter_scale_control/run/release/image1024_filter31_batch16.scale-folded.json`.
The same-padding gap is about 2.5%; native-padding CuPy is still about 2.40x
faster. Long-axis kernel coverage cannot by itself remove padding inflation.
Inspection then identified an application-level omission: the existing public
compact-real lowering already accepts 1080x1080. The filter runner had retained
its power-of-two preparation choice. The new explicit `--padding-policy fast`
uses real `next_fast_len` for cuButterfly and its same-padding Torch control;
the report records the actual transformed extent independently of the original
input manifest. The power-of-two cohort remains intact. This corrects the
earlier claim that this application necessarily required new arbitrary-length
FFT kernels.

The smaller-padding control is now measured on the same unchanged library.
All three backends transform 1080x1080, use the same original image/filter, and
cache the fixed filter spectrum. Padding, spectral product and crop remain in
the measured application call. Values below are complete-call microseconds.

| Batch / independent process | cuButterfly | CuPy | Torch | CuPy / cuButterfly |
| --- | ---: | ---: | ---: | ---: |
| 1 / first | 137.22 | 314.88 | 183.91 | 2.295x |
| 16 / first | 1327.21 | 1440.77 | 1463.19 | 1.086x |
| 16 / rotated order, DEFAULT cache replay | 1324.13 | 1436.26 | 1463.30 | 1.085x |

The two B=16 processes agree directionally; this is preliminary repeated-process
evidence, not a significance claim. Relative L2 error is about 3.34e-7 for
cuButterfly and 3.69e-7 for CuPy. No floating acceptance threshold was added.
The new kernel build is not credited for these application-only improvements.
Raw reports are under `results/e2e_unified_20261013/filter_fast_control`,
`filter_fast_control_p2`, and `filter_fast_b1_control`, each in `release/`.

Complete CKKS evidence remains separate unfinished work.

## New build application evidence

The new shared library SHA256 is
`ddb7d70cc8e0fac900c0b912f03d7aae18855a807d06f0dff4a0ebad1bcb665b`.
On GPU1, the public tuned-cache filter run at B=16 reports 1323.93 us for
cuButterfly, 1431.65 us for CuPy and 1458.69 us for Torch. All three transform
1080x1080. Against the same FP32 CPU convolution, their relative L2 errors are
4.257e-7, 4.783e-7 and 4.778e-7 respectively. This is distribution-only
evidence. The historical archive still contains its FP64 diagnostic array;
the new report explicitly records that it was not the validation reference.
Raw report: `results/e2e_unified_20261013/filter_new_b16_cpu32_t1/release/`.

Initial matched FNO trials on GPU1 use three distinct Darcy input batches,
fixed random weights, and two independent processes per shape. The complete
call includes input copy, output clone and synchronization in both modes.
Median-of-process-medians in milliseconds:

| FNO shape / batch | Torch ordinary | cuButterfly ordinary | Torch graph | cuButterfly graph |
| --- | ---: | ---: | ---: | ---: |
| 64x64 / 4 | 2.74131 | 2.90453 | 0.624510 | 0.606179 |
| 128x128 / 4 | 2.78901 | 2.96737 | 1.039555 | 0.982995 |

The graph path is about 1.030x/1.058x faster, but ordinary cuButterfly calls
still take about 6% longer. These are distinct results. Graph outputs equal
the corresponding backend's ordinary outputs, weights remain unchanged and
changing inputs produce changing outputs. GPU-to-GPU output differences are
application integration diagnostics, not same-precision CPU accuracy evidence.
`results/e2e_unified_20261013/fno_summary.json` links the four complete reports.
The table above is the pre-removal control. Separate measurements after
removing duplicate adapter views used the same library, weights, inputs and
actual plan mappings.
The revised public adapter performs input contiguity handling once and avoids
redundant input views; all shape, dtype, alias and stream checks remain.

| Final FNO shape / batch | Torch ordinary | cuButterfly ordinary | Torch graph | cuButterfly graph |
| --- | ---: | ---: | ---: | ---: |
| 64x64 / 4 | 2.72858 | 2.85591 | 0.623415 | 0.605298 |
| 128x128 / 4 | 2.75338 | 2.87602 | 1.038970 | 0.983483 |

The ordinary-call speed ratios are 0.9554x/0.9574x, so a roughly 4.5% latency
gap remains. Graph ratios are 1.0299x/1.0564x. Both processes agree on the
direction, not universal applicability or statistical significance. Output
hashes match the preceding control. `fno_hotpath_summary.json` links the raw
measurements and compares their identities. Thirty-one related CPU tests
passed after the wrapper change.

The final B=1 filter runs on GPU1 report cuButterfly 139.67/167.94 us,
CuPy 722.33/261.53 us and Torch 186.88/181.45 us in two differently ordered
processes. cuButterfly remains faster in both runs, but the CuPy variation is
too large to use a single B=1 speedup as a stable headline. Keep these raw
runs separate from the steadier B=16 result; both use CPU FP32 validation and
the second requires public cache replay. The directories are
`filter_new_b1_cpu32_t0` and `filter_new_b1_cpu32_t1`.

QuaRot's first B=1/B=4 application processes both selected the fused short
FWHT provider using public MEASURE with an explicit eight-candidate budget.
The full legal inventory remains larger; this bounded search is not a global
optimality claim. Prefill and all 32 decode outputs match the unchanged GPU
application exactly. Its arithmetic is independently checked against the CPU
in the FWHT unit tests. Application measurements and later cache-replay trials
remain distinct from the numerical oracle and exclude preparation/search.

Both cache-replay processes have now completed on GPU2. Each replays all four
public FWHT plans and preserves all 33 application outputs exactly. Full
prefill-512 plus decode-32 application latencies in milliseconds are:

| Batch / process | Original QuaRot | Public cuButterfly | Original / cuButterfly |
| --- | ---: | ---: | ---: |
| 1 / measured search, baseline first | 2493.716 | 2455.360 | 1.01562x |
| 1 / DEFAULT cache, treatment first | 2497.129 | 2461.664 | 1.01441x |
| 4 / measured search, treatment first | 2597.747 | 2563.544 | 1.01334x |
| 4 / DEFAULT cache, baseline first | 2638.318 | 2617.890 | 1.00780x |

These are repeated small gains, not a statistical-significance claim. B=1
prefill-only native latency varied from 75.55 to 91.41 ms, so its larger second
prefill speedup is not presented as a stable improvement. Complete measured
search took about 105/102 seconds for B=1/B=4, separately from execution;
cached construction reduced the B=1 setup to 33 ms. The runtime winner is
`short-fwht-fp32-final-storage` at all four application requests, not an
application-specific standalone library. The raw reports are
`quarot_b{1,4}_budget8_t{0,1}/release/quarot.json` under this milestone's results
directory. The aborted exhaustive-search attempt is retained at
`quarot_b1_t0` and is not included in any timing summary.

The remaining performance gap demonstrated here is FNO ordinary submission.
Its graph-mode gains do not erase that negative result. This milestone does
not complete whole-project CKKS evidence, migration qualification or an
all-operator performance matrix.

## Current application summary and attribution

Speedup is comparator complete-call latency divided by cuButterfly latency,
excluding compilation, search and graph capture. Filter and FNO use GPU1;
QuaRot uses GPU2. Both are A100 80GB, but their results are not pooled. FNO
values are medians of two process medians; QuaRot ranges retain both process
ratios. The filter rows use the B=16 same-precision CPU-reference cache-replay
run. These preliminary measurements do not establish statistical significance.
Earlier tables retain their absolute-time and historical-control roles.

| 场景 | 类型 | 对照 | 加速比 |
| --- | --- | --- | ---: |
| 图像滤波 1024²，31² 固定滤波器，B=16，1080² padding | 2D 实数 FFT，FP32 | CuPy，同 padding，缓存滤波器频谱 | 1.081x |
| 图像滤波 1024²，31² 固定滤波器，B=16，1080² padding | 2D 实数 FFT，FP32 | PyTorch/cuFFT，同 padding，缓存滤波器频谱 | 1.102x |
| FNO 64²，B=4，普通调用 | 2D 实数 FFT，FP32 | PyTorch/cuFFT，普通调用 | 0.955x |
| FNO 128²，B=4，普通调用 | 2D 实数 FFT，FP32 | PyTorch/cuFFT，普通调用 | 0.957x |
| FNO 64²，B=4，CUDA Graph | 2D 实数 FFT，FP32 | PyTorch/cuFFT，CUDA Graph | 1.030x |
| FNO 128²，B=4，CUDA Graph | 2D 实数 FFT，FP32 | PyTorch/cuFFT，CUDA Graph | 1.056x |
| QuaRot 随机权重 7B，B=1，prefill512 + decode32 | FWHT，FP16 存储 / FP32 计算 | 原生 QuaRot + Dao FWHT | 1.014–1.016x |
| QuaRot 随机权重 7B，B=4，prefill512 + decode32 | FWHT，FP16 存储 / FP32 计算 | 原生 QuaRot + Dao FWHT | 1.008–1.013x |
| CKKS 完整应用 | NTT | HEonGPU | 尚无完整结果 |

B=1 filter results remain above with their observed variance rather than being
condensed into a stable speedup. Shared random model weights do not establish
model quality. The standalone-to-application gap has distinct causes:

- **Different requests.** FNO uses 64²/128² compact real transforms with FFT
  batch 96 (model batch 4, hidden channels 24), four forward and four inverse
  calls per model forward. Both directions select measured thread-shared axis
  cores. QuaRot uses N=32/64, not long FWHT transforms. Complex candidate
  diagnostics and long-transform gains cannot stand in for these requests.
  Even complete public 2048² real FFT at B=16 gives 1.123x forward but only
  1.001x inverse versus Torch. This is a direction-dependent example, not an
  attribution of the current 1080² filter, whose request is different.
- **FNO submission and integration.** Ordinary execution is slower by
  127.3/122.6 us at 64²/128², while matched graph execution saves 18.1/55.5 us.
  The same plans, weights and inputs are used, and graph outputs match ordinary
  outputs. This supports a material submission/integration effect without
  isolating Python, FFI, allocation or GPU idle gaps individually. CUDA-event
  elapsed time includes gaps between submissions; it is not a sum of busy
  kernel times. Selection measures device execution with CUDA Graph, not the
  entire eager application. Alternative-plan eager ranking is not measured here.
- **QuaRot phase composition.** At B=4, prefill improves 1.069x, while decode-32
  improves only 1.007–1.013x and takes about 2.4 seconds versus about 0.16 seconds
  for prefill. Independently timed phase medians are not added to reconstruct
  E2E. The current FWHT provider fuses scaling and final narrowing with zero
  storage-conversion kernels; the old composed-adapter defect is repaired.
  Current model traces do not establish the exact FWHT fraction, SM utilization
  or bandwidth saturation. The accelerated region is only part of the model,
  but its exact contribution remains to be profiled in this build.
- **Complete filter costs.** Timing includes padding, spectral multiply,
  inverse transform and crop, with the reusable filter spectrum cached for all
  backends. Earlier 2048² padding inflation is repaired. Current 1.081x/1.102x
  gains cannot be equated with an isolated complex FFT candidate's speedup.

The remaining attribution gap is matched operator-boundary and kernel-timeline
profiling in these applications, especially inverse real FFT and QuaRot decode.
No new GPU experiment was run for this result audit.
