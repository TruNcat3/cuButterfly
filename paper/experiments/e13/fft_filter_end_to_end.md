# Classic FFT filtering contract

`run_fft_filter_end_to_end.py` measures FP32 batched 2D linear convolution with
`same` cropping. Preparation defaults to 512/1024/2048 square images crossed
with 31/127 square filters and B=1. `--sizes`, `--filters` and `--batches` select
a bounded cohort. Batch entries contain independent images and one shared
asymmetric signed filter. Output storage belongs to each returned call.

Graph is the primary mode for fixed-shape repeated applications. Every backend
refreshes device image and kernel buffers, replays its full convolution graph,
and copies the cropped result into independent output storage. These input
refreshes and output ownership copies are timed. Captured temporary allocations
are reused; capture, plan construction, selection, compilation, host/device
transfers and CPU validation are outside timing. The default eager mode remains
an integration diagnostic: its padding, FFTs, multiply, crop/output copy and
application allocations are timed, without Graph input refreshes.

`--cache-filter-spectrum` precomputes one fixed filter spectrum outside timing
for every backend. The timed call then includes image padding/FFT, spectrum
multiply, inverse FFT and crop/output copy. The kernel input refresh remains
timed in Graph mode even though the cached filter makes it invariant. Replacing
the filter requires recapture. Without this flag, each complete batch call also
pads and transforms its shared filter once. cuButterfly forward-normalized FFTs
require compensation by the transform area; cached runs fold that compensation
into the invariant filter once. All rounding errors remain reported.

cuButterfly and optional Torch/cuFFT use power-of-two padding by default;
`--padding-policy fast` uses CuPy's real `next_fast_len`. CuPy always uses its
native fast padding. Reports record actual padding per method. The FP32 CPU
Torch FFT reference uses cuButterfly's actual padding; it preserves FP32 storage,
complex64 FFT intermediates and backward normalization. The archived SciPy FP64
reference is a separate preparation diagnostic.

New `fft-filter-complete-call-v2` reports distinguish `contract` (actual padding,
execution and filter policy) from `preparation_contract` (unchanged input archive
metadata). Each row records its `call_contract`, whole-batch latency, per-image
amortization, throughput and full error distributions. Historical preparation
fields named `cpu_fp32_error` used the older SciPy/NumPy helper and must not be
treated as the runtime same-precision oracle. Historical reports are preserved;
the [CPU evidence index](../../../results/e2e_alignment_20261019/fft/README.md)
resolves their stale filter-contract descriptions from actual run fields.

Graph validation changes the image, restores the original image, checks exact
restoration and retained prior output, and records CPU errors for both inputs.
Finite results are required. The user's error-report-only policy does not infer
an application threshold: default `accepted=null` and the runner's legacy
`formal_performance_eligible=false` mean numerical acceptance is unassessed.
They do not invalidate ordinary full-call timing or turn it into profiler time.
Report speed alongside its errors without claiming a numerical threshold passed.
Both `--atol` and `--rtol` evaluate a separately chosen criterion when requested.

```bash
/home/wt/yes/envs/cubutterfly/bin/python -m unittest discover -s paper/experiments/e13 -p test_fft_filter_end_to_end.py -v
/home/wt/yes/envs/cubutterfly/bin/python paper/experiments/e13/run_fft_filter_end_to_end.py --prepare --output <new-input-directory> --sizes 1024 --filters 31 --batches 16
# Requires an exclusively assigned A100 80GB UUID and an explicit library.
CUDA_VISIBLE_DEVICES=<assigned-uuid> CUBUTTERFLY_COMPILE_MODE=research /home/wt/yes/envs/cubutterfly/bin/python paper/experiments/e13/run_fft_filter_end_to_end.py --run --output <input-directory> --library <library.so> --run-id graph-v2 --execution-mode graph --padding-policy fast --cache-filter-spectrum --torch-control
```

The default timing protocol uses CUDA events, five warmups and seven samples
of ten calls. Repetitions are samples from one process. For independent process
runs use distinct IDs and `--process-index 1`, `2`, `3`; the method order rotates
and is saved. Existing reports resume only with matching code, library, hardware,
arguments, preparation and execution contract. Under `scripts/run_isolated_gpu.py`,
reports go to `XD_REPORT_DIR`; prepared input archives stay in `--output`.
`--require-cache` fails if a public plan did not replay `user_cache`.
`--calibration-identity` records host rank1 provenance only: public native xD
selection does not consume that manifest, and actual selection is recorded in
`plan_metadata` and `filter_plan_metadata`.
