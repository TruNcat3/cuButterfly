# Reusable FNO CUDA Graph integration

`paper/tools/run_fno_graph_integration_20261013.py` compares four matched
application modes: torch eager, cuButterfly eager, torch graph, and cuButterfly
graph. All use identical real Darcy batches, seeded model weights (or an
explicit checkpoint), compact inverse semantics, and FP32 with TF32 disabled.
Seeded weights support implementation consistency and performance evidence;
they do not establish trained model quality.

The treatment uses the public `cubutterfly.real_fft.RealFFT` adapter and
default automatic plan selection, with `CUBUTTERFLY_PLAN_CACHE` receipts and
actual mappings preserved. It does not request explicit mappings. The frozen
E13 loader, FNO definition, FFT patch boundary, and dataset are reused. This is
a distinct changing-input application protocol from the earlier fixed-device
input graph diagnostic in `fno_dispatch_optimization_20261012.py`.

The optional `cubutterfly.torch_graph.InferenceGraph` wrapper captures one
inference function on a declared non-default stream. Inputs must preserve
shape, dtype, and device, and calls must be serialized on that stream. The
callable, its eval-mode modules, weights, and operator plans must remain alive
and unchanged. Each call copies the new input, replays, and returns a clone
with independent storage. Capture, compilation, search, and warmup are setup
costs. Input copy and output clone are timed for both eager and graph modes;
wall timing also includes synchronization. Device events and complete wall
time are both reported. Input transfer from CPU is outside this protocol.

Multiple distinct real input batches are checked against same-backend eager
and torch eager outputs. Absolute-error quantiles, relative L2 error, and
finite status are recorded without a numerical acceptance threshold. Finite
outputs and unchanged model weights are required. `--process-trial-index`
rotates the four-mode execution order; aggregate fresh-process trials outside
the runner without combining eager and graph speedups.

Run through the existing isolated GPU launcher and pass the new library:

```bash
XD_REPORT_DIR=/path/to/report \
  /home/wt/yes/envs/cubutterfly/bin/python \
  paper/tools/run_fno_graph_integration_20261013.py \
  --library /path/to/libcuntt.so --size 64 --batch 4 \
  --process-trial-index 0 --output /path/to/report/fno.json
```

`--size 128`, `--batch 1`, and larger matched batches use the same protocol.
At least two full distinct input batches must exist in the selected payload.
GPU identity includes UUID, model, and memory capacity. `XD_REPORT_DIR` selects
the default cache directory; an explicit `CUBUTTERFLY_PLAN_CACHE` takes priority.

## GPU1 results on 2026-10-13

The bounded validation completed at 09:43 +08 on NVIDIA A100 80GB PCIe,
UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`, using
`build-e2e-20261013/libcuntt.so` SHA256
`ddb7d70cc8e0fac900c0b912f03d7aae18855a807d06f0dff4a0ebad1bcb665b`.
Each size used B4, three distinct real Darcy input batches, 30 measured
changing-input calls per mode, and two independently launched processes with
rotated mode order. The table reports medians of process wall-time medians;
speedup is torch/cuButterfly within the same execution mode.

| Size | Mode | Torch wall ms | cuButterfly wall ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | Eager | 2.741309 | 2.904528 | 0.943805 |
| 64 | Graph | 0.624510 | 0.606179 | 1.030241 |
| 128 | Eager | 2.789014 | 2.967371 | 0.939894 |
| 128 | Graph | 1.039555 | 0.982995 | 1.057539 |

Eager remains slower at both sizes. The graph result is a separate reusable
application execution mode applied equally to both backends; it does not
establish an eager speedup or completion of a broader E2E performance goal.
The graph speedup ranges across the two processes are 1.029007-1.031477 at
64 and 1.056840-1.058238 at 128. Two processes and B4 alone do not establish
general performance across batches, hardware, or trained models.

Both graph backends are exactly equal to their own eager outputs for all
three validation batches, and all four modes produce three distinct output
hashes in every process. Outputs are finite and model state remains unchanged.
cuButterfly versus torch has maximum absolute error `2.9802322387695312e-8`
at 64 and `3.725290298461914e-8` at 128; maximum relative L2 errors are
`6.486594962432287e-8` and `6.717859607129288e-8`. Per-input quantiles remain
in the reports without introducing a numerical acceptance threshold.

Public DEFAULT selected and cached the compact R2C/C2R plans through runtime
measurement during preparation in all four processes. Actual axis resources
are thread-shared, EPT8, block cap16, and recurrence-root64 twiddles. Search
and capture remain outside measured calls; no new adapter was created during
timing. These trials independently populated fresh caches; they do not claim
cross-process cache-replay validation. Model weights are seeded random weights.

Raw samples, errors, complete mappings, and cache files are retained in
[64 trial0](../../../results/e2e_unified_20261013/fno_64_b4_trial0/release/fno.json),
[64 trial1](../../../results/e2e_unified_20261013/fno_64_b4_trial1/release/fno.json),
[128 trial0](../../../results/e2e_unified_20261013/fno_128_b4_trial0/release/fno.json), and
[128 trial1](../../../results/e2e_unified_20261013/fno_128_b4_trial1/release/fno.json).
Cache paths inside the JSON record the original attempt location; the guard
moves successful artifacts into `release/plans.cache`. The
[summary receipt](../../../results/e2e_unified_20261013/fno_summary.json)
preserves aggregate wall and device-event numbers.

Two initial 64 trial0 attempts failed before timing because the model state
contains non-Tensor metadata with named callables. The runner digest now
handles that state, a CPU regression covers it, and the failed attempts remain
under `fno_64_b4_trial0/attempts/`. No core library or public FFT source changed
during this FNO cohort. GPU1 was released after the fourth successful process.

## Bounded adapter hotpath follow-up

The follow-up only removes duplicate input contiguous/reshape handling in the
new `PublicPatch`, and the unnecessary input reshape in public `RealFFT` after
the leading-dimension product has already been checked. All stream, dtype,
shape, output layout, and alias checks remain. The frozen E13 runner and the
library binary are unchanged. The affected CPU tests report `31 passed`.

Four new guarded processes used the same 64/128 B4 parameters and rotation.
The original control reports above remain intact. Aggregate wall times are:

| Size | Mode | Torch wall ms | cuButterfly wall ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | Eager | 2.728578 | 2.855908 | 0.955415 |
| 64 | Graph | 0.623415 | 0.605298 | 1.029932 |
| 128 | Eager | 2.753376 | 2.876015 | 0.957358 |
| 128 | Graph | 1.038970 | 0.983483 | 1.056419 |

The eager gap narrows in this cohort but remains a negative result: both sizes
still take longer than torch. Separate process timing also varies, so the
before/after difference is bounded evidence for this local change and is not
a general speedup claim. No further optimization or profiling was added.

Each mode again produces three distinct outputs, both graphs match their own
eager outputs exactly, and all outputs are finite with unchanged model state.
All optimized output hashes and cache/mapping hashes are identical to their
paired original controls. Maximum GPU-Torch differences are unchanged.
These GPU-Torch comparisons are application regressions, not mathematical
truth; operator correctness uses the same-precision CPU oracle protocol.

Raw reports are
[64 trial0](../../../results/e2e_unified_20261013/fno_hotpath_64_b4_trial0/release/fno.json),
[64 trial1](../../../results/e2e_unified_20261013/fno_hotpath_64_b4_trial1/release/fno.json),
[128 trial0](../../../results/e2e_unified_20261013/fno_hotpath_128_b4_trial0/release/fno.json), and
[128 trial1](../../../results/e2e_unified_20261013/fno_hotpath_128_b4_trial1/release/fno.json).
The [hotpath receipt](../../../results/e2e_unified_20261013/fno_hotpath_summary.json)
links both cohorts. GPU1 was released at 10:17:33 +08 after the fourth process.
