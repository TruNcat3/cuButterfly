# Bounded integer image registration

`paper/tools/run_phase_correlation_graph_20261013.py` implements batched 2D
phase-correlation registration with a fixed cached reference spectrum. Its
application basis is scikit-image's Fourier image-registration algorithm:
[phase_cross_correlation](https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation).
The local older scikit-image implementation at
`/home/wt/yes/lib/python3.7/site-packages/skimage/feature/register_translation.py`
also supplies the FFT, cross-spectrum, inverse FFT, absolute peak, and signed
shift sequence. This runner uses integer resolution and circular translations,
without subpixel refinement, masks, or disambiguation.

All backends compute `reference_fft * conj(moving_fft)`, divide by
`max(abs(cross), float32_epsilon)` (epsilon exactly `2**-23`), run the backward
normalized inverse real FFT, and decode the absolute correlation peak into the
shift to apply to the moving image. The epsilon rule is explicitly frozen here;
this is not an exact reproduction of every scikit-image version's normalization
constant. Inputs are finite, seeded, independent random reference images with
known per-image circular translations, changed across calls. Random images
establish implementation correctness and performance, not natural-image quality.

The bounded matrix is square sizes 1024 and 2048 with batches 4 and 16. Each
process measures Torch/cuFFT, public `RealFFT`, and CuPy/cuFFT eager and complete
CUDA Graph calls. Reference FFT caching, plan search, warmup, and capture are
excluded equally. Device input copy and independent shift output copy are timed
equally; correctness includes changing inputs and preserved output ownership.
Validation restores the original input after the changed inputs and compares
actual returned shifts exactly with same-backend eager results. CuPy capture owns
a dedicated memory pool kept alive with the graph, so later ordinary output
allocations cannot recycle captured temporary addresses.
CPU Torch FP32 correlations are the numerical reference. Reports preserve
correlation errors and exact known integer-peak recovery, without a new floating
point threshold. Optional transform-pair graphs are nonadditive diagnostics;
their ratios must not be called exact FFT fractions of application time.

Use the existing environment and isolation wrapper, with the GPU UUID and
deadline supplied by the experiment owner:

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/run_isolated_gpu.py \
  --gpu-uuid GPU_UUID --report-dir REPORT_DIR --until ISO_DEADLINE -- \
  /home/wt/yes/envs/cubutterfly/bin/python paper/tools/run_phase_correlation_graph_20261013.py \
  --library build-e2e-20261013/libcuntt.so --size 1024 --batch 4 \
  --output phase_correlation.json --transform-diagnostic
```

The runner honors `XD_REPORT_DIR` for relative output and default cache paths.
Reports contain actual public selection metadata, cache hashes, code and library
hashes, software versions, GPU UUID/model/memory, input identities, and raw times.
`--cpu-only --output /tmp/phase_cpu.json` validates the full CPU reference without
initializing CUDA or loading CuPy. GPU 1 is reserved by the coordinating agent;
this implementation task does not authorize GPU execution.
