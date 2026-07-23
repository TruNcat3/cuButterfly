# Cross-GPU Experiment Matrix

## Purpose

The cross-GPU experiment tests whether calibrated hardware demands predict a
near-optimal butterfly mapping. Vendor peak specifications are context only;
the model consumes the measured feedback, processing-unit, shared-exchange,
barrier, and explicit stage-pipeline capabilities.

The table is stored in `configs/hardware/cross_gpu_matrix.csv`. V100 is the
measured reference. A100, H100, and RTX 4090 rows are placeholders and must not
enter performance plots until their `status` becomes `measured`.

## Capture Protocol

Build for the target architecture, then collect its calibrated profile:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=<SM>
cmake --build build -j

TRIALS=5 OUTPUT=results/hardware_capabilities_raw.csv \
  ./scripts/benchmark_hardware_capabilities.sh
python3 scripts/summarize_hardware_capabilities.py \
  results/hardware_capabilities_raw.csv \
  --output results/hardware_capabilities_<gpu>.json
```

Then run the common semantic and large-length matrix:

```bash
cmake --build build --target test
python3 scripts/sweep_cubutterfly_designs.py \
  --operators fwht fft xor-zeta \
  --backends hierarchical cufft \
  --logNs 12 14 16 18 20 \
  --precisions fp32 --normalizations none \
  --tile-thread-options 128 256 \
  --hierarchical-local-stages 6 8 10 \
  --target-points 4194304 --warmup 50 --repeat 100 --trials 5 \
  --output results/cubutterfly_large_<gpu>_raw.csv
```

For each GPU, record the predicted mapping before inspecting sweep performance.
Report top-1 accuracy, top-3 recall, predicted/measured slowdown, and how the
selected `local_stages`, radix, threads, `Us`, and residency change relative to
V100.
