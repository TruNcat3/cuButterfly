# cuButterfly Summary And Example

[Documentation index](README.md) | [Repository](../README.md)

## Project summary

cuButterfly is a CUDA library and research artifact for regular butterfly
computations, including FFT, NTT, FWHT, structured 2x2 and zeta transforms.
It separates the layered operator graph, architecture mapping, local arithmetic
core and target hardware profile. The planner uses these layers to select a
legal implementation for a workload.

Start with [Getting Started](getting_started.md) for installation and builds,
and [Unified Planner](unified_planner.md) for the current planning workflow.
The release and result documents identify the GPU and workload behind each
reported measurement.

## Example: verify one FWHT workload

Build the library for the target NVIDIA GPU using the Getting Started guide.
From the repository root, run the following command with the `build` directory
produced by the manual CMake build:

```bash
./build/cubutterfly_bench --operator fwht --backend temporal-tile \
  --local-exchange warp-register --precision fp32 \
  --logN 15 --batch 128 --verify
```

This selects an FP32 FWHT workload with length 32,768 and batch size 128.
The `--verify` option compares the result with the CPU reference. Record the
GPU, build and complete command when saving a result for later comparison.

## Example: check repository documentation

The metadata and local-link checks can be run with Python without a GPU:

```bash
python3 scripts/check_repository.py
```

A successful run reports the number of checked Markdown, JSON and Python files.
Continue with the [Programming Guide](programming_guide.md) for application
integration or [Reproducibility](reproducibility.md) for experiment protocols.
