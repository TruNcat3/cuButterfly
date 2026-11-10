# Reproducible experiments

This directory holds executable experiments independently of the manuscript.

- `workloads/` contains workload definitions, target configurations and application runners. The `e13/` runners cover FFT filtering and QuaRot; the CKKS and TFHE runners use pinned HEonGPU dependencies.
- `tools/` contains CPU preparation, GPU measurement, mechanism controls, profile analysis, result export and plotting programs.
- `snapshots/` contains frozen JSON/CSV evidence used by result checks and visualizations.
- `provenance/` preserves source manifests and historical evidence identifiers. Historical source paths and hashes describe the original measurements; directory relocation does not revalidate or remeasure them.

Start with the public configuration-driven runner in `benchmark/README.md` for operator comparisons. Use `python experiments/tools/prepare_experiments.py --help` for the bounded experiment queue, `prepare_end_to_end.py --help` for application preparation, and `prepare_core_dataflow_ablation.py --help` for controlled mechanism experiments. Preparation and compilation run on the CPU; measurement requires an available GPU. Check each program's help for explicit input and output locations.

Frozen snapshots are reusable evidence, not new measurements. Some historical preparation programs require raw result bundles that are not included in a source checkout; they report the missing inputs rather than reconstructing or inventing results. Store new measurements in a separate results directory and retain their semantic, hardware, build and timing identities.
