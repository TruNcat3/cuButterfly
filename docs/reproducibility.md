# Reproducibility

## 1. Capture The Environment

Record these before interpreting a result:

```bash
nvidia-smi
nvcc --version
cmake --version
./build/cubutterfly_bench --list-capabilities
./build/cuntt_hardware_microbench --help
```

The hardware capture script writes a machine-readable profile:

```bash
./scripts/benchmark_hardware_capabilities.sh
```

For a new GPU, copy `configs/hardware/gpu_profile_template.json`, fill measured
fields, and leave the V100 profile unchanged.

## 2. Correctness Before Timing

```bash
cmake --build build --target test
compute-sanitizer --tool memcheck ./build/cubutterfly_tests
```

For any selected benchmark mapping, add `--verify`. Verification is intentionally
outside repeated kernel timing.

## 3. Timing Protocol

- use Release builds and an explicit CUDA architecture;
- record and explicitly select the CUDA and host compiler pair;
- keep the input, coefficients, and output resident for kernel timing;
- report H2D and D2H separately;
- use enough warmup to reach stable clocks;
- record independent process trials and report the median;
- keep total points approximately constant for multi-length comparisons;
- never compare native bit-reversed output against required natural output
  without charging the conversion.

A representative common-operator sweep is:

```bash
python3 scripts/sweep_cubutterfly_designs.py \
  --operators fwht fft xor-zeta \
  --logNs 8 10 12 14 16 18 20 \
  --target-points 4194304 --warmup 50 --repeat 100 --trials 5 \
  --output results/reproduction_raw.csv

python3 scripts/summarize_cubutterfly_designs.py \
  results/reproduction_raw.csv \
  --output results/reproduction_summary.csv
```

The controlled single-GPU cross-workload suite validates semantic grouping,
runs correctness preflights, randomizes trial order, and persists each sample:

```bash
python3 scripts/run_comprehensive_suite.py --mode full \
  --output results/comprehensive_v100_full_raw.csv

python3 scripts/summarize_comprehensive_suite.py \
  results/comprehensive_v100_full_raw.csv \
  --output results/comprehensive_v100_full_summary.csv \
  --markdown results/comprehensive_v100_full_report.md \
  --require-stable
```

The current protocol fixes `N * batch = 2^22` for length comparisons. Do not
use it to infer independent batch scaling; that requires holding `N` fixed and
sweeping batch separately.

Generate and run the orthogonal length/batch suite with:

```bash
python3 scripts/generate_scaling_suite.py \
  --spec config/v100_scaling_space.json \
  --output config/v100_scaling_suite.json

python3 scripts/run_comprehensive_suite.py \
  --manifest config/v100_scaling_suite.json --mode full \
  --output results/v100_scaling_full_raw.csv

python3 scripts/summarize_scaling_suite.py \
  results/v100_scaling_full_raw.csv \
  --output results/v100_scaling_full_summary.csv \
  --markdown results/v100_scaling_full_report.md
```

Rows below the default 0.020 ms timing floor are retained for transparency but
cannot define the reported peak or saturation batch.

After all raw records are present, regenerate the V100 scaling, selector,
external-baseline, and NCU analyses together without modifying raw data:

```bash
./scripts/reproduce_v100_analysis.sh
```

Evaluate the V100-calibrated selector without using the target shape's own
timing:

```bash
python3 scripts/select_mapping.py --evaluate --top-k 3 \
  --evaluation-output results/v100_mapping_selector_evaluation.csv \
  --metrics-output results/v100_mapping_selector_metrics.json
```

Refresh matching-protocol external baselines with:

```bash
python3 scripts/run_external_baseline_suite.py --resume \
  --fht-python /home/wt/.conda/envs/cubutterfly-baselines/bin/python \
  --gpuntt-binary /tmp/gpuntt_merge_gap_bench \
  --output results/v100_external_baselines_raw.csv

python3 scripts/summarize_external_baseline_suite.py \
  results/v100_external_baselines_raw.csv \
  --output results/v100_external_baselines_summary.csv \
  --markdown results/v100_external_baselines_report.md
```

## 4. Generated Design Points

The build invokes:

```bash
python3 scripts/generate_design_points.py \
  --spec config/v100_design_points.json \
  --output-dir /tmp/cubutterfly-generated
```

The generated manifest lists exact `(operator, core, precision, logN,
tile_threads)` combinations. Unsupported points throw during plan construction.
Do not treat a silently selected fallback as a measured generated core.

## 5. External Baselines

### Dao Fast Hadamard Transform

```bash
./scripts/install_external_fht.sh
python3 scripts/benchmark_external_fht.py --sm70-patched \
  --logNs 8 10 12 14 15 --dtypes fp16 bf16 fp32 \
  --target-points 4194304 --warmup 50 --repeat 100 --trials 5 \
  --output results/external_dao_fht_v100_raw.csv
```

The upstream revision and V100 gencode-only patch are recorded in the detailed
large-results report and `patches/dao_fast_hadamard_sm70.patch`.

### GPU-NTT

The comparator is `benchmarks/gpuntt_merge_gap_bench.cu`. Its compile command,
pinned revision, modulus, layout, and steady-clock protocol are in
`gpu_ntt_gap_analysis.md`. Keep the external checkout outside this repository.

### cuFFT

cuFFT uses the same `cubutterfly_bench` process and timing interface:

```bash
./build/cubutterfly_bench --operator fft --backend cufft \
  --precision fp32 --logN 20 --batch 4 \
  --warmup 50 --repeat 100 --csv
```

## 6. Nsight Compute

Administrator-enabled representative collection:

```bash
sudo -E ./scripts/profile_ncu.sh
sudo -E ./scripts/profile_fft_units_ncu.sh
sudo -E ./scripts/profile_fft_cta_mapping_ncu.sh
sudo chown -R "$USER:$USER" results/ncu results/ncu_fft_units results/ncu_fft_cta_mapping
```

Reduce raw NCU CSV files with:

```bash
python3 scripts/summarize_ncu.py results/ncu/*.csv \
  --output results/ncu_summary.csv
```

The parser locates the raw NCU header rather than assuming it is the first CSV
line. Keep raw counters when publishing a performance claim.

## 7. Nsight Systems

```bash
./scripts/profile_cubutterfly_fft_nsys.sh
```

Binary `.nsys-rep` and `.sqlite` files are ignored because they are
machine/tool-version specific. Reduced kernel summaries and trace CSV files are
tracked.

## 8. Cross-GPU Protocol

1. capture the new GPU profile and microbenchmarks;
2. predict a reduced candidate set before inspecting the sweep winner;
3. generate legal processing units for that architecture;
4. run identical semantic shapes and external baselines;
5. compare predicted and measured rankings;
6. update `configs/hardware/cross_gpu_matrix.csv` from placeholder to measured.

This protocol is necessary to demonstrate portability of the methodology. A
successful build on another architecture is not cross-GPU validation.
