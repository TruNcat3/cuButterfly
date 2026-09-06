# Reproducibility

## 1. Capture The Environment

For a new installation, calibrate the local GPU before reusing any mapping
table. The one-command wrapper builds, tests, installs, and profiles the
current device:

```bash
./scripts/install_with_hardware_profile.sh \
  --build-dir build \
  --prefix "$HOME/.local" \
  --profile-dir "$HOME/.local/share/cuButterfly/hardware/local"
```

This produces a device-fingerprinted `hardware_profile.json` and a parameterized
`unfolding_model.csv`. See [Hardware Profile Initialization](hardware_profile_install.md)
for the profile schema, update policy, and mismatch check.

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

The cross-operator v0.8 matrix uses a first-class operator/physical-group
manifest. It is the preferred protocol for comparing the current v0.6
incumbent, searched v0.8 candidates, and available library controls:

```bash
python3 scripts/run_cross_operator_comparison.py \
  --manifest config/v100_cross_operator_profiles.json \
  --batches 1 4 16 64 --warmup 100 --repeat 100 --trials 3 \
  --output-dir results/cross_operator_v08
```

For Nsight Compute, run from the repository root as an administrator and keep
the generated `profiles.json` beside the raw captures:

```bash
sudo -E ./scripts/profile_cross_operator_v08_ncu.sh \
  --output-dir results/ncu_cross_operator_v08
```

`scripts/summarize_ncu.py --metadata ... --require-metadata` rejects a capture
that cannot be associated with a declared operator, preventing a filename or
kernel-name heuristic from mixing FFT, NTT, and coefficient-free transforms.

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

### Numeric-Regime Mapping

Generate the large numeric manifest on demand, then run only its quick tier for
screening. The 5,690-case expanded manifest is intentionally ignored rather
than checked into Git:

```bash
python3 scripts/generate_numeric_regime_suite.py \
  --output results/v100_numeric_regime_suite.json
python3 scripts/run_comprehensive_suite.py \
  --manifest results/v100_numeric_regime_suite.json --mode quick \
  --output results/v100_numeric_regime_quick_raw.csv
python3 scripts/analyze_numeric_regime_cliffs.py \
  results/v100_numeric_regime_quick_raw.csv \
  --manifest results/v100_numeric_regime_suite.json \
  --events results/v100_numeric_regime_events.csv \
  --regimes results/v100_numeric_regimes.json \
  --report results/v100_numeric_regime_quick_report.md
```

Focused full-timing, NCU, adaptive-coverage, and final selector commands are
versioned in [Numeric-Regime Mapping Study](next_phase_numeric_regimes.md). The
privileged counter step is run from the repository root:

```bash
python3 scripts/generate_numeric_boundary_ncu.py
sudo -E ./scripts/profile_numeric_boundaries_ncu.sh
python3 scripts/analyze_numeric_boundary_ncu.py \
  results/ncu_numeric_boundaries/summary.csv \
  --timing-analysis results/v100_numeric_confirmed_followup_analysis.csv \
  --output results/v100_numeric_boundary_ncu_analysis.csv \
  --report results/v100_numeric_boundary_ncu_analysis.md
```

### Library/Base/Search Matrix

```bash
python3 scripts/run_external_baseline_suite.py \
  --manifest config/v100_three_way_comparison.json \
  --fht-python /home/wt/.conda/envs/cubutterfly-baselines/bin/python \
  --output results/v100_three_way_raw.csv
python3 scripts/summarize_external_baseline_suite.py \
  results/v100_three_way_raw.csv \
  --output results/v100_three_way_summary.csv
python3 scripts/summarize_three_way_comparison.py \
  results/v100_three_way_summary.csv \
  --archived-baselines results/v100_external_baselines_summary.csv \
  --internal-raw results/v100_numeric_coverage_raw.csv \
  --internal-manifest results/v100_numeric_coverage_suite.json \
  --output results/v100_three_way_comparison.csv \
  --internal-output results/v100_three_way_internal.csv \
  --metrics results/v100_three_way_metrics.json \
  --report results/v100_three_way_comparison.md
```

The NTT library rows in this matrix come from the checked-in matching-protocol
same-V100 archive when the GPU-NTT comparator is unavailable. The summarizer
labels that provenance; do not describe those rows as interleaved with the
current FFT/FWHT refresh.

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

The matched v0.6/v0.7 NTT matrix is reproduced with:

```bash
python3 scripts/run_external_baseline_suite.py \
  --manifest config/v100_v06_v07_comparison.json \
  --output results/v100_v06_v07_r4_refresh_raw.csv
python3 scripts/summarize_external_baseline_suite.py \
  results/v100_v06_v07_r4_refresh_raw.csv \
  --output results/v100_v06_v07_r4_refresh_summary.csv
python3 scripts/summarize_v06_v07_comparison.py \
  results/v100_v06_v07_r4_refresh_summary.csv \
  --output results/v100_v06_v07_r4_refresh_comparison.csv \
  --metrics results/v100_v06_v07_r4_refresh_metrics.json \
  --markdown results/v100_v06_v07_r4_refresh.md
```

Use `--resume` on the collection command only when the manifest and existing
raw file describe the same protocol. The runner keys completion by case ID and
trial.

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

### Hierarchical NTT dataflow

```bash
MODE=screen ./scripts/benchmark_hierarchical_dataflow.sh
MODE=space WARMUP=20 REPEAT=100 ./scripts/benchmark_hierarchical_dataflow.sh
sudo -E ./scripts/profile_hierarchical_dataflow_ncu.sh
```

`screen` holds total points near `2^22` while sweeping `logN=12..20` and both
word widths. `space` scans rows/subgraphs per CTA, serial subgraph depth, and
thread count at `logN=20,batch=4`; every result is reference-verified.

Administrator-enabled representative collection:

```bash
sudo -E ./scripts/profile_ncu.sh
sudo -E ./scripts/profile_fft_units_ncu.sh
sudo -E ./scripts/profile_fft_cta_mapping_ncu.sh
sudo -E ./scripts/profile_structured_2x2_ncu.sh
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
## 9. Bounded HybridDataflow/cuFFTDx Envelope

The first cuFFTDx integration is intentionally bounded to one block-resident
local tile. To measure it against the native mixed-dataflow kernel, standalone
cuFFTDx block launch, and external cuFFT, run:

```bash
BIN=$PWD/build-cuda118-cufftdx2/cubutterfly_bench \
LOG_NS="8 9 10" BATCHES="1 4 16 64" TRIALS=3 \
OUTPUT_DIR=$PWD/results/hybrid_cufftdx_adapter \
./scripts/benchmark_hybrid_cufftdx_adapter.sh
```

The script writes `raw.csv`, `summary.csv`, and `analysis.md`. It uses
`--verify` by default and records unsupported/resource-failed points with an
error status rather than treating them as performance results. Use the Python
entry point directly with `--dry-run` to inspect generated commands on a host
without a CUDA device.
