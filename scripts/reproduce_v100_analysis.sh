#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

python3 scripts/generate_scaling_suite.py \
  --spec config/v100_scaling_space.json \
  --check config/v100_scaling_suite.json

python3 scripts/generate_fft_pipeline.py \
  --check config/v100_fft_pipeline_candidates.json

python3 scripts/summarize_fft_pipeline.py \
  results/v100_fft_pipeline_vectorized_raw.csv \
  --output results/v100_fft_pipeline_vectorized_summary.csv \
  --markdown results/v100_fft_pipeline_vectorized_report.md

python3 scripts/generate_fft_dispatch.py \
  --summary results/v100_fft_pipeline_vectorized_summary.csv \
  --output config/v100_fft_dispatch.json

python3 scripts/analyze_fft_pipeline_model.py \
  results/v100_fft_pipeline_vectorized_summary.csv \
  --output results/v100_fft_pipeline_model_evaluation.csv \
  --calibrated-output results/v100_fft_pipeline_model_calibrated_evaluation.csv \
  --metrics results/v100_fft_pipeline_model_metrics.json \
  --markdown results/v100_fft_pipeline_model_report.md

python3 scripts/summarize_scaling_suite.py \
  results/v100_scaling_full_raw.csv \
  --output results/v100_scaling_full_summary.csv \
  --markdown results/v100_scaling_full_report.md

python3 scripts/select_mapping.py --evaluate --top-k 3 \
  --evaluation-output results/v100_mapping_selector_evaluation.csv \
  --metrics-output results/v100_mapping_selector_metrics.json

python3 scripts/summarize_external_baseline_suite.py \
  results/v100_external_baselines_raw.csv \
  --output results/v100_external_baselines_summary.csv \
  --markdown results/v100_external_baselines_report.md

python3 scripts/summarize_comprehensive_suite.py \
  results/comprehensive_v100_full_raw.csv \
  --output results/comprehensive_v100_full_summary.csv \
  --markdown results/comprehensive_v100_full_report.md

python3 scripts/summarize_comprehensive_suite.py \
  results/comprehensive_v100_vectorized_fft_raw.csv \
  --output results/comprehensive_v100_vectorized_fft_summary.csv \
  --markdown results/comprehensive_v100_vectorized_fft_report.md \
  --require-stable

python3 scripts/summarize_ncu.py \
  results/ncu_scaling_crossovers/fft*.csv \
  results/ncu_scaling_crossovers/fwht*.csv \
  --output results/ncu_scaling_crossovers/summary.csv

python3 scripts/analyze_scaling_ncu.py \
  results/ncu_scaling_crossovers/summary.csv \
  --output results/ncu_scaling_crossovers/attribution.csv \
  --markdown results/ncu_scaling_crossovers/attribution.md

python3 scripts/summarize_ncu.py \
  results/ncu_fft_vectorized/post_vector_*.csv \
  --output results/ncu_fft_vectorized/summary.csv

python3 scripts/analyze_fft_vectorized_ncu.py \
  results/ncu_scaling_crossovers/summary.csv \
  results/ncu_fft_vectorized/summary.csv \
  --pre-timing-summary results/v100_fft_pipeline_summary.csv \
  --post-timing-summary results/v100_fft_pipeline_vectorized_summary.csv \
  --batch 16 \
  --output results/ncu_fft_vectorized/attribution.csv \
  --markdown results/ncu_fft_vectorized/attribution.md

python3 scripts/analyze_fp64_fft_robustness.py \
  results/fp64_robustness_batch_raw.csv \
  --output results/fp64_robustness_batch_summary.csv \
  --markdown results/fp64_robustness_batch_analysis.md

python3 scripts/summarize_cubutterfly_designs.py \
  results/fp64_robustness_mapping_coarse_raw.csv \
  --output results/fp64_robustness_mapping_coarse_summary.csv

python3 scripts/summarize_cubutterfly_designs.py \
  results/fp64_robustness_crossover_*_raw.csv \
  --output results/fp64_robustness_crossover_summary.csv

python3 scripts/summarize_cubutterfly_designs.py \
  results/fp64_robustness_logN*_confirm_raw.csv \
  results/fp64_robustness_cufft_confirm_raw.csv \
  --output results/fp64_robustness_confirm_summary.csv

python3 scripts/summarize_ncu.py \
  results/ncu_fp64_fft/fp64_*.csv \
  --output results/ncu_fp64_fft/summary.csv

python3 scripts/analyze_fp64_fft_ncu.py \
  results/ncu_fp64_fft/summary.csv \
  --timing-summary results/fp64_address_logN16_summary.csv \
  --output results/ncu_fp64_fft/analysis.csv \
  --markdown results/ncu_fp64_fft/analysis.md

python3 scripts/check_repository.py
printf 'V100 derived analyses reproduced from checked-in raw records.\n'
