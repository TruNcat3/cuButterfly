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
  results/v100_fft_pipeline_raw.csv \
  --output results/v100_fft_pipeline_summary.csv \
  --markdown results/v100_fft_pipeline_report.md

python3 scripts/generate_fft_dispatch.py \
  --summary results/v100_fft_pipeline_summary.csv \
  --output config/v100_fft_dispatch.json

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

python3 scripts/summarize_ncu.py \
  results/ncu_scaling_crossovers/fft*.csv \
  results/ncu_scaling_crossovers/fwht*.csv \
  --output results/ncu_scaling_crossovers/summary.csv

python3 scripts/analyze_scaling_ncu.py \
  results/ncu_scaling_crossovers/summary.csv \
  --output results/ncu_scaling_crossovers/attribution.csv \
  --markdown results/ncu_scaling_crossovers/attribution.md

python3 scripts/check_repository.py
printf 'V100 derived analyses reproduced from checked-in raw records.\n'
