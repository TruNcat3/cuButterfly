# Single-GPU Comprehensive Benchmark

The comprehensive suite compares operators and implementations on one GPU
without mixing incompatible semantics. Its canonical case list is
[`config/v100_comprehensive_suite.json`](../config/v100_comprehensive_suite.json).
The current measured interpretation is in
[`comprehensive_v100_results.md`](comprehensive_v100_results.md).

## Test Layers

1. **Correctness preflight** runs every implementation with `batch=1`, one
   execution, and the benchmark's CPU/reference verification.
2. **Resident performance** uses the same `2^22` total points per semantic
   group, CUDA-event kernel time, randomized case/trial order, and no host-device
   transfers in the ranked metric.
3. **Comparison grouping** requires the same operator, precision, direction,
   layout contract, length, and batch. A declared reference is used when one is
   runnable; otherwise ratios are explicitly relative to the group fastest.

`quick` uses 5 warmups, 20 repetitions, and 3 trials over representative short
and long FFT, NTT, FWHT, and XOR-zeta cases. `full` contains every quick case and
uses 1000 unmeasured warmups to stabilize V100 DVFS, 100 measured repetitions,
5 trials, FP64, mixed precision, inverse,
in-place/strided layouts, and additional semantic coverage.

## Run

Inspect commands without executing them:

```bash
python3 scripts/run_comprehensive_suite.py --mode quick --dry-run
```

Run or resume the quick suite:

```bash
python3 scripts/run_comprehensive_suite.py --mode quick \
  --output results/comprehensive_v100_quick_raw.csv

python3 scripts/run_comprehensive_suite.py --mode quick --resume \
  --output results/comprehensive_v100_quick_raw.csv
```

Summarize it:

```bash
python3 scripts/summarize_comprehensive_suite.py \
  results/comprehensive_v100_quick_raw.csv \
  --output results/comprehensive_v100_quick_summary.csv \
  --markdown results/comprehensive_v100_quick_report.md \
  --require-stable
```

Run the full suite after the quick data path is clean:

```bash
python3 scripts/run_comprehensive_suite.py --mode full \
  --output results/comprehensive_v100_full_raw.csv
```

`--only fft18 ntt-u60` restricts execution by case-id or group substring.
`--resume` uses `(case_id, trial)` as the durable completion key and rewrites
the CSV after every successful sample.

## Current External Coverage

cuFFT and VkFFT are runnable in the current build and are included directly for
FP32 forward contiguous FFT groups. Dao-AILab FHT cannot be rerun from the
active environment because PyTorch is absent. The GPU-NTT comparator executable
is also absent. Both gaps are recorded in the manifest with paths to the
existing same-V100 evidence; the summary does not present those archived
numbers as fresh trials.

The internal NTT radix-2 row is an ablation reference, not an external-library
reference. Groups without a runnable external reference must not be used to
claim library superiority.

The default stability gate requires the central range to be at most 3%. With
five trials, the central range removes one sample from each end; the full range
is still reported. A case whose central range passes but full range does not is
labelled `stable-with-outlier`, never silently filtered. On this V100, short
per-process warmups can alternate between the 1312 MHz application clock and
the 1530 MHz boost clock; mixed-clock central samples are rejected. Locked
clocks are preferable when available.
