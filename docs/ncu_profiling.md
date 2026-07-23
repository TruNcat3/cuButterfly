# Nsight Compute Profiling

## Purpose

The profiling run compares three implementations under the same modulus,
length, batch, warmup, cache policy, and clock policy:

- cuNTT with the original first-pass cross twiddle;
- cuNTT with fused coset-NTT twiddles;
- GPU-NTT Merge, when `GPUNTT_BIN` is provided.

It collects kernel time, DRAM bytes and utilization, L1/L2 hit rates, executed
instructions, integer instructions, active warps, barrier stalls, long
scoreboard stalls, registers, and shared memory.

## Enable counters

The current server returns `ERR_NVGPUCTRPERM`. An administrator must allow GPU
performance-counter access. One persistent Linux configuration is:

```bash
sudo sh -c 'printf "%s\n" "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
  > /etc/modprobe.d/nvidia-profiler.conf'
```

This normally requires reloading the NVIDIA kernel modules or rebooting. Verify
that no `ERR_NVGPUCTRPERM` remains with:

```bash
/usr/local/cuda-11.8/bin/ncu \
  --metrics gpu__time_duration.sum \
  --launch-count 1 \
  build/cuntt_bench --logN 16 --batch 1 --backend hybrid2d \
  --warmup 0 --repeat 1
```

## Build the comparators

Build cuNTT normally. The GPU-NTT harness can be linked against the local
GPU-NTT checkout with:

```bash
/usr/local/cuda-11.8/bin/nvcc -std=c++17 -O3 -arch=sm_70 -ccbin g++-9 \
  -I /tmp/GPU-NTT-compare/src/include \
  benchmarks/gpuntt_merge_gap_bench.cu \
  /tmp/GPU-NTT-compare/build-v100-118/src/libntt-1.0.a \
  -o /tmp/gpuntt_merge_gap_bench
```

## Collect

Run from the cuNTT project root. The script uses 1000 warmups, skips those
kernel launches, and profiles only the following transform. NCU holds the V100
at its base profiling clock and does not flush caches between replay passes.
For `logN=16/18/20`, the script explicitly gives `first`, fused Shoup, and
fused Barrett the same fused-optimal block mapping, so their counter delta
isolates twiddle organization and modular reduction rather than rows or thread
count. Placement and reduction are emitted as separate CSV fields.

```bash
GPUNTT_BIN=/tmp/gpuntt_merge_gap_bench \
  ./scripts/profile_ncu.sh 16 64

GPUNTT_BIN=/tmp/gpuntt_merge_gap_bench \
  ./scripts/profile_ncu.sh 20 4
```

To collect only a missing GPU-NTT file without overwriting the cuNTT captures:

```bash
./scripts/profile_ncu.sh 20 4 \
  --gpuntt-only /tmp/gpuntt_merge_gap_bench
```

The explicit option also works through `sudo`, which may otherwise discard
the `PROFILE_CUNTT` and `GPUNTT_BIN` environment variables.

Raw files are written to `results/ncu/`. To change the output path or profiler:

```bash
OUTPUT_DIR=/tmp/ncu-log20 \
NCU=/usr/local/cuda-11.8/bin/ncu \
GPUNTT_BIN=/tmp/gpuntt_merge_gap_bench \
  ./scripts/profile_ncu.sh 20 4
```

To collect only the three native-layout `compact-stage` kernels at `logN=20`:

```bash
PROFILE_CUNTT=0 PROFILE_COMPACT=1 \
  OUTPUT_DIR=results/ncu_compact \
  ./scripts/profile_ncu.sh 20 4

python3 scripts/summarize_ncu.py \
  results/ncu_compact/*_logN20.csv \
  --output results/ncu_compact_summary_logN20.csv
```

Set `GPUNTT_BIN=/tmp/gpuntt_merge_gap_bench` on the collection command to
include GPU-NTT in the same run.

When NCU requires `sudo`, use the explicit option because `sudo` commonly
removes `PROFILE_COMPACT`:

```bash
sudo OUTPUT_DIR=results/ncu_compact \
  ./scripts/profile_ncu.sh 20 4 --compact-only
sudo chown -R "$USER:$USER" results/ncu_compact

python3 scripts/summarize_ncu.py \
  results/ncu_compact/*_logN20.csv \
  --output results/ncu_compact_summary_logN20.csv
```

If clock control is not permitted even after counters are enabled, change
`--clock-control base` to `--clock-control none` in the script and record the
SM clock separately with `nvidia-smi`.

## Summarize

Pivot the raw NCU rows into one row per kernel:

```bash
python3 scripts/summarize_ncu.py \
  results/ncu/*_logN16.csv \
  --output results/ncu_summary_logN16.csv

python3 scripts/summarize_ncu.py \
  results/ncu/*_logN20.csv \
  --output results/ncu_summary_logN20.csv

column -s, -t < results/ncu_summary_logN16.csv | less -S
```

For the steady-state CUDA-event reduction A/B test (three trials per mode and
length), run:

```bash
./scripts/benchmark_fused_reduction.sh
python3 scripts/summarize_fused_reduction.py \
  results/fused_reduction_ab.csv \
  --output results/fused_reduction_summary.csv
```

The expected fused signature is fewer integer instructions than `first`, with
more root-table traffic inside the second pass. Barrett halves the large fused
table footprint but performs more integer instructions. At
`logN=20`, compare `dram_read_mib`, `l2_hit_pct`, `dram_peak_pct`, integer
instructions, and `long_scoreboard_stall_pct` first; these distinguish table
capacity and bandwidth from modular-arithmetic cost.

The script assumes two cuNTT kernels at all supported Hybrid2D lengths, two
GPU-NTT kernels through `logN=16`, and three GPU-NTT kernels above it.
