# Nsight Compute Profiling

## v0.7 Global-Load Sector Attribution

The v0.6/v0.7 comparison requires two distinct measurements: aggregate
L1TEX/DRAM counters and source-correlated coefficient request counters. Build
an otherwise unchanged Release binary with CUDA line information, then run the
privileged capture:

```bash
./scripts/build_v07_sector_attribution.sh
sudo -E ./scripts/profile_v07_sector_attribution_ncu.sh
```

The script verifies forward and inverse v0.6/v0.7 d6 results before profiling.
It captures the full d0/d3/d4/d5/d6 aggregate sequence, then retains
SourceCounters `.ncu-rep` files and exports `cuda,sass` CSV for v0.6 and d6.
Outputs are written under `results/ncu_v07_sector_attribution/`; the analyzer
keeps predicted and source-measured sectors in separate columns. See the
[root-cause audit](v07_sector_root_cause.md) for the stage model and the
resident-subgraph pipeline boundary.

## APPT tail physical cores

The matched V100 pass compares v0.6 10+10, warp and CTA online cores, and the
full, split, and shared/register dependency-closed tails:

```bash
sudo -E "$PWD/scripts/profile_appt_tail_cores_ncu.sh"
```

It writes raw CSV, `summary.csv`, `analysis.csv`, and `analysis.md` under
`results/ncu_appt_tail_cores/`. The counter set includes global and local
sectors, shared bank conflicts, registers, shared bytes, active warps, and the
barrier/scoreboard/MIO stalls needed to distinguish a spilled retained state
from an issue-limited shared-memory tail. The script excludes derived summary
CSV files when rerun.

The archived `ncu_appt_tail_cores_selected` pass isolates readiness counters.
The `ncu_appt_tail_coeff_cache` pass then captures CTA-local coefficient-tree
reuse for the producer and low tail in a separate directory:

```bash
cd /home/wt/git/Hermes/cuNTT-v05
sudo -E env OUTPUT_DIR="$PWD/results/ncu_appt_tail_coeff_cache" \
  "$PWD/scripts/profile_appt_tail_cores_ncu.sh"
```

The script uses the coefficient-cache-specific role weights selected by the
event-timing neighborhood scan. Reproduce the before/after reduction with:

```bash
python3 scripts/compare_appt_tail_ncu.py \
  results/ncu_appt_tail_cores_selected/summary.csv \
  results/ncu_appt_tail_coeff_cache/summary.csv \
  --markdown results/ncu_appt_tail_coeff_cache/comparison.md
```

The source now additionally retains the tree through the high tail. Preserve
the first-cache attribution and capture that increment under
`results/ncu_appt_tail_coeff_cache_full` rather than overwriting it:

```bash
sudo -E env OUTPUT_DIR="$PWD/results/ncu_appt_tail_coeff_cache_full" \
  "$PWD/scripts/profile_appt_tail_cores_ncu.sh"
```

Do not pass the older readiness capture as
`ABLATION_BASELINE`: that would conflate coefficient reuse and role rebalance
with the readiness-only ablation.

## APPT Static Output and Online Writer

The static-layout capture compares the mature v0.6 kernel with fragment widths
8/16/32 under both static and natural output contracts:

```bash
cd /home/wt/git/Hermes/cuNTT-v05
sudo -E env BIN="$PWD/build-cuda118/cuntt_bench" \
  "$PWD/scripts/profile_appt_static_layout_ncu.sh"
```

`summarize_ncu.py` receives only raw profiler CSV files; the script excludes
its own `summary.csv` before reduction. `analyze_appt_static_layout_ncu.py`
reports global load/store sectors relative to matched v0.6, DRAM utilization,
registers, and local sectors. CUDA-event scans remain the timing authority;
NCU replay is used to attribute the sector and synchronization mechanisms.

## Hybrid Dataflow NTT

The dedicated script profiles primary, stage-space, layout, and handoff points
using absolute repository paths:

```bash
sudo -E "$PWD/scripts/profile_hybrid_dataflow_ncu.sh"
```

It writes raw reports and `summary.csv` under
`results/ncu_hybrid_dataflow_td/`. Global load/store sectors establish the
single input/output boundary independently of replay-cache behavior; DRAM
bytes alone may read zero when NCU replay keeps buffers in L2. Bank conflicts,
barrier stalls, active warps, registers, shared memory, and waves explain the
cost of each unfolding choice.

## FFT resident mapping

The `logN=12` FFT decomposition compares two-pass, resident 64x64, direct 4096,
and cuFFT kernels:

```bash
sudo -E ./scripts/profile_fft_resident_ncu.sh
sudo chown -R "$USER:$USER" results/ncu_fft_resident
```

The resulting counter analysis is recorded in
`results/ncu_fft_resident_analysis.md`. The key distinction is between the
two-pass external traffic boundary and the resident kernel's internal
shared-exchange/control overhead.

The new whole-transform `logN=14` winner can be compared directly with cuFFT:

```bash
sudo -E ./scripts/profile_fft_direct14_ncu.sh
sudo chown -R "$USER:$USER" results/ncu_fft_direct14
```

This profiles the 1024-thread direct point and cuFFT with the same transform
count and counter set. Non-admin runs on the current server fail with
`ERR_NVGPUCTRPERM`.

## FFT two-dimension mapping

The `logN=18` architecture search selected a 256/256-thread `9+9` mapping over
the previous fixed 512/512 point. Profile the tuned, fixed, asymmetric, and
cuFFT cases with:

```bash
sudo -E ./scripts/profile_fft_architecture_ncu.sh
sudo chown -R "$USER:$USER" results/ncu_fft_architecture
```

Override `LOG_N`, `BATCH`, and `LOCAL_STAGES` to reuse the script for another
selected split.

The paired 128-bit boundary optimization has a dedicated controlled capture.
It profiles the pre-existing 512/512-thread mapping, the newly selected
256/128-thread mapping, and cuFFT at `logN=20`, batch 16. The reducer compares
the fixed mapping with the archived pre-vector capture before attributing the
additional mapping-selection gain:

```bash
sudo -E ./scripts/profile_fft_vectorized_ncu.sh
sudo chown -R "$USER:$USER" results/ncu_fft_vectorized
```

The script writes raw CSV, a pivoted summary, `attribution.csv`, and
`attribution.md`. Override `BATCH` only when a matching archived pre-vector
label exists.

The archived V100 capture shows that vectorizing the same fixed mapping reduces
warp instructions by 19.6%, CUDA-event time by 10.5%, and NCU replay time by
9.3%, while DRAM writes remain unchanged. See
`results/ncu_fft_vectorized/attribution.md`. The selected mapping is faster in
CUDA-event timing but not under replay, reinforcing that NCU time is not the
performance authority.

The APPT register-tail physical-core comparison profiles the mature v0.6
kernel, the previous calibrated radix-4 point, radix-4 with the newly matched
role mapping, and radix-8 with that same mapping:

```bash
sudo -E "$PWD/scripts/profile_appt_physical_cores_ncu.sh"
sudo chown -R "$USER:$USER" results/ncu_appt_physical_cores
```

Use the CUDA-event table in `results/appt_physical_cores/analysis.md` as the
timing authority. The NCU capture is for load/store sectors, active warps,
barrier stalls, scoreboard stalls, cache behavior, and register attribution.
The script also writes `results/ncu_appt_physical_cores/analysis.md`. Its
controlled table compares role rebalancing and radix-8 against the same
radix-4 mapping; do not rank the persistent cooperative kernels by replay
time, especially at the uint64 batch-16 point.

After the physical-core pass, profile the grouped producer request layouts:

```bash
sudo -E "$PWD/scripts/profile_appt_grouped_producer_ncu.sh"
sudo chown -R "$USER:$USER" results/ncu_appt_grouped_producer
```

This holds the `7+7+6` graph, arithmetic core, fragment layout, and grouped-core
role weights fixed while varying the physical `a` group over 8/16/32. Radix-4
retains its independently calibrated weights. The generated analysis compares
global sectors, L1/L2 hit rates, instructions, stalls, and registers against
matched register-tail radix-4.

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

## Length/Batch Crossover Targets

The orthogonal V100 scaling sweep selects FFT `logN=14/18/20` and FWHT
`logN=15` crossover points for counter attribution. Collect all selected
low/intermediate/saturated batches with administrator-enabled counters:

```bash
sudo -E ./scripts/profile_scaling_crossovers_ncu.sh
sudo chown -R "$USER:$USER" results/ncu_scaling_crossovers

python3 scripts/summarize_ncu.py \
  results/ncu_scaling_crossovers/*.csv \
  --output results/ncu_scaling_crossovers/summary.csv

python3 scripts/analyze_scaling_ncu.py \
  results/ncu_scaling_crossovers/summary.csv \
  --output results/ncu_scaling_crossovers/attribution.csv \
  --markdown results/ncu_scaling_crossovers/attribution.md
```

The script requests base profiling clocks. Compare grid waves, active warps,
DRAM throughput, register/shared-memory limits, barrier stalls, and long
scoreboard stalls across batch before comparing absolute NCU kernel time with
the steady-state CUDA-event table.

## HybridDataflow Fused-Role Attribution

After the CUDA-event role screen, profile the controlled `Ur/Td` pairs with:

```bash
sudo -E ./scripts/profile_hybrid_dataflow_roles_ncu.sh
```

Results are written to `results/ncu_hybrid_dataflow_roles/summary.csv`. Compare
warp and integer instruction counts, shared bank conflicts, barrier stalls,
registers per thread, shared bytes per block, and waves per SM. The primary
question is whether fusion reduces materialized-edge work before register
pressure or an underfilled `Td/Ur` packet removes the gain.

## HierarchicalDataflow Numeric Boundary

Profile matched Hybrid2D and one-launch hierarchical points at the `logN=20`
32/64-bit residency crossover:

```bash
sudo -E ./scripts/profile_hierarchical_dataflow_ncu.sh
```

The output is `results/ncu_hierarchical_dataflow/summary.csv`. Compare DRAM
bytes first to verify the one-boundary traffic contract, then registers/shared
memory, waves, active warps, integer instructions, barrier stall, and long
scoreboard stall. The expected distinction is five resident CTAs/SM for the
selected 32-bit physical point versus two for the 64-bit point. The script
captures both Hybrid2D pass kernels and the single cooperative hierarchical
kernel, then writes plan-level totals to `plan_totals.csv` and the matched
interpretation to `analysis.md`.

To isolate the resident physical unit while holding that outer graph fixed,
run:

```bash
sudo -E ./scripts/profile_hierarchical_cores_ncu.sh
sudo -E env BIN="$PWD/build-v07/cuntt_bench" ./scripts/profile_appt_pipeline_ncu.sh
```

This writes `results/ncu_hierarchical_cores/analysis.csv` and `analysis.md`,
including instruction, stall, register, and effective CTA-residency deltas for
the native dataflow and mature Hybrid2D radix-4 schedules.

To compare the homogeneous CTA control with the generated warp-granular
physical cores, run:

```bash
sudo -E ./scripts/profile_homogeneous_10x10_ncu.sh
sudo -E ./scripts/profile_homogeneous_warp_10x10_ncu.sh
sudo -E ./scripts/profile_homogeneous_vector_radix4_ncu.sh
```

The first command attributes outer data-time traversal while holding the
radix-4 core fixed. The second compares v0.6, one 1024-point transform per
warp, and the 128/256-thread versions of the 256-point-per-warp prefix. Compare
registers and occupancy first, then warp/integer instruction counts, global
sectors, wait stalls, and DRAM utilization. CUDA-event timings in
`results/homogeneous_warp_10x10/` remain the timing authority.

The third command is the focused current-build comparison for the mature v0.6
radix-4 control, selected dual-warp128 coefficient-reuse depth 4, plain
dual-warp128 vector-radix4, and vector coefficient-broadcast depths 3--7. It
writes eight validated raw CSV files and `summary.csv` under
`results/ncu_homogeneous_vector_radix4/`.

For the cheaper stage-6 root-cause closure, profile only v0.6, vector d6, and
the dual-bank vector d7 candidate:

```bash
sudo -E ./scripts/profile_vector_stage6_distribution_ncu.sh
```

This writes `results/ncu_vector_stage6_distribution/summary.csv` and
`analysis.md`. The generated decision requires both a material load-sector
reduction and a kernel-time improvement before recommending d7; otherwise d6
remains the performance default.

After that attribution, compare the aligned vector-load replacement against
both shuffle controls:

```bash
sudo -E ./scripts/profile_packed_stage6_ncu.sh
```

The script validates forward and inverse execution for both packed variants,
then profiles v0.6, vector d6, vector d7, packed vector16, and packed lane32
with base clock control. It flushes caches across replay passes and also
collects L1/L2 hit rates. Results are written to
`results/ncu_packed_stage6_lane32/{summary.csv,analysis.md}`; the earlier
vector16-only capture remains in `results/ncu_packed_stage6/`. Promotion requires the
lane32 point to preserve d7's sector reduction and improve fixed-clock time;
the global selector is not changed by the script.

After the lane32 event schedule screen, confirm its three neighboring CTA
splits against v0.6 and d7:

```bash
sudo -E ./scripts/profile_packed_stage6_lane32_schedule_ncu.sh
```

This profiles `84:76`, `85:75`, and `86:74` with base clocks and flushed replay
caches. It writes
`results/ncu_packed_stage6_lane32_schedule/{summary.csv,analysis.md}` and only
marks a point selector-eligible when it beats both fixed-clock controls.

The next physical-core comparison replaces four sequential warp-register rows
with one shared radix-4 packet:

```bash
sudo -E ./scripts/profile_packet_shared_radix4_ncu.sh
```

It compares v0.6, d7, lane32, and packet128 at base clock. Besides time and
sectors, it collects ADU/ALU/CBU/LSU/XU instruction-pipe counts plus global and
shared SASS instruction classes. The generated report is
`results/ncu_packet_shared_radix4/analysis.md`.

To compare aggregate handoff, per-packet polling, and identity-preserving
wave-bitmap readiness at batch 16, 32, and 64, run:

```bash
sudo -E ./scripts/profile_packet_streaming_ncu.sh
```

This writes nine raw captures, `summary.csv`, and `analysis.md` under
`results/ncu_packet_readiness_modes/`. Per-packet uses the selected
`ready_window=4/2/16`; wave-bitmap uses `16/8/8` for batch 16/32/64. The
validated unpadded per-packet control remains in
`results/ncu_packet_polling_backoff/`; the rejected padded capture is retained
in `results/ncu_packet_streaming_padded_staging/`. The preflight verifies
forward and inverse execution for both publication modes before privileged
profiling begins.

After the readiness capture, the same command profiles the current
stage-0/1 compute-layout ablation under
`results/ncu_packet_compute_layout/`: aggregate, bitmap interleaved-row, and
bitmap warp-row at batch 16/32/64. The script verifies both bitmap layouts in
forward and inverse directions. The completed capture confirms the mapping
model: warp rows reduce online global-load sectors by about 8.7% and
shared-load conflicts by about 19%, with 1.6%-2.1% fewer LSU instructions.
Warm CUDA-event timing remains neutral at batch 32/64, so the candidate is an
attribution control rather than a dispatch default. Service-weight and
wave-barrier follow-ups are recorded in `results/packet_warp_rows_weights/`
and `results/packet_folded_barriers/`; neither exposes a remaining pipeline
imbalance large enough to change selection.

`results/ncu_packet_streaming_q_frontier/` is a rejected synchronization
ablation. It showed that aggregating notifications reduces CBU work, but clean
rebuilds exposed a cross-producer publication race. Do not use its timing in a
selector or headline comparison.

The parameterized three-level diagnostic compares the mature packet128
`10+10` codelet, the historical `7+7+6` wave, `6+6+8`, and a finer
128-thread/four-row task:

```bash
sudo -E ./scripts/profile_three_level_partitions_ncu.sh
```

It writes four fixed-clock captures plus `summary.csv` under
`results/ncu_three_level_partitions/`. This command is retained for later
physical-core attribution. First run the `M/G` Event screen:

```bash
./scripts/benchmark_resident_execution_groups.sh
```

It holds each logical partition fixed while changing resident edge lowering.
Only the Event winners should proceed to matched NCU attribution; a
fixed-three-level NCU result cannot select the architecture-level decomposition.

The selected matched capture is:

```bash
sudo -E ./scripts/profile_resident_execution_groups_ncu.sh
```

It profiles `M4/G4 generic`, `M4/G2 generic`, `M4/G2 dataflow-radix4`, and
the physically equivalent `M2/G2 dataflow-radix4` at batch 1/16. The first
pair attributes boundary materialization, the second attributes physical-core
quality, and the last pair checks that logical M does not change counters once
the physical execution schedule is identical.

The completed V100 capture is in
`results/ncu_resident_execution_groups/analysis.md`. G4-to-G2 lowering removes
2.56x--2.73x load sectors and 4.91x--5.01x warp instructions. Replacing the
generic G2 unit with `dataflow-radix4` removes another 2.62x--2.64x load
sectors, reduces registers from 72 to 40, and raises active warps by
1.93x--1.99x. Physically identical M2/M4 points differ by at most 1.27% in
sector/instruction counters, so no further logical-M NCU sweep is justified.
