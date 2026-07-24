# Next Phase: v0.3.0

## Objective

Turn cuButterfly's measured V100 design space into a counter-calibrated mapping
selector. The result must predict a small set of efficient mappings from the
operator contract, shape, and GPU service profile, while keeping the local
butterfly computation unit replaceable.

This is the next architecture-level step. Adding more isolated fast kernels is
useful only when it fills a missing candidate class or tests a selector
hypothesis.

## Starting Point

- `v0.2.0` is the archived V100 comprehensive baseline on `main`.
- The new orthogonal suite contains 184 cases and 920 process trials.
- Length and batch are now independent axes; fixed total points are no longer
  used to infer saturation.
- The scan establishes real mapping crossovers, including FWHT `logN=15`, and
  distinct FFT launch-limited and throughput-limited regions.
- Cross-GPU rows are still placeholders.
- P1 is implemented over 49 internal multi-candidate shapes; it reaches 93.88%
  top-1 accuracy and 1.0049x geometric-mean regret.
- P2 is complete with 120 matching-protocol Dao FHT/GPU-NTT samples.
- P0 is complete with 22 profiled implementation-shape pairs and 37 kernels;
  the attribution separates grid coverage, residency, and useful work/warp.

## Work Packages

### P0: Mechanism Attribution

Run `scripts/profile_scaling_crossovers_ncu.sh` and reduce the output with
`scripts/summarize_ncu.py`. Analyze counters at these boundaries:

| Question | Cases | Required attribution |
|:--|:--|:--|
| Why does direct FFT need batch concurrency? | FFT `logN=14`, batch 16/256/1024 | CTA waves, occupancy, registers, launch amortization |
| Why does cuButterfly saturate early? | FFT `logN=18`, batch 2/16/64 | DRAM/L2 service, active warps, issue stalls, composition cost |
| Why is the long FFT ceiling lower? | FFT `logN=20`, batch 2/8/16 | the same services plus prefix/suffix imbalance |
| Why does the preferred unit change? | FWHT `logN=15`, batch 4/16 | occupancy and exchange cost for online versus warp-register |

Output: `results/ncu_scaling_crossovers/summary.csv` plus a counter-attribution
report. A performance explanation is accepted only when its predicted trend
matches at least two measured batch points.

### P1: Calibrated Selector

Implement a selector over the existing descriptor
`M=(Us,Ts,Ud,Td,Ub,Tb,Hs,Rs,Rd,Rb,L,F,Q)`. Its first version should:

1. reject illegal candidates from static resource and semantic constraints;
2. estimate launch concurrency, residency, data movement, and synchronization;
3. rank mapping families separately from replaceable processing units;
4. return top-k candidates with human-readable limiting factors;
5. fall back to measurement when the model confidence is low.

Evaluate by holding out complete `(operator, logN, batch)` shapes. Report top-1
accuracy, top-3 recall, geometric-mean regret, worst regret, and search-space
reduction. The initial V100 gate is top-3 recall at least 90% with geometric
mean regret at most 5%; otherwise keep the selector labeled experimental.

### P2: Baseline Refresh

Rerun Dao-AILab FHT and GPU-NTT with the comprehensive-suite controls. Every
paper-facing comparison must record library revision, modulus or precision,
direction, output order, placement, batch, clock policy, warmups, repetitions,
process trials, and correctness result.

XOR-zeta remains an internal architecture ablation until a maintained tuned
external implementation with matching semantics is available.

### Future: Cross-GPU Transfer

This work is deferred until the repository moves to a newer GPU host. After
P0-P2 are frozen on V100, capture one newer GPU profile, generate a
reduced candidate set without using its timing results, and then measure it.
The primary portability metric is prediction regret before recalibration; a
second result may show regret after a small calibration set.

## Decision Gates

| Gate | Continue when | Reconsider when |
|:--|:--|:--|
| G1: counters | bottlenecks explain observed batch trends | counters contradict the proposed resource model |
| G2: selector | top-3 recall and regret meet the V100 target | exhaustive search remains necessary for common shapes |
| G3: baselines | matching semantics and protocols are reproducible | library contracts cannot be aligned |

Cross-GPU transfer is intentionally not a `v0.3.0` decision gate.

## Deferred Scope

- production-compatible cuFFT replacement API;
- broad Tensor Core FFT claims without end-to-end accuracy and composition
  accounting;
- universal superiority claims;
- additional operators that do not test a new dependency or hardware service;
- large multidimensional or non-power-of-two transforms before the selector is
  validated on the current regular graph family.

## Immediate Commands

```bash
# Repository and generated-manifest checks
cmake --build build -j
/usr/bin/ctest --test-dir build --output-on-failure

# Recollect counters only when an administrator-enabled run is required
sudo -E ./scripts/profile_scaling_crossovers_ncu.sh
sudo chown -R "$USER:$USER" results/ncu_scaling_crossovers

# The profiling script also emits this summary; rerun explicitly if needed
python3 scripts/summarize_ncu.py results/ncu_scaling_crossovers/*.csv \
  --output results/ncu_scaling_crossovers/summary.csv

python3 scripts/analyze_scaling_ncu.py \
  results/ncu_scaling_crossovers/summary.csv \
  --output results/ncu_scaling_crossovers/attribution.csv \
  --markdown results/ncu_scaling_crossovers/attribution.md

# Regenerate every derived V100 analysis from immutable raw records
./scripts/reproduce_v100_analysis.sh
```
