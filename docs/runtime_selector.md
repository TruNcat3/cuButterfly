# Runtime Mapping Selector

The runtime selector turns the measured V100 design space into a plan
configuration. An application specifies the transform semantics and workload;
plan construction resolves the mapping family, decomposition, processing unit,
and physical parameters before any execution is submitted.

```cpp
cuntt::ButterflyConfig config;
config.op = cuntt::ButterflyOperator::Fwht;
config.precision = cuntt::ButterflyPrecision::Fp32;
config.placement = cuntt::ButterflyPlacement::OutOfPlace;
config.log_n = 15;
config.batch = 16;
config.auto_select = true;

cuntt::ButterflyPlan plan(config);
const auto& resolved = plan.config();
const auto& decision = plan.selection();
```

`resolved` is the physical configuration used by the kernels. `decision`
reports the calibration target, selected implementation, interpolation
confidence, predicted kernel time, and hardware-oriented reason. Selection
happens once during plan construction; `execute_async` remains allocation-free
and contains no selection branch.

NTT uses the same contract through `PlanConfig::auto_select` and
`Plan::selection()`.

## V100 Coverage

| Operator | Calibrated semantics | Lengths |
|:--|:--|:--|
| FFT | FP32, forward, contiguous, in-place | logN 8, 14, 18, 20 |
| FWHT | FP32, forward, contiguous, out-of-place | logN 8, 15, 20 |
| Legacy XOR-zeta name | uint32, forward, contiguous, out-of-place | logN 8, 20 |

Canonical `subset-zeta` and `superset-zeta` use the same explicit mapping
families but are not labeled calibrated until their new sweep is recorded.
| NTT | 32-bit natural or 64-bit natural/bit-reversed forward contracts | logN 12, 16, 20 |

The selector refuses unsupported semantics, lengths, builds, and devices. It
does not silently reuse V100 thresholds on another GPU. FFT auto-selection
requires a build with cuFFTDx enabled.

## Evidence Flow

The runtime implementation has two generated evidence inputs:

1. `config/v100_fft_dispatch.json` records the measured long-FFT mapping and
   batch thresholds.
2. `scripts/generate_runtime_selector.py` extracts latency anchors for all
   covered operators from `results/v100_scaling_full_summary.csv` during CMake
   configuration.

Latency is interpolated in log-batch/log-latency space with slope constrained
to `[0,1]`, matching the validated offline selector. Calls below or above the
measured batch range are labeled `extrapolated-low` or `extrapolated-high` in
`SelectionInfo`; they are not presented as measured values. `calibrated-anchor`
means that the implementation family has an archived measurement at that
batch. It is a selector estimate, not an execution-time guarantee; the finer
long-FFT dispatch may select another physical mapping within that family.

The CLI exposes the same path:

```bash
./build/cubutterfly_bench --auto-select --operator fwht --precision fp32 \
  --placement out-of-place --logN 15 --batch 16 --verify

./build/cuntt_bench --auto-select --logN 16 --batch 16 --verify
```

CSV output includes both resolved physical parameters and selection metadata,
so an automatic run can be reproduced later as an explicit mapping.
