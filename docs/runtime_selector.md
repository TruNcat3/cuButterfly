# Runtime Mapping Selector

The runtime selector turns the measured target-device design space into a plan
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
reports the calibration target, selected implementation, selection confidence,
predicted kernel time, and hardware-oriented reason. Selection
happens once during plan construction; `execute_async` remains allocation-free
and contains no selection branch. The runtime registry is the only source of a
calibrated decision. A record is eligible only when its hardware identity,
compile-policy fingerprint, operator semantics, length, batch, direction,
layout, and strides match the request exactly. Among eligible records the
selector chooses the lowest measured kernel time.

When no exact record exists, the portable inventory still validates and builds
an executable mapping, but it is reported as `unmeasured-feasible` with
`predicted_kernel_ms = 0`. This is an explicit coverage signal, not an
interpolated performance estimate. Installation calibration must measure every
FFT semantic stratum that should be auto-selected, including requested length
and batch ranges. Records from another compile policy (`precompiled`,
`on-demand`, or `research`) are not reused because generated code and resource
behavior can differ.

NTT uses the same contract through `PlanConfig::auto_select` and
`Plan::selection()`.

The same measured-first rule applies to every butterfly family.  The
operator-specific part is the candidate inventory, not the evidence rule:
FFT, NTT, FWHT, zeta, and structured 2x2 requests all require an exact
hardware/compile/semantic record before a mapping is called calibrated.  A
rank-one request uses the normal `auto` lowering because `xd` denotes a
multidimensional lowering in the public API.  Rank two and higher requests
may use `xd` or `composite` according to the manifest.  The benchmark runner
keeps that execution mode in each case record, so a rank-one FFT cannot be
silently rejected by an unconditional xD flag.

For a bounded installation or replay, set `CUBUTTERFLY_SEARCH_BUDGET` in the
workload manifest.  This bounds candidate enumeration while retaining the
measured result and its coverage metadata.  A full installation can set the
budget to zero explicitly; a quick experiment should use a positive budget
and report it with the timing rows.

## Historical V100 Coverage Example

The following table is a historical coverage example, not a portable default.
An A100, H100, or another device must populate its own exact registry records
under the same contract before those rows are labeled calibrated.

| Operator | Calibrated semantics | Lengths |
|:--|:--|:--|
| FFT | FP32, forward, contiguous, in-place | logN 8, 14, 18, 20 |
| FWHT | FP32, forward, contiguous, out-of-place | logN 8, 15, 20 |
| Legacy XOR-zeta name | uint32, forward, contiguous, out-of-place | logN 8, 20 |
| NTT | 32-bit natural or 64-bit natural/bit-reversed forward contracts | logN 12, 16, 20 |

Canonical `subset-zeta` and `superset-zeta` use the same explicit mapping
families but are not labeled calibrated until their new sweep is recorded.

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

The offline calibration tools may interpolate stage-service curves for search
and candidate ranking. Those artifacts are not consumed directly by the C++
runtime selector. A runtime decision is labeled calibrated only after its
complete mapping has been measured and promoted to the hardware registry.

The CLI exposes the same path:

```bash
./build/cubutterfly_bench --auto-select --operator fwht --precision fp32 \
  --placement out-of-place --logN 15 --batch 16 --verify

./build/cuntt_bench --auto-select --logN 16 --batch 16 --verify
```

CSV output includes both resolved physical parameters and selection metadata,
so an automatic run can be reproduced later as an explicit mapping.
