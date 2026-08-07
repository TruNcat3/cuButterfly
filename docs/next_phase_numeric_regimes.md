# Numeric-Regime Mapping Study

## Research Question

The next phase does not begin by moving the current V100 winner table to a new
GPU. It first asks a more fundamental question on fixed hardware:

> How do numeric representation, arithmetic semantics, workload shape, and
> physical processing-unit cost move the resource cliffs and change the best
> space-time mapping?

The current evidence already shows that no single answer is sufficient. FP32
and FP64 select different radix/thread combinations, broadcast and per-stage
coefficients change register pressure, batch changes launch and saturation
behavior, and resident Structured 2x2 crosses shared-memory and register
boundaries between `logN=13` and `logN=15`. The goal is to turn those examples
into a conditional method rather than a larger lookup table.

## Independent Descriptors

The study separates four descriptor groups:

| Group | Controlled variables | Hardware demand affected |
|:--|:--|:--|
| numeric representation | FP16/BF16/FP32/FP64, integer/modular word width, accumulator width | register bytes, issue mix, conversion and memory traffic |
| arithmetic semantics | FFT twiddles, NTT modulus/reduction, FWHT signs, Structured coefficient reuse or per-stage tables | coefficient traffic, temporary lifetime, integer/FP pipelines, instruction count |
| workload shape | `logN`, batch, direction, normalization, stride, placement, output order | grid waves, launch amortization, boundary traffic and layout work |
| realization | `Us/Ts/Ud/Td`, decomposition, radix, threads, EPT, residence, processing unit | CTA resources, occupancy, synchronization and global exchange |

Hardware properties are held fixed during this phase. This lets the experiment
identify which changes come from the workload and numeric contract before GPU
generation is introduced as another variable.

## Hypotheses

1. Resource cliffs are predictable from allocated state and work per CTA, but
   their performance effect is conditional on batch-derived grid waves.
2. Equal byte width is not sufficient: coefficient entropy/reuse and arithmetic
   pipeline mix can change the winner even when occupancy is unchanged.
3. Mapping preference is piecewise stable within regimes separated by launch,
   occupancy, register, shared-memory, and bandwidth boundaries.
4. A selector trained on descriptors and boundary neighborhoods will generalize
   to held-out numeric/workload regimes better than one trained on shape IDs or
   uniformly sampled points.

## Experiment Design

The first pass uses an orthogonal screening matrix to estimate main effects and
important interactions. It varies one numeric or semantic contract at matched
length/batch points while retaining the same candidate mappings. The second
pass concentrates measurements around observed or predicted boundaries instead
of expanding every Cartesian combination.

Each retained row records both performance and explanatory features:

- median and trial range, effective throughput, and external-baseline ratio
  when semantics can be matched;
- allocated registers/shared memory, resident CTA/warps, occupancy upper bound,
  grid waves, concurrency fraction, and limiting resource;
- DRAM traffic, instruction mix, replay/stalls, and shared-bank conflicts for
  a small counter-attribution subset;
- selected mapping/core, runner-up distance, and whether the point lies on a
  measured or predicted cliff.

Comparisons must not mix natural and bit-reversed output, normalized and
unnormalized inverse, different accumulation precision, or different placement
contracts. Those are model inputs, not benchmark noise.

## Work Packages

1. Extend the manifest schema and summarizers with the descriptor groups above.
2. Run matched precision/arithmetic controls for FFT, NTT, FWHT, and Structured
   2x2 at low, crossover, and saturated batches.
3. Detect change points in length and batch, then launch focused decomposition,
   radix, thread, EPT, coefficient-policy, and core scans around each boundary.
4. Capture NCU only for winner changes that the static resource model cannot
   explain.
5. Evaluate the selector by holding out complete precision, arithmetic-policy,
   batch-region, and operator/core regimes.

## Definition Of Done

This phase is complete when:

1. every claimed crossover has matched semantics, repeated CUDA-event timing,
   an explicit limiting-resource classification, and a nearby control point;
2. the paper-facing result is a regime/boundary model, not a list of per-shape
   winners;
3. held-out-regime top-k recall and performance regret are reported, including
   negative cases where the descriptors are insufficient;
4. the runtime selector consumes the new descriptors and exposes the reason for
   its choice;
5. one reduced matrix can later be rerun on a second GPU to test transfer of the
   method without making cross-GPU measurement the present optimization target.

Cross-GPU validation remains necessary for a portability claim, but it follows
this phase. Its purpose will be to test whether the conditional model transfers,
not merely whether V100 tile choices happen to remain fast.

## Implemented Harness

The v0.5 branch implements the first fixed-V100 screening harness. The numeric
contracts are FP16 and BF16 storage with either native-width or FP32
accumulation, plus FP32, FP64, uint32 zeta, and uint32/uint64 NTT controls.
FFT, FWHT, and Structured 2x2 use the same floating-point length set and mapping
candidates; the integer cases use matched radix candidates. Each comparison group shares
an exact batch; batch anchors are generated around the predicted `0.5`, `1`,
`2`, and `4` grid-wave boundaries and include the adjacent integer batches at
the one-wave boundary.

Every generated case carries storage, compute, and accumulator widths;
arithmetic/coefficient semantics; estimated register and shared-memory cost;
resident CTA/warp limits; grid waves; L2 working-set ratio; and limiting
resource. Every workload cell also has an explicit external-baseline status.
In particular, BF16 cuFFT is recorded as unsupported on SM70 rather than being
silently omitted, and V100 BF16 native arithmetic is marked as emulated.

Generate the manifest on demand so the expanded 5,690-case matrix is not a
large checked-in artifact:

```bash
python3 scripts/generate_numeric_regime_suite.py \
  --output results/v100_numeric_regime_suite.json

python3 scripts/run_comprehensive_suite.py \
  --manifest results/v100_numeric_regime_suite.json \
  --mode quick --output results/v100_numeric_regime_quick_raw.csv
```

The quick tier contains 736 candidate cases and is intended for screening.
Use `--only <cell-or-case-token>` for focused boundary scans and `--resume` for
long runs. Promote selected neighborhoods to `--mode full`; do not begin with
all full cases.

Analyze repeated measurements and evaluate complete-regime generalization:

```bash
python3 scripts/analyze_numeric_regime_cliffs.py \
  results/v100_numeric_regime_quick_raw.csv \
  --manifest results/v100_numeric_regime_suite.json \
  --events results/v100_numeric_regime_events.csv \
  --regimes results/v100_numeric_regimes.json \
  --report results/v100_numeric_regime_quick_report.md

python3 scripts/evaluate_numeric_regime_selector.py \
  results/v100_numeric_regime_quick_raw.csv \
  --manifest results/v100_numeric_regime_suite.json \
  --records results/v100_numeric_selector_holdouts.csv \
  --metrics results/v100_numeric_selector_metrics.json

python3 scripts/build_numeric_piecewise_selector.py \
  results/v100_numeric_regime_quick_raw.csv \
  --manifest results/v100_numeric_regime_suite.json \
  --followups results/v100_numeric_confirmed_followup_analysis.csv \
  --calibration-raw results/v100_numeric_confirmed_followup_raw.csv \
  --calibration-manifest results/v100_numeric_confirmed_followup_suite.json \
  --model results/v100_numeric_piecewise_selector.json \
  --records results/v100_numeric_piecewise_holdouts.csv \
  --metrics results/v100_numeric_piecewise_metrics.json \
  --report results/v100_numeric_piecewise_report.md
```

A measured winner change is confirmed only when the median advantage is at
least 2% and the trial ranges do not overlap. Runtime-selector promotion also
requires coverage of all four complete holdouts (numeric contract, arithmetic
policy, batch region, and operator), top-3 recall at least 95%, geometric-mean
regret at most 1.02, and worst regret at most 1.10. Otherwise the result remains
`measurement-required`.

## First Quick Screen

The first V100 quick run completed all 736 candidates and 2,208 timed samples.
It produced 294 comparable regimes, 29 confirmed and 67 ambiguous mapping
crossovers, and 41 cases where changing the numeric pipeline changed the best
mapping at fixed operator/shape. Complete-regime evaluation retained the
measured winner in the top three for every evaluated holdout, but geometric
mean regret was 1.047 and worst regret was 2.078. The selector therefore remains
`measurement-required`; no runtime table is promoted from this screen.

The complete interpretation and evidence paths are in
[V100 Numeric-Regime Quick Screening](../results/v100_numeric_regime_quick_report.md).

The 29 initially confirmed crossover neighborhoods were then repeated with
five trials, 1000 warmups, 100 timed iterations, and adjacent controls. Twenty-
three reproduce in the same direction, three reverse with separated trial
ranges, and three no longer have separated ranges. All nine length-axis events
reproduce; all six unstable events are batch-axis boundaries. The focused
evidence is in [V100 Numeric-Regime Confirmed Follow-Ups](../results/v100_numeric_confirmed_followup_report.md).

Reproduce the focused timing and generate the reduced NCU capture with:

```bash
python3 scripts/generate_numeric_followup_suite.py \
  --output results/v100_numeric_confirmed_followup_suite.json
python3 scripts/run_comprehensive_suite.py \
  --manifest results/v100_numeric_confirmed_followup_suite.json --mode full \
  --output results/v100_numeric_confirmed_followup_raw.csv
python3 scripts/analyze_numeric_followups.py \
  results/v100_numeric_confirmed_followup_raw.csv \
  --manifest results/v100_numeric_confirmed_followup_suite.json \
  --output results/v100_numeric_confirmed_followup_analysis.csv \
  --summary results/v100_numeric_confirmed_followup_summary.json \
  --report results/v100_numeric_confirmed_followup_report.md

python3 scripts/generate_numeric_boundary_ncu.py
sudo -E scripts/profile_numeric_boundaries_ncu.sh
```

The generated NCU script contains 12 paired profiles for only the three
reversed and three statistically unconfirmed batch events. Stable length
crossovers are not profiled again unless a later model cannot explain them.

The paired capture is now complete. It separates three mechanisms:

1. radix-4 often wins by reducing warp instruction work, even when it increases
   register footprint or synchronization stall;
2. the Structured 2x2 reversal is online-composition overhead: the online path
   executes 1.745x the hierarchical warp instructions and raises barrier stall
   from 13.5% to 34.9%;
3. FP64 `logN=8,batch=961` is the only unstable pair that changes the actual
   resident-CTA capacity bound, leaving a 1.8% NCU near-tie after trading fewer
   instructions for lower residency and higher barrier pressure.

The detailed counter table is [Numeric Boundary NCU Attribution](../results/v100_numeric_boundary_ncu_analysis.md).

## Piecewise Selector With Abstention

The NCU attribution rules out a single monotonic explanation for batch
crossovers. A radix can execute fewer instructions while consuming more
registers or barrier cycles, and the dominant term changes as the grid crosses
wave boundaries. The selector therefore does not extrapolate a global nearest
neighbor model across numeric contracts or force a decision at every batch.

For an exact `(operator, precision, accumulation, logN)` contract, a quick
anchor is stable only when its winner has at least a 15% median advantage and
its trial range does not overlap the runner-up. A five-trial full anchor uses a
5% threshold with the same separation requirement. An interval is automatic
only when its two batch anchors are stable and name the same winner. Different winners,
weak/overlapping evidence, an unseen contract or length, and out-of-range batch
values all return `measurement-required` with a candidate set.

The first quick-only model auto-selected 40 of 294 held-out shapes (13.61%). An
adaptive pass then promoted weak, non-crossover anchors to full timing:

```bash
python3 scripts/generate_numeric_coverage_suite.py \
  --calibration-raw results/v100_numeric_confirmed_followup_raw.csv \
  --calibration-manifest results/v100_numeric_confirmed_followup_suite.json \
  --followups results/v100_numeric_confirmed_followup_analysis.csv \
  --output results/v100_numeric_coverage_suite.json
python3 scripts/run_comprehensive_suite.py \
  --manifest results/v100_numeric_coverage_suite.json --mode full \
  --output results/v100_numeric_coverage_raw.csv --skip-verify
python3 scripts/analyze_numeric_coverage.py \
  results/v100_numeric_coverage_raw.csv \
  --manifest results/v100_numeric_coverage_suite.json \
  --output results/v100_numeric_coverage_analysis.csv \
  --summary results/v100_numeric_coverage_summary.json \
  --report results/v100_numeric_coverage_report.md
```

Of 99 promoted shapes, 66 retain the quick winner with a separated gap of at
least 5%, 33 remain near-ties, and none reverse. After overlaying those full
measurements, leave-one-batch-out validation auto-selects 86 of 310 shapes
(27.74%). Top-1 agreement is 98.84%, geometric-mean regret is 1.00013x, and
worst regret is 1.01112x. The regret gates pass for the calibrated subset, while
the global selector remains measurement-required.
This distinction is important: the result validates a performance-safe stable
region, not a universal mapping model. See
[V100 Numeric Piecewise Selector](../results/v100_numeric_piecewise_report.md)
and [Coverage Expansion](../results/v100_numeric_coverage_report.md).

Rebuild the final model by adding the coverage suite as a second calibration
pair to the earlier `build_numeric_piecewise_selector.py` command:

```bash
python3 scripts/build_numeric_piecewise_selector.py \
  results/v100_numeric_regime_quick_raw.csv \
  --manifest results/v100_numeric_regime_suite.json \
  --followups results/v100_numeric_confirmed_followup_analysis.csv \
  --calibration-raw results/v100_numeric_confirmed_followup_raw.csv \
  --calibration-manifest results/v100_numeric_confirmed_followup_suite.json \
  --calibration-raw results/v100_numeric_coverage_raw.csv \
  --calibration-manifest results/v100_numeric_coverage_suite.json \
  --model results/v100_numeric_piecewise_selector.json \
  --records results/v100_numeric_piecewise_holdouts.csv \
  --metrics results/v100_numeric_piecewise_metrics.json \
  --report results/v100_numeric_piecewise_report.md
```
