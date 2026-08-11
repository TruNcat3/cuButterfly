# Experimental Results

## 1. Measurement Policy

The checked-in results use resident kernel time unless explicitly labelled
end-to-end. Comparisons pin GPU, CUDA version, workload shape, precision or
modulus, direction, normalization, placement, output order, warmups, repeats,
and external-library revision. Raw and summarized CSV files are retained.

The reference system is a Tesla V100-SXM2-16GB (`sm_70`, 80 SMs, 6 MiB L2)
with CUDA 11.8. Cross-GPU rows are placeholders until measured.

Performance is reported only after correctness preflight. Ratios are computed
from matched resident CUDA-event times; a value above `1.0x` means
cuButterfly has higher throughput. Rows below the established `0.020 ms`
timing floor remain useful launch-latency observations but are not counted as
stable performance claims.

## 2. What The Results Establish

The experiments test the mapping model in three steps. First, cross-operator
and processing-unit scans test whether the same four-factor vocabulary remains
valid when arithmetic changes. Second, orthogonal length/batch and decomposition
scans test whether hardware-selected factors move as concurrency, residence,
and local-unit requirements change. Third, NCU fixed-mapping comparisons test
whether the proposed service bottleneck changes in the direction predicted by
an optimization. A fast isolated row is not sufficient evidence for the
architecture claim unless its semantics and mechanism are also controlled.

The controlled full-suite cross-section and its evidence boundaries are
reported in [V100 Comprehensive Results](comprehensive_v100_results.md). It is
the current authority for claims spanning multiple lengths, precisions,
semantics, and runnable FFT libraries; the focused experiments below retain
their original protocols.

The [V100 Library/Base/Search Comparison](v100_three_way_comparison.md) is the
current authority for separating the gain from mapping and processing-unit
search from the remaining position against cuFFT, Dao FHT, and GPU-NTT. The
[Numeric-Regime Mapping Study](next_phase_numeric_regimes.md) is the authority
for FP16/BF16 and integer regime boundaries, focused full-timing confirmation,
paired NCU attribution, and selector abstention. Neither replaces the broader
comprehensive semantic matrix.

The follow-up [V100 Length/Batch Scaling](v100_scaling_results.md) experiment
holds transform length fixed while varying batch, exposing saturation points
and mapping crossovers hidden by constant-total-point comparisons.

The [V100 Mapping Selector](v100_mapping_selector.md) performs
leave-one-complete-shape-out validation over the internal multi-candidate rows.
The [matching-protocol external refresh](v100_external_baselines.md) is the
current authority for Dao FHT and GPU-NTT comparisons with matching semantics.
Targeted [V100 counter attribution](v100_ncu_attribution.md) explains the
length/batch saturation and FWHT mapping crossover in hardware-service terms.
The focused [Structured 2x2 V100 Results](structured_2x2_v100_results.md)
experiment tests a stage-parameterized dense processing unit across the same
mapping families. Its follow-up generated register codelet separates mapping
and transport gains from the remaining dense-pair arithmetic cost.

The v0.6 [General-Shape V100 Selection Results](general_shape_results.md) keep
the initial Bluestein-only diagnostic separate from the five-trial selector
follow-up. The first lowering reached only 0.105x-0.179x of direct cuFFT; the
new composition selector reaches FFT parity through direct-core abstention and
substantially improves NTT/FWHT/Structured physical-core choices. Embedded
boundaries remain explicit and must not be merged with bare-core library
comparisons. The follow-up removes contiguous output scatter and fuses input
for selected warp-register FWHT: `32767 -> 32768,batch=256` reaches 0.11759 ms,
within 1.01x of the archived pre-padded Dao core. Structured input fusion is a
measured negative candidate and retains the pack path.

The evidence chain is therefore:

```text
same graph, different operators
    -> different mapping winners
    -> hardware services explain the crossover
    -> a small calibrated candidate set recovers the measured winner
```

The leave-one-complete-shape-out selector reaches `93.88%` top-1 accuracy,
`100%` top-3 recall, and `1.0049x` geometric-mean regret across the internal
NTT/FWHT/XOR-zeta candidate rows. In the separate 48-candidate FFT pipeline
experiment, static scoring alone has `1.0762x` geometric-mean regret, while a
small measured calibration set reduces top-3 regret to `1.0007x`. This negative
static-model result is why the method combines hardware constraints with
calibration instead of claiming a universal analytic tile rule.

### Common Mapping Across Operators

At `N=256`, a controlled eight-warp stage pipeline selects the interior
`Us=4` point for NTT, FFT, FWHT, and XOR-zeta, with 1.90x-2.40x speedup over
`Us=1` inside that family. Its absolute performance is below specialized
temporal kernels, so the result validates the unfolding trade-off rather than
the queue-based implementation.

An expanded search selects different local realizations:

| Operator | Best local realization | Median | Reference |
|:--|:--|--:|:--|
| FWHT FP32 | radix-4 temporal, 128 threads | 0.051292 ms | 327.09 Gbutterfly/s |
| XOR-zeta uint32 | radix-4 temporal, 128 threads | 0.050156 ms | 334.50 Gbutterfly/s |
| FFT FP32 | radix-4 temporal, 128 threads | 0.110346 ms | 75.9% cuFFT throughput |

This operator dependence is expected: the graph is common, while arithmetic,
coefficient traffic, and local transport change the hardware balance.

### Processing-Unit Reuse

The Dao-derived warp-register FWHT unit reaches 92%-100% of the same-machine
Dao FP32 baseline from `logN=8..15`, while remaining behind the common
cuButterfly stride, placement, inverse, and batching interface.

The generated FP32 CTA DFT8 reaches 92.4% of cuFFT throughput at `N=8`. Its
selected spatial factor changes with length, and it crosses below the existing
shared radix-8 schedule after `N=64`. This identifies layout synchronization as
a composition cost rather than presenting the codelet as a universal winner.

The long FFT results make the separation more explicit. Logical stage
decomposition count and physical execution-group count are independent: adjacent
segments may be fused when a CTA can retain their state. Segment length,
prefix/suffix CTA threads, elements per thread, coefficient policy, and boundary
layout are selected independently. The same architecture can therefore use a
scalar unit for coverage and a cuFFTDx unit where its local precision and size
contract is legal.

### Online Reordering

At `logN=20`, fusing the boundary permutation and retaining the suffix in one
CTA improves the prior hierarchy by 1.52x for FWHT/XOR-zeta and 1.87x for FFT.
Nsight Systems shows the FFT suffix shrinking from ten kernels to one. The
permuted prefix itself becomes about 25% slower, which confirms that online
reordering is beneficial only when the eliminated downstream work is larger
than its transaction cost.

### FFT Against cuFFT And VkFFT

The latest matching-protocol FP32 long-transform comparison uses five process
trials, 1000 warmups, 100 timed repetitions, randomized execution order, and
correctness preflight:

| Shape | cuButterfly | cuFFT | VkFFT | cuButterfly/cuFFT |
|:--|--:|--:|--:|--:|
| `logN=18`, batch 16 | 0.195942 ms | 0.211098 ms | 0.201103 ms | 1.077x |
| `logN=20`, batch 4 | 0.214733 ms | 0.210330 ms | 0.206520 ms | 0.979x |

The broader mapping scan finds low-batch points above cuFFT at both lengths,
but the saturated `logN=20` batch-8/16 rows remain at `0.962x/0.954x`. This is
a narrow remaining ceiling, not evidence that every FFT shape exceeds cuFFT.
The counters attribute the long-path gap to useful work per resident warp and
local exchange rather than missing occupancy alone.

The expanded FP64 experiment covers all two-segment totals `logN=14..18` whose
segments are each `logN=7..9`. It independently selects decomposition and both
CTA/EPT dimensions, then alternates cuButterfly and cuFFT execution order over
five trials. Among 25 stable length/batch shapes, 20 have a higher
cuButterfly median and 19 have non-overlapping faster trial ranges. All 13
stable `logN=17/18` shapes are faster.

| Shape with `2^22` total points | Split and physical mappings | cuButterfly | cuFFT | Ratio |
|:--|:--|--:|--:|--:|
| `logN=14`, batch 256 | `7+7`, `128/4 + 128/4` | 0.340173 ms | 0.344433 ms | 1.013x |
| `logN=15`, batch 128 | `7+8`, `128/4 + 128/8` | 0.338852 ms | 0.342753 ms | 1.012x |
| `logN=16`, batch 64 | `8+8`, `128/8 + 128/8` | 0.339364 ms | 0.343040 ms | 1.011x |
| `logN=17`, batch 32 | `8+9`, `128/8 + 256/8` | 0.346307 ms | 0.395213 ms | 1.141x |
| `logN=18`, batch 16 | `9+9`, `256/8 + 256/8` | 0.354877 ms | 0.421970 ms | 1.189x |

At the profiled FP64 `logN=16`, batch-64 point, hoisting invariant global
addresses and advancing XOR-swizzled shared pointers by compile-time strides
reduces prefix warp instructions by 26.8% and prefix replay by 8.3% relative
to the pre-optimization XOR kernel. Total replay becomes 2.5% lower than
cuFFT, although the complete path still executes 1.83x its warp instructions,
3.62x its integer instructions, and about 114x its shared-bank conflicts. The
result validates the predicted address-path mechanism while identifying the
remaining physical-unit cost.

### NTT Against GPU-NTT

With the same V100, 60-bit prime, resident timing, and output semantics:

| Workload | cuButterfly | GPU-NTT | Throughput ratio |
|:--|--:|--:|--:|
| `logN=16`, natural Hybrid2D | 0.274104 ms | 0.291133 ms | 1.062x |
| `logN=18`, natural Hybrid2D | 0.342482 ms | 0.355820 ms | 1.039x |
| `logN=20`, native bit-reversed compact stage | 0.341320 ms | 0.396186 ms | 1.161x |

NCU localizes the original `logN=20` Hybrid2D gap to expanded fused-root
traffic and long-scoreboard stalls. Compact roots remove the native-layout gap;
a required natural-order permutation changes the ranking and is reported
separately.

### Numeric Regimes And Library/Base/Search

The fixed-V100 numeric study contains 185 workload cells. Its 736 quick cases
and 2,208 samples find 29 confirmed and 67 ambiguous winner crossovers. Full-
protocol follow-up keeps 23 directions, reverses three, and leaves three with
overlapping trial ranges. All nine length-axis changes reproduce; all six
unstable events lie on batch boundaries. Twelve paired NCU captures attribute
those cases to instruction/register tradeoffs, online-composition
synchronization, and one FP64 resident-CTA capacity transition.

Adaptive timing of 99 non-boundary anchors produces 66 stable anchors, 33
near-ties, and no reversals. The final piecewise selector deliberately abstains
outside same-winner stable intervals:

| Metric | Result |
|:--|--:|
| evaluated leave-one-batch-out shapes | 310 |
| automatically selected | 86 (27.74%) |
| top-1 agreement in selected region | 98.84% |
| geometric-mean regret | 1.00013x |
| worst regret | 1.01112x |

The global complete-regime model remains `measurement-required`; its 1.047x
geometric-mean and 2.078x worst regret reject universal extrapolation from the
quick screen. The validated result is a bounded stable region with explicit
abstention.

The representative library/base/search matrix then asks where the selected
point stands after search. All local rows use five randomized process trials,
1000 warmups, 100 timed iterations, correctness preflight, and resident CUDA-
event timing. GPU-NTT rows are archived same-V100 measurements with matching
protocol and semantics, but are not interleaved with the current refresh.

| Operator | Shapes | Search/base geomean | Search/library geomean | Faster/parity/slower |
|:--|--:|--:|--:|:--|
| FFT | 4 | 2.931x | 1.015x | 1/3/0 |
| FWHT | 3 | 3.768x | 1.054x | 2/1/0 |
| NTT | 3 | 1.089x | 1.313x | 3/0/0 |
| overall selected matrix | 10 | 2.349x | 1.109x | 6/4/0 |

The status counts use a +/-3% parity band. These ten representative rows show
that mapping/core search closes large fixed-base deficits and reaches parity or
advantage in this matrix; they do not establish universal superiority across
all precisions, lengths, batches, or semantics.

## 3. Remaining Gaps

| Area | Current boundary | Needed evidence or implementation |
|:--|:--|:--|
| FP32 FFT | selected direct and long shapes reach parity or advantage; saturated `logN=20` batch 8/16 remains at `0.962x/0.954x` cuFFT | reduce long-path local exchange and excess useful-work cost without losing residence |
| FP64 FFT | 20/25 stable shapes have higher median throughput; five `logN=14..16` crossover shapes remain 0.1%-3.2% behind | reduce launch and boundary overhead; exhaustive remapping alone did not remove the deficits |
| conditional mapping model | the numeric piecewise selector safely covers 27.74% of held-out batch shapes and abstains elsewhere | expand stable intervals across stride, direction, normalization, coefficient policy, and complete-regime holdouts without relaxing regret gates |
| workload coverage | representative length/batch points are measured, but cliff neighborhoods are uneven | use boundary-focused scans across length, batch, stride, direction, and normalization rather than a uniformly larger grid |
| cross GPU | only V100 fully measured | after the conditional model is established, test whether its descriptors and boundary predictions transfer to another GPU generation |
| automatic selection | V100 calibrated selectors meet current regret gates | add numeric/workload descriptors and validate held-out-regime regret before cross-GPU calibration |
| XOR-zeta baseline | internal comparisons only | pinned same-machine external implementation |
| public API | device-pointer, stream-aware plans and caller-owned workspace are implemented | broaden production compatibility only where demanded by operator coverage |
| numeric coverage | current listed precisions/moduli | additional FFT mixed precision and NTT reduction contracts |

## 4. Result Locations

| Topic | Report | Primary records |
|:--|:--|:--|
| numeric-regime mapping | `next_phase_numeric_regimes.md` | `v100_numeric_regime_quick_raw.csv`, `v100_numeric_confirmed_followup_raw.csv`, `v100_numeric_coverage_raw.csv` |
| numeric boundary NCU | `results/v100_numeric_boundary_ncu_analysis.md` | `ncu_numeric_boundaries/summary.csv`, `v100_numeric_boundary_ncu_analysis.csv` |
| numeric piecewise selector | `results/v100_numeric_piecewise_report.md` | `v100_numeric_piecewise_selector.json`, `v100_numeric_piecewise_metrics.json` |
| library/base/search matrix | `v100_three_way_comparison.md` | `v100_three_way_comparison.csv`, `v100_three_way_metrics.json` |
| comprehensive V100 suite | `comprehensive_v100_results.md` | `comprehensive_v100_full_raw.csv`, `comprehensive_v100_full_summary.csv` |
| orthogonal length/batch scaling | `v100_scaling_results.md` | `v100_scaling_full_raw.csv`, `v100_scaling_full_summary.csv` |
| calibrated V100 selector | `v100_mapping_selector.md` | `v100_mapping_selector_evaluation.csv`, `v100_mapping_selector_metrics.json` |
| scaling-crossover counters | `v100_ncu_attribution.md` | `ncu_scaling_crossovers/summary.csv`, `ncu_scaling_crossovers/attribution.csv` |
| structured 2x2 mapping | `structured_2x2_v100_results.md` | `structured_2x2_v100_raw.csv`, `structured_2x2_v100_summary.csv` |
| refreshed external baselines | `v100_external_baselines.md` | `v100_external_baselines_raw.csv`, `v100_external_baselines_summary.csv` |
| V100 NTT baseline | `v100_initial_results.md` | `hybrid2d_matrix*.csv` |
| NTT vs GPU-NTT | `gpu_ntt_gap_analysis.md` | `fused_vs_gpuntt.csv`, `gpu_ntt_gap_same_modulus.csv` |
| processing units | `processing_unit_design_space.md` | `processing_units_v100_*.csv` |
| cross operator | `cubutterfly_cross_operator_results.md` | `cubutterfly_design_sweep_v100_*.csv` |
| large transforms | `cubutterfly_large_results.md` | `cubutterfly_large_v100_*.csv`, `cubutterfly_online_*.csv` |
| CTA DFT8 | `fft_cta_space_time_results.md` | `fft_cta_*_v100_raw.csv` |
| FP64 length/batch robustness | `results/fp64_robustness_batch_analysis.md` | `fp64_robustness_batch_raw.csv`, `fp64_robustness_batch_summary.csv` |
| FP64 address-path NCU | `results/ncu_fp64_fft/analysis.md` | `ncu_fp64_fft/summary.csv`, `ncu_fp64_fft/analysis.csv` |
| hardware model | `hardware_mapping_methodology.md` | `hardware_capabilities_v100.json`, `v100_compact_mapping_model.csv` |
| NCU | `ncu_profiling.md` | `ncu_summary_logN20.csv`, `ncu_fft_units_analysis.md` |

The detailed reports contain commands, revisions, caveats, and negative
results omitted from this overview.
