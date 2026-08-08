# Research Status: V100 Evidence Closure

## Scope

cuButterfly is a single-GPU research artifact for hardware-mapped space-time
parallelism across regular butterfly computations. V100 is the only fully
measured hardware target. Cross-GPU transfer is explicitly future work and is
not used to support any current portability claim.

## Research Claims And Evidence

| Claim | Evidence | Current result |
|:--|:--|:--|
| Logical factorization is independent of physical execution | controlled `D2/D3/D4 -> P2` FFT lowering | the same physical schedule agrees within 1.8% |
| Processing units are replaceable mapping parameters | scalar, register, warp, cuFFTDx, TurboFFT, WMMA, radix and reduction alternatives | winners change with operator, length, batch and precision |
| Boundary realization affects the optimum without changing the paradigm | scalar versus paired 128-bit online cuFFTDx boundaries | best-point gain is 0.6%-13.5% across six long-FFT shapes |
| The physical-unit gain has a counter-level mechanism | fixed-mapping pre/post NCU capture at `logN=20`, batch 16 | CUDA-event -10.5%, replay -9.3%, warp instructions -19.6%, unchanged DRAM writes |
| Static hardware constraints are useful but insufficient for final ranking | 48-candidate FFT static-score evaluation | top-1 geomean regret 1.0762x; worst 1.4220x |
| A small hardware calibration set can replace exhaustive timing | leave-one-batch-out FFT interpolation | top-3 geomean regret 1.0007x, worst 1.0020x using 6.2% of candidates |
| The selection method extends beyond FFT | leave-one-complete-shape-out selector over NTT/FWHT/XOR-zeta | 93.88% top-1, 100% top-3, 1.0049x geomean regret |
| Numeric and batch preference is piecewise rather than globally smooth | 736-case screen, 29 full follow-ups, 12 paired NCU profiles, 99 adaptive anchors | all 9 length crossovers reproduce; 6 batch events are unstable; bounded selector coverage is 27.74% with 1.01112x worst regret |

The static-model failure is part of the result, not a row to hide. Resource
legality, occupancy, waves, and issued work provide a useful shortlist, but do
not capture launch-limited low-batch behavior accurately enough. The supported
method is therefore hierarchical: static legality and service constraints,
then a small measured calibration set, then dispatch.

## Current Performance Boundary

The latest matching-protocol FP32 long-FFT comparison uses five process trials,
1000 warmups, 100 timed repetitions, correctness preflight, and resident CUDA
event time.

| Shape | cuButterfly | cuFFT | VkFFT | cuButterfly/cuFFT |
|:--|--:|--:|--:|--:|
| `logN=18`, batch 16 | 0.195942 ms | 0.211098 ms | 0.201103 ms | 1.077x |
| `logN=20`, batch 4 | 0.214733 ms | 0.210330 ms | 0.206520 ms | 0.979x |
| FP64 `logN=14`, batch 256 | 0.340173 ms | 0.344433 ms | not measured | 1.013x |
| FP64 `logN=16`, batch 64 | 0.339364 ms | 0.343040 ms | not measured | 1.011x |
| FP64 `logN=17`, batch 32 | 0.346307 ms | 0.395213 ms | not measured | 1.141x |
| FP64 `logN=18`, batch 16 | 0.354877 ms | 0.421970 ms | not measured | 1.189x |

The broader design scan reaches 1.085x cuFFT at `logN=18`, batch 2 and 1.030x
at `logN=20`, batch 2. At saturated `logN=20` batch 8/16 it reaches
0.962x/0.954x. These focused results use a different batch protocol and must
not be mixed into the comprehensive rows above.

Matching-protocol external measurements show parity or advantage over Dao FHT
on the qualified large FWHT shapes and 1.109x-1.425x throughput over GPU-NTT on
the matched natural-order NTT shapes. XOR-zeta remains an internal architecture
ablation because no maintained matching external baseline has been identified.

In the current ten-shape library/base/search matrix, searched configurations
are 2.349x faster than fixed radix-2 base mappings and 1.109x faster than the
matched external rows by geometric mean. The per-operator library ratios are
1.015x FFT, 1.054x FWHT, and 1.313x NTT, with six rows faster and four within a
+/-3% parity band. This is a representative matrix, not a universal result;
the GPU-NTT rows are archived same-machine matching-protocol measurements rather
than interleaved rows from the current refresh.

## Semantic Coverage

| Operator | Numeric coverage | Measured lengths | Additional contracts |
|:--|:--|:--|:--|
| FFT | FP16/BF16 storage with native-width or FP32 accumulation, FP32, FP64 | 3, 8, 12, 14, 16, 18, 20 | forward/inverse, normalized inverse, in/out-of-place, stride 2 |
| NTT | 30-bit/32-bit and 60-bit/64-bit | 12, 16, 20 | forward/inverse, natural and native bit-reversed order |
| FWHT | FP16/BF16 storage with native-width or FP32 accumulation, FP32, FP64 | 8, 12, 15, 20 | forward, multiple exchange and residency choices |
| XOR-zeta | uint32 | 8, 12, 20 | forward/inverse, multiple mapping families |
| Structured 2x2 | FP16/BF16 storage with native-width or FP32 accumulation, FP32, FP64 | 8, 12, 15, 20 | broadcast/per-stage matrices, forward/inverse, shared/register cores |

Coverage establishes generality of the abstraction, not universal superiority.
The original FP64 scalar row remains 0.642x cuFFT in the comprehensive suite.
A controlled follow-up first scans 264 scalar mappings and confirms the winner
at 0.658x, then replaces the local unit with double-precision cuFFTDx and reaches
0.893x with table twiddles. Register recurrence reduces the selected prefix's
table lookups and raises the final result to 0.948x. This isolates most of the
former deficit in the physical unit and its precision-specific CTA/EPT mapping.
The controlled NCU capture shows recurrence reduces prefix replay time by
17.5% and warp instructions by 6.9%, while spending 8.4% more FP64 instructions
with unchanged registers, shared allocation, and waves/SM. Prefix DRAM peak
rises from 60.2% to 72.9%; suffix time is effectively unchanged. The remaining 5.5%
event-time gap is now localized to shared exchange and address work: total
shared-bank conflicts remain unchanged and warp instructions are still 2.06x
cuFFT.
An XOR-swizzled prefix then redistributes the same-capacity shared slot layout
and reaches 0.353526 ms, or 0.970x cuFFT. The complete 36-point remap retains
the same `256/4 + 128/8` winner, so this is a boundary-layout gain rather than
a mapping-selection artifact. NCU confirms that it cuts prefix shared conflicts
by 53.5% and prefix replay by 5.3%, raising peak DRAM utilization from 72.9% to
77.0%; the suffix remains unchanged. Its 10.0% increase in prefix warp
instructions identifies swizzled address generation as the next optimization
target. The implemented strength reduction computes each swizzled pointer once,
advances it by a compile-time stride, and hoists invariant transform/global
address work out of the staging loops. It lowers linear recurrence from
0.361708 ms to 0.345027 ms and XOR swizzle from 0.353526 ms to 0.342098 ms.
The latter matches the 0.342856 ms cuFFT measurement at 1.002x throughput; five
trial ranges are 0.341893-0.342170 ms and 0.342610-0.343040 ms, respectively.
This is parity evidence rather than a claim of a robust lead. Updated NCU
counters confirm the predicted mechanism: versus the pre-optimization XOR
kernel, prefix warp instructions fall 26.8%, prefix replay falls 8.3%, and
prefix shared conflicts fall 8.6%; suffix replay changes by less than 0.4%.
The full XOR path now replays 2.5% below cuFFT while executing 1.83x its warp
instructions, 3.62x its integer instructions, and about 114x its shared
conflicts. Prefix registers rise from 48 to 64 and the register block limit
falls from 5 to 4, so strength reduction exchanges occupancy for substantially
less dynamic address work.

The FP64 adapter now instantiates 128/256/512-point segments and covers every
two-segment composition whose dimensions are both `logN=7..9`. A 648-point
coarse scan selects distinct mappings for `logN=14..18`; a subsequent
five-trial, execution-order-interleaved matrix covers batch 1..64. Applying the
existing 0.020-ms timing floor leaves 25 stable shapes: 20 have higher median
throughput than cuFFT and 19 have completely separated faster trial ranges.
All 13 stable `logN=17/18` shapes are faster. The five remaining median deficits
are confined to `logN=14..16` crossover batches and range from 0.1% to 3.2%.

## Remaining Work

The v0.5 quick screen now covers FP16/BF16 accumulation contracts, FP32/FP64,
uint32 zeta, and uint32/uint64 NTT. It finds 29 confirmed and 67 ambiguous
mapping crossovers. Complete-regime top-3 recall is 1.0, but geometric-mean and
worst regret are 1.047 and 2.078, so global transfer remains
`measurement-required`. A conservative piecewise model now auto-selects only
between same-winner stable batch anchors. Its leave-one-batch-out coverage is
27.74% after adaptive full timing, with 1.00013x geometric-mean and 1.01112x
worst regret. Of 99 weak non-boundary anchors, 66 become stable and 33 remain
near-ties; none reverse direction. This validates a bounded region with
explicit abstention; it does not promote the model to a global runtime selector.

1. **Conditional cliff model:** expand the current stable-interval model and determine how numeric width, coefficient
   policy, modular reduction state, local-unit instruction mix, length, batch,
   stride, direction, and normalization move residency cliffs and mapping
   crossovers on the fixed V100 hardware target.
2. **Boundary-focused selector validation:** hold out complete numeric and
workload regimes, not just individual shapes, and report top-k recall and
   regret around launch, occupancy, register, shared-memory, and bandwidth
boundaries.
3. **Remaining performance margins:** explain and reduce the FP32 `logN=20`
   saturated-batch gap, the five FP64 `logN=14..16` crossover deficits, and the
   Structured register cliff at `logN=13..15` without conflating physical-core
   improvements with the scheduling contribution.
4. **Cross-GPU transfer:** only after the conditional single-GPU model is
   established, capture a newer GPU profile and test whether the learned
   descriptors and predicted boundaries transfer before and after calibration.

The repository does not currently claim a production cuFFT API replacement,
non-power-of-two coverage, universal library superiority, or architecture
portability beyond V100.

## Reproduction Map

- FFT mapping and vectorization: `results/v100_fft_pipeline_vectorized_*`
- FFT prediction evaluation: `results/v100_fft_pipeline_model_*`
- latest comprehensive FFT: `results/comprehensive_v100_vectorized_fft_*`
- cross-operator semantics: `results/comprehensive_v100_full_*`
- calibrated selector: `results/v100_mapping_selector_*`
- numeric quick/follow-up/coverage: `results/v100_numeric_*`
- numeric boundary attribution: `results/ncu_numeric_boundaries/`
- library/base/search matrix: `results/v100_three_way_*`
- matching external baselines: `results/v100_external_baselines_*`
- counter attribution: `results/ncu_scaling_crossovers/`
- vectorized FFT attribution: `results/ncu_fft_vectorized/`
- FP64 scalar/core/mapping validation: `results/fp64_*logN16*`
- FP64 length/batch robustness: `results/fp64_robustness_*`
- FP64 privileged counter command: `scripts/profile_fp64_fft_ncu.sh`

Run `./scripts/reproduce_v100_analysis.sh` to regenerate derived artifacts from
the checked-in raw measurements. It does not recollect timings or counters.
