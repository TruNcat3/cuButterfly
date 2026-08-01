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

The broader design scan reaches 1.085x cuFFT at `logN=18`, batch 2 and 1.030x
at `logN=20`, batch 2. At saturated `logN=20` batch 8/16 it reaches
0.962x/0.954x. These focused results use a different batch protocol and must
not be mixed into the comprehensive rows above.

Matching-protocol external measurements show parity or advantage over Dao FHT
on the qualified large FWHT shapes and 1.109x-1.425x throughput over GPU-NTT on
the matched natural-order NTT shapes. XOR-zeta remains an internal architecture
ablation because no maintained matching external baseline has been identified.

## Semantic Coverage

| Operator | Numeric coverage | Measured lengths | Additional contracts |
|:--|:--|:--|:--|
| FFT | FP16/FP32 accumulation, FP32, FP64 | 3, 8, 12, 14, 16, 18, 20 | forward/inverse, normalized inverse, in/out-of-place, stride 2 |
| NTT | 30-bit/32-bit and 60-bit/64-bit | 12, 16, 20 | forward/inverse, natural and native bit-reversed order |
| FWHT | FP32, FP64 | 8, 12, 15, 20 | forward, multiple exchange and residency choices |
| XOR-zeta | uint32 | 8, 12, 20 | forward/inverse, multiple mapping families |

Coverage establishes generality of the abstraction, not universal superiority.
FP64 FFT remains a clear processing-unit gap at 0.642x cuFFT in the current
comprehensive suite.

## Remaining Work

1. **Cross-GPU transfer:** capture a newer GPU profile, predict without its
   timings, and report pre/post-calibration regret. This is the only deferred
   architecture-level validation item.

The repository does not currently claim a production cuFFT API replacement,
non-power-of-two coverage, universal library superiority, or architecture
portability beyond V100.

## Reproduction Map

- FFT mapping and vectorization: `results/v100_fft_pipeline_vectorized_*`
- FFT prediction evaluation: `results/v100_fft_pipeline_model_*`
- latest comprehensive FFT: `results/comprehensive_v100_vectorized_fft_*`
- cross-operator semantics: `results/comprehensive_v100_full_*`
- calibrated selector: `results/v100_mapping_selector_*`
- matching external baselines: `results/v100_external_baselines_*`
- counter attribution: `results/ncu_scaling_crossovers/`
- vectorized FFT attribution: `results/ncu_fft_vectorized/`

Run `./scripts/reproduce_v100_analysis.sh` to regenerate derived artifacts from
the checked-in raw measurements. It does not recollect timings or counters.
