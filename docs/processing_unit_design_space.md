# Processing-Unit Design Space

The canonical operator-independent architecture and candidate-state contract is
defined in [Complete Butterfly Design Space](butterfly_design_space.md). The
[FFT Architecture Design Space](fft_design_space.md) is its validated FFT
projection. This document focuses on implemented processing units and measured
selections.

## Architectural Boundary

cuButterfly owns the two-dimensional space-time mapping. A processing unit is
a replaceable implementation of one or more equivalent butterfly stages; it
does not own global scheduling, tile residency, or the online permutation.
The unit is described by four independent fields:

```text
P = (stage_group, arithmetic_core, coefficient_form, local_exchange)
```

- `stage_group`: one, two, or three fused binary stages (`radix2/4/8`).
- `arithmetic_core`: operator-specific pair computation, such as FFT
  four-multiply or Gauss three-multiply complex arithmetic.
- `coefficient_form`: native constants, Shoup precomputation, or Barrett
  reciprocal precomputation.
- `local_exchange`: shared-memory, warp-register, or an imported local
  codelet transport.

The architectural mapping parameters remain
`(local_stages, reorder_columns, tile_threads, prefix_threads, suffix_threads, backend, ...)`. Consequently a
core can be replaced without changing the mapping model, and each GPU can
search the product of both spaces.

## Implemented Units

| Operator | Stage group | Arithmetic/coefficient choices | Mapping support |
|:--|:--|:--|:--|
| FFT FP32/FP64 | radix-2, fused radix-4, fused radix-8 | four-multiply, Gauss three-multiply, FP32 thread-register DFT8 | temporal, hierarchical, online-reorder; generated DFT8 point |
| FFT FP32 optional | cuFFTDx block/direct FFT, TurboFFT generated FFT | imported local codelet transport and arithmetic | cuFFTDx online dimensions `logN=3..12`, TurboFFT `logN=7..10` |
| FFT FP64 optional | cuFFTDx block FFT | imported double-precision local codelet | temporal `logN=3..10`; online `logN=16` with an `8+8` decomposition |
| FFT mixed | temporally fused DFT8 matrix | FP16 WMMA input with FP32 accumulation/output | generated temporal point |
| FWHT FP32/FP64 | radix-2, fused radix-4, fused radix-8 | add/subtract; FP32 register-vector/XOR-swizzle exchange | temporal, hierarchical, online-reorder |
| XOR-zeta uint32 | radix-2, fused radix-4, fused radix-8 | add/subtract modulo `2^32` | temporal, hierarchical, online-reorder |
| NTT 32/64-bit | radix-2, fused radix-4, fused radix-8 | independent Shoup or Barrett modular multiplication | Hybrid2D |

`warp-hybrid` and `stage-pipeline` currently expose radix-2 only. cuFFT is an
opaque vendor baseline, not an internal unit. The FP32 FWHT `warp-register`
unit isolates the register-vector, warp shuffle, and XOR-swizzled cross-warp
exchange used by Dao-AILab FHT behind the same local-unit contract. It retains
cuButterfly batch, stride, placement, inverse, and normalization semantics.
Query the compiled matrix with:

```bash
build/cubutterfly_bench --list-capabilities
```

The imported choices are selected through `fft_core=cufftdx-block` and
`fft_core=turbofft-generated`. They deliberately use the same plan and timing
contract as the in-tree units. cuFFTDx is also composable with the long
`online-reorder` backend: each pass embeds a block FFT, retains per-thread
fragments in registers, uses shared memory for exchange and a tiled transpose,
and fuses the cross twiddle into the first-pass epilogue. TurboFFT remains a
standalone generated global kernel and therefore needs a device-codelet form
before the same composition is possible.

## Imported FFT Unit Results

The V100 comparison uses FP32 forward, contiguous out-of-place batches with
`2^22` total points, 20 warmups, 100 timed repetitions, and five trials. Values
are medians; throughput above 1 means faster than cuFFT for the same batch.

| logN | cuFFT ms | CTA DFT8 ms | cuFFTDx ms | cuFFTDx / cuFFT | TurboFFT ms | TurboFFT / cuFFT |
|--:|--:|--:|--:|--:|--:|--:|
| 3 | 0.086313 | 0.090798 | 0.149279 | 0.578x | n/a | n/a |
| 4 | 0.084265 | 0.102072 | 0.099113 | 0.850x | n/a | n/a |
| 5 | 0.084746 | 0.099830 | 0.084808 | 0.999x | n/a | n/a |
| 6 | 0.084122 | 0.092037 | 0.084019 | **1.001x** | n/a | n/a |
| 7 | 0.084541 | 0.095672 | 0.084490 | **1.001x** | 0.087634 | 0.965x |
| 8 | 0.084152 | 0.101601 | 0.084736 | 0.993x | 0.087388 | 0.963x |
| 9 | 0.084920 | 0.101540 | 0.084756 | **1.002x** | 0.083702 | **1.015x** |
| 10 | 0.085903 | 0.107971 | 0.084808 | **1.013x** | 0.186931 | 0.460x |

cuFFTDx reaches parity at `logN=5..10` but is not a replacement for the
specialized batched scheduling used by cuFFT at very small sizes. TurboFFT is
competitive at `logN=7..9`; its T4-selected `logN=10` mapping is a poor V100
point, confirming that imported arithmetic does not remove the need for a
hardware-specific organization search. TurboFFT `logN=10` also measured about
`2.4e-4` maximum absolute error on the targeted random test, so its automated
unit test uses a documented `3e-4` FP32 threshold.

Raw and derived records are
`results/fft_processing_units_v100_{raw,summary}.csv`; reproduce them with
`scripts/benchmark_fft_processing_units.sh`.

## Long cuFFTDx Composition

For `N=N1*N2`, the long path performs two local block FFT passes. The first
pass reads the `N1` dimension, applies `W_N^(k1*n2)` in registers, and writes
the transposed intermediate layout. The second pass transforms `N2` and writes
natural order. The original point stages four large local FFTs together so the
transpose edges use fully occupied 32-byte memory sectors. `cross_twiddle` selects either
one table lookup per element or two initial lookups followed by a per-thread
register recurrence.

That four-FFT packaging is now a default rather than a constant. For the
cuFFTDx path, `prefix_threads` and `suffix_threads` independently select
128/256/512/1024-thread budgets, while each dimension independently selects
EPT 4/8/16 from the current V100 codegen manifest. The architecture derives
`FFTsPerBlock = threads*EPT/local_size` for each dimension. Invalid points
whose budget cannot cover one local FFT are rejected before launch. This exposes
the spatial width of both dimensions without changing the two-pass dataflow.

The following V100 medians use `2^22` total FP32 complex points, 20 warmups,
100 repetitions, and five trials:

| logN | Best split | Cross twiddle | Generic online ms | cuFFTDx online ms | Speedup | cuFFT ms | cuFFTDx/cuFFT throughput |
|--:|:--|:--|--:|--:|--:|--:|--:|
| 12 | `6+6` | table | 0.448748 | 0.211917 | 2.118x | 0.088893 | 41.9% |
| 14 | `5+9` | recurrence | 0.440914 | 0.203510 | 2.167x | 0.122757 | 60.3% |
| 16 | `6+10` | recurrence | 0.445317 | 0.225987 | 1.971x | 0.179292 | 79.3% |
| 18 | `10+8` | recurrence | 0.468081 | 0.240364 | 1.947x | 0.211661 | 88.1% |
| 20 | `10+10` | recurrence | 0.639396 | 0.235971 | 2.710x | 0.209521 | 88.8% |

The selected split and twiddle policy vary with length. This is the intended
separation: the architecture selects the two-dimensional flow and hardware
mapping, while an established processing core supplies each resident local
transform. Reproduce the table with
`scripts/benchmark_fft_cufftdx_long.sh`.

### Independent two-dimension mapping

A hierarchical search first ranks split and twiddle policy at the default
512/512 mapping, expands both thread axes only for the leading splits, and then
remeasures finalists. On the same V100 with `2^22` points it selects:

| logN | Split | Twiddle | Prefix/suffix threads | cuButterfly ms | cuFFT ms | Throughput ratio |
|--:|:--|:--|:--|--:|--:|--:|
| 16 | `6+10` | recurrence | `128/512` | 0.212685 | 0.180702 | 0.850x |
| 18 | `9+9` | recurrence | `256/256` | 0.212019 | 0.226877 | **1.070x** |
| 20 | `10+10` | recurrence | `512/512` | 0.245180 | 0.223232 | 0.910x |

The `logN=18` point was independently confirmed with seven trials of 20 warmups
and 100 repetitions. More threads do not monotonically improve performance:
they increase FFTs per CTA and shared-memory footprint while reducing the number
of independently schedulable CTAs. The optimum therefore depends on both local
dimension lengths and the transform batch.

Reproduce the constrained search with:

```bash
./scripts/explore_fft_architecture.py \
  --logNs 16 18 20 --target-points 4194304 \
  --output-prefix results/fft_architecture_explore_v100
```

Primary records are `results/fft_architecture_explore_v100_{search,confirm,summary}.csv`
and `results/fft_architecture_logN18_confirm_v100_{confirm,summary}.csv`.

### FP64 processing-unit isolation

The FP64 `logN=16`, batch-64 case separates mapping and local-unit effects in
three ordered experiments. A 264-point scalar scan covers decomposition,
radix, complex multiplication, CTA width, and reorder columns. It moves the
original 0.534313 ms point to a 0.521492 ms confirmed result (`8+8`, radix-4,
128 threads), or 0.658x cuFFT throughput. Replacing both local dimensions with
`Precision<double>()` cuFFTDx codelets and independently scanning their CTA/EPT
axes selects prefix `256/4` and suffix `128/8`.

| FP64 implementation | Median ms | Throughput vs cuFFT |
|:--|--:|--:|
| scalar comprehensive point | 0.534313 | 0.642x |
| scalar mapping winner | 0.521492 | 0.658x |
| cuFFTDx `8+8`, table twiddle | 0.383949 | 0.893x |
| cuFFTDx `8+8`, recurrence twiddle | 0.361708 | 0.948x |
| cuFFT | 0.342927 | 1.000x |

The final three rows use five independent trials, 1000 warmups, and 100 timed
repetitions; all FP64 cuFFTDx local, online-forward, and normalized-inverse
paths pass automated correctness tests. The imported table unit reduces the
selected scalar latency by 26.4%, and recurrence raises that reduction to
30.6%, so the former FP64 deficit is primarily a physical-core and mapping
maturity issue rather than evidence against the architecture-level
space/time decomposition. `scripts/profile_fp64_fft_ncu.sh` fixes the scalar,
table, recurrence, and cuFFT mappings for privileged boundary attribution.
Reproduce the timing sequence with `scripts/benchmark_fp64_fft_units.sh` and
the counters with `scripts/profile_fp64_fft_ncu.sh`. Raw and ranked records are
`results/fp64_*logN16*.csv`.

The first prefix optimization replaces four EPT4 twiddle-table reads per
thread with two initial reads and register recurrence. The selected mapping
remains prefix `256/4`, suffix `128/8` across the full 36-point rescan, while
latency falls from 0.383949 ms to 0.361708 ms (5.8%). Simple `pitch+1` shared
padding is a negative result: prefix-only and both-pass variants take 0.398039
and 0.397609 ms, while suffix-only is statistically neutral at 0.383765 ms.
The conflict count therefore identifies real exchange overhead, but uniform
padding is not an efficient realization on V100.

The fixed privileged capture confirms the recurrence mechanism. It reduces
prefix replay from 247.136 us to 207.776 us (15.9%), warp instructions by 6.9%,
and raises peak DRAM use from 60.8% to 72.2%. This costs 8.4% more prefix FP64
instructions but does not change registers, shared allocation, or waves/SM.
The suffix changes by only 0.1%, confirming that the measured gain belongs to
the prefix twiddle policy. Against cuFFT, the recurrence path transfers 1.006x
the bytes and executes 1.023x the FP64 instructions, but still executes 2.06x
the warp instructions and incurs about 180x the shared-bank conflicts. The
remaining work is therefore shared exchange and general address work, not
twiddle service, occupancy, or FP64 arithmetic throughput. See
`results/ncu_fp64_fft/analysis.md`.

### Resident and direct granularity experiment

At `logN=12`, one 4096-value transform fits in a V100 CTA. Three physical
organizations were measured with the same `2^22`-point protocol:

| Organization | Physical boundary | Best mapping | Median ms | Throughput vs cuFFT |
|:--|:--|:--|--:|--:|
| cuFFTDx two-pass 64x64 | two launches, global scratch | table | 0.213207 | 41.6% |
| resident composed 64x64 | one launch, shared 64x65 tile | recurrence, 1024 threads | 0.120412 | 73.7% |
| direct cuFFTDx 4096 | one launch, whole-transform unit | 512 threads | 0.088996 | 99.7% |
| cuFFT | vendor plan | vendor selected | 0.088750 | 100.0% |

The resident composed point proves that the intermediate global boundary is
not required at this size. Its remaining gap is not launch overhead: replacing
the 128 separately executed 64-point local FFTs with one 4096-point equivalent
unit closes it. Equivalent-unit granularity is therefore a hardware-dependent
architecture parameter alongside spatial width, temporal depth, CTA threads,
tile pitch, and cross-twiddle policy.

The direct-unit search was then extended to every `logN=11..14` whole-transform
cuFFTDx specialization. A contiguous compile-time layout removes generic
stride/tail address work. Independent five-trial runs produce 1.006x, 1.012x,
0.965x, and 1.029x cuFFT throughput at `logN=11,12,13,14`, respectively. The
best V100 CTA shapes are 256, 512, 512, and 1024 threads. `cuobjdump` reports 63
registers/thread and no local-memory spill for the winning `logN=14` point;
the 512-thread alternative uses about 90 registers/thread.

The same ownership policy is not universally beneficial. A manually composed
persistent 7+7 `logN=14` CTA reaches only 0.227922 ms, while the direct 16384
unit reaches 0.131543 ms and cuFFT reaches 0.135383 ms in the final run.
This separates the architecture decision (whole-transform temporal ownership)
from the processing-core implementation (manual symmetric composition versus
cuFFTDx). It also shows why the equivalent unit must remain selectable rather
than fixed.

Reproduce the searches with:

```bash
./scripts/benchmark_fft_resident_logN12.sh
./scripts/benchmark_fft_persistent_logN14.sh
./scripts/benchmark_fft_direct_units.sh
```

The privileged counter comparison is prepared in
`scripts/profile_fft_resident_ncu.sh`.
Its V100 result confirms that resident and direct have the same approximately
64 MiB external traffic, while resident executes 46.4% more warp instructions,
incurs 2.0x the shared-store conflicts, and has higher barrier stall. See
`results/ncu_fft_resident_analysis.md`.
The corresponding `logN=14` direct-versus-cuFFT capture is prepared in
`scripts/profile_fft_direct14_ncu.sh`.

## V100 Selection Evidence

The following medians use a Tesla V100-SXM2-16GB, contiguous forward data,
approximately `2^22` total points, three trials, and resident kernel time. Each
entry is the best thread count for that unit.

| Shape/operator | radix-2 ms | radix-4 ms | radix-8 ms | Selected |
|:--|--:|--:|--:|:--|
| `logN=8` FFT FP32 | 0.130314 | 0.110572 | **0.110551** | radix-8, 64 threads |
| `logN=8` FWHT FP32 | 0.055869 | **0.051364** | 0.053330 | radix-4, 128 threads |
| `logN=8` XOR-zeta | 0.052408 | **0.050094** | 0.051896 | radix-4, 128 threads |
| `logN=20` FFT FP32 | 0.826112 | **0.674355** | 0.689306 | radix-4, 256 threads |
| `logN=20` FWHT FP32 | 0.377037 | **0.372787** | 0.373094 | radix-4, 256 threads |
| `logN=20` XOR-zeta | 0.378317 | **0.372634** | 0.373299 | radix-4, 256 threads |

The imported register hierarchy changes the FWHT result materially. The next
table uses five trials, 20 warmups, 100 repetitions, FP32, and the same `2^22`
total points. Dao is the locally built upstream external baseline; ratios above
one mean cuButterfly is slower.

| logN | shared best ms | warp-register ms | Dao FP32 ms | speedup vs shared | ratio vs Dao |
|--:|--:|--:|--:|--:|--:|
| 8 | 0.055828 | **0.043131** | 0.043172 | 1.29x | 1.00x |
| 10 | 0.061481 | **0.044626** | 0.043960 | 1.38x | 1.02x |
| 12 | n/a | **0.045199** | 0.044073 | n/a | 1.03x |
| 14 | n/a | **0.055992** | 0.053617 | n/a | 1.04x |
| 15 | n/a | **0.069243** | 0.063949 | n/a | 1.08x |

This is the intended architectural claim: a previously validated local core
can be inserted as one processing-unit choice without taking ownership of the
global space-time mapping. The remaining long-length gap is localized to the
unit implementation and code generation, rather than the public execution
contract.

The compiled V100 resource envelope makes the length dependence explicit:

| logN | threads/CTA | registers/thread | exchange shared memory | resource implication |
|--:|--:|--:|--:|:--|
| 8 | 32 | 30 | 1 KiB | block-count bound; one warp per CTA |
| 10 | 128 | 32 | 4 KiB | full thread occupancy is feasible |
| 12 | 256 | 32 | 16 KiB | shared memory limits residency before registers |
| 13 | 256 | 64 | 32 KiB | at most three CTAs/SM from shared memory |
| 14 | 256 | 128 | 32 KiB | at most two CTAs/SM from registers |
| 15 | 256 | 255 | 32 KiB | at most one CTA/SM from registers |

Thus `local_exchange` is not a binary preference. Increasing temporal extent
removes global boundaries but grows the per-thread live set and eventually
collapses CTA residency. A GPU-specific selector should maximize useful
resident work subject to register-file and shared-memory capacity; on a GPU
with a different register file, shared-memory partition, or warp size, the
breakpoints and thread schedule must be regenerated while the mapping paradigm
remains unchanged. Register counts above come from `cuobjdump
--dump-resource-usage build/cubutterfly_bench`; occupancy statements are
resource upper bounds for V100, not profiler-measured achieved occupancy.

Gauss three-multiply is correct but does not improve these V100 FP32 cases. Its
best times differ from four-multiply by less than 0.5%; the extra dependent
adds offset the nominal multiply reduction. It therefore remains a searched
choice rather than a default.

For 60-bit NTT with fused cross-twiddles, radix-4 is also selected: 0.268237 ms
at `logN=12` and 0.475955 ms at `logN=20`. Radix-8 takes 0.432998 ms and
0.538419 ms respectively. The larger register live set and reduced number of
independent local units outweigh one fewer synchronization group. This is a
useful negative result: larger fused units are not intrinsically better.

The modular arithmetic axis is also shape dependent. With radix-4 and fused
cross-twiddles, the V100 medians are:

| Shape | Shoup ms | Barrett ms | Barrett change | Selected |
|:--|--:|--:|--:|:--|
| 30-bit, 32-bit word, `logN=12` | **0.157901** | 0.169830 | +7.6% | Shoup |
| 30-bit, 32-bit word, `logN=20` | 0.325939 | **0.324096** | -0.6% | Barrett |
| 30-bit, 64-bit word, `logN=12` | **0.268646** | 0.377293 | +40.4% | Shoup |
| 30-bit, 64-bit word, `logN=20` | **0.476262** | 0.537958 | +13.0% | Shoup |
| 60-bit, 64-bit word, `logN=12` | **0.268646** | 0.377395 | +40.5% | Shoup |
| 60-bit, 64-bit word, `logN=20` | **0.476262** | 0.538163 | +13.0% | Shoup |

Barrett's only win is a small long-transform 32-bit case. Extending its wider
reduction temporaries across both local passes is expensive on the V100,
especially for radix-8. This again argues for selecting a tuple rather than
assigning one arithmetic core to the architecture.

Raw and ranked records are stored in:

- `results/processing_units_v100_logN8_{raw,summary}.csv`
- `results/processing_units_v100_logN20_{raw,summary}.csv`
- `results/ntt_processing_units_v100{,_raw}.csv`
- `results/ntt_unit_reduction_v100{,_raw}.csv`
- `results/register_fwht_v100_{raw,summary}.csv`
- `results/external_dao_fht_v100_summary.csv`

## Reproduction

```bash
python3 scripts/sweep_cubutterfly_designs.py \
  --backends temporal-tile --logNs 8 \
  --compute-units radix2 radix4 radix8 \
  --complex-multiplies four-mul gauss3 \
  --target-points 4194304 --warmup 10 --repeat 50 --trials 3 \
  --output results/processing_units_v100_logN8_raw.csv

python3 scripts/summarize_cubutterfly_designs.py \
  results/processing_units_v100_logN8_raw.csv \
  --output results/processing_units_v100_logN8_summary.csv

python3 scripts/sweep_cubutterfly_designs.py \
  --operators fwht --precisions fp32 --backends temporal-tile \
  --compute-units radix2 --local-exchanges shared warp-register \
  --logNs 8 10 12 14 15 --target-points 4194304 \
  --warmup 20 --repeat 100 --trials 5 \
  --output results/register_fwht_v100_raw.csv

python3 scripts/sweep_matrix.py --logN 12 20 --bits 60 --word-bits 64 \
  --units radix2 radix4 radix8 --cross-twiddle fused \
  --mod-multiplies shoup barrett --verify \
  --output results/ntt_processing_units_v100.csv
```

The conclusion is not that radix-4 or radix-8 is universally best. It is that
the architecture exposes enough unit structure for a hardware-specific search
to choose correctly without changing the space-time parallel paradigm.
