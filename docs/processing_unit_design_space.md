# Processing-Unit Design Space

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
`(local_stages, reorder_columns, tile_threads, backend, ...)`. Consequently a
core can be replaced without changing the mapping model, and each GPU can
search the product of both spaces.

## Implemented Units

| Operator | Stage group | Arithmetic/coefficient choices | Mapping support |
|:--|:--|:--|:--|
| FFT FP32/FP64 | radix-2, fused radix-4, fused radix-8 | four-multiply, Gauss three-multiply, FP32 thread-register DFT8 | temporal, hierarchical, online-reorder; generated DFT8 point |
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
