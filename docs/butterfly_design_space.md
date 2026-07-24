# Operator-Independent Butterfly Design Space

This document is the canonical architecture description for cuButterfly. The
machine-readable form is
[`config/butterfly_architecture_space.json`](../config/butterfly_architecture_space.json);
[`scripts/butterfly_design_space.py`](../scripts/butterfly_design_space.py)
validates it and derives architecture demand. The FFT-specific specification is
a checked projection of this general space, not a second architecture model.

## 1. Scope And Separation

A design point is the tuple

```text
D = (G, A, P, L, F, Q | H)
```

| Object | Owns | Does not own |
|:--|:--|:--|
| `G` graph semantics | operator, length, batch, direction, normalization, order, stride | physical execution |
| `A` architecture | graph factorization, space/time unfolding, residence, handoff, schedule | butterfly arithmetic circuit |
| `P` processing unit | radix/codelet, arithmetic core, coefficient form, local exchange | global graph schedule |
| `L` layout | global and intermediate permutations, bank mapping, vectorization | arithmetic semantics |
| `F` realization | threads, warps, CTAs, buffering, pipeline timing, synchronization | logical unfolding definition |
| `Q` generation/selection | specialization state, search policy, optimization objective | hardware limits |
| `H` hardware | hierarchy, execution resources, storage capacities, transport rates | transform semantics |

This separation is the central contract. cuFFTDx, TurboFFT, a tensor-core
codelet, or a better modular multiplier changes `P`. CUDA block shape and
barriers change `F`. Neither change the meaning of the architecture factors in
`A`. Conversely, replacing two global passes by a resident composition changes
`A`, even if the same processing unit is used.

## 2. Architecture Invariant

For a binary butterfly graph with `K=log2(N)` stages and `D=N/2` independent
cells per stage, the basic unfolding is

```text
             space       time
stage axis     Us          Ts       Us * Ts >= K
data axis      Ud          Td       Ud * Td >= D
batch axis     Ub          Tb       Ub * Tb >= B
```

`Us` is the number of dependent stages materialized in space. `Ts` is their
temporal fold count. `Ud` is independent cell width, while `Td` serializes that
work. `Ub` replicates the mapping across independent transforms and `Tb` folds
the batch. The useful-lane fraction is

```text
eta = K/(Us*Ts) * D/(Ud*Td) * B/(Ub*Tb).
```

The first two rows are the paper's two-dimensional space-time paradigm. The
batch row is deliberately separate: it does not change graph dependencies, but
it changes occupancy, replicated compute, and total coefficient/boundary
bandwidth on a GPU.

At fixed physical cell budget `C=Us*Ud`, ignoring tails,
`Ts*Td=K*D/C` is constant. Moving resources from `Ud` to `Us` therefore does
not create arithmetic throughput. It trades boundary bandwidth for interstage
handoff and live state:

```text
boundary words/cycle   = 2 * Ud * Ub
interstage words/cycle = 2 * Ud * (Us - 1) * Ub
coefficient streams    <= Us * Ud * Ub  (coefficient-bearing operators)
```

The checked example in
`results/butterfly_architecture_fixedC_logN16.csv` holds `C=128`, `Ub=1`, and
`K=16`. `Us=1..16` preserves 4096 ideal body cycles while boundary demand falls
from 256 to 16 words/cycle and interstage demand rises from 0 to 240. This is
the architecture-level reason an optimum depends on the target hardware.

## 3. Multi-Dimensional Factorization

The graph may be factored into any ordered positive composition

```text
K = k0 + k1 + ... + k(r-1).
```

Each logical dimension has its own complete mapping

```text
Mi = (ki, Us_i, Ts_i, Ud_i, Td_i, Hs_i, Rs_i, Rd_i, Pi, Li, Fi).
```

The factors, core, thread count, EPT, exchange mechanism, and residence may be
asymmetric across dimensions. A boundary descriptor connects every adjacent
pair and records its permutation, cross coefficient placement/form, and
producer/consumer width. Thus `6+10` and `10+6`, or cuFFTDx EPT `4+16` and
`16+4`, are distinct points. Rank one is a whole-transform unit; rank two is
the current long FFT/Hybrid2D form; higher rank covers deeper compositions.

An online reorder is a choice of boundary layout and producer epilogue. It can
remove a standalone transpose kernel, but it is not free: its stores, address
generation, synchronization scope, and resulting consumer coalescing belong to
`L` and `F` and must be measured.

## 4. Hardware-Hierarchy Mapping

A logical unfolding factor is insufficient to describe a GPU mapping. Each is
factored over the physical hierarchy:

```text
Us = product(Us_lane, Us_thread, Us_warp, Us_cta, ...)
Ud = product(Ud_lane, Ud_thread, Ud_warp, Ud_cta, ...)
Ub = product(Ub_lane, Ub_thread, Ub_warp, Ub_cta, ...)
```

The supported levels are lane, thread, warp, CTA, cluster, grid, device, and
node. Ownership and transport must also be stated. For example, `Us=8` can mean
eight register stages in one thread, two stages per warp across four CTA
groups, or an eight-stage producer-consumer pipeline. These have the same
logical factors but different register pressure, barriers, occupancy, and
handoff bandwidth.

The hardware profile supplies capacities and rates rather than preferred tile
constants. A legalizer maps `A/P/L/F` demand to `H`:

| Demand | Principal hardware constraints |
|:--|:--|
| stage/data/batch spatial factors | lane, warp, CTA, grid concurrency |
| live state and coefficient residence | registers, shared memory, cache capacity |
| handoff and permutation | shuffle/shared/global/network bandwidth and barriers |
| arithmetic core | scalar, SIMD, tensor, or pipeline issue rates |
| temporal folds and pipeline II | latency hiding, occupancy, synchronization rate |
| vector/layout choice | alignment, memory transactions, bank conflicts |

Consequently the V100 optimum is evidence for one `H`, not an architecture
constant. A different GPU regenerates hierarchy factors and `F/P/L` choices
while preserving the invariant in Section 2.

## 5. Operator Projections

The architecture only requires a regular staged pair-update graph. Operator
projections provide the semantic constraints that a processing unit must
preserve.

| Operator | Cell/coefficient property | Current status |
|:--|:--|:--|
| FFT | complex weighted sum/difference; stage coefficients; floating error | implemented |
| NTT | modular weighted sum/difference; modulus and exactness contract | implemented |
| FWHT | coefficient-free symmetric sum/difference | implemented |
| XOR-zeta | coefficient-free asymmetric semiring update | implemented |
| subset/Mobius | asymmetric forward/inverse semiring update | candidate projection |
| structured butterfly | operator-defined 2x2 linear or semiring map | candidate projection |

Coefficient-free operators set coefficient streams to zero. Asymmetric
operators constrain legal reordering and in-place updates. FFT normalization
and error tolerance, and NTT modulus/root validity, remain `G/P` contracts and
cannot be discarded when comparing performance.

## 6. Coverage Of Design Freedoms

The following table is the completeness checklist. A proposed optimization is
inside the model only if it changes at least one listed object.

| Freedom | Represented by |
|:--|:--|
| operator, precision, direction, order, stride, batch | `G`, operator projection |
| number/order of factor dimensions | `A.factorization_rank`, stage composition |
| stage, data, and batch space/time unfolding | `A: Us/Ts, Ud/Td, Ub/Tb` |
| independent mapping per factor dimension | `A.per_dimension_mapping` |
| register/shared/global/distributed residence | `A: Rs/Rd/Rb` |
| handoff, online reorder, transpose, Stockham, swizzle | `A.dimension_boundary`, `L` |
| cross twiddle/root placement and generation | boundary descriptor, coefficient supply, `P` |
| radix, imported/generated codelet, tensor core | `P` |
| lane-to-node physical placement | `A.hierarchical_unfolding`, `H.hierarchy` |
| threads, EPT, units/CTA, CTAs/work | `P`, `F` |
| pipeline II/depth/buffers and synchronization | `F` |
| tails, padding, bank conflicts, vector width | `A.tail_policy`, `L` |
| runtime/generation/search state | `Q` |
| latency/throughput/traffic/energy/error objective | `Q.objectives` |
| capacity, arithmetic, transport, barrier limits | `H` |

Out of scope are irregular sparse graphs whose partner relation cannot be
expressed as staged pair updates, communication faults, and distributed
placement policy above the declared node level. Adding one of these requires a
schema revision rather than silently treating it as a tile parameter.

## 7. Current Runtime Projection

The schema binds every public backend to an architecture family. NTT maps
`baseline`, `tile256`, `hybrid2d`, `compact-stage`, and `stage-pipeline`.
The common interface maps `temporal-tile`, `hierarchical`, `online-reorder`,
`warp-hybrid`, and `stage-pipeline`; `cufft` is explicitly an external vendor
baseline. Tests reject a binding to a nonexistent family.

This is implementation coverage, not design-space coverage. `abstract`,
`awaiting-codegen`, `requires-new-kernel`, and `hardware-infeasible` are valid
states. They prevent an uncompiled point from being confused with a concept
that the architecture cannot express.

Processing-core bindings prevent another kind of false point. Each named core
maps to a legal stage group, arithmetic class, coefficient subset, and local
exchange subset. For example, `scalar-radix4` derives `stage_group=radix4`,
while `cufftdx-block` derives `full-codelet/imported-codelet`. The enumerator
will not form `scalar-radix4/radix2` or attach a twiddle stream to FWHT.

## 8. Validation And Enumeration

Validate the general schema and its FFT projection:

```bash
python3 scripts/butterfly_design_space.py --validate-only
```

Reproduce the fixed-compute architecture experiment:

```bash
python3 scripts/butterfly_design_space.py \
  --operator ntt --logN 16 --factorization-rank 1 \
  --Us 1 2 4 8 16 --fixed-spatial-cells 128 \
  --stage-residence shared --data-residence shared \
  --output results/butterfly_architecture_fixedC_logN16.csv
```

Enumerate batch folding independently:

```bash
python3 scripts/butterfly_design_space.py \
  --operator fft --logN 12 --factorization-rank 2 \
  --Us 4 --Ud 16 --batch 10 --Ub 1 2 4
```

Construct a filtered product across processing-unit and realization choices:

```bash
python3 scripts/butterfly_design_space.py \
  --operator fft --logN 12 --factorization-rank 2 \
  --Us 4 --Ud 16 --batch 10 --Ub 1 4 \
  --stage-residence shared --data-residence shared \
  --operator-core scalar-radix4 cufftdx-block \
  --coefficient-form native-table recurrence \
  --kernel-form online-reorder --threads 256 \
  --max-candidates 1000 --output results/fft_filtered_space.csv
```

Unspecified processing/layout/realization axes use their first legal value.
Specified lists form a filtered Cartesian product after dependent core
constraints are applied. `--max-candidates` guards against accidentally
materializing an unbounded product; hierarchical search should enumerate broad
architecture points first and expand `P/L/F` only for survivors.

The general validator runs during CMake configure and under CTest. Its tests
cover projection validity, ordered multi-dimensional factorization, fixed-cell
conservation, operator-dependent coefficient demand, residence, batch
unfolding, hierarchy composition, legal core binding, filtered products, and
runtime-family bindings.

## 9. Completeness Contract

The schema is complete for the stated class when all of the following hold:

1. Graph meaning is fully specified by `G` and an operator projection.
2. Every stage, data, batch, and factor-dimension mapping has explicit space,
   time, residence, handoff, and boundary choices in `A/L`.
3. Local arithmetic and transport are replaceable `P` choices.
4. Every logical factor decomposes to an explicit hardware hierarchy in `F/H`.
5. Generation status and comparison objective are recorded in `Q`.
6. A measured result records enough fields to reconstruct this tuple and never
   compares points with different semantic contracts.

This contract is stronger than listing kernel knobs: it states which decisions
are architectural, how they compose, and which hardware demand changes when
any decision moves.
