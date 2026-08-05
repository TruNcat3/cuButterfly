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
