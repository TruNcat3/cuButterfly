# Structured 2x2 V100 Results

## Scope

This focused experiment tests whether a dense, stage-parameterized real `2x2`
processing unit can reuse the same two-axis space/time mappings as FWHT. It is
not an external-library comparison: an arbitrary sequence of local matrices
has no matching cuBLAS or cuFFT semantic contract.

The measured GPU is a Tesla V100-SXM2-16GB (`sm_70`). Every transform uses the
broadcast matrix `[1, 0.25; -0.5, 1]`, out-of-place forward semantics, and about
`2^22` total points. Each candidate has three trials, ten warmups, and fifty
timed repetitions. The search covers FP32/FP64, radix-2/4/8, 128/256 threads,
and the legal temporal, hierarchical, online-reorder, warp-hybrid, and
stage-pipeline mappings at `logN=8/12/16/20`.

A follow-up experiment adds the generated FP32 `warp-register` matrix core at
`logN=3..15`. Its comparison points use the same GPU and approximately `2^22`
total points, with 20 warmups, 100 repetitions, and five trials.

## Best Measured Mappings

| Precision | logN | Batch | Best mapping | Local setting | Median ms | Gbutterfly/s |
|:--|--:|--:|:--|:--|--:|--:|
| FP32 | 8 | 16,384 | temporal tile | radix-4, 128 threads | 0.052982 | 316.660 |
| FP32 | 12 | 1,024 | hierarchical | radix-8, 256 threads, 10 local stages | 0.143954 | 174.819 |
| FP32 | 16 | 64 | online reorder | radix-4, 128 threads, `8+8` | 0.314184 | 106.799 |
| FP32 | 20 | 4 | online reorder | radix-8, 256 threads, `10+10` | 0.324895 | 129.097 |
| FP64 | 8 | 16,384 | temporal tile | radix-4, 256 threads | 0.091587 | 183.184 |
| FP64 | 12 | 1,024 | hierarchical | radix-4, 256 threads, 10 local stages | 0.275149 | 91.463 |
| FP64 | 16 | 64 | online reorder | radix-4, 128 threads, `8+8` | 0.329011 | 101.986 |
| FP64 | 20 | 4 | online reorder | radix-8, 128 threads, `10+10` | 0.344699 | 121.680 |

All eight selected mappings pass forward and inverse verification at batch 4
in both precisions. The full GPU correctness executable also covers the five
mapping families and stage-specific coefficient lists. The 16 preflight rows
are retained in
[`structured_2x2_v100_best_verify.csv`](../results/structured_2x2_v100_best_verify.csv).

## Architecture Result

The selected schedule changes with length and precision even though the graph
paradigm and local matrix semantics remain fixed:

1. `logN=8` keeps the complete transform in one temporal tile.
2. `logN=12` maximizes on-chip stage residence, then executes the short global
   suffix through the hierarchical path.
3. `logN=16/20` prefers two balanced online-reordered dimensions, avoiding a
   long sequence of global stage kernels.
4. FP32 and FP64 often select different radix or thread counts because the
   coefficient/value width changes register, shared-memory, and instruction
   demand.

At `logN=8`, stage-pipeline `Us=4` is 1.874x faster than `Us=1`, matching the
best unfolding factor observed for FWHT and Boolean zeta. This supports the
narrow architecture claim that the stage-space unfolding remains useful after
the pair operation changes from two additions to a dense matrix multiply.

## Generated Register Core

The generated core maps one transform to one CTA. Each thread owns a vector of
four-value register chunks. The first two stages are thread-local, later stages
use warp shuffles, and cross-warp/chunk stages use an XOR-swizzled shared
exchange. The same transport is used by the FWHT register core; only the pair
operation changes to the selected stage matrix.

| logN | Batch | Register ms | Shared-core control ms | Register speedup | Gbutterfly/s |
|--:|--:|--:|--:|--:|--:|
| 8 | 16,384 | 0.043858 | 0.052982 | 1.208x | 382.536 |
| 10 | 4,096 | 0.047903 | 0.060047 | 1.254x | 437.794 |
| 12 | 1,024 | 0.051241 | 0.143954 | 2.810x | 491.127 |
| 15 | 128 | 0.088003 | 0.273521 | 3.108x | 357.459 |

The `logN=8/12` controls are the original best mappings. The `logN=10/15`
controls are focused shared-core reruns because those lengths were outside the
original four-length scan. The result changes the preferred physical unit for
resident FP32 transforms, but does not remove the long-transform design space:
FP64 and `logN>15` still require the general shared, hierarchical, or online
paths, and low-batch occupancy can still favor decomposition.

Compiled V100 resource usage has no local-memory spill. The general per-stage
table uses 58/34/43/76/128/214 registers/thread at
`logN=8/10/12/13/14/15`. Consequently register pressure is a measurable design
parameter rather than an assumed monotonic function of length; generated code
shape and CTA width both affect the allocation.

## Coefficient Policy

The follow-up generator emits two implementations per resident length. A
single matrix selects `broadcast-register`, which loads one `float4` and reuses
it across all stages. A list of `logN` matrices selects `per-stage-table`. The
A/B control repeats the same matrix in that list, so both paths compute the
same transform.

| logN | Broadcast ms | Per-stage ms | Broadcast speedup | Broadcast Gbutterfly/s |
|--:|--:|--:|--:|--:|
| 8 | 0.043448 | 0.043817 | 1.008x | 386.142 |
| 10 | 0.043295 | 0.047913 | 1.107x | 484.390 |
| 12 | 0.047421 | 0.051139 | 1.078x | 530.685 |
| 15 | 0.080517 | 0.087992 | 1.093x | 390.691 |

At `logN=8/10/12`, broadcast reduces registers/thread from 58/34/43 to
35/32/40. At `logN=15` it instead raises allocation from 214 to 254, but remains
9.3% faster with no local-memory spill. Reusing coefficients therefore removes
meaningful load/lifetime work, but it is not universally a lower-occupancy-cost
policy. The policy remains an explicit generated choice rather than an
architecture invariant.

## Matched FWHT Attribution

Under the same shared-memory mapping, Structured2x2 is within 1.2% of FWHT at
FP32 `logN=12`, within 1.1% at FP32 `logN=16`, and within 4.2% for the measured
FP64 points. At FP32 `logN=8`, the same-mapping dense unit is 3.5% slower than
FWHT. These results show that the added arithmetic is mostly hidden by exchange
and memory costs at these fixed-total-point shapes.

The initial exception was FP32 FWHT `logN=12`: its generated warp-register
codelet reached 0.043930 ms, while the best generic Structured2x2 mapping took
0.143954 ms. The new matrix codelet reduces Structured2x2 to 0.051241 ms,
recovering most of that transport/residency gap. This confirms that the old
3.28x difference was a processing-unit implementation gap rather than evidence
against the mapping paradigm.

The initial order-alternating per-stage register-core comparison isolates the
remaining dense pair arithmetic. Structured2x2 is 1.64%, 7.27%, 13.74%, and
27.06% slower than FWHT at `logN=8/10/12/15`, respectively. A new alternating-
order broadcast/FWHT control measures Structured at 0.74% slower, 3.01%
faster, 5.44% slower, and 16.30% slower for those lengths. Thus coefficient
reuse brings the dense unit to practical parity through `logN=12`; the
`logN=15` arithmetic and 254-register footprint remain visible. The increasing
difference at the longer resident points is expected:
both paths now use the same one-kernel resident transport, so the four
multiplications and two additions per dense pair are no longer hidden behind
shared/global exchange to the same degree. This residual is a physical-unit
specialization opportunity, not a reason to change the architecture schedule.

The initial sequential FP32 `logN=20` scan measured Structured2x2 13% faster
than the same-mapping FWHT point. An order-alternating ten-trial confirmation
does not reproduce that result: Structured2x2 has a 0.324923 ms median versus
0.319527 ms for FWHT, or 1.69% lower throughput. NCU base-clock time agrees,
placing Structured2x2 1.36% behind. The initial advantage is retained as raw
evidence of a frequency/order-sensitive scan, but it is not a performance or
mechanism claim.

## Counter Attribution

The current focused NCU capture contains all nine planned A/B cases and 15
kernels, including both coefficient policies.
Its main findings are:

1. At `logN=8`, the dense unit executes 2.00x the FP32 instructions but takes
   only 1.060x the time under the identical temporal radix-4 mapping. Both
   retain more than 91% active warps.
2. At `logN=12`, matched hierarchical paths differ by 3.0%. The dense prefix
   uses 39 versus 23 registers/thread and reduces prefix active warps from
   92.6% to 70.7%; the two suffix kernels remain essentially equal.
3. The FWHT warp-register codelet is 3.08x faster than hierarchical FWHT and
   moves 3.04x less DRAM traffic because it keeps all stages in one kernel.
   This observation motivated the now-implemented Structured register core.
4. At `logN=20`, dense and FWHT paths move the same DRAM volume. Structured2x2
   executes 1.20x the warp instructions and 1.98x the FP32 instructions, while
   low DRAM utilization allows most of the extra arithmetic to overlap.

The complete counter table and interpretation are in
[`ncu_structured_2x2_attribution.md`](../results/ncu_structured_2x2_attribution.md).
The first refreshed capture included the generated per-stage Structured
register row. At
`logN=12`, it reduces the hierarchical path from three kernels to one, DRAM
traffic from 96.49 to 32.00 MiB, and NCU base-clock time from 147.104 to 55.648
us, a 2.64x improvement. Against the same-transport FWHT register core, it has
1.010x the DRAM traffic but 2.07x the warp instructions, 1.80x the FP32 thread
instructions, 43 versus 32 registers/thread, and 56.1% versus 66.2% active
warps. Shared bank conflicts are negligible for both. This rules out global
layout and shared-bank behavior as the primary remaining cause and attributes
the residual to dense-pair arithmetic plus its register/occupancy cost. The
NCU time gap is 18.70% under base clocks; the CUDA-event steady-state authority
remains the 13.74% gap reported above.

The coefficient-policy recapture supersedes that row for the optimized
broadcast path. Broadcast takes 52.992 us versus 56.544 us for the equivalent
per-stage table, a 1.067x NCU speedup consistent with the 1.078x CUDA-event
result. FP32 instructions remain 100.66 M and DRAM changes by less than 0.6%,
while registers fall from 43 to 40, active warps rise from 56.1% to 65.5%, and
long-scoreboard stall falls from 25.5% to 19.2%. Against FWHT, broadcast has
essentially equal active warps (65.5% versus 66.0%) and DRAM (1.006x), but
1.80x FP32 and 2.05x warp instructions. The current `logN=12` residual is
therefore dense arithmetic, not an achieved-occupancy or data-layout collapse.

## Reproduction

The timing records are
[`structured_2x2_v100_raw.csv`](../results/structured_2x2_v100_raw.csv) and
[`structured_2x2_v100_summary.csv`](../results/structured_2x2_v100_summary.csv).
The matched FWHT records use the `fwht_matched_v100_*` names. The focused
cross-operator `logN=8` protocol is stored in
`butterfly_operators_structured_v100_*`. The order-alternating long-transform
confirmation is `structured_fwht_log20_interleaved_v100_raw.csv`.
The generated-core timing records are `structured_register_v100_raw.csv` and
`structured_register_v100_summary.csv`; the alternating FWHT control is
`structured_register_vs_fwht_v100_raw.csv`. Focused shared controls use the
`structured_shared_log10*` files.
The coefficient-policy A/B records are
`structured_broadcast_policy_v100_raw.csv` and
`structured_broadcast_policy_v100_summary.csv`; compiled resources are in
`structured_register_resources_v100.csv`. The alternating FWHT confirmation is
`structured_broadcast_vs_fwht_v100_raw.csv`.

Run the counter attribution with administrator performance-counter permission:

```bash
sudo --preserve-env=PATH \
  OUTPUT_DIR="$PWD/results/ncu_structured_2x2_register" \
  ./scripts/profile_structured_2x2_ncu.sh
```

NCU replay time is attribution evidence only. CUDA-event medians above remain
the performance authority.
