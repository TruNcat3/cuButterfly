# NTT comparison and bounded search repair

The October 17 investigation separates two effects. The earlier 1.472x NTT
headline describes a different 214-pair population. The pre-repair 1.085x
covers five word64 requests. Within the current population, a real search omission
also selected a slower mapping for N65536, B4, forward NTT. The existing
Hybrid2D template already contains a faster legal mapping.

Research revision: `0.9.3.11-search.1`. This revision changes finite public
search sampling and adds trace records. It adds no arithmetic kernel and no
GPU model, size or batch exception. The original measured library is retained
at `results/ntt_diagnostic_20261017/original_libcuntt.so` with its original
SHA256 `cc78cfad96c1cba5ea9b64afef24f9e4f062aedf4e41d18c0e9faa9b7c29a169`.

## Population and numerical evidence

The [historical audit](ntt_comparison_history_20261017.md) identifies every
headline and the complete semantic intersections. RC2 combines word32/word64,
cyclic/coset/negacyclic transforms, different moduli and larger batches. Its
word64 subgroup is 1.334x. The October 12 four-request intersection with today's
matrix has GM 1.195x; today's same requests have GM 1.166x. Those runs used
different physical GPUs and builds. Adding today's B4 loss, 0.814x, lowers
the five-request GM to 1.085x. None of these calculations removes a loss.

The [66-case numerical review](install_comparison_results_20261017.md) checks
194 retained same-precision output artifacts. All 131 floating implementation
records are finite and all 63 integer implementation records are exact.
These counts include cuButterfly and external CPU/GPU implementations.
cuButterfly contributes 37 floating and 29 integer requests. Floating error
distributions remain reports without a newly assigned acceptance threshold.

## Why the public search missed the better mapping

The original B4 search logged 20,758 candidates and a 512-measurement budget.
The old sampler selected indices 0, 40, 81, 121, and so on. All 30 Hybrid2D
candidates occupied the contiguous interval 1--30 or 2--31, depending on the
platform-prior prefix. The sampler skipped every one. Only the separately
inserted first incumbent retained a Hybrid2D option.

The shared sampler now distributes budget across execution families before
sampling within a family. Butterfly keys include backend, compute core,
local exchange and stage overlap. NTT keys include backend, compute core,
hierarchical core and stage overlap. Small families saturate and release
their unused quotas to larger families. The incumbent and original inventory
order remain intact. Budget zero still enumerates the full inventory; a
budget smaller than the family count cannot cover every family.

This repair addresses candidate admission. A cost model cannot rank a mapping
after the finite runtime schedule has omitted it. Installation calibration,
public MEASURE search and explicit replay remain separate evidence. The
new sampled search needs current finalist confirmation; old calibration does
not by itself qualify the changed sampler. The B4 and remaining four
confirmations below have now completed.

## Controlled execution diagnostic

The [finite replay and sweep](../results/ntt_diagnostic_20261017/paired_sweep/release/results.json)
uses the original library on A100 80GB PCIe UUID
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`. It preserves the five original
contracts and tests the Hybrid2D neighborhood plus four shared configurations
and two other backend requests for the B4 case. There are 39 exact passing
candidate/request records and one explicit unsupported CompactStage request.
The original five selections receive three alternating timing rounds within
one process. Additional candidates receive one round. These are diagnostic
measurements, not independent process replications or replacement public rows.

| N65536/B4 forward word64 mapping | Complete Graph latency | GPU-NTT / cuButterfly |
| --- | ---: | ---: |
| Original public choice, n1_log=6, radix4, rows4 | about 32 us | 0.826--0.830x on replay |
| Existing n1_log=8, radix4, rows4 | 24.498 us | 1.116x in the diagnostic sweep |

Both use the same template, precision, modulus, input and required output
order. Planning and compilation stay outside the 20-call Graph timing.
The [original trace](../results/ntt_diagnostic_20261017/paired_sweep/release/cubutterfly_trace.json)
contains five calls, with first-pass total 112.86 us and second-pass total
46.40 us. The first pass accounts for about 71% of their summed kernel time.
Profiler durations describe this diagnostic and are not the Graph latency.

The launch equations explain the imbalance. With n1_log=6 and rows4, the first
pass has B*64/4=64 CTAs, fewer than the GPU's 108 SMs, and each row spans 1024
elements. The second pass has 1024 CTAs and rows of 64 elements. With n1_log=8,
both passes have 256 CTAs and rows of 256 elements. This is evidence of a
missed organization choice, not evidence that HBM bandwidth is saturated or
that the theoretical framework fails. It also does not prove global optimality.

A [paired trace](../results/ntt_diagnostic_20261017/balanced_trace/release/results.json)
on the second A100 80GB, UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`,
confirms the launch geometry for both choices with the same original binary.
Each trace contains five calls. Registers remain 32 per thread in both stages;
the improvement does not require a different arithmetic core.

| Mapping | First / second pass CTAs | First / second shared bytes per CTA | First / second summed kernel time, us |
| --- | ---: | ---: | ---: |
| Original n1_log=6 | 64 / 1024 | 32800 / 2080 | 112.353 / 46.432 |
| Balanced n1_log=8 | 256 / 256 | 8224 / 8224 | 54.397 / 64.317 |

The first pass becomes much shorter while the second becomes longer. Their
combined trace duration falls from 158.785 to 118.714 us. Separately measured
complete Graph latency falls from a three-round median of 31.437 us to
24.477 us in one diagnostic round. These traces describe kernel scheduling
and duration, not NCU utilization or stall counters.

The matched GPU-NTT trace contains two core launches and one bit-reversal copy
per complete call. Across five calls their summed durations are 102.018 and
31.614 us respectively. The copy is part of the requested natural-order
contract and remains inside timing. The speedup therefore compares complete
ordered transforms; it does not establish a faster bare arithmetic core.

The current public NTT tuner still ranks by its ordinary host-submitted event
loop. The external comparison uses CUDA Graphs. The trace explicitly labels
this difference; changing the measurement objective is a separate remaining
issue. The sampling repair is validated against the final complete Graph call.

This follow-up can still compile newly admitted modules between measurements.
The existing CPU broker accepts frozen mappings, but the public NTT entry does
not yet export its exact post-incumbent, post-budget schedule before creating
a plan. The full-inventory listing is not that schedule. Consequently, this
run excludes compilation from execution timing but does not establish fully
CPU-prepared public search. Completing that export is a separate engineering
task; the run does not substitute a guessed candidate list.

## Additional external FFT comparisons

The new baseline adapter composes vendor transforms in groups of at most three
consecutive axes. Rank4 uses 3+1 and rank8 uses 3+3+2. It preallocates two
packing buffers and includes every packing copy, vendor execution, output
copy and required normalization in the captured call. It preserves logical
xD semantics and never flattens them into a different one-dimensional FFT.

The [grouped supplement](../results/ntt_diagnostic_20261017/fft_gaps_grouped/release/supplement.json)
was paired on the second A100 80GB PCIe UUID
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`, with explicit replay of the frozen
cuButterfly choices. Both requests keep the original inputs, checked by hash.

| Request | cuButterfly us | cuFFT composition us | VkFFT composition us |
| --- | ---: | ---: | ---: |
| FP32 rank4, 8x8x8x8, B4, forward | 8.243 | 17.101 | 16.384 |
| FP32 rank8, eight axes of 2, B16, normalized inverse | 11.827 | 32.358 | 31.744 |

All outputs are finite against the same-precision CPU reference. Rank8 outputs
are exact on this input; rank4 relative L2 errors are 1.364e-7, 1.059e-7 and
1.262e-7 respectively. These four external records are explicitly classified
as `external-axis-composition`. They close four adapter slots, leaving eleven
GPU slots without a compatible implementation. They remain separate from
native-rank1--3 vendor averages. The earlier single-axis prototype is retained
as diagnostic evidence but is superseded by the stronger grouped baseline.

## Validation and remaining scope

The bounded-sampler and planner-trace CPU suites pass 13 tests. The grouped
adapter, synchronous/asynchronous runner and pipeline suites pass 228 tests.
The shared library builds successfully. The finite GPU sweep verifies exact
NTT outputs, and four grouped FFT baseline records complete real CUDA Graph
execution with same-precision error artifacts.

The repaired public 512-candidate NTT search for N65536/B4 completed under
`results/ntt_diagnostic_20261017/automatic_b4`. It automatically selected
Hybrid2D radix4 with n1_log=8 and rows4. Across three alternating timing rounds,
the median complete Graph latencies are 23.450 us for cuButterfly and 25.805 us
for GPU-NTT, a ratio of 1.1004x. Every output matches the exact CPU reference.
Plan construction, including JIT compilation and candidate verification, took
1356.660 seconds and is explicitly excluded from these execution latencies.

The remaining four original requests completed under `automatic_remaining`
on the same GPU and new library. A deliberate
[execution-window adjustment](../results/ntt_diagnostic_20261017/remaining_window_adjustment.md)
retains the incomplete first attempt as diagnostic evidence. A second-card
N65536/B1 search also completed separately; its timing does not enter the
primary five-point mean. The reproducible
[follow-up summarizer](../results/ntt_diagnostic_20261017/summarize_followup.py)
retains the old, replay and new public-search cohorts separately. The new
public-search cohort has five valid Graph timing rounds per request, one common
GPU UUID and one library hash. Its five-point geometric mean is **1.1765x**.
The per-request ratios are 1.434x (N4096 inverse), 1.260x (N4096 forward),
0.993x (N65536 forward), 1.141x (N65536 inverse), and 1.100x (N65536 B4
forward). The N65536 forward B1 point remains approximately tied and is the
remaining weak point in this five-case sample.
The original 66-case table remains frozen. Broader mechanism/application
claims, untested semantic combinations and CPU timing without host contention
remain separate work; these finite supplements do not mark them complete.
