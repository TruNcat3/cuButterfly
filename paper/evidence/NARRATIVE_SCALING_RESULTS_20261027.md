# M1 Local Publication Scaling Results, 2026-10-27

## Outcome And Scope

The separate A100 PCIe 40GB cohort completed all 18 workloads, 54 organization
cells and 162 timed trials. Across the six workloads of each operator, the
geometric mean of `median(local0) / median(local2)` is 1.196x for FFT, 1.106x for
FWHT and 1.199x for NTT. Wider grouping is not monotonically better:
`local4` gives 0.932x, 1.103x and 0.816x respectively. These are descriptive
ratios from three trials per cell, not significance tests or guaranteed gains.
Several trial ranges overlap substantially or contain large variation.

The supported claim is local publication/register-residency scaling within this
fixed scalar radix-2 implementation and finite domain. Local grouping changes
ownership, synchronization and resource demand together. It does not isolate a
single latency-hiding mechanism, prove a global theory, cover the complete HHD
space, or establish competitive performance against optimized libraries.
`local0` is the fixed shared-publication control, not a claimed searched or
optimized conventional baseline. FP64 was selected before measurement; no FP32
replacement cells are included.

## Accepted Evidence And Protocol

The authoritative entry point is the
[per-case collection](../../results/narrative_gap_20261027/m1_gpu40_cases/collection.json),
SHA256 `2d033a697c14308e8a052dbfee3e3035942e605728809bddeca919a512ef975f`.
Its 18 records reference original release paths and SHA256 values; this report
read every referenced release and independently checked each hash and the
queue's `validate_release` contract. The contract checks each subset's study,
source, binaries, hardware, compile mode and protocol identities, accepted
preflights, complete trial IDs, mappings and comparison acceptance. No timings
were copied from another attempt or assigned a fabricated parent-study identity.
The [collection manifest](../../results/narrative_gap_20261027/m1_gpu40_cases/manifest.json)
retains the parent study and each immutable subset's identities.

Hardware is NVIDIA A100-PCIE-40GB, compute capability 8.0, UUID
`GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e`, CUDA global memory 42285268992 bytes,
NVML memory 40960 MiB, driver 550.127.05. The explicit static capacity model is
[a10040_hardware_model.json](../../results/narrative_gap_20261027/a10040_hardware_model.json),
SHA256 `4d6f1513411d307a2a560dd8d4654c26e4bc14478640397cb6f7276316f6d0c8`.
The original 80GB attempts and interrupted whole-study 40GB attempts are retained
only as diagnostic/quarantined history. Neither contributes accepted data here;
this is a separate hardware cohort, not fallback pooling.

Every cell is forward, unnormalized, out-of-place, natural external order,
shared-iterative, scalar radix-2, linear shared layout, no overlap and
`batch_tile_count=1`. FFT/FWHT use FP64 storage and native FP64 arithmetic;
FFT uses four-multiply complex arithmetic and table coefficients. NTT uses
word64 Shoup multiplication with exact prime `576460756061519873`.
At logN16 the physical groups are 8+8 with 128 threads; at logN20 they are
10+10 with 256 threads. B is 1, 4 or 16.

Each cell has one full-batch verification preflight and three timed processes,
each with 10 warmups and 20 repeats. Timed results are kernel-event means per
full batched transform call; reported central values are medians of those three
process results. The trial orders alternate local0/local2/local4 and its reverse.
Preflight timing, host references, transfers and setup are not used as kernel
measurements. Three trial values are not 60 independent observations and do not
provide a confidence interval. Guard acceptance is not a stability guarantee.

## All Workloads And Raw Trial Ranges

Each timing column contains all three raw trial means in trial-ID order,
in milliseconds. Thus its minimum and maximum give the complete observed
three-trial range without discarding an extreme. `R2` and `R4` are ratios of
the two separate medians, `median(local0)/median(local2 or local4)`; greater
than one favors grouping. They are not medians of paired trial ratios.
Case IDs directly index the original release/hash records in the collection.

| Case | local0: three trials (ms) | local2: three trials (ms) | local4: three trials (ms) | R2 | R4 |
| --- | --- | --- | --- | --- | --- |
| fft-fp64-n16-b1 | .025805/.024013/.016333 | .021760/.023603/.023654 | .025446/.031539/.027699 | 1.017 | .867 |
| fft-fp64-n16-b4 | .061594/.056166/.056269 | .048691/.041165/.051610 | .077773/.048845/.081971 | 1.156 | .724 |
| fft-fp64-n16-b16 | .210688/.210432/.210688 | .172288/.185088/.163994 | .300237/.265933/.174029 | 1.223 | .792 |
| fft-fp64-n20-b1 | .279040/.260147/.259482 | .219904/.217344/.209510 | .259942/.194714/.247654 | 1.197 | 1.050 |
| fft-fp64-n20-b4 | .866560/.799642/.842342 | .678195/.556800/.740762 | .793754/.862618/.900045 | 1.242 | .976 |
| fft-fp64-n20-b16 | 3.083827/3.334758/2.329754 | 2.254489/2.686362/2.023731 | 2.525235/2.395392/2.396877 | 1.368 | 1.287 |
| fwht-fp64-n16-b1 | .017459/.016794/.018278 | .015821/.015770/.016282 | .016947/.018278/.011008 | 1.104 | 1.030 |
| fwht-fp64-n16-b4 | .030720/.031744/.036454 | .028570/.025293/.029491 | .021606/.021914/.034253 | 1.111 | 1.449 |
| fwht-fp64-n16-b16 | .075827/.085862/.084326 | .071270/.083814/.079462 | .094720/.094771/.056371 | 1.061 | .890 |
| fwht-fp64-n20-b1 | .101939/.107213/.107264 | .083456/.083610/.092211 | .091597/.063744/.101325 | 1.282 | 1.170 |
| fwht-fp64-n20-b4 | .413389/.413030/.413491 | .390195/.360243/.278016 | .369357/.363571/.380416 | 1.148 | 1.119 |
| fwht-fp64-n20-b16 | 1.479373/1.549005/1.479475 | 1.028557/1.551411/1.552794 | 1.494477/1.426074/1.038336 | .954 | 1.037 |
| ntt-word64-n16-b1 | .030566/.025907/.026726 | .026317/.020378/.024269 | .033894/.020531/.030976 | 1.101 | .863 |
| ntt-word64-n16-b4 | .055757/.053709/.058829 | .054323/.049562/.049510 | .083354/.083302/.052531 | 1.125 | .669 |
| ntt-word64-n16-b16 | .180736/.141210/.163174 | .177920/.098611/.147814 | .162816/.257485/.291789 | 1.104 | .634 |
| ntt-word64-n20-b1 | .255539/.241203/.298547 | .212838/.145818/.212736 | .279194/.326758/.286003 | 1.201 | .893 |
| ntt-word64-n20-b4 | 1.046630/.893850/1.033062 | .704768/.533043/.735642 | 1.032858/1.032653/1.033421 | 1.466 | 1.000 |
| ntt-word64-n20-b16 | 4.020224/2.474086/2.466202 | 2.629888/1.991014/2.007143 | 3.309209/2.747392/2.747955 | 1.233 | .900 |

The NTT logN20 B4 local4 ratio is 1.000198 before rounding, effectively a tie
at the displayed precision. A formal stability conclusion cannot follow from
this tiny difference. FFT logN16 B1 local0 spans .016333--.025805 ms; FWHT
logN20 B16 local2 spans 1.028557--1.552794 ms; NTT logN20 B16 local0 spans
2.466202--4.020224 ms. These ranges are retained, not filtered away.

| Operator (six workloads each) | R2 geometric mean | R2 min--max | R4 geometric mean | R4 min--max |
| --- | --- | --- | --- | --- |
| FFT | 1.196 | 1.017--1.368 | .932 | .724--1.287 |
| FWHT | 1.106 | .954--1.282 | 1.103 | .890--1.449 |
| NTT | 1.199 | 1.101--1.466 | .816 | .634--1.000 |

Each operator gets its own equal-workload geometric summary. There is no pooled
arithmetic mean across operators, sizes, batches or raw times. Local2 has a
positive median ratio in 6/6 FFT, 5/6 FWHT and 6/6 NTT workloads; this count is
descriptive, not a significance result. Local4 is slower in all three logN16
FFT cases and all three logN16 NTT cases. The worst NTT ratio, .634x at B16,
means local4 takes about 1.578x the local0 median time.

## N And Batch Throughput Scaling

Point throughput is `N*B/(median_ms*1e6)` Gpoints/s. It counts complex points
for FFT and scalar/residue points for FWHT/NTT; it is not a cross-operator FLOP
comparison. Increasing B from 1 to 4 or 16 increases total work by 4x or 16x.
The following are throughput ratios, not latency speedups; entries are ordered
local0/local2/local4. The final column gives absolute B16 point throughput.

| Operator | logN | B4/B1 throughput | B16/B1 throughput | B16 Gpoints/s |
| --- | --- | --- | --- | --- |
| FFT | 16 | 1.707/1.939/1.425 | 1.824/2.192/1.667 | 4.977/6.086/3.943 |
| FFT | 20 | 1.235/1.282/1.148 | 1.350/1.542/1.653 | 5.440/7.442/7.000 |
| FWHT | 16 | 2.200/2.215/3.093 | 3.313/3.186/2.863 | 12.435/13.196/11.070 |
| FWHT | 20 | 1.037/.928/.992 | 1.159/.862/1.028 | 11.340/10.814/11.765 |
| NTT | 16 | 1.917/1.959/1.487 | 2.621/2.627/1.925 | 6.426/7.094/4.072 |
| NTT | 20 | .989/1.207/1.108 | 1.653/1.696/1.665 | 6.781/8.359/6.105 |

For example FFT logN20 local2 grows from .217344 ms at B1 to 2.254489 ms at
B16: total batch latency increases about 10.373x, while point throughput
increases 1.542x because work increased 16x. Conversely FWHT logN20 local2
has .862x the B1 throughput at B16, so this cohort does not support a uniform
batch-scaling benefit.

At fixed B, increasing logN16 to logN20 increases N by 16x and radix-2 stage
work `N*log2(N)` by 20x. The table gives point-throughput ratios
`16*median_ms(N16)/median_ms(N20)`, again local0/local2/local4. It is not a
fixed-work speedup; physical groups and thread counts also change with N.

| Operator | B1: N20/N16 | B4: N20/N16 | B16: N20/N16 |
| --- | --- | --- | --- |
| FFT | 1.477/1.738/1.790 | 1.069/1.149/1.443 | 1.093/1.223/1.775 |
| FWHT | 2.606/3.028/2.960 | 1.229/1.269/.949 | .912/.820/1.063 |
| NTT | 1.673/1.825/1.733 | .864/1.125/1.290 | 1.055/1.178/1.499 |

These finite sweeps show that larger total work does not imply proportionally
larger throughput, and larger N does not uniformly improve point throughput.
The raw trial variation limits stronger scaling conclusions.

## Numerical Gates

All 54 preflights have `correct=1` and `verified_batches=B`. In particular, all
18 NTT cells verify every batch by exact integer equality, with the exact
decimal modulus and word_bits=64 recorded in each release sample. NTT has no
floating `max_error` field; no rounded floating zero is used as evidence of
integer exactness. The timed processes do not repeat verification:
`verified_batches=0` and `correct=-1` there are expected, not extra correctness
claims.

All 36 floating preflight cells print `max_error=0.000000`. The benchmark
prints fixed six-decimal output and converts the internally computed double
maximum to float for this field, so this means a rounded reported maximum,
not bitwise equality or an error distribution. FFT's gate compares its double
maximum complex absolute error with 1e-10. FWHT's gate compares its double
maximum absolute error with `1e-10*sqrt(max(1,N/1024))`, namely 8e-10 at logN16
and 3.2e-9 at logN20. These existing limits were not changed. The pass flags
and rounded fields do not complete full error-distribution validation or the
user's floating-point acceptance verdict; both remain unfinished.

## Source, Modules And Runtime Mapping Audit

The [40GB preparation record](../../results/narrative_gap_20261027/m1_gpu40/preparation.json)
binds all 54 case/variant pairs to 18 batch-independent request IDs and records
each generated-source, module and compiler-manifest path and full SHA256.
All 18 source/module/manifest hashes were checked against the retained files.
All 54 runtime preflight resolved mappings were independently projected through
the current `compile_request(..., policy="research")`; every resulting request
hash matches the recorded case/variant request ID. This confirms the requested
physical/local grouping projection, including local4's two-stage remainder,
without requesting new compilation or GPU execution.

Generated local2/local4 sources instantiate `shared_resident_kernel` with the
recorded group offsets and explicit local partitions; local0 uses
`shared_iterative_kernel`. In `src/shared_resident.cuh::run_leaf`, each thread
owns all `2^width` register values in a leaf and invokes the same scalar
`op.apply` radix-2 primitive for each internal stage. This is source-primitive
invariance, not identical machine code across organizations. The changed
fusion/publication schedule and compiler resource differences are intentional.

Every runtime cell reports two physical execution groups: offsets 0/8 with
8 stages and 128 threads at logN16, or offsets 0/10 with 10 stages and 256
threads at logN20. Both groups report the requested local partition: empty
for local0, 2+2+2+2 or 2+2+2+2+2 for local2, and 4+4 or 4+4+2 for local4.
All cells use the same recorded runtime fingerprint
`a4e20fbc5253a083e2bb40b12427b2d727c2b44a55c9312176496c63a93579f7:research`.
This fingerprint is not a per-module runtime-load attestation; the preparation
module identities and matched request projections are retained separately.

The following module cache-identity prefixes distinguish all 18 modules; full
identities and module binary SHA256 values are in the preparation record.
The register counts are the two physical groups' runtime-reported compiler
registers per thread, not achieved occupancy or profiler counters. Reuse across
B retains the same module request.

| Operator | logN | Organization | Cache identity prefix | Registers/thread, groups 0/1 |
| --- | --- | --- | --- | --- |
| FFT | 16 | local0 | `5ec79c06f178395e` | 32/27 |
| FFT | 16 | local2 | `b08d490d7c82c023` | 42/42 |
| FFT | 16 | local4 | `d8d6e6686d8b0287` | 138/138 |
| FFT | 20 | local0 | `b27daffc0e71614f` | 32/27 |
| FFT | 20 | local2 | `a10e70d659f3dcbd` | 42/40 |
| FFT | 20 | local4 | `78b1fef60d939ace` | 106/104 |
| FWHT | 16 | local0 | `dba46f65cabce6f4` | 20/20 |
| FWHT | 16 | local2 | `3631c738ecb22b65` | 26/26 |
| FWHT | 16 | local4 | `5772b94bd2928759` | 55/55 |
| FWHT | 20 | local0 | `f5d03bcd6a1f551f` | 20/20 |
| FWHT | 20 | local2 | `8b762c9c49eb9b3e` | 26/26 |
| FWHT | 20 | local4 | `9425b47760133cd0` | 55/55 |
| NTT | 16 | local0 | `56bf2abb589683b2` | 30/32 |
| NTT | 16 | local2 | `e769e512d265d66a` | 32/32 |
| NTT | 16 | local4 | `e94c70fc2ec23790` | 128/168 |
| NTT | 20 | local0 | `f4ed78c52b0da163` | 28/32 |
| NTT | 20 | local2 | `0be522eadee9927e` | 36/32 |
| NTT | 20 | local4 | `534a9299634d791a` | 124/166 |

The source-derived underownership warning remains important. For an 8-stage
tile, width2 has 64 leaf owners among 128 launched threads, while width4 has
only 16/128. For a 10-stage tile, width2 has 256/256 and width4 64/256; the
local4 width-2 remainder has 256/256. These are logical owner counts, not
measured occupancy. The logN16 local4 regressions are consistent with an
ownership/resource tradeoff, but timing alone does not prove that mechanism
caused them. Wider register residency is not unconditionally beneficial.

## Counters And Trace

The [matched nine-cell profile analysis](NARRATIVE_SCALING_PROFILE_20261027.md)
records actual kernels, launches, resources and selected traffic/stall counters
for logN20 B16. Local4 raises registers, lowers active warps and raises barrier
stall percentages; local2 has nearly unchanged DRAM traffic. Shared-publication
counts, L2 traffic volume and spills were not measured, so these observations
bound a resource/ownership interpretation without establishing a unique cause.

## Reproduction And Related Artifacts

The [preparation report](NARRATIVE_SCALING_PREPARATION_20261027.md) records finite
cohort construction, compiler policy and capacity checks. The parent-owned
[plot data](../generated/narrative_scaling_20261027.json) derives from this
collection; original releases remain authoritative. This analysis changed no
runtime, thresholds, GPU results, plots or TeX and launched no GPU work.
