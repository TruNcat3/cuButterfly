# M1 Matched Local-Publication Profiles, 2026-10-27

## Outcome And Scope

All nine logN20, B16 cells now have complete FFT/FWHT/NTT profile releases. Actual kernels confirm the local-publication intervention: local0 uses `shared_iterative_kernel`; local2/local4 use `shared_resident_kernel` with the requested leaf partitions. The strongest consistent physical observation is not a monotone benefit: local4 increases registers, reduces measured active warps and increases barrier-stall percentage for both groups of every operator. Local2's DRAM traffic is nearly unchanged from local0. These records support a resource/ownership tradeoff and limit an HBM-saving explanation, but do not directly measure shared-publication counts or establish a unique performance cause.

This is a local, read-only analysis of the following released profiles and their matched ordinary timing journals. No GPU work, new timing, runtime change or extrapolation to unprofiled workloads was performed. The [main M1 results](NARRATIVE_SCALING_RESULTS_20261027.md) remain authoritative for the 18-workload timing result and source-level ownership analysis; this report covers only three workloads / nine organization cells / 18 main kernel instances.

| Evidence ID / operator | Profile manifest | Matched ordinary journal |
| --- | --- | --- |
| F / FFT FP64 | [fft/release/profile/manifest.json](../../results/narrative_gap_20261027/m1_profiles/fft/release/profile/manifest.json) | [case 005 timing.json](../../results/narrative_gap_20261027/m1_gpu40_cases/cases/005/guards/0000/release/timing.json) |
| H / FWHT FP64 | [fwht/release/profile/manifest.json](../../results/narrative_gap_20261027/m1_profiles/fwht/release/profile/manifest.json) | [case 011 timing.json](../../results/narrative_gap_20261027/m1_gpu40_cases/cases/011/guards/0000/release/timing.json) |
| N / NTT word64 | [ntt/release/profile/manifest.json](../../results/narrative_gap_20261027/m1_profiles/ntt/release/profile/manifest.json) | [case 017 timing.json](../../results/narrative_gap_20261027/m1_gpu40_cases/cases/017/guards/0000/release/timing.json) |

All three manifests record `status=complete`, `failures=[]`; all nine variants record both profilers succeeded and benchmark binary hashes matching the timing journal. Their `full_population=true` means the entire **one-case subset** was profiled, not all 18 M1 workloads. Hardware is the same A100-PCIE-40GB UUID `GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e`, not the original 80GB refresh. FFT/FWHT binary SHA256 is `abeebc62cdbd4a8a34ff9f97d64bdce2000798d014870d81923912b787d4eb6e`; NTT is `93ca7db81fa80f3b8c1c2f724c064bf385abb7ddba1e4942059e9b27dbb3077f`.

## Ordinary Performance, Not Profiler Performance

All times below are milliseconds for the full batched call. Ordinary trials use 10 warmups, 20 repeats, three independent processes; values are trial 0/1/2 and their median, without dropping first trials. S = median(local0) / median(cell), so S>1 favors the cell. Range is observed min-max, not a confidence interval. Forward, unnormalized, out-of-place requests retain scalar radix-2, stage partition 10+10, no overlap and batch tile count 1. NTT q is the exact integer `576460756061519873`.

| Cell | Ordinary trials 0 / 1 / 2 | Median | Range | S |
| --- | --- | ---: | --- | ---: |
| F-local0 | 3.083827 / 3.334758 / 2.329754 | 3.083827 | 2.329754-3.334758 | 1.0000 |
| F-local2 | 2.254489 / 2.686362 / 2.023731 | 2.254489 | 2.023731-2.686362 | 1.3679 |
| F-local4 | 2.525235 / 2.395392 / 2.396877 | 2.396877 | 2.395392-2.525235 | 1.2866 |
| H-local0 | 1.479373 / 1.549005 / 1.479475 | 1.479475 | 1.479373-1.549005 | 1.0000 |
| H-local2 | 1.028557 / 1.551411 / 1.552794 | 1.551411 | 1.028557-1.552794 | 0.9536 |
| H-local4 | 1.494477 / 1.426074 / 1.038336 | 1.426074 | 1.038336-1.494477 | 1.0374 |
| N-local0 | 4.020224 / 2.474086 / 2.466202 | 2.474086 | 2.466202-4.020224 | 1.0000 |
| N-local2 | 2.629888 / 1.991014 / 2.007143 | 2.007143 | 1.991014-2.629888 | 1.2326 |
| N-local4 | 3.309209 / 2.747392 / 2.747955 | 2.747955 | 2.747392-3.309209 | 0.9003 |

Thus local2 loses to local0 by this FWHT median, and NTT local4 loses to local0. FFT local4 is slower than local2 but remains faster than local0 by the median; FWHT local4 has a small median benefit despite its resource loss. Trial ranges overlap substantially. These outcomes are retained; counter selection does not replace or improve them. No significance, equivalence or tuned-conventional-baseline claim follows from these three-trial summaries.

NCU 2024.1.0.0 (build 33681293) uses kernel replay, three passes per kernel, `--cache-control none`, `--clock-control base`, and warns about uncontrolled caches. NSYS is 2023.4.4.54. Each profiler invokes warmup 0 / repeat 1 / `--csv`, without timed verification, unlike the ordinary protocol. Duration below is each profiler's own kernel record, not the benchmark's heavily perturbed event time. g0/g1 are offsets 0/10, respectively; pairs are never pooled across profilers or used as speedup evidence.

| Cell | NCU kernel ms, g0 / g1 | NSYS kernel ms, g0 / g1 |
| --- | --- | --- |
| F-local0 | 1.274976 / 2.324736 | 1.280021 / 2.310816 |
| F-local2 | 1.050592 / 2.000160 | 0.944006 / 1.777448 |
| F-local4 | 1.303200 / 2.474624 | 1.164368 / 2.215642 |
| H-local0 | 0.422048 / 1.242592 | 0.320076 / 0.980741 |
| H-local2 | 0.324896 / 1.229152 | 0.283275 / 1.064167 |
| H-local4 | 0.493696 / 1.111520 | 0.330349 / 0.712378 |
| N-local0 | 1.060000 / 2.955328 | 1.052932 / 2.935877 |
| N-local2 | 0.763232 / 2.417024 | 0.675159 / 2.147530 |
| N-local4 | 1.299200 / 3.270496 | 0.754937 / 2.021157 |

For example, NTT local4's NSYS durations are lower than local0's while its ordinary median is worse. That disagreement is a concrete reason not to infer performance rankings from these single, cold profiler captures.

## Actual Kernels And Physical Resources

The exact demangled template instances are reconstructed by the following notation, omitting only the common function argument list retained in each raw CSV. `O_F = cuntt::traits::FftOperator<cuntt::Complex64,double>`, `O_H = cuntt::traits::FwhtOperator<double>`, and `O_N = cuntt::traits::NttStageOperatorT<unsigned long>`. In every cell, `O` is its operator above and `o` is 0 for g0 and 10 for g1:

```text
local0: cuntt::detail::shared_iterative_kernel<O,20,o,10,0,0,1>
local2: cuntt::detail::shared_resident_kernel<O,20,o,10,0,0,2,2,2,2,2>
local4: cuntt::detail::shared_resident_kernel<O,20,o,10,0,0,4,4,2>
```

Both NCU and NSYS identify exactly these two main kernels per process, one instance per group, and no separate setup kernel. This attests to the emitted kernel family and operator/partition template parameters, not identical machine instructions between organizations or a full runtime-load attestation for every module hash. The two compiled kernels' register counts agree with the corresponding module-resource pairs in the main results report.

Every one of the 18 NCU records launches grid **(16384,1,1)** and block **(256,1,1)**. These are profiler launch fields, not the generic `resources` projection. FFT dynamic shared storage is 32768 bytes/block; FWHT/NTT is 16384. All have zero static shared and 1024 driver bytes, giving reported totals 33792 and 17408 bytes/block, respectively. This capacity does not shrink with local grouping; unchanged allocation must not be read as unchanged shared-access counts.

The following percentages are the exact recorded metrics: active = `sm__warps_active.avg.pct_of_peak_sustained_active`; barrier = `smsp__warp_issue_stalled_barrier_per_warp_active.pct`; long-score = `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`. Active measures active warps, not how many lanes own useful butterfly work. Stall percentages are per active warp, not additive elapsed-time fractions or counts of barriers.

| Cell | Registers/thread g0 / g1 | Active % g0 / g1 | Barrier % g0 / g1 | Long-score % g0 / g1 |
| --- | --- | --- | --- | --- |
| F-local0 | 32 / 27 | 49.33 / 49.05 | 13.03 / 14.26 | 44.19 / 61.47 |
| F-local2 | 42 / 40 | 49.16 / 48.86 | 14.22 / 12.24 | 51.38 / 61.64 |
| F-local4 | 106 / 104 | 24.46 / 24.40 | 42.72 / 50.17 | 32.16 / 35.91 |
| H-local0 | 20 / 20 | 97.22 / 97.48 | 18.55 / 21.87 | 19.71 / 14.80 |
| H-local2 | 26 / 26 | 96.67 / 97.27 | 16.51 / 20.72 | 30.45 / 19.30 |
| H-local4 | 55 / 55 | 48.71 / 47.61 | 39.64 / 27.32 | 34.37 / 44.38 |
| N-local0 | 28 / 32 | 98.00 / 97.93 | 11.52 / 18.19 | 24.32 / 23.62 |
| N-local2 | 36 / 32 | 73.39 / 97.69 | 10.81 / 19.26 | 27.86 / 16.54 |
| N-local4 | 124 / 166 | 24.50 / 12.28 | 52.94 / 42.66 | 17.67 / 31.03 |

NCU's `launch__occupancy_limit_registers` block ceilings are F: 8/8, 5/6, 2/2; H: 10/10, 8/8, 4/4; N: 8/8, 6/8, 2/1 for local0/local2/local4 in order. These are register-resource ceilings, not measured CTA residency. Other constraints also exist: for example N-local4 g1 reports a shared-memory occupancy ceiling of one block too, despite the same requested shared bytes. The tables do not attribute every active-warp change solely to register pressure or assume fixed shared-memory configuration/carveout.

## Main-Kernel Traffic

DRAM columns are exact bytes from `dram__bytes_read.sum` and `dram__bytes_write.sum`; L2 is `lts__t_sector_hit_rate.pct`. This L2 metric is a **hit percentage only**: no L2 read/write byte or sector-count metric was requested, so L2 traffic volume is unavailable and cannot be reconstructed from DRAM bytes and hit percentage. There are no imputed zeroes for absent metrics.

| Kernel record | DRAM read bytes | DRAM write bytes | L2 hit % |
| --- | ---: | ---: | ---: |
| F-local0 g0 | 298476032 | 275206144 | 79.43 |
| F-local0 g1 | 543267840 | 280795520 | 79.55 |
| F-local2 g0 | 299029248 | 275302528 | 79.74 |
| F-local2 g1 | 542676992 | 280307584 | 78.26 |
| F-local4 g0 | 293474048 | 275120256 | 78.89 |
| F-local4 g1 | 538267008 | 277998208 | 79.31 |
| H-local0 g0 | 134257280 | 134699392 | 62.24 |
| H-local0 g1 | 139108352 | 140572416 | 90.53 |
| H-local2 g0 | 134225920 | 134955648 | 62.40 |
| H-local2 g1 | 141795968 | 143499648 | 90.31 |
| H-local4 g0 | 134226432 | 136428928 | 61.87 |
| H-local4 g1 | 137722496 | 140525056 | 90.03 |
| N-local0 g0 | 134339456 | 137689088 | 87.38 |
| N-local0 g1 | 400011392 | 145646848 | 87.84 |
| N-local2 g0 | 134281216 | 138670976 | 85.71 |
| N-local2 g1 | 401508864 | 144842240 | 85.92 |
| N-local4 g0 | 134305792 | 141163904 | 85.04 |
| N-local4 g1 | 400341504 | 145863680 | 86.49 |

These are per-main-kernel counters under replay, not an input/boundary/coefficient/output attribution, shared-memory traffic or complete-process HBM accounting. NSYS reports the same transfer inventory for all organizations of an operator: FFT two HtoD copies totaling 285.213 decimal MB and one DtoH copy 268.435 MB; FWHT one HtoD and one DtoH, 134.218 MB each; NTT three HtoD totaling 150.995 MB and one DtoH 134.218 MB. Those are rounded NSYS report values. The process includes allocation, input/coefficient setup and output retrieval, with no operator-only capture range. Host transfers are not added to this DRAM table as if they were matched operator HBM demand.

## Interpretation And Remaining Gaps

The captures support **implementation identity and a bounded ownership/resource tradeoff**, not a universal latency-hiding mechanism. They verify a fixed scalar operator trait with the expected iterative versus resident leaf families and constant external launch geometry. In conjunction with the source audit already recorded in the main report, this is consistent with retaining local values across multiple radix-2 stages and publishing at leaf boundaries. It does not show a different arithmetic primitive, increased overlap, a different global boundary or fewer HBM passes.

Local2 raises registers modestly while retaining FFT/FWHT active-warps percentages; NTT g0 falls from 98.00% to 73.39% while g1 stays near 98%. Its ordinary median improves for FFT and NTT, not FWHT. Per-group DRAM counters remain close to local0 across all three operators, with no systematic L2 hit-rate increase. Thus an HBM-volume reduction is not supported as the explanation for these local2 benefits; shared-access/instruction scheduling changes remain plausible but are not measured directly.

Local4 raises registers sharply and lowers active-warps percentages in all six groups. Its barrier-stall percentage rises in all six, even though the source organization has fewer leaf-publication boundaries. Fewer source barriers/publications therefore must not be equated with less measured barrier waiting, fewer barrier instructions or lower synchronization cost. Logical width4 underownership (64 of 256 threads per width4 leaf, versus 256 of 256 for width2 in this tile) comes from the existing source audit, not from these active-warps counters. The latter cannot reveal lane utilization inside a warp or distinguish ownership imbalance from instruction scheduling and resource effects. Local4's ordinary timing is not uniformly worse than local0, so the resource pattern is not by itself a sufficient performance predictor.

Unavailable in this metric set: shared load/store transactions or bytes; dynamic barrier instruction/publication counts; L2 traffic volume; spill/local-memory traffic; instruction mix; useful active-lane/owner utilization; repeated-counter uncertainty; and time-resolved per-leaf ownership/publication latency. Register use is not proof of spills, barrier-stall percentage is not a barrier count, and long-scoreboard percentage does not identify an exclusive memory level or establish the critical path. One capture per profiler/cell, kernel replay, cold timing and uncontrolled caches further limit causal attribution. No profiler correctness check replaces the ordinary full-batch preflights or the floating-point numerical evidence limits in the main report. No additional search or experiment is implied by these gaps.

## Provenance And Reproduction

All three source-journal SHA256 values were independently checked against their manifest bindings. The profile runner is `09b8805f3148ce0c2076e52a535a39383c5f4590d6f3e159d106a5c0d3fdf8e1`, controlled runner `6fa72cbeb58bf279abdace807f858e1dc50d8eb1e2992c236a62df503db3918e`. Exact profiler commands and measured source identities are retained in the manifests; no current source file is substituted for them.

| Source | SHA256 |
| --- | --- |
| F ordinary journal | `07722e41477320cf7980cb14fb7071ddd94d411e14c599b8b287ef63a5b7c7dc` |
| H ordinary journal | `6b9a68461d7b9132b100fdfa1baec9efae9da14185006c948cd5274eaec9160f` |
| N ordinary journal | `6047ffa17a41fa0013cf29a4e571de97f4b8425edd0dc6c21e9b3e6d1e85e5cc` |
| F profile manifest | `25893b6a160c0208684231f1d40ae923596ab5d387f8318631da2098432253db` |
| H profile manifest | `d4221effbe2c42db5308dc5726510a942b48ff4f97fef44f7b355dbb87e7a13e` |
| N profile manifest | `b2e0a4c1df619f10f8185e315dd84c5a90d8c8dc1a6e232dc790589b517a2cb4` |

Each following pair hashes `ncu.csv` / `nsys.nsys-rep`, respectively. Paths are under the corresponding linked profile manifest's directory, then `<case>/<local>/`; case names are `fft-fp64-n20-b16`, `fwht-fp64-n20-b16`, `ntt-word64-n20-b16`. These release files, not any earlier attempt directories named in command history, supply every counter and duration above.

| Cell | NCU CSV SHA256 | NSYS report SHA256 |
| --- | --- | --- |
| F-local0 | `63f679d94cbe3e7e8e32b985943ca0abba82b9c77c5912fe29f7cd820a6e1833` | `1414bcec64efa9980309a55a680856f092ff54ca18a7d87f8b270cee14a3eea6` |
| F-local2 | `5c781a3b5fd897117770f99442b642c7fe6eb3b0b07d90140c78f787e6609672` | `b62abe45b05ac9433dc1930dc60f4173564566f3d2407cdc4f0856fdd1da57da` |
| F-local4 | `9e2ab7b52b4505a44fc21c88425fa6480762b5942e9bb54c1b731d49ea114533` | `867ed63982603ead7e26b1ac8f547cb5e2b28b3387be35953297d43d0b560cd0` |
| H-local0 | `1bcb0885714a21695f94cde20b8f8bf82fc5dc236da709c92f48e445e10d5500` | `5b3a0c3bc1f971a56117ca4b6da94703de9fc948d8240a3e7494f9498c89c222` |
| H-local2 | `2499b2d960c19ab5df05762c4b8ed28218a74bdf2c0659aede58c0d7940c4129` | `1ed3ac89c56b0bf9614aef3af055b282ca44186f2046b9dd6b2d20c3c8b3907c` |
| H-local4 | `334fd02a7a1504d50ee305d36a32b3b77f0ebb1821ba7e60255d3c7f91a2450c` | `f96f5ef37dcdadca53478812689e7437a666c58b144ef139381630e527c778aa` |
| N-local0 | `7b208795cafb332a0fa785b04e6edac254ddaff018163ca430d40a4af47bc6fb` | `31f75d2140eb8f256ea0e4c86f5bdbaacef090887bc5a39446aa65d13c374105` |
| N-local2 | `4a60e44c98d05f6acb3ec4b6d900035417e024da120482d319127799941180c4` | `30f07d362ab7ea1db5d12b7fff343565895be79f92c054e608623ee9d1b967b6` |
| N-local4 | `93b76a0c8f495879be1aa5412aa3ae4bceef28fbe5ae4f14f42be1774f81f02b` | `240374a9c612b479b165d72cde6e993f70da11d7ac811001ebbc3a62d325b111` |

CPU extraction parsed the CSVs structurally and ran `nsys stats --sqlite <temporary-file> --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum --format csv <release-report>` for each of the nine reports; all returned zero. Temporary exports were removed at completion; released artifacts were not modified. This agent-authored analysis is not a declaration of human scientific approval or submission readiness.
