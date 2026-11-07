# M3 finite core/organization crossing results, 2026-10-27

The tested stronger grouping has **no median timing win** in any of the eight within-core organization comparisons: FWHT columns 4 loses to columns 1 for both local algorithms at B=4/16; NTT data-time 2 loses to data-time 1 for both cores at B=4/16. Individual trials sometimes reverse that ranking. These are bounded negative results, with substantial variation in several cells. They do not establish that grouping is generally harmful, that a particular latency mechanism caused the losses, or that either reference organization is a tuned conventional baseline.

This report analyzes existing artifacts only. It adds no GPU measurement, test, search or runtime change. It uses the [readiness audit](CORE_CROSSING_READINESS_20261027.md) for the core definitions and support boundaries; the source check below only confirms what NTT data-time changes physically.

## Cohorts and validity

| Retained evidence | Scope and recorded status |
| --- | --- |
| [Mixed parent timing](../../results/narrative_gap_20261027/m3_gpu40/guard/release/timing.json) | `complete-with-gaps`: retain its eight FWHT cells only. Its eight NTT preflights failed because the attempted study used the rounded composite modulus. They have zero timed trials and are excluded from all tables below. |
| [Corrected NTT retry timing](../../results/narrative_gap_20261027/m3_ntt_retry/guard/release/timing.json) | `complete`: eight NTT cells, eight accepted exact preflights and 24 timed trials. This is a separate corrected request cohort, not a relabeling of the failed NTT attempts. |

Both retained cohorts identify NVIDIA A100-PCIE-40GB, UUID `GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e`, NVML capacity 40960 MiB, CUDA-reported capacity 42285268992 bytes, SM80, driver 550.127.05, build `build-e13-cufftdx`, research compile policy and runtime fingerprint `a4e20fbc5253a083e2bb40b12427b2d727c2b44a55c9312176496c63a93579f7:research`. Raw identity and guard records remain in the linked result directories. These are A100 40GB results, not the original refresh's A100 80GB measurements.

The protocol is three independent timed benchmark processes per cell, each with 100 warmups and 100 repetitions. Preflight uses `--verify --verify-batches 0` outside timing. Trials 0 and 2 visit variants in listed order; trial 1 reverses the four-variant order. Each raw value below is one process's reported kernel time, not one invocation or an end-to-end application time. Setup, allocation, references and compilation are outside this kernel-time boundary. FWHT uses two kernels/call; NTT uses one cooperative kernel/call, as identified in readiness. No sample has been dropped.

NTT preflight records `correct=1` and `verified_batches=4/16` for all eight cells, supporting exact full-batch reference agreement. FWHT records `correct=1`, full verified batch and rounded `max_error=0.000000` for all eight cells. That printed zero is not a full error distribution or completion of the user's floating-point numerical acceptance policy. `performance_accepted` on timings denotes the controlled runner's admission checks; it does not remove this floating-point evidence gap.

## All 16 measured cells

Times are **milliseconds**. Trial columns follow trial indices 0, 1, 2; range is the observed minimum-maximum, not a confidence interval. FWHT: N=65536, forward unnormalized out-of-place FP32/native arithmetic. NTT: N=1048576, forward natural input/output word64, Shoup, q=`576460756061519873`.

| Operator | B | Core / organization | Trial 0 | Trial 1 | Trial 2 | Median | Range |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| FWHT | 4 | r2-c1 | 0.037304 | 0.036024 | 0.035973 | 0.036024 | 0.035973-0.037304 |
| FWHT | 4 | r2-c4 | 0.025948 | 0.047043 | 0.041329 | 0.041329 | 0.025948-0.047043 |
| FWHT | 4 | r4-c1 | 0.036004 | 0.041042 | 0.031242 | 0.036004 | 0.031242-0.041042 |
| FWHT | 4 | r4-c4 | 0.039260 | 0.039250 | 0.041462 | 0.039260 | 0.039250-0.041462 |
| FWHT | 16 | r2-c1 | 0.072090 | 0.086948 | 0.109558 | 0.086948 | 0.072090-0.109558 |
| FWHT | 16 | r2-c4 | 0.099174 | 0.092826 | 0.069632 | 0.092826 | 0.069632-0.099174 |
| FWHT | 16 | r4-c1 | 0.094556 | 0.091290 | 0.091290 | 0.091290 | 0.091290-0.094556 |
| FWHT | 16 | r4-c4 | 0.098703 | 0.098652 | 0.095293 | 0.098652 | 0.095293-0.098703 |
| NTT | 4 | shared-r4-dt1 | 0.772659 | 0.466893 | 0.466227 | 0.466893 | 0.466227-0.772659 |
| NTT | 4 | shared-r4-dt2 | 0.637839 | 0.586086 | 0.585636 | 0.586086 | 0.585636-0.637839 |
| NTT | 4 | warp-r2-dt1 | 1.083044 | 1.083228 | 1.083105 | 1.083105 | 1.083044-1.083228 |
| NTT | 4 | warp-r2-dt2 | 1.212580 | 1.214628 | 1.211238 | 1.212580 | 1.211238-1.214628 |
| NTT | 16 | shared-r4-dt1 | 1.798277 | 1.620839 | 1.614449 | 1.620839 | 1.614449-1.798277 |
| NTT | 16 | shared-r4-dt2 | 1.855611 | 1.856051 | 1.854372 | 1.855611 | 1.854372-1.856051 |
| NTT | 16 | warp-r2-dt1 | 3.897856 | 3.867924 | 3.881431 | 3.881431 | 3.867924-3.897856 |
| NTT | 16 | warp-r2-dt2 | 4.221532 | 4.261048 | 4.272630 | 4.261048 | 4.221532-4.272630 |

## Crossed comparisons

Define S = median(reference time) / median(treatment time); S>1 favors treatment. The artifact field named `median_treatment_over_baseline` uses this same speedup convention, despite its name (`scripts/run_controlled_experiment.py:918`). It is a ratio of per-cell medians, not the median of paired ratios. Parentheses give the range of the three raw paired ratios. Each row reports all four comparisons for that request.

| Request | Organization, first core | Organization, second core | Core replacement, first organization | Core replacement, second organization |
| --- | --- | --- | --- | --- |
| FWHT B4 | c1/c4, r2: **0.8716** (0.7658-1.4376) | c1/c4, r4: **0.9171** (0.7535-1.0457) | r2/r4, c1: 1.0006 (0.8777-1.1514) | r2/r4, c4: 1.0527 (0.6609-1.1985) |
| FWHT B16 | c1/c4, r2: **0.9367** (0.7269-1.5734) | c1/c4, r4: **0.9254** (0.9254-0.9580) | r2/r4, c1: 0.9524 (0.7624-1.2001) | r2/r4, c4: 0.9409 (0.7307-1.0048) |
| NTT B4 | dt1/dt2, shared-r4: **0.7966** (0.7961-1.2114) | dt1/dt2, warp-r2: **0.8932** (0.8918-0.8942) | shared/warp, dt1: 0.4311 (0.4305-0.7134) | shared/warp, dt2: 0.4833 (0.4825-0.5260) |
| NTT B16 | dt1/dt2, shared-r4: **0.8735** (0.8706-0.9691) | dt1/dt2, warp-r2: **0.9109** (0.9077-0.9233) | shared/warp, dt1: 0.4176 (0.4159-0.4614) | shared/warp, dt2: 0.4355 (0.4340-0.4396) |

FWHT's small B4 core differences are not stable evidence of superiority: every FWHT core-replacement paired range crosses one. c4 has no median benefit in either core row, even though some individual trials favor c4. NTT's shared-radix4 configuration is faster than the warp-radix2 configuration in every retained trial at either data-time, but this comparison includes their required different spatial mapping. It cannot isolate the effect of radix alone.

Only three trials per cell are available. For example, FWHT B4 r2-c4 spans 0.025948-0.047043 ms, while NTT B4 shared-r4-dt1 starts at 0.772659 ms and then reports 0.466893/0.466227 ms. The first trial remains included; no explanation such as cache, frequency, thermal state or launch noise is established by these records. There are no confidence intervals, statistical significance claims or cross-cohort aggregate speedup. These observations complete the finite timing intervention even with negative organization medians; they do not trigger another optimization matrix.

## Exact mapping differences and physical interpretation

All 16 artifact comparisons have `status=measured`, `performance_accepted=true`, and `actual_changed_axes` equal the declared controls. A separate read-only recursive comparison of the full resolved mapping objects found the following exhaustive differences. `i in {0,1}` denotes both physical stages; this notation specifies each leaf, not a blanket array change permission.

| Factor | Exact full resolved mapping delta | Fixed contract / interpretation |
| --- | --- | --- |
| FWHT c1 -> c4, either radix | `reorder_columns: 1 -> 4`; no other mapping leaf differs. | Online-reorder, scalar FFT-core field, shared local exchange, partition `[8,8]`, 128 threads, direct-strided/global-scratch boundary, no overlap. The existing suffix groups four independent columns; the prefix is unchanged. |
| FWHT r2 -> r4, either columns | `compute_unit: radix2 -> radix4`; no other mapping leaf differs. | Existing scalar paired-stage versus per-stage local algorithm. It changes local fusion/synchronization and belongs to the core factor. The resolved mapping includes inherited `cufftdx-block` segment descriptors, but `fft_core=scalar` and FWHT's audited implementation dispatch determine this path; those descriptors do not establish use of an external cuFFTDx FWHT leaf. |
| NTT dt1 -> dt2, either core | `subgraph_mappings.i.data_time: 1 -> 2` and the corresponding derived `execution_group_mappings.i.data_time: 1 -> 2`; four leaves total. | Hierarchical-dataflow, word64/Shoup, partition `[10,10]`, full-scratch one-buffer boundary, 256 threads/role, data_space=1, role_stages=1, token_interleave=1, coefficient_reuse_stages=0, cta_weight=1, target_ctas_per_sm=2, no stage overlap. |
| NTT shared-r4 -> warp-r2, either dt | Both `subgraph_mappings.i.core` and `execution_group_mappings.i.core`: `homogeneous-radix4 -> homogeneous-warp-radix2`; both arrays' `.i.units_per_cta: 4 -> 8`; eight leaves total. | A core replacement with its required unit mapping. Top-level `compute_unit=radix4` is unchanged metadata; the explicit per-stage core fields identify the actual requested leaves. |

The [readiness audit](CORE_CROSSING_READINESS_20261027.md) already traces both NTT leaves. A bounded confirmation against unchanged `src/ntt.cu` shows the general templates' data-time changes `ceil(transforms/data_time)` packets and the consecutive-transform inner loop for a spatial task: shared-radix4 at `src/ntt.cu:3220` and `:3233`, warp-radix2 at `:3420` and `:3435`. Consumers still wait for the transform's producer completion at `:3242` and `:3443`. However, the subsequently captured shared-radix4 dt1 uses the specialized wrapper, not that general template; the dispatch distinction below qualifies a pure traversal interpretation. The arithmetic local leaf remains fixed within each core pair. Neither dt2 nor this capture proves intra-tile prefetch, overlap or a relaxed full-transform readiness requirement.

FWHT's recorded group summaries reflect suffix launch/storage changes: at c1/c4, suffix CTAs are 1024/256 for B4 and 4096/1024 for B16, with 1024/4096 shared bytes per CTA; prefix CTAs/storage remain 1024 or 4096 and 1024 bytes. Workspace is unchanged at 1048576 bytes for B4 and 4194304 for B16. Counts are launch descriptors, not measured occupancy.

NTT's `resources.execution_groups` is a generic projection: it reports `core=dataflow-radix4`, `data_time=1`, `live_shared_bytes=0` for all cells, with `compiler_resources_known=false`. These values do not replace the explicit resolved per-stage mapping or establish actual resident grid/shared use. In particular, zero reported shared bytes must not be interpreted as the shared-radix4 leaf using no shared memory. The matched B16 profile below confirms the readiness audit's 32800 bytes/CTA dynamic storage for shared-radix4 and zero dynamic shared for warp-radix2, and supplies physical launch/register counters without retroactively changing the generic projection.

## Rounded modulus provenance and identities

The [original attempted study](../../results/narrative_gap_20261027/m3/attempted_study.json) preserves NTT q=`576460756061519900`, produced by losing integer precision during JS configuration handling. That value is composite; all eight original NTT preflights were rejected, and no timing from them is included. The [corrected source study](../../configs/experiments/core_crossing_20261027.json) and [frozen NTT-only retry study](../../results/narrative_gap_20261027/m3_ntt_retry/study.json) retain q=`576460756061519873` as an exact integer. The parent mixed cohort remains `complete-with-gaps`; accepted FWHT rows are reused without rewriting its status or failed attempts.

| Artifact or code identity | SHA256 |
| --- | --- |
| Original attempted study / mixed cohort study identity | `e9b733229c6ebcd5311f58e6d15dc139573087a33bc56d08f10106fac32ba578` |
| Corrected four-request source study | `eb683b8e913680aba039b365649909e367d3c65ff07a91f0c12a667f338903c7` |
| Frozen NTT-only retry study | `accdc99771b194e8a0fa30258517bc5c181ad4d71466a7865ea376412d2b28f7` |
| Mixed parent timing, retaining FWHT only here | `b67c5b0670d631db77d024b270497292f71f01ee6c7bbb3064ca5ddca6ee0bc8` |
| Corrected NTT timing | `c880b85f04b598aa685b6ffbb7c2ef0333018fa9d78dcba55bb04ed20ece927d` |
| Controlled runner used by mixed FWHT parent, after nested-control repair | `210fcec35b4fd36372a0fe849c30caf4cc564b68ce1959ef0a9a310a6e41a6ea` |
| Controlled runner used by NTT retry, with CPU modulus validation | `6fa72cbeb58bf279abdace807f858e1dc50d8eb1e2992c236a62df503db3918e` |
| `cubutterfly_bench` binary, FWHT | `abeebc62cdbd4a8a34ff9f97d64bdce2000798d014870d81923912b787d4eb6e` |
| `cuntt_bench` binary, same in both identities | `93ca7db81fa80f3b8c1c2f724c064bf385abb7ddba1e4942059e9b27dbb3077f` |
| `src/ntt.cu`, read-only data-time confirmation | `db136b3c3d19ccfde7129fd8a73b964602bef37182698a52111dda8c256e2547` |

The older pre-nested-control runner hash `42f486006e4411721facb5e23dfd43b76ce4630124e31f46920c8f379fb32039` in readiness is not the runner used for either retained timing cohort. CPU modulus validation was added after the mixed attempt: it rejects float/bool/string moduli, word-range violations, composites and unsupported roots before preparation, reusing the existing primality helper. This report runs no new validation tests and makes no claim that the old attempt passed those later checks.

## Matched B16 NTT profile

The isolated four-cell profile release is complete: [manifest](../../results/narrative_gap_20261027/m3_ntt_profile/release/profile/manifest.json), `status=complete`, `failures=[]`, same GPU UUID and benchmark binary as the NTT timing journal. Only B16 was selected (`selected_case_count=1`, `population_case_count=2`, `full_population=false`); this is not a B4 or FWHT profile. Earlier interrupted attempts outside `release/` are not used. Each capture runs one benchmark process with warmup 0 / repeat 1, unlike ordinary timing's 100 / 100. NCU 2024.1.0 uses kernel replay (three passes), `--cache-control none` and `--clock-control base`; it warns about uncontrolled caches. NSYS is 2023.4.4.54. These are diagnostic captures, not replacement performance trials.

Each NCU CSV and NSYS kernel summary contains exactly one main pipeline kernel, not a separate setup kernel. Initial ready/trace clearing and cooperative grid synchronization are inside that kernel (`src/ntt.cu:3044` and `:3191`); their work is not separable from its counters. NSYS additionally records five HtoD copies (167.756 decimal MB total, rounded report value) and one DtoH copy (134.218 MB) in every case. Those whole-process transfers include setup/input/output, not operator-only device traffic; they are excluded from the main-kernel DRAM table and never summed as complete HBM demand. The captures have no operator-only NVTX range.

Kernel abbreviations: **S** = `hierarchical_streaming_10x10_kernel<unsigned long,4>`; **G** = `hierarchical_homogeneous_two_level_kernel<unsigned long,10,10,4>`; **W** = `hierarchical_homogeneous_warp_two_level_kernel<unsigned long,10,10,8>`. The raw CSVs retain full signatures. All blocks are `(256,1,1)`; grids below are `(x,1,1)`.

| B16 cell | Kernel | Grid x | Registers/thread | Shared bytes/block: dynamic + static + driver = total | Ordinary median ms | NSYS kernel ms | NCU kernel ms |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| shared-r4-dt1 | S | 216 | 48 | 32800 + 16 + 1024 = 33840 | 1.620839 | 2.554910 | 2.599648 |
| shared-r4-dt2 | G | 216 | 52 | 32800 + 16 + 1024 = 33840 | 1.855611 | 2.621214 | 2.954848 |
| warp-r2-dt1 | W | 108 | 239 | 0 + 0 + 1024 = 1024 | 3.881431 | 5.437625 | 6.370560 |
| warp-r2-dt2 | W | 108 | 239 | 0 + 0 + 1024 = 1024 | 4.261048 | 6.537995 | 7.284320 |

Shared totals are `launch__shared_mem_per_block`, not just application storage. Allocated shared is 33920/1024 bytes for shared/warp. `launch__occupancy_limit_registers` is 5/4/1/1 blocks and `launch__occupancy_limit_shared_mem` is 4/4/8/8; these are reported resource ceilings, not observed resident-block counts. The cooperative launch actually has two CTAs/SM for shared and one for warp (108 SMs). This directly contradicts treating the generic B16 `grid_ctas=16384` or zero shared projection as physical launch resources.

| B16 cell | DRAM read bytes | DRAM write bytes | L2 sector hit % | Active warps % | Barrier stall % | Long-scoreboard stall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared-r4-dt1 | 467927168 | 277344896 | 64.71 | 23.88 | 3.69 | 38.77 |
| shared-r4-dt2 | 419321472 | 273916160 | 68.21 | 23.95 | 3.51 | 42.11 |
| warp-r2-dt1 | 648571008 | 437813120 | 73.64 | 12.50 | 0.02 | 13.47 |
| warp-r2-dt2 | 803514112 | 571881600 | 71.37 | 12.50 | 0.02 | 14.34 |

Exact metric names are `dram__bytes_read.sum`, `dram__bytes_write.sum`, `lts__t_sector_hit_rate.pct`, `sm__warps_active.avg.pct_of_peak_sustained_active`, `smsp__warp_issue_stalled_barrier_per_warp_active.pct`, and `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`. Active-warps percentage is the measured active-cycle metric, not a theoretical occupancy claim; the two stall percentages are per active warp, not additive fractions of elapsed runtime. DRAM values are main-kernel counters under replay, not a breakdown of semantic input/boundary/output or a whole-process HBM accounting. No spills/local-memory counters were requested; high registers do not establish spilling.

### What this explains, and what it does not

The shared-r4 dt contrast is **not a matched wrapper implementation**. The guard at `src/ntt.cu:6942` selects G whenever either role's dt differs from one; dt1 falls through to S at `:7253`. S removes the runtime packet loop. Both wrappers call the same arithmetic templates: producer `hierarchical_resident_radix4<false,Word,10,4>` at `:3142` / `:3316`, consumer `<true,Word,10,4>` at `:3114` / `:3282`. Thus the arithmetic primitive is held fixed, but wrapper specialization and traversal change together. Warmup is not a dispatch input: `execute` only repeats launch in its warmup loop (`:8332`); the same explicit mapping selects the wrapper in ordinary and profiled runs.

For shared-r4 dt2, registers rise 48->52, but active warps are essentially unchanged and both DRAM byte counters fall while L2 hit rate rises. Long-scoreboard stall percentage rises. The data therefore do not support an explanation that dt2 is slower because it necessarily increases DRAM traffic, shared-memory usage or reduces measured active warps. Runtime loop/indexing and changed scheduling are plausible contributors, but this profile cannot separate them from each other or establish the critical-path cause.

For warp-r2 dt2, the kernel symbol, register count, grid, shared memory and active-warps percentage are unchanged. Read/write DRAM counters rise and L2 hit rate falls; long-scoreboard percentage rises slightly while barrier percentage remains 0.02. That supports an association between the traversal change and worse observed traffic/locality in this capture, not proof that traffic alone causes the ordinary timing slowdown. Relative to shared-r4, warp-r2 has much higher register use, a register ceiling of one CTA/SM, half the measured active-warps percentage and more DRAM traffic, despite lower barrier/long-scoreboard percentages. These observations identify resource/traffic differences between the implemented cores; they do not isolate radix, derive a bottleneck from a single counter, or prove that eliminating shared memory is beneficial.

A future strict shared-r4 generic dt1/dt2 control needs both points routed through the **existing** generic launcher, retaining the same local arithmetic leaf. No public force-generic switch was found in the bounded launch/CLI inspection; an explicit test-only dispatch override or direct harness would be needed, with its identity recorded. Existing generic dt2/dt4 would avoid this dispatch boundary but answer a different contrast. No runtime change or new experiment was made here.

CPU-only extraction used `nsys stats --sqlite <temporary-file> --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv <release-report>` for each variant. Temporary SQLite exports were removed on completion; original reports/results were unchanged. The tables transcribe CSV values without imputation or aggregation across profilers.

| Profile artifact (under release/profile) | SHA256 |
| --- | --- |
| `manifest.json` | `a39f3a95855935e687f8948b2bb554709eaec523c6a700660ccbfae4628cfd39` |
| `shared-r4-dt1/ncu.csv` | `bfd758b61793c073fcf1f494e1d44743d2c6bf43607eb5f3580fa382953e3319` |
| `shared-r4-dt2/ncu.csv` | `ba5bf7f80fa6820cfde897c17c2c3fd72f005eeb45d2e61da44fd264e0392bf0` |
| `warp-r2-dt1/ncu.csv` | `337ff40a6b4268c8ce69963b912d42a96b27a06ea98bcb89bac1489e6c6cfd5e` |
| `warp-r2-dt2/ncu.csv` | `1f066fb9edeaedb3875277315c10554a6e1724122fb610c07a7b0fba13ae40f6` |
| `shared-r4-dt1/nsys.nsys-rep` | `3fa3ea0a48a2698237e13db07a6ffd49b70a42e8419ed06d7c3dff2457ca4eef` |
| `shared-r4-dt2/nsys.nsys-rep` | `6e55784686ed6828db15cf92f170f7bf7f29f9b7d2b2c3c9acc22cc89041ca6e` |
| `warp-r2-dt1/nsys.nsys-rep` | `5c4a6744615eba7f9515e303cb62188f4865f6865359ff64d27cf497147686d8` |
| `warp-r2-dt2/nsys.nsys-rep` | `f4ffe9abc96890b86a38aabc88e602fe6e31755250b054ba40136a130d94545a` |

Variant paths in this table are relative to `ntt-word64-n20-b16-core-data-time/`; no earlier attempt is substituted. The profile runner hash recorded by the manifest is `09b8805f3148ce0c2076e52a535a39383c5f4590d6f3e159d106a5c0d3fdf8e1`; controlled runner remains `6fa72cbeb58bf279abdace807f858e1dc50d8eb1e2992c236a62df503db3918e` and source timing hash remains `c880b85f04b598aa685b6ffbb7c2ef0333018fa9d78dcba55bb04ed20ece927d`.

## Remaining evidence boundary

The finite crossing now has 16 measured cells and 48 raw timed trials, with exact NTT preflight evidence and a restricted FWHT numerical record. Its organization deltas are observable on both core choices, but the tested stronger settings provide no median win. The designs are explicit interventions; neither c1/dt1 nor c4/dt2 is established as an optimized conventional-versus-hybrid comparator. No finite-budget re-search or independent selected-plan confirmation was performed here.

The B16 NTT matched capture now supplies physical launch/resources and selected traffic/stall counters. It does not supply repeated counter uncertainty, spills, instruction mix, per-role timing/readiness waits, a matched generic shared dt1/dt2 pair, B4 profiles or FWHT profiles. Profiler timing is visibly different from ordinary timing and cannot replace it. Full FWHT floating error distributions and the existing acceptance policy also remain open. These gaps prevent a unique causal explanation and remain separate from the completed negative timing result; no universal causal or performance claim follows from this bounded table.
