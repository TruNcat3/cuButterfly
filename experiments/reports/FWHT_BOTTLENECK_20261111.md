# FWHT comparison and bottleneck checkpoint — 11 November 2026

The historical FWHT lead and the new FP16 losses describe different arithmetic,
baselines and builds. The FP16 losses remain real complete-call observations for
the named DEFAULT selections. This checkpoint audits existing measurements and
CPU-side source/binary evidence; it adds no GPU timing or NCU capture. The GPU is
currently busy. Static resources identify hypotheses, not a measured bottleneck.

## Historical comparison contracts

Ratios below are competitor/cuButterfly; values below one are losses.

| Evidence | Arithmetic and requests | Interface and result | Identity |
| --- | --- | --- | --- |
| Main66 frozen study | FP32 storage/compute; ten Dao pairs: five forward/five normalized inverse; N256/B1, N4096/B32, N32768/B1, plus rank2/3/4/8 | Supplied output, including Dao output adaptation; GM 1.474332×, 9/10 nominal wins | A100 80GB PCIe UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`; library `fac10905…` |
| November 7 return-contract audit | Same ten FP32 requests; warp-register provider | Supplied-output GM 1.236–1.242×; independent-return GM 0.936–0.938× across three processes | Same UUID as main66; library `775a85c8…` |
| November 7 native repair | Same ten FP32 requests; fixed original mappings, packed-layout repair | Supplied-output GM 1.320–1.330×, 10/10 wins each process; independent-return GM 1.013–1.031× | A100 80GB PCIe UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`; repaired library `41f43216…` |
| November 11 HadaCore probes | Native FP16; N256/4096/32768 × B1/16/256 forward, plus two controls | Supplied-output FP16 competition; six medium/large forward requests lose, detailed below | Same UUID as repair; acceptance library `775a85c8…`, not repaired `41f43216…` |

Out-of-place comparisons use 10 warmups, 20 complete calls per CUDA Graph and
the median of seven replays. Required GPU copies/scales are timed; setup,
compilation and CPU verification are excluded. Independent-return Graphs retain
20 separate outputs; supplied-output Graphs reuse one buffer. The interfaces
therefore cannot be subtracted to estimate pure copy cost. The new in-place
control uses single-call events with reset outside timing and stays separate.

The frozen snapshot contains FP32/FP64 FWHT requests, but no matching native
FP16 N4096/N32768 history. The FP16 acceptance report's similarly sized performance
rows concern FFT. Within this audited evidence, no old FP16 FWHT record establishes
a same-contract version regression.

## Observed FP16 weak region

These are retained measurements, not new runs or a calibrated aggregate.

| N | B | cuButterfly µs | HadaCore µs | Hada/cu | cu/Hada latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 14.5408 | 4.5056 | 0.310 | 3.23× |
| 4096 | 16 | 15.3088 | 4.9152 | 0.321 | 3.11× |
| 4096 | 256 | 20.2240 | 10.3424 | 0.511 | 1.96× |
| 32768 | 1 | 111.6672 | 14.1312 | 0.127 | 7.90× |
| 32768 | 16 | 117.4016 | 16.7936 | 0.143 | 6.99× |
| 32768 | 256 | 393.0624 | 67.7888 | 0.172 | 5.80× |

Three layers must remain distinct:

- **Core:** cuButterfly native FP16 rounds each radix-2 addition/subtraction stage;
  its radix-4 codelet retains that dependency tree. The prepared CPU reference is
  `FP16-per-stage-rounded-Sylvester`. HadaCore uses FP16 MMA, internally normalized
  subtransform coefficients rounded to FP16, then a timed FP32 rescale and FP16
  writeback. It is not stage-rounding equivalent. HadaCore relative L2 error is
  about 0.00089–0.00104 here; N32768/B1 cuButterfly matches its prepared reference
  exactly. Both retain error-report-only status.
- **Selection and organization:** current DEFAULT selects scalar/shared
  `temporal-tile-radix4`, one temporal stage and 128 threads. N32768 uses `[15]`.
  The selection reason explicitly reports `hardware-rate-prior` and unproven
  timing-protocol compatibility. This establishes the actual public selection's
  weakness, not the best executable mapping's frontier or a controlled dataflow
  ablation.
- **Interface:** both new out-of-place sides write supplied output, and HadaCore's
  required normalization adaptation is timed. The losses cannot be dismissed
  by substituting the historical independent-return contract.

## Static resource evidence and bounded next checks

CPU inspection of the actual `775a85c8…` binary gives 16 registers/thread and zero
stack/local storage for native FP16 temporal radix-4 at N256/4096/32768. Dynamic
shared memory is N×2 bytes. HadaCore FP16 uses 256 threads, 39 registers/thread and
8 KiB shared at N4096; at N32768 it uses 512 threads, 127 registers/thread and
64 KiB shared. Both implementations launch grid=B, one CTA per transform.

These facts do not support a register-spill explanation. Small batch can constrain
both implementations' CTA count, but cannot alone explain the losses retained at
B256 or their size. Core instruction throughput, shared exchange and block shape
remain hypotheses until matched runtime evidence is available. Temporal candidate
generation currently includes 128/256/64/32 threads; the kernel can accept
512/1024. Whether those larger blocks improve this workload remains unmeasured.

The new [trace harness](../../scripts/trace_fwht_comparator.py) prepares five fixed
input cases with explicit mappings for controlled profiling. It is not automatic
library comparison evidence. No current NCU trace has been collected, so this
checkpoint reports no measured stall, occupancy or bandwidth bottleneck.

CPU preparation is complete for 24 bounded diagnostic jobs: ten original-point
timings, eight 256/512-thread variants with all other mapping fields held fixed,
four NCU captures and two Nsight Systems captures. The local idle executor has a
30-minute window, stops on foreign compute work or a failed measurement, and
preserves incomplete status if the card remains busy. Its status is recorded at
`results/fwht_trace_20261111/idle-executor-status.json`. The quiet 4 MiB Xorg
graphics context is explicitly permitted; zero utilization and the idle-memory
limit are still required. Prepared jobs are not measured results.

There is a separate **same-FP32 historical speed discrepancy** worth retaining:
main66 N32768/B1 forward supplied-output cuButterfly is 6.8608 µs, versus 16.522 µs
in the repaired November 7 report (about 2.41× longer). Dao is about 18.125 versus
18.165 µs. The old mapping is `shared-iterative-radix2`, partition `[8,7]`, 256
threads; the later report uses the warp-register provider. Builds, mappings and
device UUIDs differ. This is a selection/path regression hypothesis requiring
matched replay, not an established regression and not explained away by the new
FP16 precision scope.

`results/fwht_trace_20261111/fp32_replay_manifest.json` preserves both complete
mappings, their common semantic contract, prepared input and a common repaired
library for a future controlled replay. The historical library path still exists
but its current checksum differs from the frozen record; it cannot authenticate
the historical binary. Import and execution of the old shared mapping in the
common library remain GPU checks, not completed compatibility claims.

## Audit sources

- [Frozen main66 snapshot](../snapshots/current_refresh/snapshot.json):
  `identities.main66`, `operators.requests/pairs`; N32768/B1 forward case
  `dbad6edfd640e16740f5017114c46c537da74f8fe5707dd9559a9f529f5dad51`.
- [Current evidence](CURRENT_EVIDENCE_20261102.md),
  [return-contract follow-up](../../docs/a100_evidence_followup_20261107.md), and
  [native repair](../../docs/fwht_native_repair_20261107.md) retain their build,
  hardware, interface and numerical-reference boundaries.
- [External comparator checkpoint](EXTERNAL_COMPARATORS_20261111.md) and
  [FP16 acceptance](FP16_PUBLIC_ACCEPTANCE_20261105.md).
- Local raw root `/home/wt/git/cuButterfly/results/comparator_extension_20261111/hadacore11`:
  `rows/fwht-fp16-n32768-b1.json` records `reference_preparation`, `mapping`,
  `selection_reason`, baseline arithmetic metadata, errors and timing samples;
  the other five named forward rows retain the remaining table entries.
  Binary-resource inspection is static evidence from this investigation, not a
  substitute for the pending NCU artifacts.
- Local static evidence root `results/fwht_trace_20261111/static` retains the
  actual library checksums, resource dumps and selected disassembly. HadaCore
  instruction summaries include FP16 symbols only. Static instruction counts
  include alternative control-flow paths and are not dynamic execution counts.
